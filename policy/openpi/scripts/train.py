import dataclasses
import functools
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import optax
import tqdm_loggable.auto as tqdm
import wandb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)

def tsne_visualize(features, task_categories, step):
    """
    Visualize contrastive learning features using t-SNE
    
    Args:
        features: [B, D] normalized features
        task_categories: [B] task categories
        step: current training step
    """
    # 检查特征和类别
    print(f"\033[96m[TSNE] Features shape: {features.shape}, dtype: {features.dtype}\033[0m")
    print(f"\033[96m[TSNE] Categories shape: {task_categories.shape}, dtype: {task_categories.dtype}\033[0m")
    print(f"\033[96m[TSNE] Unique categories: {np.unique(task_categories)}\033[0m")
    
    # Convert to numpy
    features = np.array(features)
    task_categories = np.array(task_categories)
    
    # 检查特征是否包含NaN或无穷大
    if np.isnan(features).any() or np.isinf(features).any():
        print("\033[91m[TSNE] Warning: Features contain NaN or Inf values\033[0m")
        # 替换NaN和无穷大
        features = np.nan_to_num(features)
    
    # Sample features to avoid slow t-SNE computation with too many data points
    max_samples = 1000
    if len(features) > max_samples:
        print(f"\033[96m[TSNE] Sampling {max_samples} points from {len(features)} total points\033[0m")
        indices = np.random.choice(len(features), max_samples, replace=False)
        features = features[indices]
        task_categories = task_categories[indices]

    # t-SNE dimensionality reduction
    try:
        print("\033[96m[TSNE] Starting t-SNE computation...\033[0m")
        tsne = TSNE(n_components=2, random_state=0, init='pca', learning_rate='auto')
        features_2d = tsne.fit_transform(features)
        print("\033[96m[TSNE] t-SNE computation complete\033[0m")
    except Exception as e:
        print(f"\033[91m[TSNE] Error in t-SNE computation: {e}\033[0m")
        return

    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Use different colors for each category
    unique_categories = np.unique(task_categories)
    print(f"\033[96m[TSNE] Plotting {len(unique_categories)} unique categories\033[0m")
    
    for cat in unique_categories:
        idx = task_categories == cat
        plt.scatter(features_2d[idx, 0], features_2d[idx, 1], label=f'Category {cat}', alpha=0.7, s=40)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f't-SNE Visualization of Contrastive Features (Step {step})')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.tight_layout()
    
    # Upload to wandb
    try:
        wandb.log({"Contrastive_Features_tSNE": wandb.Image(plt)}, step=step)
        print("\033[96m[TSNE] Successfully logged to wandb\033[0m")
    except Exception as e:
        print(f"\033[91m[TSNE] Error logging to wandb: {e}\033[0m")
        
    plt.close()

def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions] | tuple[_model.Observation, _model.Actions, at.Int[at.Array, "b"]],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, task_indices: at.Int[at.Array, "b"] | None = None
    ):
        loss_result = model.compute_loss(rng, observation, actions, train=True, task_indices=task_indices)
        
        # 处理不同类型的损失返回值
        if isinstance(loss_result, dict):
            # 如果返回字典，使用total_loss作为主损失
            total_loss = jnp.mean(loss_result["total_loss"])
            diffusion_loss = jnp.mean(loss_result["diffusion_loss"])
            contrastive_loss_cl1 = jnp.mean(loss_result.get("contrastive_loss_cl1", 0.0))
            
            # 提取可视化数据
            cl1_features = loss_result.get("cl1_features", None)
            cl1_task_categories = loss_result.get("cl1_task_categories", None)
            
            return total_loss, {
                "diffusion_loss": diffusion_loss,
                "contrastive_loss_cl1": contrastive_loss_cl1,
                "cl1_features": cl1_features,
                "cl1_task_categories": cl1_task_categories
            }
        else:
            # 如果返回单一损失值（扩散损失）
            chunked_loss = loss_result
            total_loss = jnp.mean(chunked_loss)
            return total_loss, {
                "diffusion_loss": total_loss,
                "contrastive_loss_cl1": 0.0,
                "cl1_features": None,
                "cl1_task_categories": None
            }

    train_rng = jax.random.fold_in(rng, state.step)
    
    # 处理可能包含task_indices的批次
    if len(batch) == 3:
        observation, actions, task_indices = batch
    else:
        observation, actions = batch
        task_indices = None
        
        # 如果批次中没有task_indices但对比学习已启用，尝试生成虚拟task_indices用于测试
        if hasattr(config.model, 'enable_contrastive_learning_cl1') and config.model.enable_contrastive_learning_cl1:
            print("\033[93m[WARNING] Creating fake task_indices for testing contrastive learning\033[0m")
            batch_size = observation.state.shape[0]
            task_indices = jnp.zeros((batch_size,), dtype=jnp.int32)

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, detailed_losses), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(model, train_rng, observation, actions, task_indices)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "diffusion_loss": detailed_losses["diffusion_loss"],
        "contrastive_loss_cl1": detailed_losses["contrastive_loss_cl1"],
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    
    # 确保cl1_features和cl1_task_categories被正确传递
    if "cl1_features" in detailed_losses:
        info["cl1_features"] = detailed_losses["cl1_features"]
    if "cl1_task_categories" in detailed_losses:
        info["cl1_task_categories"] = detailed_losses["cl1_task_categories"]
    
    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        num_workers=config.num_workers,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    
    # 检查批次格式，确保包含task_indices
    print("\033[92m[DATA] Batch type:", type(batch), "length:", len(batch), "\033[0m")
    if len(batch) >= 3:
        print("\033[92m[DATA] Batch contains task_indices with shape:", batch[2].shape, "\033[0m")
    else:
        print("\033[91m[DATA] Batch does not contain task_indices, this will prevent contrastive learning\033[0m")
        # 检查数据配置是否包含task_index
        if hasattr(config.data, 'repack_transforms'):
            repack_dict = None
            for transform in config.data.repack_transforms.inputs:
                if hasattr(transform, 'mapping'):
                    repack_dict = transform.mapping
                    break
            if repack_dict:
                print("\033[92m[DATA] Repack mapping:", repack_dict, "\033[0m")
                if 'task_index' not in repack_dict:
                    print("\033[91m[DATA] task_index not found in repack mapping\033[0m")
    
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    # 检查模型是否启用了对比学习
    model = nnx.merge(train_state.model_def, train_state.params)
    if hasattr(config.model, 'enable_contrastive_learning_cl1'):
        print(f"\033[92m[MODEL] Contrastive learning enabled: {config.model.enable_contrastive_learning_cl1}\033[0m")
        if config.model.enable_contrastive_learning_cl1:
            if hasattr(model, 'contrastive_module_cl1'):
                print("\033[92m[MODEL] contrastive_module_cl1 exists in model\033[0m")
            else:
                print("\033[91m[MODEL] contrastive_module_cl1 NOT found in model despite being enabled\033[0m")
    else:
        print("\033[91m[MODEL] enable_contrastive_learning_cl1 not found in config\033[0m")

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    # 添加特征收集变量
    accumulated_features = []
    accumulated_categories = []
    accumulation_count = 0
    accumulation_target = 10  # 收集5个batch的数据
    is_collecting = False    # 是否正在收集数据的标志
    
    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )
    print("\033[91maction_dim: ", config.model.action_dim, "\033[0m")
    print("\033[91mdual_arm_separate_denoise: ", config.model.dual_arm_separate_denoise, "\033[0m") 
    print("\033[91menable_contrastive_learning_cl1: ", config.model.enable_contrastive_learning_cl1, "\033[0m")
    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items() if k not in ["cl1_features", "cl1_task_categories"])
            pbar.write(f"Step {step}: {info_str}")
            wandb.log({k: v for k, v in reduced_info.items() if k not in ["cl1_features", "cl1_task_categories"]}, step=step)
            infos = []
        
        # 检查是否到达可视化间隔点
        if hasattr(config, 'cl1_tsne_interval') and step % config.cl1_tsne_interval == 0 and step > 0:
            # 如果当前不在收集状态，则开始收集数据
            if not is_collecting:
                print(f"\033[92m[COLLECT] Starting to collect data for the next {accumulation_target} batches\033[0m")
                # 清空之前可能存在的数据
                accumulated_features = []
                accumulated_categories = []
                accumulation_count = 0
                is_collecting = True
        
        # 如果正在收集数据，则处理当前批次
        if is_collecting:
            cl1_features = info.get("cl1_features")
            cl1_task_categories = info.get("cl1_task_categories")
            
            if cl1_features is not None and cl1_task_categories is not None:
                try:
                    # 将特征和类别从设备中获取
                    cl1_features = jax.device_get(cl1_features)
                    cl1_task_categories = jax.device_get(cl1_task_categories)
                    
                    # 检查数据是否为NaN或无穷大
                    if not (np.isnan(cl1_features).any() or np.isinf(cl1_features).any() or 
                            np.isnan(cl1_task_categories).any() or np.isinf(cl1_task_categories).any()):
                        accumulated_features.append(cl1_features)
                        accumulated_categories.append(cl1_task_categories)
                        accumulation_count += 1
                        print(f"\033[96m[COLLECT] Added batch {accumulation_count}/{accumulation_target}, "
                              f"total samples: {sum(f.shape[0] for f in accumulated_features)}\033[0m")
                        
                        # 如果已收集足够的批次，进行可视化并重置收集状态
                        if accumulation_count >= accumulation_target:
                            try:
                                # 合并收集的特征和类别
                                combined_features = np.vstack(accumulated_features)
                                combined_categories = np.concatenate(accumulated_categories)
                                
                                print(f"\033[92m[VISUALIZE] Combined {accumulation_count} batches, "
                                      f"total samples: {combined_features.shape[0]}\033[0m")
                                
                                # 执行t-SNE可视化
                                tsne_visualize(combined_features, combined_categories, step)
                                
                                # 重置收集状态
                                is_collecting = False
                                print("\033[92m[COLLECT] Data collection completed and visualized\033[0m")
                            except Exception as e:
                                print(f"\033[91m[ERROR] Failed to visualize collected features: {e}\033[0m")
                                # 即使可视化失败，也重置收集状态以避免无限收集
                                is_collecting = False
                except Exception as e:
                    print(f"\033[91m[ERROR] Failed to process batch for collection: {e}\033[0m")
        
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
