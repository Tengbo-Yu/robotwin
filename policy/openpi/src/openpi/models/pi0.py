import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.models import CL1 as _contrastive  # 导入对比学习模块
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


def get_task_category(task_index):
    """根据task_index获取任务类别"""
    task_index = jnp.asarray(task_index)
    
    # 定义任务类别范围
    categories = jnp.array([
        (0, 34),    # 类别0: 0-34
        (35, 96),   # 类别1: 35-96
        (97, 138),  # 类别2: 97-138
        (139, 169), # 类别3: 139-169
        (170, 199), # 类别4: 170-199
        (200, 209), # 类别5: 200-209
    ])
    
    # 创建条件检查
    conditions = []
    for i, (start, end) in enumerate(categories):
        conditions.append((task_index >= start) & (task_index <= end))
    
    # 使用jnp.select来根据条件选择类别
    category = jnp.select(conditions, jnp.arange(len(categories)), default=-1)
    
    return category


def infonce_loss(image_features, text_features, task_indices, temperature=0.07):
    """
    基于任务类别的InfoNCE对比学习损失
    
    Args:
        image_features: [B, D] 图像特征
        text_features: [B, D] 文本特征  
        task_indices: [B] 任务索引
        temperature: 温度参数
        
    Returns:
        对比学习损失
    """
    batch_size = image_features.shape[0]
    
    # 获取任务类别
    task_categories = jax.vmap(get_task_category)(task_indices)
    
    # 计算特征相似度矩阵
    # 归一化特征
    image_features = image_features / (jnp.linalg.norm(image_features, axis=-1, keepdims=True) + 1e-8)
    text_features = text_features / (jnp.linalg.norm(text_features, axis=-1, keepdims=True) + 1e-8)
    
    # 计算相似度矩阵 [B, B]
    similarity_matrix = jnp.dot(image_features, text_features.T) / temperature
    
    # 创建任务类别掩码矩阵 [B, B]
    # 相同类别的任务为正例(True)，不同类别为负例(False)
    task_category_matrix = task_categories[:, None] == task_categories[None, :]
    
    # 创建对角线掩码，避免自己和自己对比
    diagonal_mask = jnp.eye(batch_size, dtype=bool)
    
    # 正例掩码：相同任务类别但不是自己
    positive_mask = task_category_matrix & (~diagonal_mask)
    
    # 计算InfoNCE损失
    # 对于每个样本，计算其与所有样本的相似度
    exp_sim = jnp.exp(similarity_matrix)
    
    # 分母：与所有样本的相似度之和（除了自己）
    denominator = jnp.sum(exp_sim * (~diagonal_mask), axis=1)
    
    # 分子：与正例的相似度之和
    numerator = jnp.sum(exp_sim * positive_mask, axis=1)
    
    # 避免除零，如果没有正例则损失为0
    valid_samples = jnp.sum(positive_mask, axis=1) > 0
    
    # 计算损失
    loss = -jnp.log(numerator / (denominator + 1e-8) + 1e-8)
    loss = jnp.where(valid_samples, loss, 0.0)
    
    # 返回平均损失
    return jnp.mean(loss)


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 14
    action_horizon: int = 50
    max_token_len: int = 48
    
    # 新增：是否对双臂分别进行降噪处理
    dual_arm_separate_denoise: bool = False
    
    # 第一个对比学习模块（CL1）相关配置
    enable_contrastive_learning_cl1: bool = False
    contrastive_loss_weight_cl1: float = 0.1
    contrastive_temperature_cl1: float = 0.07
    contrastive_projection_dim_cl1: int = 256

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        # 匹配所有LLM参数的正则表达式
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant: # 如果lora，冻结所有和llm有关的参数，同时微调VLM和动作专家的参数
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


class Pi0(_model.BaseModel):
    def __init__(self, config: Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
            )
        )
        llm.lazy_init(rngs=rngs, method="init")
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        # 动作投影
        self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
        
        # ----------------update----------------
        # 存储配置以便在其他方法中使用
        self.config = config
        
        # 如果启用双臂分别降噪，添加额外的投影层
        if config.dual_arm_separate_denoise:
            # 假设action_dim=14，每个手臂7维
            single_arm_dim = config.action_dim // 2
            self.left_arm_in_proj = nnx.Linear(single_arm_dim, action_expert_config.width, rngs=rngs)
            self.left_arm_out_proj = nnx.Linear(action_expert_config.width, single_arm_dim, rngs=rngs)
            self.right_arm_in_proj = nnx.Linear(single_arm_dim, action_expert_config.width, rngs=rngs)
            self.right_arm_out_proj = nnx.Linear(action_expert_config.width, single_arm_dim, rngs=rngs)
        
        # 第一个对比学习模块（CL1）
        if config.enable_contrastive_learning_cl1:
            self.contrastive_module_cl1 = _contrastive.create_contrastive_learning_module(
                input_dim=paligemma_config.width,
                projection_dim=config.contrastive_projection_dim_cl1,
                rngs=rngs
            )
        
        # ----------------update----------------

    @at.typecheck
    def embed_prefix( # prefix 包含img instruction
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
                
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix( # suffix 包含state， noisy action，timestep
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # add a single state token
        state_token = self.state_proj(obs.state)[:, None, :]
        tokens.append(state_token)
        input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
        # image/language inputs do not attend to state or actions
        ar_mask += [True]

        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        
        # ----------------update----------------
        # 根据配置选择处理方式
        if self.config.dual_arm_separate_denoise:
            # 双臂分别处理模式
            single_arm_dim = self.config.action_dim // 2
            # 分离左臂和右臂动作
            left_arm_actions = noisy_actions[..., :single_arm_dim]  # 前7维
            right_arm_actions = noisy_actions[..., single_arm_dim:]  # 后7维
            
            # 分别投影左臂和右臂动作
            left_arm_tokens = self.left_arm_in_proj(left_arm_actions)
            right_arm_tokens = self.right_arm_in_proj(right_arm_actions)
            
            # mix timestep + action information using an MLP for each arm
            time_tokens_left = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            time_tokens_right = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            
            left_arm_time_tokens = jnp.concatenate([left_arm_tokens, time_tokens_left], axis=-1)
            left_arm_time_tokens = self.action_time_mlp_in(left_arm_time_tokens)
            left_arm_time_tokens = nnx.swish(left_arm_time_tokens)
            left_arm_time_tokens = self.action_time_mlp_out(left_arm_time_tokens)
            
            right_arm_time_tokens = jnp.concatenate([right_arm_tokens, time_tokens_right], axis=-1)
            right_arm_time_tokens = self.action_time_mlp_in(right_arm_time_tokens)
            right_arm_time_tokens = nnx.swish(right_arm_time_tokens)
            right_arm_time_tokens = self.action_time_mlp_out(right_arm_time_tokens)
            
            # 将左臂和右臂tokens按顺序添加
            tokens.append(left_arm_time_tokens)
            tokens.append(right_arm_time_tokens)
            input_mask.append(jnp.ones(left_arm_time_tokens.shape[:2], dtype=jnp.bool_))
            input_mask.append(jnp.ones(right_arm_time_tokens.shape[:2], dtype=jnp.bool_))
            # image/language/state inputs do not attend to action tokens
            ar_mask += [True] + ([False] * (self.action_horizon - 1))  # 左臂
            ar_mask += [True] + ([False] * (self.action_horizon - 1))  # 右臂
        # ----------------update----------------
        else:
            # 原始的整体处理模式
            # mix timestep + action information using an MLP
            action_tokens = self.action_in_proj(noisy_actions)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            tokens.append(action_time_tokens)
            input_mask.append(jnp.ones(action_time_tokens.shape[:2], dtype=jnp.bool_))
            # image/language/state inputs do not attend to action tokens
            ar_mask += [True] + ([False] * (self.action_horizon - 1))
            
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False, task_indices: at.Int[at.Array, "b"] | None = None
    ) -> at.Float[at.Array, "*b ah"] | dict[str, at.Float[at.Array, "*b ah"]]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape) # 生成随机噪声
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions # 对真实动作添加噪声
        u_t = noise - actions # 噪声减去真实动作

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions
        )
        
        # ----------------update----------------
        # 根据配置选择输出处理方式
        if self.config.dual_arm_separate_denoise:
            # 双臂分别处理模式
            single_arm_dim = self.config.action_dim // 2
            # suffix_out包含: [state_token, left_arm_tokens, right_arm_tokens]
            # 提取左臂和右臂的输出（跳过state_token）
            left_arm_start = 1  # 跳过state token
            left_arm_end = left_arm_start + self.action_horizon
            right_arm_start = left_arm_end
            right_arm_end = right_arm_start + self.action_horizon
            
            left_arm_out = suffix_out[:, left_arm_start:left_arm_end]
            right_arm_out = suffix_out[:, right_arm_start:right_arm_end]
            
            # 分别投影左臂和右臂的输出
            left_arm_v_t = self.left_arm_out_proj(left_arm_out)
            right_arm_v_t = self.right_arm_out_proj(right_arm_out)
            
            # 合并左臂和右臂的预测
            v_t = jnp.concatenate([left_arm_v_t, right_arm_v_t], axis=-1)
            
        # ----------------update----------------
        else:
            # 原始的整体处理模式
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        # 计算主要的扩散损失
        diffusion_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        
        # 如果启用CL1对比学习且在训练模式下，添加对比学习损失
        contrastive_loss_cl1 = 0.0
        if self.config.enable_contrastive_learning_cl1 and train and task_indices is not None:
            # 在此处直接提取对比学习特征
            image_features, text_features = self._extract_contrastive_features_cl1(observation)
            
            if image_features is not None and text_features is not None:
                # 计算CL1对比学习损失
                contrastive_loss_cl1 = self.contrastive_module_cl1.compute_contrastive_loss(
                    image_features, 
                    text_features,
                    task_indices,
                    temperature=self.config.contrastive_temperature_cl1
                )
                
                # 将对比学习损失添加到主损失中
                # 由于diffusion_loss的形状是[batch, action_horizon]，我们需要将contrastive_loss广播到相同形状
                contrastive_loss_expanded = jnp.broadcast_to(
                    contrastive_loss_cl1, diffusion_loss.shape
                )
                
                total_loss = diffusion_loss + self.config.contrastive_loss_weight_cl1 * contrastive_loss_expanded
                
                # 返回详细损失信息用于wandb记录
                return {
                    "total_loss": total_loss,
                    "diffusion_loss": diffusion_loss,
                    "contrastive_loss_cl1": jnp.broadcast_to(contrastive_loss_cl1, diffusion_loss.shape)
                }
        
        # 如果没有对比学习，只返回扩散损失
        return diffusion_loss
    
    def _extract_contrastive_features_cl1(self, obs: _model.Observation):
        """提取用于CL1对比学习的特征，在compute_loss方法内部调用"""
        if not hasattr(self, 'contrastive_module_cl1'):
            return None, None
            
        # 收集图像tokens和masks
        image_tokens_list = []
        image_masks_list = []
        
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            image_tokens_list.append(image_tokens)
            image_masks_list.append(obs.image_masks[name])
        
        # 提取图像特征
        image_features = None
        if image_tokens_list:
            image_features = self.contrastive_module_cl1.extract_image_features(
                image_tokens_list, image_masks_list
            )
        
        # 提取文本特征
        text_features = None
        if obs.tokenized_prompt is not None:
            text_tokens = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            text_features = self.contrastive_module_cl1.extract_text_features(
                text_tokens, obs.tokenized_prompt_mask
            )
        
        return image_features, text_features

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache
            )
            assert prefix_out is None

            # ----------------update----------------
            # 根据配置选择输出处理方式
            if self.config.dual_arm_separate_denoise:
                # 双臂分别处理模式
                single_arm_dim = self.config.action_dim // 2
                # suffix_out包含: [state_token, left_arm_tokens, right_arm_tokens]
                # 提取左臂和右臂的输出（跳过state_token）
                left_arm_start = 1  # 跳过state token
                left_arm_end = left_arm_start + self.action_horizon
                right_arm_start = left_arm_end
                right_arm_end = right_arm_start + self.action_horizon
                
                left_arm_out = suffix_out[:, left_arm_start:left_arm_end]
                right_arm_out = suffix_out[:, right_arm_start:right_arm_end]
                
                # 分别投影左臂和右臂的输出
                left_arm_v_t = self.left_arm_out_proj(left_arm_out)
                right_arm_v_t = self.right_arm_out_proj(right_arm_out)
                
                # 合并左臂和右臂的预测
                v_t = jnp.concatenate([left_arm_v_t, right_arm_v_t], axis=-1)
            # ----------------update----------------
            else:
                # 原始的整体处理模式
                v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0