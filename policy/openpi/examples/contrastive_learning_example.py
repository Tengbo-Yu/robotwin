#!/usr/bin/env python3
"""
OpenPI对比学习示例

本示例展示如何在Pi0模型中启用基于任务类别的对比学习功能。
对比学习能够学习图像和文本特征之间的对应关系，提高模型的泛化能力。

任务分类:
- 0-34: 类别0 
- 35-96: 类别1
- 97-138: 类别2
- 139-169: 类别3
- 170-199: 类别4
- 200-209: 类别5
"""

import jax
import jax.numpy as jnp
import numpy as np
from openpi.models.pi0 import Pi0Config, Pi0
from openpi.models import model as _model


def create_contrastive_learning_config():
    """创建启用对比学习的Pi0配置"""
    config = Pi0Config(
        # 基本配置
        action_dim=14,
        action_horizon=50,
        max_token_len=48,
        
        # 启用对比学习
        enable_contrastive_learning=True,
        contrastive_loss_weight=0.1,      # 对比学习损失权重
        contrastive_temperature=0.07,     # 温度参数，控制相似度分布的锐利程度
        contrastive_projection_dim=256,   # 对比学习投影维度
        
        # 可选：启用双臂分别降噪
        dual_arm_separate_denoise=False,
    )
    return config


def create_dummy_batch(batch_size=8):
    """创建一个模拟批次数据，包含不同任务类别"""
    
    # 创建模拟的观察数据
    image_shape = (batch_size, 224, 224, 3)
    images = {
        "base_0_rgb": jnp.ones(image_shape, dtype=jnp.float32),
        "left_wrist_0_rgb": jnp.ones(image_shape, dtype=jnp.float32),
        "right_wrist_0_rgb": jnp.ones(image_shape, dtype=jnp.float32),
    }
    
    image_masks = {
        "base_0_rgb": jnp.ones(batch_size, dtype=jnp.bool_),
        "left_wrist_0_rgb": jnp.ones(batch_size, dtype=jnp.bool_),
        "right_wrist_0_rgb": jnp.ones(batch_size, dtype=jnp.bool_),
    }
    
    # 创建模拟的状态和动作数据
    state = jnp.ones((batch_size, 14), dtype=jnp.float32)
    actions = jnp.ones((batch_size, 50, 14), dtype=jnp.float32)
    
    # 创建模拟的tokenized prompt
    tokenized_prompt = jnp.ones((batch_size, 48), dtype=jnp.int32)
    tokenized_prompt_mask = jnp.ones((batch_size, 48), dtype=jnp.bool_)
    
    # 创建不同任务类别的task_indices
    # 确保batch中包含不同类别的任务
    task_indices = jnp.array([
        5,    # 类别0 (0-34)
        15,   # 类别0 (0-34) 
        45,   # 类别1 (35-96)
        65,   # 类别1 (35-96)
        120,  # 类别2 (97-138)
        150,  # 类别3 (139-169)
        180,  # 类别4 (170-199)
        205,  # 类别5 (200-209)
    ])
    
    observation = _model.Observation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
    )
    
    return observation, actions, task_indices


def train_with_contrastive_learning():
    """演示如何使用对比学习进行训练"""
    
    print("创建启用对比学习的Pi0配置...")
    config = create_contrastive_learning_config()
    
    print("初始化模型...")
    rng = jax.random.PRNGKey(42)
    model = config.create(rng)
    
    print("创建模拟数据...")
    observation, actions, task_indices = create_dummy_batch()
    
    print(f"任务索引: {task_indices}")
    print(f"对应的任务类别: {[_get_task_category_cpu(idx) for idx in task_indices]}")
    
    print("\n开始训练步骤...")
    
    # 计算包含对比学习的损失
    train_rng = jax.random.PRNGKey(123)
    loss = model.compute_loss(
        train_rng, 
        observation, 
        actions, 
        train=True,
        task_indices=task_indices
    )
    
    print(f"训练损失形状: {loss.shape}")
    print(f"平均损失: {jnp.mean(loss):.4f}")
    
    # 评估模式（不计算对比学习损失）
    eval_loss = model.compute_loss(
        train_rng, 
        observation, 
        actions, 
        train=False,
        task_indices=task_indices
    )
    
    print(f"评估损失形状: {eval_loss.shape}")
    print(f"评估平均损失: {jnp.mean(eval_loss):.4f}")
    
    # 生成动作（推理）
    print("\n生成动作...")
    sample_rng = jax.random.PRNGKey(456)
    predicted_actions = model.sample_actions(sample_rng, observation, num_steps=10)
    
    print(f"预测动作形状: {predicted_actions.shape}")
    print(f"预测动作范围: [{jnp.min(predicted_actions):.3f}, {jnp.max(predicted_actions):.3f}]")


def _get_task_category_cpu(task_index):
    """CPU版本的任务类别获取函数，用于打印"""
    if 0 <= task_index <= 34:
        return 0
    elif 35 <= task_index <= 96:
        return 1
    elif 97 <= task_index <= 138:
        return 2
    elif 139 <= task_index <= 169:
        return 3
    elif 170 <= task_index <= 199:
        return 4
    elif 200 <= task_index <= 209:
        return 5
    else:
        return -1


def analyze_task_distribution():
    """分析任务分布情况"""
    print("任务类别分布分析：")
    categories = [
        (0, 34, "类别0"),
        (35, 96, "类别1"), 
        (97, 138, "类别2"),
        (139, 169, "类别3"),
        (170, 199, "类别4"),
        (200, 209, "类别5"),
    ]
    
    for start, end, name in categories:
        count = end - start + 1
        print(f"{name}: 任务{start}-{end} (共{count}个任务)")
    
    total_tasks = sum(end - start + 1 for start, end, _ in categories)
    print(f"\n总任务数: {total_tasks}")


def contrastive_learning_tips():
    """对比学习使用技巧"""
    print("\n对比学习使用技巧：")
    print("1. 超参数调优：")
    print("   - contrastive_loss_weight: 控制对比学习损失的权重 (建议: 0.01-0.5)")
    print("   - contrastive_temperature: 控制相似度分布锐利程度 (建议: 0.05-0.2)")
    print("   - contrastive_projection_dim: 投影维度 (建议: 128-512)")
    
    print("\n2. 训练策略：")
    print("   - 确保每个batch包含不同类别的任务")
    print("   - 可以使用渐进式训练：先训练主任务，后添加对比学习")
    print("   - 监控对比学习损失是否收敛")
    
    print("\n3. 故障排除：")
    print("   - 如果对比学习损失过大，降低contrastive_loss_weight")
    print("   - 如果对比学习没有效果，检查task_indices是否正确传递")
    print("   - 确保数据加载器包含task_index字段")


if __name__ == "__main__":
    print("OpenPI对比学习示例")
    print("=" * 50)
    
    # 分析任务分布
    analyze_task_distribution()
    
    print("\n" + "=" * 50)
    
    # 演示训练过程
    train_with_contrastive_learning()
    
    print("\n" + "=" * 50)
    
    # 显示使用技巧
    contrastive_learning_tips()
    
    print("\n对比学习功能配置完成！") 