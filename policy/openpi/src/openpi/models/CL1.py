"""
对比学习模块 - 基于任务类别的图像文本对比学习
支持根据task_index进行任务分类，并使用InfoNCE损失进行对比学习
"""

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import einops
from typing_extensions import override

from openpi.shared import array_typing as at


def get_task_category(task_index):
    """根据task_index获取任务类别
    
    Args:
        task_index: 任务索引
        
    Returns:
        任务类别 (0-5)，如果不在范围内返回-1
    """
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


class ContrastiveLearningModule(nnx.Module):
    """对比学习模块"""
    
    def __init__(self, input_dim: int, projection_dim: int, rngs: nnx.Rngs):
        """
        Args:
            input_dim: 输入特征维度
            projection_dim: 投影后的特征维度
            rngs: 随机数生成器
        """
        self.image_projection = nnx.Linear(input_dim, projection_dim, rngs=rngs)
        self.text_projection = nnx.Linear(input_dim, projection_dim, rngs=rngs)
        self.projection_dim = projection_dim
    
    def extract_image_features(self, image_tokens, image_masks):
        """
        从图像tokens中提取特征
        
        Args:
            image_tokens: 图像tokens列表，每个元素形状为[B, S, D]
            image_masks: 对应的mask列表，每个元素形状为[B]
            
        Returns:
            pooled_image_features: [B, D] 池化后的图像特征
        """
        pooled_features = []
        
        for tokens, mask in zip(image_tokens, image_masks):
            # 对图像tokens进行mask加权平均池化
            mask_expanded = einops.repeat(
                mask.astype(jnp.float32),
                "b -> b s",
                s=tokens.shape[1],
            )[:, :, None]
            
            masked_tokens = tokens * mask_expanded
            pooled_feature = jnp.sum(masked_tokens, axis=1) / (jnp.sum(mask_expanded, axis=1) + 1e-8)
            pooled_features.append(pooled_feature)
        
        # 平均所有图像特征
        if pooled_features:
            return jnp.mean(jnp.stack(pooled_features, axis=0), axis=0)
        else:
            return None
    
    def extract_text_features(self, text_tokens, text_mask):
        """
        从文本tokens中提取特征
        
        Args:
            text_tokens: [B, S, D] 文本tokens
            text_mask: [B, S] 文本mask
            
        Returns:
            pooled_text_features: [B, D] 池化后的文本特征
        """
        # 对文本tokens进行mask加权平均池化
        text_mask_expanded = text_mask.astype(jnp.float32)[:, :, None]
        masked_text_tokens = text_tokens * text_mask_expanded
        pooled_text_features = jnp.sum(masked_text_tokens, axis=1) / (jnp.sum(text_mask_expanded, axis=1) + 1e-8)
        
        return pooled_text_features
    
    def compute_contrastive_loss(self, image_features, text_features, task_indices, temperature=0.07):
        """
        计算对比学习损失
        
        Args:
            image_features: [B, D] 原始图像特征
            text_features: [B, D] 原始文本特征
            task_indices: [B] 任务索引
            temperature: 温度参数
            
        Returns:
            对比学习损失
        """
        if image_features is None or text_features is None:
            return 0.0
        
        # 投影到对比学习空间
        projected_image_features = self.image_projection(image_features)
        projected_text_features = self.text_projection(text_features)
        
        # 计算InfoNCE损失
        return infonce_loss(
            projected_image_features, 
            projected_text_features, 
            task_indices,
            temperature=temperature
        )


def create_contrastive_learning_module(input_dim: int, projection_dim: int, rngs: nnx.Rngs) -> ContrastiveLearningModule:
    """创建对比学习模块的工厂函数"""
    return ContrastiveLearningModule(input_dim, projection_dim, rngs)


# 用于测试的示例函数
def test_contrastive_learning():
    """测试对比学习模块的功能"""
    import jax.random as random
    
    # 测试参数
    batch_size = 8
    feature_dim = 256
    projection_dim = 128
    
    # 创建测试数据
    rng = random.key(42)
    rng, sub_rng1, sub_rng2, sub_rng3, sub_rng4 = random.split(rng, 5)
    
    # 模拟图像和文本特征
    image_features = random.normal(sub_rng1, (batch_size, feature_dim))
    text_features = random.normal(sub_rng2, (batch_size, feature_dim))
    
    # 模拟任务索引 (0-209范围)
    task_indices = random.randint(sub_rng3, (batch_size,), 0, 210)
    
    # 创建对比学习模块
    cl_module = create_contrastive_learning_module(
        input_dim=feature_dim,
        projection_dim=projection_dim,
        rngs=nnx.Rngs(sub_rng4)
    )
    
    # 计算对比学习损失
    loss = cl_module.compute_contrastive_loss(
        image_features, text_features, task_indices, temperature=0.07
    )
    
    print(f"测试任务索引: {task_indices}")
    print(f"任务类别: {jax.vmap(get_task_category)(task_indices)}")
    print(f"对比学习损失: {loss}")
    print("对比学习模块测试完成！")
    
    return loss


if __name__ == "__main__":
    test_contrastive_learning() 