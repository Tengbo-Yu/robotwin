# OpenPI对比学习指南

本指南介绍如何在OpenPI的Pi0模型中使用基于任务类别的对比学习功能。

## 概述

对比学习是一种自监督学习技术，通过学习图像和文本特征之间的对应关系来提高模型的泛化能力。在Pi0模型中，我们实现了基于任务类别的对比学习，相同类别的任务作为正例，不同类别的任务作为负例。

## 任务分类

根据您的需求，任务被分为以下6个类别：

- **类别0**: 任务0-34 (35个任务)
- **类别1**: 任务35-96 (62个任务)
- **类别2**: 任务97-138 (42个任务)
- **类别3**: 任务139-169 (31个任务)
- **类别4**: 任务170-199 (30个任务)
- **类别5**: 任务200-209 (10个任务)

总计210个任务，分为6个类别。

## 配置参数

在`Pi0Config`中添加以下对比学习相关参数：

```python
from openpi.models.pi0 import Pi0Config

config = Pi0Config(
    # 基本配置
    action_dim=14,
    action_horizon=50,
    max_token_len=48,
    
    # 启用对比学习
    enable_contrastive_learning=True,
    contrastive_loss_weight=0.1,      # 对比学习损失权重
    contrastive_temperature=0.07,     # 温度参数
    contrastive_projection_dim=256,   # 投影维度
)
```

### 参数说明

- `enable_contrastive_learning`: 是否启用对比学习功能
- `contrastive_loss_weight`: 对比学习损失在总损失中的权重 (建议范围: 0.01-0.5)
- `contrastive_temperature`: InfoNCE损失中的温度参数，控制相似度分布的锐利程度 (建议范围: 0.05-0.2)
- `contrastive_projection_dim`: 特征投影到对比学习空间的维度 (建议范围: 128-512)

## 使用方法

### 1. 数据准备

确保您的数据加载器包含`task_index`字段：

```python
# 在transforms.py中使用PromptFromLeRobotTask确保task_index被保留
transform = PromptFromLeRobotTask(tasks=your_task_mapping)
```

### 2. 模型训练

```python
import jax
from openpi.models.pi0 import Pi0Config

# 创建配置
config = Pi0Config(enable_contrastive_learning=True)
model = config.create(jax.random.PRNGKey(42))

# 训练时传入task_indices
loss = model.compute_loss(
    rng=train_rng,
    observation=observation,
    actions=actions,
    train=True,
    task_indices=task_indices  # 必须提供
)
```

### 3. 模型评估

```python
# 评估时可以不传入task_indices或设置train=False
eval_loss = model.compute_loss(
    rng=eval_rng,
    observation=observation,
    actions=actions,
    train=False
)
```

## 技术实现

### 特征提取

1. **图像特征**: 对每个图像视角的tokens进行mask加权全局平均池化
2. **文本特征**: 对tokenized prompt进行mask加权平均池化
3. **特征融合**: 多个图像视角的特征取平均

### InfoNCE损失

对比学习使用InfoNCE损失函数：

```
L = -log(sum(exp(sim(i,j)/τ) for j in positives) / sum(exp(sim(i,k)/τ) for k ≠ i))
```

其中：
- `sim(i,j)` 是图像特征i和文本特征j的余弦相似度
- `τ` 是温度参数
- `positives` 是相同任务类别的样本

### 正负例定义

- **正例**: 同一batch中相同任务类别的图像-文本特征对
- **负例**: 同一batch中不同任务类别的图像-文本特征对

## 训练策略

### 1. 渐进式训练

建议采用渐进式训练策略：

```python
# 阶段1: 只训练主任务
config_stage1 = Pi0Config(enable_contrastive_learning=False)

# 阶段2: 添加对比学习
config_stage2 = Pi0Config(
    enable_contrastive_learning=True,
    contrastive_loss_weight=0.01  # 从小权重开始
)
```

### 2. 批次构造

确保每个batch包含不同类别的任务：

```python
def create_balanced_batch(dataset, batch_size=32):
    # 确保batch中包含多个任务类别
    samples_per_category = batch_size // 6
    batch = []
    for category in range(6):
        category_samples = sample_from_category(dataset, category, samples_per_category)
        batch.extend(category_samples)
    return shuffle(batch)
```

### 3. 监控指标

训练时监控以下指标：

- 主要扩散损失
- 对比学习损失
- 总损失
- 不同任务类别间的特征相似度

## 超参数调优

### contrastive_loss_weight

- **0.01-0.05**: 轻微的对比学习约束
- **0.1-0.2**: 中等强度的对比学习
- **0.3-0.5**: 强对比学习约束

### contrastive_temperature

- **0.05**: 非常锐利的分布，强调最相似的样本
- **0.07**: 默认值，平衡锐利度和稳定性
- **0.1-0.2**: 更平滑的分布，考虑更多样本

### contrastive_projection_dim

- **128**: 轻量级投影，计算效率高
- **256**: 默认值，平衡表达能力和效率
- **512**: 更强的表达能力，但计算开销更大

## 故障排除

### 问题1: 对比学习损失过大

**症状**: 对比学习损失远大于主损失
**解决方案**: 
- 降低`contrastive_loss_weight`
- 增加`contrastive_temperature`

### 问题2: 对比学习没有效果

**症状**: 模型性能没有提升
**解决方案**:
- 检查`task_indices`是否正确传递
- 确保batch中包含多个任务类别
- 增加`contrastive_loss_weight`

### 问题3: 训练不稳定

**症状**: 损失震荡或发散
**解决方案**:
- 使用渐进式训练
- 降低学习率
- 检查批次构造是否平衡

## 性能优化

### 1. 内存优化

对比学习会增加额外的计算开销：

```python
# 在配置中调整projection_dim来控制内存使用
config = Pi0Config(
    enable_contrastive_learning=True,
    contrastive_projection_dim=128,  # 降低维度减少内存
)
```

### 2. 计算优化

对比学习只在训练时计算：

```python
# 推理时自动跳过对比学习计算
predictions = model.sample_actions(rng, observation)
```

## 示例代码

完整的使用示例请参考 `examples/contrastive_learning_example.py`。

## 参考文献

1. Chen, T., et al. "A simple framework for contrastive learning of visual representations." ICML 2020.
2. He, K., et al. "Momentum contrast for unsupervised visual representation learning." CVPR 2020.
3. Oord, A., et al. "Representation learning with contrastive predictive coding." arXiv 2018. 