# 预训练权重指导的正则化 (Pretrain-Guided Regularization)

## 概述

本文档介绍了在Drop工程中实现的**预训练权重指导的正则化**功能，该功能借鉴了[LoRASculpt](https://github.com/...)的思想，旨在解决细粒度参数剪枝后LoRA参数与预训练权重重要区域冲突的问题。

## 动机

### 问题背景

在持续学习场景下，我们对LoRA进行了两级剪枝：

1. **粗粒度剪枝（层级）**：将LoRA层分为independent和shared
2. **细粒度剪枝（参数级）**：对independent层进行结构化剪枝（按rank维度剪枝）

然而，细粒度剪枝后保留的ΔW参数可能会与预训练权重的重要区域发生冲突，导致：
- **灾难性遗忘**：破坏预训练模型的关键知识
- **性能下降**：在新任务和旧任务上的表现都变差

### 解决方案

引入**预训练权重指导的正则化项**，在训练过程中惩罚ΔW在预训练权重重要区域的变化：

```
Loss = Task_Loss + λ * Σ ||M ⊙ ΔW||_F / N
```

其中：
- `M`: 预训练权重重要性mask
- `ΔW = B @ A`: LoRA的增量权重
- `⊙`: 逐元素乘法（Hadamard product）
- `||·||_F`: Frobenius范数
- `λ`: 正则化系数（默认1e-3）
- `N`: 层数

## 实现细节

### 1. 预训练权重重要性Mask计算

在细粒度剪枝完成后，计算每个LoRA层的预训练权重重要性mask：

```python
def compute_pretrain_importance_masks(self, model):
    """
    计算预训练权重的重要性mask
    
    步骤：
    1. 获取预训练权重 W_0
    2. L2归一化: W_norm = W_0 / ||W_0||_2
    3. 计算重要性分数: S = |1 / log(|W_norm| + ε)|
    4. 应用tanh缩放: M = tanh(ω * S)
    """
    for name, module in model.named_modules():
        # 获取预训练权重
        W_0 = module.base_layer.weight.data
        
        # L2归一化
        W_norm = W_0 / (torch.norm(W_0, p=2) + 1e-15)
        
        # 计算重要性分数
        S = torch.abs(1.0 / torch.log(torch.abs(W_norm) + 1e-15))
        
        # 应用tanh缩放
        M = torch.tanh(self.omega * S)
        
        # 存储mask
        self.pretrain_masks[name] = M.detach()
```

**重要性计算原理**：
- 预训练权重的绝对值越大，该位置越重要
- 使用log函数放大差异，使重要性分布更明显
- tanh函数将重要性限制在[0, 1]范围内

### 2. 正则化损失计算

在每个训练step，计算正则化损失：

```python
def compute_pretrain_guided_regularization(self, model, cur_task: int):
    """
    计算预训练权重指导的正则化损失
    
    对每个independent层：
    1. 计算ΔW = B @ A
    2. 获取预训练权重重要性mask M
    3. 计算正则化项: ||M ⊙ ΔW||_F
    4. 累加并平均
    """
    reg_loss = 0.0
    layer_count = 0
    
    for name, module in model.named_modules():
        # 只对independent层且有mask的层计算
        if name not in self.fine_grained_masks:
            continue
        if name not in self.pretrain_masks:
            continue
            
        # 获取LoRA参数
        weight_A = lora_A.loraA[cur_task].mlp.weight
        weight_B = lora_B.loraB[cur_task].mlp.weight
        
        # 计算ΔW
        delta_W = torch.matmul(weight_B, weight_A)
        
        # 获取预训练权重重要性mask
        M = self.pretrain_masks[name]
        
        # 计算正则化项
        masked_delta_W = M * delta_W
        reg_loss += torch.norm(masked_delta_W, p='fro')
        layer_count += 1
    
    # 平均化并应用正则化系数
    if layer_count > 0:
        reg_loss = self.reg_lambda * reg_loss / layer_count
        
    return reg_loss
```

### 3. 集成到训练流程

在`LLaVATrainer.compute_loss()`中集成正则化损失：

```python
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    # 计算任务损失
    loss = super().compute_loss(model, inputs, ...)
    
    # 添加正则化损失（每个step都计算）
    if self.args.lora_drop_enable:
        pruner = get_lora_drop_pruner()
        reg_loss = pruner.compute_pretrain_guided_regularization(model, self.args.cur_task)
        loss = loss + reg_loss
    
    return loss
```

## 配置参数

### 训练参数

在`TrainingArguments`中添加了两个新参数：

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `pretrain_reg_lambda` | 1e-3 | 正则化系数λ，控制正则化强度 |
| `pretrain_reg_omega` | 1.0 | 预训练权重重要性mask的缩放因子ω |

### 配置文件示例

在`task2.json`中添加：

```json
{
  "fine_grained_prune_ratio": 0.1,
  "fine_grained_start_task": 1,
  "fine_grained_min_keep_ratio": 0.4,
  "fine_grained_max_keep_ratio": 0.9,
  "pretrain_reg_lambda": 0.001,
  "pretrain_reg_omega": 1.0
}
```

## 使用方法

### 1. 训练配置

确保配置文件中包含正则化参数：

```bash
# task2.json
{
  "pretrain_reg_lambda": 0.001,  # 正则化系数
  "pretrain_reg_omega": 1.0      # mask缩放因子
}
```

### 2. 运行训练

```bash
bash scripts/MCITlib/Train/Taskn.sh \
    configs/modal_configs/llava.json \
    configs/data_configs/UCIT/VizWiz.json \
    configs/train_configs/Drop/LLaVA/UCIT/train/task2.json
```

### 3. 监控训练

训练过程中，每100步会打印损失分解：

```
[Step 100] Loss breakdown: task=2.3456, reg=0.001234, total=2.3468
[Step 200] Loss breakdown: task=2.1234, reg=0.001156, total=2.1246
```

## 超参数调优

### pretrain_reg_lambda (正则化系数)

控制正则化项的强度：

- **较小值 (1e-4 ~ 1e-3)**：轻微约束，适合任务差异较大的场景
- **中等值 (1e-3 ~ 1e-2)**：平衡约束，推荐默认值
- **较大值 (1e-2 ~ 1e-1)**：强约束，适合需要严格保持预训练知识的场景

**调优建议**：
1. 从默认值1e-3开始
2. 如果遗忘严重，增大到1e-2
3. 如果新任务性能不佳，减小到1e-4

### pretrain_reg_omega (mask缩放因子)

控制预训练权重重要性mask的动态范围：

- **较小值 (0.5 ~ 1.0)**：mask值分布更平滑，约束更均匀
- **中等值 (1.0 ~ 2.0)**：推荐默认值
- **较大值 (2.0 ~ 5.0)**：mask值分布更极端，重要区域约束更强

**调优建议**：
1. 通常保持默认值1.0即可
2. 如果需要更强的区分度，可以增大到2.0

## 实验结果

### 性能对比

| 方法 | MFT | MFN | MAA | BWT |
|------|-----|-----|-----|-----|
| Drop-relative (无细粒度剪枝) | 74.41 | 68.41 | 81.32 | -7.20 |
| Drop-relative (细粒度剪枝，无正则化) | 72.89 | 67.62 | 79.71 | -6.33 |
| Drop-relative (细粒度剪枝+正则化) | **TBD** | **TBD** | **TBD** | **TBD** |

### 预期效果

添加预训练权重指导的正则化后，预期：

1. **减少遗忘**：BWT指标提升
2. **保持性能**：MFT和MAA指标接近或超过无细粒度剪枝的版本
3. **参数效率**：在保持性能的同时，享受细粒度剪枝带来的参数减少

## 与LoRASculpt的对比

### 相同点

1. **核心思想**：都使用预训练权重的重要性来指导LoRA训练
2. **Mask计算**：都基于预训练权重的幅度计算重要性mask
3. **正则化形式**：都使用Frobenius范数作为正则化项

### 不同点

| 特性 | LoRASculpt | Drop (本实现) |
|------|-----------|--------------|
| **剪枝策略** | 非结构化剪枝（保留Top-10%参数） | 结构化剪枝（按rank维度剪枝） |
| **剪枝对象** | 基于LoRA参数本身的重要性 | 基于LoRA-Drop的layer-wise importance |
| **自适应性** | 固定保留比例（10%） | 自适应保留比例（40%~90%，根据importance） |
| **应用场景** | 单任务微调 | 持续学习（多任务） |
| **Mask应用** | 硬约束（参数直接置零） | 软约束（通过正则化项） |

### 优势

相比LoRASculpt，本实现的优势：

1. **更适合持续学习**：结合LoRA-Drop的粗粒度剪枝，更好地处理多任务场景
2. **更灵活的剪枝**：自适应保留比例，重要层保留更多参数
3. **结构化剪枝**：保持低秩结构的完整性，不破坏A和B的协同关系
4. **按投影类型归一化**：每个投影类型（Q/K/V/O等）内部独立归一化importance，避免跨类型比较的不公平

## 代码结构

### 修改的文件

1. **CoIN/peft/tuners/lora_drop.py**
   - 添加`pretrain_reg_lambda`和`pretrain_reg_omega`参数
   - 添加`compute_pretrain_importance_masks()`方法
   - 添加`compute_pretrain_guided_regularization()`方法
   - 在`compute_fine_grained_masks()`后自动调用mask计算

2. **llava/train/llava_trainer.py**
   - 重写`compute_loss()`方法
   - 集成正则化损失到总损失
   - 添加损失分解的日志输出

3. **llava/train/train_MOE.py**
   - 在`TrainingArguments`中添加正则化参数
   - 更新pruner初始化，传递正则化参数

4. **configs/train_configs/Drop/LLaVA/UCIT/train/task2.json**
   - 添加`pretrain_reg_lambda`和`pretrain_reg_omega`配置

### 关键方法调用流程

```
训练开始
  ↓
Phase 1: Warm-up + Importance Evaluation
  ↓
compute_layer_decisions() [粗粒度剪枝]
  ↓
compute_fine_grained_masks() [细粒度剪枝]
  ↓
compute_pretrain_importance_masks() [计算预训练权重mask]
  ↓
Phase 2: Training with Regularization
  ↓
每个training step:
  compute_loss()
    ↓
    task_loss = forward + backward
    ↓
    reg_loss = compute_pretrain_guided_regularization()
    ↓
    total_loss = task_loss + reg_loss
  ↓
  apply_fine_grained_masks() [应用剪枝mask]
```

## 调试和验证

### 1. 检查Mask是否计算

训练开始时应该看到：

```
开始计算预训练权重重要性mask (omega=1.0)
  计算了 172 个预训练权重重要性mask
```

### 2. 检查正则化损失

每100步应该看到：

```
[Step 100] Loss breakdown: task=2.3456, reg=0.001234, total=2.3468
```

如果`reg=0.000000`，说明正则化没有生效，需要检查：
- `pretrain_masks`是否为空
- `fine_grained_masks`是否为空
- `cur_task`是否 >= `fine_grained_start_task`

### 3. 可视化Mask

可以使用以下脚本查看预训练权重重要性mask：

```python
import torch
import matplotlib.pyplot as plt

# 加载pruner状态
state = torch.load('lora_drop_pruner.pt')
pretrain_masks = state.get('pretrain_masks', {})

# 可视化某一层的mask
layer_name = 'base_model.model.model.layers.0.self_attn.q_proj'
if layer_name in pretrain_masks:
    mask = pretrain_masks[layer_name]
    plt.imshow(mask.cpu().numpy(), cmap='hot')
    plt.colorbar()
    plt.title(f'Pretrain Importance Mask: {layer_name}')
    plt.show()
```

## 常见问题

### Q1: 正则化损失为0？

**可能原因**：
1. `pretrain_masks`为空 → 检查`compute_pretrain_importance_masks()`是否被调用
2. `fine_grained_masks`为空 → 检查细粒度剪枝是否生效
3. `cur_task < fine_grained_start_task` → 检查任务ID配置

**解决方法**：
- 在`compute_fine_grained_masks()`后添加打印，确认mask计算
- 检查配置文件中的`fine_grained_start_task`设置

### Q2: 训练速度变慢？

**原因**：每个step都计算正则化损失，增加了计算开销

**解决方法**：
1. 减少正则化计算频率（每N步计算一次）
2. 使用更小的`pretrain_reg_lambda`减少影响
3. 优化mask计算（预计算并缓存）

### Q3: 如何选择合适的lambda？

**建议**：
1. 从默认值1e-3开始
2. 观察训练日志中的`reg_loss`值
3. 如果`reg_loss`远小于`task_loss`（<1%），可以增大lambda
4. 如果`reg_loss`接近`task_loss`（>10%），应该减小lambda

### Q4: Omega参数如何影响结果？

**影响**：
- Omega越大，mask的动态范围越大，重要区域和非重要区域的差异越明显
- Omega越小，mask的值越接近，约束更均匀

**建议**：
- 通常保持默认值1.0
- 如果需要更强的区分度，可以尝试2.0
- 不建议超过5.0，可能导致梯度不稳定

## 参考文献

1. **LoRASculpt**: [论文链接或GitHub链接]
2. **LoRA-Drop**: [相关论文]
3. **DISCO**: [相关论文]

## 更新日志

- **2026-04-24**: 初始实现，添加预训练权重指导的正则化功能
- **2026-04-24**: 完善文档，添加使用说明和调试指南

## 联系方式

如有问题或建议，请联系：[您的联系方式]