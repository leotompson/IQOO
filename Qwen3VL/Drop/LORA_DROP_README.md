# LoRA-Drop 集成说明

## 概述

LoRA-Drop 是一种高效的 LoRA 参数剪枝方法，通过评估每层 LoRA 的输出重要性来决定哪些层需要独立的新 LoRA，哪些层可以共享跨层 LoRA，从而减少持续学习中的参数增长。

## 核心原理

### 1. 重要性评估

- **数据采样**：随机抽取 10% 训练数据作为评估集
- **预热训练**：因为 LoRA-B 零初始化，需要先训练 warmup_steps（默认 50 步）
- **重要性计算**：`importance = mean(||ΔW × x||²)` 对每层 LoRA 输出

### 2. 剪枝策略

对每个投影类型（q/k/v/o/gate/up/down）独立处理：

1. 层级归一化 importance 到 [0, 1]
2. 按 importance 降序排列
3. 累积求和 < 0.9 的层：保留独立的新 LoRA
4. 累积求和 ≥ 0.9 的层：共享一个跨层 LoRA

最终每个投影类型有 1 个共享 LoRA，共 7 个。

### 3. 参数共享

不重要的层指向同一个共享 LoRA 参数，显著减少参数量：

```
重要层（前 90%）：  layer_0.q_proj → 独立 LoRA_0
                    layer_1.q_proj → 独立 LoRA_1
                    ...
不重要层（后 10%）：layer_28.q_proj → 共享 LoRA_q
                    layer_29.q_proj → 共享 LoRA_q
                    layer_30.q_proj → 共享 LoRA_q
                    layer_31.q_proj → 共享 LoRA_q
```

## 使用方法

### 1. 启用 LoRA-Drop

在训练脚本中添加参数：

```bash
deepspeed llava/train/train_mem_MOE.py \
    --lora_enable True \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_drop_enable True \
    --lora_drop_warmup_steps 50 \
    --lora_drop_eval_ratio 0.1 \
    --lora_drop_threshold 0.9 \
    ...
```

### 2. 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--lora_drop_enable` | False | 是否启用 LoRA-Drop |
| `--lora_drop_warmup_steps` | 50 | 预热步数（让 B 从零初始化更新） |
| `--lora_drop_eval_ratio` | 0.1 | 用于评估的数据比例（10%） |
| `--lora_drop_threshold` | 0.9 | 累积重要性阈值（前 90% 独立） |

### 3. 修改训练配置

编辑 `/data/taosen/code/MCITlib/configs/train_configs/Conv/LLaVA/UCIT/train/task2.json`（及后续任务）：

```json
{
  "lora_drop_enable": true,
  "lora_drop_warmup_steps": 50,
  "lora_drop_threshold": 0.9
}
```

或直接在 `Taskn.sh` 中添加：

```bash
deepspeed --include localhost:$GPU_LIST llava/train/train_mem_MOE.py \
    --lora_enable True --lora_r $RANK --lora_alpha $((RANK * 2)) \
    --lora_drop_enable True \
    --lora_drop_warmup_steps 50 \
    --lora_drop_threshold 0.9 \
    ...
```

## 训练流程

### Task 1（无剪枝）

```bash
bash scripts/MCITlib/Train/Task1.sh \
    configs/modal_configs/llava.json \
    configs/data_configs/UCIT/ImageNet-R.json \
    configs/train_configs/Conv/LLaVA/UCIT/train/task1.json
```

Task 1 不启用 LoRA-Drop（`cur_task=0` 时自动跳过）。

### Task 2+（启用剪枝）

```bash
bash scripts/MCITlib/Train/Taskn.sh \
    configs/modal_configs/llava.json \
    configs/data_configs/UCIT/ArxivQA.json \
    configs/train_configs/Conv/LLaVA/UCIT/train/task2.json
```

训练过程分为两个阶段：

**阶段 1：重要性评估（10% 数据）**

1. 随机抽取 10% 训练数据作为评估集
2. 前 warmup_steps（默认 50 步）：预热训练，让 LoRA-B 从零初始化更新
3. 继续训练评估集：每个 forward 自动收集 importance（`||ΔW × x||²`）
4. 评估结束后：
   - 计算每层归一化 importance
   - 按投影类型分组，累积阈值剪枝
   - 应用共享参数（创建 7 个跨层 LoRA）
   - 保存剪枝状态到 `lora_drop_pruner.pt`

**阶段 2：完整训练（100% 数据）**

5. 使用剪枝后的 LoRA 结构
6. 用全部训练数据进行正式的指令微调
7. 保存最终模型

## 输出示例

**阶段 1：重要性评估**

```
============================================================
LoRA-Drop Phase 1: Warmup and Importance Evaluation
============================================================
Using 500 samples (10.0%) for importance evaluation
Running warmup + evaluation for 150 steps...

============================================================
LoRA-Drop: Computing layer importance and applying pruning...
============================================================
  q_proj: 28 independent, 4 shared
  k_proj: 29 independent, 3 shared
  v_proj: 27 independent, 5 shared
  o_proj: 30 independent, 2 shared
  gate_proj: 26 independent, 6 shared
  up_proj: 28 independent, 4 shared
  down_proj: 29 independent, 3 shared

LoRA-Drop Statistics:
  Total layers: 224
  Independent LoRA layers: 197
  Shared LoRA layers: 27
  Parameter reduction: 12.05%

Applying LoRA-Drop pruning...
  Created shared LoRA for q_proj from base_model.model.model.layers.28.self_attn.q_proj
  Created shared LoRA for k_proj from base_model.model.model.layers.29.self_attn.k_proj
  ...

LoRA-Drop Summary:
  Independent LoRA layers: 197
  Shared LoRA layers: 27
  Shared LoRA groups: 7

Saved LoRA-Drop pruner state to output/task2/lora_drop_pruner.pt
============================================================
LoRA-Drop Phase 2: Full training with pruned LoRA structure
============================================================
```

**阶段 2：完整训练**

```
Training with 5000 samples (100%)...
[Normal training logs...]
```

## 预期效果

### 参数量对比

假设 LLaMA-7B，32 层，rank=64：

- **无 LoRA-Drop**：224 层 × 2 (A+B) × rank × dim ≈ 每任务 28M 参数
- **有 LoRA-Drop（12% 共享）**：197 独立 + 7 共享 ≈ 每任务 24.6M 参数
- **节省**：~12% 参数量

### 性能影响

根据 LoRA-Drop 论文，在保持性能的前提下可减少 10-30% 参数：

- **阈值 0.9**：保守剪枝，性能几乎无损
- **阈值 0.8**：激进剪枝，可能轻微性能下降（<1%）

## 调试与验证

### 1. 检查 importance 收集

在训练日志中搜索：

```bash
grep "LoRA-Drop" train.log
```

应该看到每层的 importance 更新。

### 2. 验证共享参数

```python
import torch
model = torch.load('output/task2/pytorch_model.bin')

# 检查共享层是否指向同一参数
layer_28_q = model['base_model.model.model.layers.28.self_attn.q_proj.lora_A.default.loraA.1.mlp.weight']
layer_29_q = model['base_model.model.model.layers.29.self_attn.q_proj.lora_A.default.loraA.1.mlp.weight']

print(layer_28_q.data_ptr() == layer_29_q.data_ptr())  # 应该为 True（共享内存）
```

### 3. 对比性能

```bash
# 无 LoRA-Drop
bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh ... task2_baseline.json

# 有 LoRA-Drop
bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh ... task2_lora_drop.json
```

对比 accuracy、BWT、FWT 等指标。

## 注意事项

1. **Task 1 不剪枝**：第一个任务需要完整的 LoRA 作为基础
2. **两阶段训练**：
   - 阶段 1（10% 数据）：预热 + 重要性评估 + 剪枝决策
   - 阶段 2（100% 数据）：使用剪枝后的结构进行完整训练
3. **阈值调优**：0.9 是保守值，可根据任务调整（0.8-0.95）
4. **内存开销**：阶段 1 的 importance 收集会略微增加前向时间（<5%）
5. **训练时间**：总训练时间 = 阶段 1（~10% 数据的训练时间）+ 阶段 2（完整训练时间）

## 文件清单

```
Drop/
├── CoIN/peft/tuners/
│   ├── lora_drop.py          # LoRA-Drop 核心实现
│   └── clitmoelora.py         # 修改：集成 importance 追踪
├── llava/train/
│   └── train_MOE.py           # 修改：添加 LoRA-Drop 参数和逻辑
└── LORA_DROP_README.md        # 本文档
```

## 引用

如果使用 LoRA-Drop，请引用：

```bibtex
@article{loradrop2024,
  title={LoRA-Drop: Efficient LoRA Parameter Pruning based on Output Evaluation},
  author={...},
  journal={arXiv preprint arXiv:...},
  year={2024}
}
```