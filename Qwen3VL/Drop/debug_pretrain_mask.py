#!/usr/bin/env python3
"""调试脚本：检查LoRA模块结构，找出为什么预训练权重mask计算失败"""

import torch
import sys

# 模拟检查模块结构
print("=" * 80)
print("检查LoRA模块结构")
print("=" * 80)

# 假设我们有一个模型
print("\n请在训练代码中添加以下调试代码：")
print("-" * 80)

debug_code = """
# 在 compute_pretrain_importance_masks() 方法开始处添加：

for name, module in model.named_modules():
    if not hasattr(module, 'lora_A') or not hasattr(module, 'lora_B'):
        continue

    print(f"\\n检查模块: {name}")
    print(f"  - 类型: {type(module)}")
    print(f"  - 有lora_A: {hasattr(module, 'lora_A')}")
    print(f"  - 有lora_B: {hasattr(module, 'lora_B')}")
    print(f"  - 有base_layer: {hasattr(module, 'base_layer')}")

    # 检查所有属性
    print(f"  - 所有属性: {[attr for attr in dir(module) if not attr.startswith('_')]}")

    # 尝试找到预训练权重
    if hasattr(module, 'base_layer'):
        print(f"    - base_layer类型: {type(module.base_layer)}")
        print(f"    - base_layer有weight: {hasattr(module.base_layer, 'weight')}")

    # 检查其他可能的属性
    for attr in ['weight', 'base_weight', 'original_weight', 'pretrained_weight']:
        if hasattr(module, attr):
            print(f"    - 找到属性: {attr}")

    # 只检查前3个模块
    if name.count('.') > 5:
        break
"""

print(debug_code)
print("-" * 80)

print("\n可能的问题：")
print("1. CoINMOELoraLinear 的预训练权重不在 base_layer 中")
print("2. 可能在其他属性中，如 weight, original_weight 等")
print("3. 需要查看 CoINMOELoraLinear 的实现")

print("\n建议：")
print("1. 查看 CoIN/peft/tuners/clitmoelora.py 中 CoINMOELoraLinear 的实现")
print("2. 找到预训练权重的实际存储位置")
print("3. 修改 compute_pretrain_importance_masks() 中的访问方式")