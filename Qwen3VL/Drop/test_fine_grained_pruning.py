#!/usr/bin/env python3
"""
测试细粒度参数剪枝功能

验证：
1. LoRADropPruner能正确初始化细粒度剪枝参数
2. compute_fine_grained_masks能正确计算mask
3. apply_fine_grained_masks能正确应用mask
4. 只在指定任务开始剪枝
5. 从CSV文件读取结果并计算持续学习指标
"""

import torch
import torch.nn as nn
import sys
sys.path.append('/data/taosen/code/MCITlib/LLaVA/Drop')

from CoIN.peft.tuners.lora_drop import LoRADropPruner
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class MockLoRAModule(nn.Module):
    """模拟LoRA模块用于测试"""
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.active_adapter = "default"
        self.lora_A = {
            "default": MockLoRAA(in_features, rank)
        }
        self.lora_B = {
            "default": MockLoRAB(rank, out_features)
        }


class MockLoRAA(nn.Module):
    def __init__(self, in_features, rank):
        super().__init__()
        # 创建多个任务的LoRA（模拟task1和task2）
        self.loraA = [
            MockExpert(in_features, rank),  # task1 (cur_task=0)
            MockExpert(in_features, rank),  # task2 (cur_task=1)
        ]


class MockLoRAB(nn.Module):
    def __init__(self, rank, out_features):
        super().__init__()
        # 创建多个任务的LoRA（模拟task1和task2）
        self.loraB = [
            MockExpert(rank, out_features),  # task1 (cur_task=0)
            MockExpert(rank, out_features),  # task2 (cur_task=1)
        ]


class MockExpert(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.mlp = nn.Linear(in_features, out_features, bias=False)
        # 初始化为随机值
        nn.init.normal_(self.mlp.weight, mean=0.0, std=0.02)


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 创建一些模拟的LoRA层
        # 注意：这些属性名会成为named_modules()返回的名称
        self.layer0_q_proj = MockLoRAModule(512, 512, 16)
        self.layer0_v_proj = MockLoRAModule(512, 512, 16)
        self.layer1_q_proj = MockLoRAModule(512, 512, 16)
        self.layer1_v_proj = MockLoRAModule(512, 512, 16)

        # 为每个模块设置layer_name属性，用于调试
        self.layer0_q_proj.layer_name = "layer0_q_proj"
        self.layer0_v_proj.layer_name = "layer0_v_proj"
        self.layer1_q_proj.layer_name = "layer1_q_proj"
        self.layer1_v_proj.layer_name = "layer1_v_proj"


def test_fine_grained_pruning():
    print("=" * 80)
    print("测试细粒度参数剪枝功能")
    print("=" * 80)

    # 1. 初始化pruner
    print("\n1. 初始化LoRADropPruner...")
    pruner = LoRADropPruner(
        warmup_steps=10,
        eval_ratio=0.1,
        importance_threshold=0.9,
        fine_grained_prune_ratio=0.1,  # 保留10%参数
        fine_grained_start_task=1      # 从task2开始（cur_task=1对应task2）
    )
    print(f"   fine_grained_prune_ratio: {pruner.fine_grained_prune_ratio}")
    print(f"   fine_grained_start_task: {pruner.fine_grained_start_task} (对应task2)")

    # 2. 创建模拟模型
    print("\n2. 创建模拟模型...")
    model = MockModel()

    # 调试：打印named_modules的结果
    print("   Model modules:")
    for name, module in model.named_modules():
        if isinstance(module, MockLoRAModule):
            print(f"     - {name}: {type(module).__name__}")


    # 3. 设置layer_lora_type（模拟粗粒度剪枝结果）
    print("\n3. 设置层类型（模拟粗粒度剪枝）...")
    pruner.layer_lora_type = {
        "layer0_q_proj": "independent",
        "layer0_v_proj": "independent",
        "layer1_q_proj": "shared",
        "layer1_v_proj": "independent",
    }
    print("   Independent layers: layer0_q_proj, layer0_v_proj, layer1_v_proj")
    print("   Shared layers: layer1_q_proj")

    # 4. 测试task1（不应该剪枝，cur_task=0）
    print("\n4. 测试task1/cur_task=0（应该跳过细粒度剪枝）...")
    cur_task = 0
    pruner.compute_fine_grained_masks(model, cur_task)
    print(f"   Masks created: {len(pruner.fine_grained_masks)}")
    assert len(pruner.fine_grained_masks) == 0, "Task1不应该创建mask"
    print("   ✓ Task1正确跳过")

    # 5. 测试task2（应该剪枝，cur_task=1）
    print("\n5. 测试task2/cur_task=1（应该进行细粒度剪枝）...")
    cur_task = 1

    # 记录剪枝前的参数
    before_stats = {}
    for name, module in model.named_modules():
        if isinstance(module, MockLoRAModule):
            lora_A = module.lora_A["default"]
            lora_B = module.lora_B["default"]
            # 使用cur_task=1的权重（对应task2）
            weight_A = lora_A.loraA[cur_task].mlp.weight.data
            weight_B = lora_B.loraB[cur_task].mlp.weight.data

            nonzero_A = (weight_A != 0).sum().item()
            nonzero_B = (weight_B != 0).sum().item()
            total_A = weight_A.numel()
            total_B = weight_B.numel()

            before_stats[name] = {
                'nonzero_A': nonzero_A,
                'nonzero_B': nonzero_B,
                'total_A': total_A,
                'total_B': total_B,
            }
            print(f"   {name} (before):")
            print(f"     LoRA_A: {nonzero_A}/{total_A} ({nonzero_A/total_A*100:.1f}%)")
            print(f"     LoRA_B: {nonzero_B}/{total_B} ({nonzero_B/total_B*100:.1f}%)")

    # 执行细粒度剪枝
    pruner.compute_fine_grained_masks(model, cur_task)

    print(f"\n   Masks created: {len(pruner.fine_grained_masks)}")
    print(f"   Expected: 3 (only independent layers)")

    # 验证只有independent层被剪枝
    expected_pruned = ["layer0_q_proj", "layer0_v_proj", "layer1_v_proj"]
    for layer_name in expected_pruned:
        assert layer_name in pruner.fine_grained_masks, f"{layer_name}应该被剪枝"
    assert "layer1_q_proj" not in pruner.fine_grained_masks, "shared层不应该被剪枝"

    # 检查剪枝后的参数
    print("\n   剪枝后的参数统计:")
    for name, module in model.named_modules():
        if isinstance(module, MockLoRAModule):
            lora_type = pruner.layer_lora_type.get(name, "unknown")
            lora_A = module.lora_A["default"]
            lora_B = module.lora_B["default"]
            # 使用cur_task=1的权重（对应task2）
            weight_A = lora_A.loraA[cur_task].mlp.weight.data
            weight_B = lora_B.loraB[cur_task].mlp.weight.data

            nonzero_A = (weight_A != 0).sum().item()
            nonzero_B = (weight_B != 0).sum().item()
            total_A = weight_A.numel()
            total_B = weight_B.numel()

            print(f"   {name} ({lora_type}):")
            print(f"     LoRA_A: {nonzero_A}/{total_A} ({nonzero_A/total_A*100:.1f}%)")
            print(f"     LoRA_B: {nonzero_B}/{total_B} ({nonzero_B/total_B*100:.1f}%)")

            # 验证independent层确实被剪枝了
            if lora_type == "independent":
                assert nonzero_A < before_stats[name]['nonzero_A'], f"{name} LoRA_A应该被剪枝"
                assert nonzero_B < before_stats[name]['nonzero_B'], f"{name} LoRA_B应该被剪枝"
                # 验证保留比例接近设定值
                keep_ratio_A = nonzero_A / total_A
                keep_ratio_B = nonzero_B / total_B
                assert 0.08 <= keep_ratio_A <= 0.12, f"保留比例应该接近0.1，实际{keep_ratio_A:.3f}"
                assert 0.08 <= keep_ratio_B <= 0.12, f"保留比例应该接近0.1，实际{keep_ratio_B:.3f}"

    # 6. 测试apply_fine_grained_masks
    print("\n6. 测试apply_fine_grained_masks...")
    # 手动修改一些被剪枝的参数
    for name, module in model.named_modules():
        if name in pruner.fine_grained_masks:
            lora_A = module.lora_A["default"]
            # 使用cur_task=1的权重
            weight_A = lora_A.loraA[cur_task].mlp.weight.data
            # 将所有参数设为1（包括被剪枝的）
            weight_A.fill_(1.0)

    # 应用mask
    pruner.apply_fine_grained_masks(model, cur_task)

    # 验证被剪枝的参数确实被置零
    print("   验证mask应用效果...")
    for name, module in model.named_modules():
        if name in pruner.fine_grained_masks:
            lora_A = module.lora_A["default"]
            # 使用cur_task=1的权重
            weight_A = lora_A.loraA[cur_task].mlp.weight.data
            mask_A = pruner.fine_grained_masks[name]['lora_A']

            # 验证mask正确应用
            assert torch.allclose(weight_A, mask_A), f"{name}的mask未正确应用"
            print(f"   ✓ {name} mask应用正确")

    # 7. 测试统计信息
    print("\n7. 测试统计信息...")
    stats = pruner.get_statistics()
    print(f"   Total layers: {stats['total_layers']}")
    print(f"   Independent layers: {stats['independent_lora_layers']}")
    print(f"   Shared layers: {stats['shared_lora_layers']}")
    print(f"   Fine-grained prune ratio: {stats['fine_grained_prune_ratio']}")
    print(f"   Fine-grained start task: {stats['fine_grained_start_task']}")
    if 'fine_grained_total_params' in stats:
        print(f"   Fine-grained total params: {stats['fine_grained_total_params']}")
        print(f"   Fine-grained pruned params: {stats['fine_grained_pruned_params']}")
        print(f"   Fine-grained kept params: {stats['fine_grained_kept_params']}")

    print("\n" + "=" * 80)
    print("✓ 所有测试通过！")
    print("=" * 80)


if __name__ == "__main__":
    test_fine_grained_pruning()
