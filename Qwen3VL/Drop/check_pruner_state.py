#!/usr/bin/env python3
"""检查保存的pruner状态"""

import torch
import sys

pruner_path = "/data/taosen/code/MCITlib/checkpoint/UCIT/LLaVA-1.5/Drop/Task2_llava_lora/lora_drop_pruner.pt"

print("=" * 60)
print("检查Task2保存的pruner状态")
print("=" * 60)

state = torch.load(pruner_path, map_location='cpu')

print(f"\n保存的keys: {list(state.keys())}")

print(f"\n基本信息:")
print(f"  warmup_steps: {state.get('warmup_steps')}")
print(f"  importance_threshold: {state.get('importance_threshold')}")
print(f"  fine_grained_prune_ratio: {state.get('fine_grained_prune_ratio')}")
print(f"  fine_grained_start_task: {state.get('fine_grained_start_task')}")

print(f"\n层信息:")
print(f"  总层数: {len(state.get('layer_lora_type', {}))}")
independent_layers = sum(1 for t in state.get('layer_lora_type', {}).values() if t == "independent")
shared_layers = len(state.get('layer_lora_type', {})) - independent_layers
print(f"  Independent层: {independent_layers}")
print(f"  Shared层: {shared_layers}")

print(f"\n细粒度剪枝信息:")
fine_grained_masks = state.get('fine_grained_masks', {})
print(f"  有mask的层数: {len(fine_grained_masks)}")

if fine_grained_masks:
    print(f"\n前5个有mask的层:")
    for i, (layer_name, masks) in enumerate(list(fine_grained_masks.items())[:5]):
        print(f"\n  {i+1}. {layer_name}")
        for matrix_name, mask in masks.items():
            total = mask.numel()
            kept = (mask > 0).sum().item()
            print(f"     {matrix_name}: shape={mask.shape}, kept={kept}/{total} ({kept/total*100:.2f}%)")

    # 统计所有mask
    total_params = 0
    kept_params = 0
    for layer_name, masks in fine_grained_masks.items():
        for matrix_name, mask in masks.items():
            total_params += mask.numel()
            kept_params += (mask > 0).sum().item()

    print(f"\n总体统计:")
    print(f"  总参数: {total_params:,}")
    print(f"  保留参数: {kept_params:,} ({kept_params/total_params*100:.2f}%)")
    print(f"  剪枝参数: {total_params - kept_params:,} ({(total_params - kept_params)/total_params*100:.2f}%)")
else:
    print("  ⚠ 没有保存任何fine_grained_masks!")