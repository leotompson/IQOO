#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
打印Q层的归一化importance（用于细粒度参数剪枝）
"""

import torch
import sys
import os

def print_q_normalized_importance(pruner_path):
    """读取并打印Q层的归一化importance"""

    if not os.path.exists(pruner_path):
        print(f"错误: 文件不存在 {pruner_path}")
        return

    state = torch.load(pruner_path, map_location='cpu', weights_only=False)

    if 'fine_grained_masks' not in state:
        print("警告: 没有找到fine_grained_masks")
        return

    fine_grained_masks = state['fine_grained_masks']

    # 筛选Q投影层
    q_layers = {}
    for layer_name, mask_info in fine_grained_masks.items():
        if 'q_proj' in layer_name and 'layers.' in layer_name:
            layer_idx = int(layer_name.split('layers.')[1].split('.')[0])
            if 'importance' in mask_info:
                q_layers[layer_idx] = {
                    'importance': mask_info['importance'],
                    'keep_ratio': mask_info.get('keep_ratio', 0),
                    'kept_ranks': mask_info.get('kept_ranks', 0)
                }

    if not q_layers:
        print("没有找到Q投影层的细粒度剪枝信息")
        return

    print("=" * 80)
    print(f"Q投影层归一化Importance (用于细粒度参数剪枝)")
    print(f"文件: {os.path.basename(pruner_path)}")
    print("=" * 80)
    print(f"{'Layer':<6} | {'Importance':<12} | {'Keep Ratio':<11} | {'Kept Ranks':<11}")
    print("-" * 80)

    # 按层号排序
    total_importance = 0
    for layer_idx in sorted(q_layers.keys()):
        info = q_layers[layer_idx]
        importance = info['importance']
        keep_ratio = info['keep_ratio']
        kept_ranks = info['kept_ranks']

        print(f"{layer_idx:<6} | {importance:<12.6f} | {keep_ratio:<11.4f} | {kept_ranks:<11}")
        total_importance += importance

    print("-" * 80)
    print(f"{'总计':<6} | {total_importance:<12.6f} | {'(归一化)':<11} | {'':<11}")

    # 统计信息
    importances = [info['importance'] for info in q_layers.values()]
    keep_ratios = [info['keep_ratio'] for info in q_layers.values()]

    print("\n" + "=" * 80)
    print("统计信息:")
    print("=" * 80)
    print(f"层数: {len(q_layers)}")
    print(f"\nImportance (归一化):")
    print(f"  总和:   {sum(importances):.6f}")
    print(f"  最大值: {max(importances):.6f} (Layer {max(q_layers.items(), key=lambda x: x[1]['importance'])[0]})")
    print(f"  最小值: {min(importances):.6f} (Layer {min(q_layers.items(), key=lambda x: x[1]['importance'])[0]})")
    print(f"  平均值: {sum(importances)/len(importances):.6f}")
    print(f"  中位数: {sorted(importances)[len(importances)//2]:.6f}")

    print(f"\nKeep Ratio:")
    print(f"  最大值: {max(keep_ratios):.4f}")
    print(f"  最小值: {min(keep_ratios):.4f}")
    print(f"  平均值: {sum(keep_ratios)/len(keep_ratios):.4f}")

    # 按importance排序显示
    print("\n" + "=" * 80)
    print("按Importance降序排列:")
    print("=" * 80)
    print(f"{'Rank':<5} | {'Layer':<6} | {'Importance':<12} | {'Keep Ratio':<11}")
    print("-" * 80)

    sorted_layers = sorted(q_layers.items(), key=lambda x: x[1]['importance'], reverse=True)
    for rank, (layer_idx, info) in enumerate(sorted_layers, 1):
        print(f"{rank:<5} | {layer_idx:<6} | {info['importance']:<12.6f} | {info['keep_ratio']:<11.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python print_q_normalized_importance.py <pruner_path>")
        print("示例: python print_q_normalized_importance.py /data/taosen/code/MCITlib/checkpoint/UCIT/LLaVA-1.5/Drop/Task2_llava_lora/lora_drop_pruner.pt")
        sys.exit(1)

    pruner_path = sys.argv[1]
    print_q_normalized_importance(pruner_path)