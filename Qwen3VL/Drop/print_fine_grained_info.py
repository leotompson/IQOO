#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取LoRA-Drop pruner状态文件，打印Q投影层的细粒度剪枝信息
包括归一化后的importance（用于参数级剪枝）
"""

import torch
import sys
import os

def print_fine_grained_info(pruner_path):
    """读取pruner文件并打印Q层的细粒度剪枝信息"""

    if not os.path.exists(pruner_path):
        print(f"错误: 文件不存在 {pruner_path}")
        return

    print(f"正在读取: {pruner_path}\n")

    # 加载pruner状态
    state = torch.load(pruner_path, map_location='cpu', weights_only=False)

    print("=" * 100)
    print("Pruner状态信息:")
    print("=" * 100)
    print(f"Keys: {list(state.keys())}\n")

    # 检查fine_grained_masks
    if 'fine_grained_masks' not in state:
        print("警告: 没有找到fine_grained_masks，可能还未进行细粒度剪枝")
        return

    fine_grained_masks = state['fine_grained_masks']
    print(f"细粒度剪枝的层数: {len(fine_grained_masks)}")

    # 筛选Q投影层
    q_layers = {}
    for layer_name, mask_info in fine_grained_masks.items():
        if 'q_proj' in layer_name:
            q_layers[layer_name] = mask_info

    if not q_layers:
        print("没有找到Q投影层的细粒度剪枝信息")
        return

    print(f"Q投影层的细粒度剪枝信息: {len(q_layers)} 层\n")

    # 打印每层的详细信息
    print("=" * 100)
    print("Q投影层细粒度剪枝详情:")
    print("=" * 100)

    # 按层序号排序
    sorted_q_layers = sorted(q_layers.items(), key=lambda x: x[0])

    for layer_name, mask_info in sorted_q_layers:
        # 提取层号
        if 'layers.' in layer_name:
            layer_idx = layer_name.split('layers.')[1].split('.')[0]
            print(f"\n{'='*100}")
            print(f"Layer {layer_idx}: {layer_name}")
            print(f"{'='*100}")

            # 打印mask_info中的所有键
            print(f"可用信息: {list(mask_info.keys())}")

            # 打印各项信息
            if 'importance' in mask_info:
                importance = mask_info['importance']
                if isinstance(importance, torch.Tensor):
                    print(f"\nImportance (归一化后，用于参数剪枝):")
                    print(f"  Shape: {importance.shape}")
                    print(f"  Min:   {importance.min().item():.6e}")
                    print(f"  Max:   {importance.max().item():.6e}")
                    print(f"  Mean:  {importance.mean().item():.6e}")
                    print(f"  Std:   {importance.std().item():.6e}")

                    # 打印前10个值作为示例
                    flat_imp = importance.flatten()
                    print(f"\n  前10个值: {flat_imp[:10].tolist()}")
                else:
                    print(f"\nImportance: {importance}")

            if 'keep_ratio' in mask_info:
                print(f"\nKeep Ratio (保留比例): {mask_info['keep_ratio']:.4f}")

            if 'kept_ranks' in mask_info:
                print(f"Kept Ranks (保留的秩数): {mask_info['kept_ranks']}")

            if 'mask_A' in mask_info:
                mask_A = mask_info['mask_A']
                if isinstance(mask_A, torch.Tensor):
                    kept_params_A = mask_A.sum().item()
                    total_params_A = mask_A.numel()
                    print(f"\nMask A:")
                    print(f"  Shape: {mask_A.shape}")
                    print(f"  保留参数: {kept_params_A}/{total_params_A} ({kept_params_A/total_params_A*100:.2f}%)")

            if 'mask_B' in mask_info:
                mask_B = mask_info['mask_B']
                if isinstance(mask_B, torch.Tensor):
                    kept_params_B = mask_B.sum().item()
                    total_params_B = mask_B.numel()
                    print(f"\nMask B:")
                    print(f"  Shape: {mask_B.shape}")
                    print(f"  保留参数: {kept_params_B}/{total_params_B} ({kept_params_B/total_params_B*100:.2f}%)")

    # 统计信息
    print("\n" + "=" * 100)
    print("统计信息:")
    print("=" * 100)

    total_kept_A = 0
    total_params_A = 0
    total_kept_B = 0
    total_params_B = 0

    for layer_name, mask_info in q_layers.items():
        if 'mask_A' in mask_info and isinstance(mask_info['mask_A'], torch.Tensor):
            mask_A = mask_info['mask_A']
            total_kept_A += mask_A.sum().item()
            total_params_A += mask_A.numel()

        if 'mask_B' in mask_info and isinstance(mask_info['mask_B'], torch.Tensor):
            mask_B = mask_info['mask_B']
            total_kept_B += mask_B.sum().item()
            total_params_B += mask_B.numel()

    if total_params_A > 0:
        print(f"\nQ层 LoRA-A 总体:")
        print(f"  保留参数: {total_kept_A}/{total_params_A} ({total_kept_A/total_params_A*100:.2f}%)")

    if total_params_B > 0:
        print(f"\nQ层 LoRA-B 总体:")
        print(f"  保留参数: {total_kept_B}/{total_params_B} ({total_kept_B/total_params_B*100:.2f}%)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python print_fine_grained_info.py <pruner_path>")
        print("示例: python print_fine_grained_info.py /data/taosen/code/MCITlib/checkpoint/UCIT/LLaVA-1.5/Drop/Task2_llava_lora/lora_drop_pruner.pt")
        sys.exit(1)

    pruner_path = sys.argv[1]
    print_fine_grained_info(pruner_path)
