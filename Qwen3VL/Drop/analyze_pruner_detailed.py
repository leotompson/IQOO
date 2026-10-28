#!/usr/bin/env python3
"""详细分析LoRA-Drop pruner状态：importance和剪枝参数保留率"""

import torch
import numpy as np

pruner_path = "/data/taosen/code/MCITlib/checkpoint/UCIT/LLaVA-1.5/Drop/Task2_llava_lora/lora_drop_pruner.pt"

print("=" * 100)
print("LoRA-Drop Pruner 详细分析")
print("=" * 100)

state = torch.load(pruner_path, map_location='cpu', weights_only=False)

layer_importance = state.get('layer_importance', {})
layer_lora_type = state.get('layer_lora_type', {})
fine_grained_masks = state.get('fine_grained_masks', {})

print(f"\n基本信息:")
print(f"  总层数: {len(layer_importance)}")
print(f"  Independent层: {sum(1 for t in layer_lora_type.values() if t == 'independent')}")
print(f"  Shared层: {sum(1 for t in layer_lora_type.values() if t == 'shared')}")
print(f"  有细粒度mask的层: {len(fine_grained_masks)}")

# 按投影类型分组
proj_types = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

for proj_type in proj_types:
    print(f"\n{'='*100}")
    print(f"{proj_type.upper()}")
    print(f"{'='*100}")

    # 收集该投影类型的所有层
    proj_layers = []
    for layer_name, lora_type in layer_lora_type.items():
        if proj_type in layer_name:
            importance = layer_importance.get(layer_name, 0.0)
            has_mask = layer_name in fine_grained_masks

            layer_info = {
                'name': layer_name,
                'type': lora_type,
                'importance': importance,
                'has_mask': has_mask
            }

            # 如果有mask，计算保留率
            if has_mask:
                masks = fine_grained_masks[layer_name]
                mask_A = masks.get('lora_A')
                mask_B = masks.get('lora_B')

                if mask_A is not None and mask_B is not None:
                    # 计算保留率
                    total_A = mask_A.numel()
                    kept_A = (mask_A > 0).sum().item()
                    keep_ratio_A = kept_A / total_A

                    total_B = mask_B.numel()
                    kept_B = (mask_B > 0).sum().item()
                    keep_ratio_B = kept_B / total_B

                    # 计算kept ranks
                    rank = mask_A.shape[0] if len(mask_A.shape) > 1 else 1
                    kept_ranks_A = (mask_A.sum(dim=1) > 0).sum().item() if len(mask_A.shape) > 1 else 1
                    kept_ranks_B = (mask_B.sum(dim=0) > 0).sum().item() if len(mask_B.shape) > 1 else 1

                    layer_info['keep_ratio_A'] = keep_ratio_A
                    layer_info['keep_ratio_B'] = keep_ratio_B
                    layer_info['total_ranks'] = rank
                    layer_info['kept_ranks_A'] = kept_ranks_A
                    layer_info['kept_ranks_B'] = kept_ranks_B

                    # 获取元数据
                    layer_info['meta_importance'] = masks.get('importance', 'N/A')
                    layer_info['meta_keep_ratio'] = masks.get('keep_ratio', 'N/A')
                    layer_info['meta_kept_ranks'] = masks.get('kept_ranks', 'N/A')

            proj_layers.append(layer_info)

    # 分别统计independent和shared
    independent_layers = [l for l in proj_layers if l['type'] == 'independent']
    shared_layers = [l for l in proj_layers if l['type'] == 'shared']

    print(f"\n总层数: {len(proj_layers)} (Independent: {len(independent_layers)}, Shared: {len(shared_layers)})")

    # 统计independent层的importance
    if independent_layers:
        importances = [l['importance'] for l in independent_layers]
        print(f"\nIndependent层 - 原始Importance统计:")
        print(f"  范围: [{min(importances):.6f}, {max(importances):.6f}]")
        print(f"  平均: {np.mean(importances):.6f}")
        print(f"  中位数: {np.median(importances):.6f}")
        print(f"  标准差: {np.std(importances):.6f}")

        # 归一化importance (模拟代码中的归一化)
        max_imp = max(importances)
        min_imp = min(importances)
        imp_range = max_imp - min_imp

        if imp_range > 0:
            normalized = [(imp - min_imp) / imp_range for imp in importances]
            print(f"\n归一化后的Importance (用于自适应剪枝):")
            print(f"  范围: [{min(normalized):.3f}, {max(normalized):.3f}]")
            print(f"  平均: {np.mean(normalized):.3f}")

            # 计算对应的保留率 (假设min=0.4, max=0.9)
            min_keep = 0.4
            max_keep = 0.9
            keep_ratios = [min_keep + (max_keep - min_keep) * norm for norm in normalized]
            print(f"\n对应的保留率 (min=0.4, max=0.9):")
            print(f"  范围: [{min(keep_ratios):.1%}, {max(keep_ratios):.1%}]")
            print(f"  平均: {np.mean(keep_ratios):.1%}")

    # 打印详细的层信息
    if independent_layers:
        print(f"\n{'='*100}")
        print(f"Independent层详细信息 (按importance排序):")
        print(f"{'='*100}")

        # 按importance排序
        independent_layers.sort(key=lambda x: x['importance'], reverse=True)

        print(f"\n{'Layer':<65} {'Importance':<12} {'Norm Imp':<10} {'Keep Ratio':<12} {'Kept Ranks':<12}")
        print("-" * 111)

        # 计算归一化importance
        importances = [l['importance'] for l in independent_layers]
        max_imp = max(importances)
        min_imp = min(importances)
        imp_range = max_imp - min_imp

        for layer in independent_layers:
            short_name = layer['name'].replace('base_model.model.model.layers.', 'L')

            # 归一化importance
            if imp_range > 0:
                norm_imp = (layer['importance'] - min_imp) / imp_range
            else:
                norm_imp = 0.5

            # 计算理论保留率
            min_keep = 0.4
            max_keep = 0.9
            theoretical_keep = min_keep + (max_keep - min_keep) * norm_imp

            # 实际保留率
            if layer['has_mask'] and 'keep_ratio_A' in layer:
                actual_keep = f"{layer['keep_ratio_A']:.1%}"
                kept_ranks = f"{layer['kept_ranks_A']}/{layer['total_ranks']}"
            else:
                actual_keep = "N/A"
                kept_ranks = "N/A"

            print(f"{short_name:<65} {layer['importance']:<12.6f} {norm_imp:<10.3f} {actual_keep:<12} {kept_ranks:<12}")

    # 打印shared层信息
    if shared_layers:
        print(f"\n{'='*100}")
        print(f"Shared层:")
        print(f"{'='*100}")
        for layer in shared_layers:
            short_name = layer['name'].replace('base_model.model.model.layers.', 'L')
            print(f"  {short_name}: importance={layer['importance']:.6f}")

print("\n" + "="*100)
print("分析完成")
print("="*100)