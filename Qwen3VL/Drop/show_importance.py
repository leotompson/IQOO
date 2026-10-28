#!/usr/bin/env python3
"""读取lora_drop_pruner.pt，展示每层的importance数值"""

import torch
import sys

pruner_path = "/data/taosen/code/MCITlib/checkpoint/UCIT/LLaVA-1.5/Drop/Task2_llava_lora/lora_drop_pruner.pt"

print("=" * 80)
print("读取LoRA-Drop Pruner状态 - Layer Importance")
print("=" * 80)

state = torch.load(pruner_path, map_location='cpu', weights_only=False)

layer_importance = state.get('layer_importance', {})
layer_lora_type = state.get('layer_lora_type', {})

print(f"\n总层数: {len(layer_importance)}")

# 按投影类型分组
proj_groups = {
    'q_proj': [],
    'k_proj': [],
    'v_proj': [],
    'o_proj': [],
    'gate_proj': [],
    'up_proj': [],
    'down_proj': [],
}

for layer_name, importance in layer_importance.items():
    lora_type = layer_lora_type.get(layer_name, "unknown")

    # 提取投影类型
    proj_type = None
    for pt in proj_groups.keys():
        if pt in layer_name:
            proj_type = pt
            break

    if proj_type:
        proj_groups[proj_type].append({
            'name': layer_name,
            'importance': importance,
            'type': lora_type
        })

# 打印每个投影类型的importance
for proj_type in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
    layers = proj_groups[proj_type]
    if not layers:
        continue

    print(f"\n{'='*80}")
    print(f"{proj_type.upper()} - {len(layers)}层")
    print(f"{'='*80}")

    # 按importance排序
    layers.sort(key=lambda x: x['importance'], reverse=True)

    # 统计
    independent_layers = [l for l in layers if l['type'] == 'independent']
    shared_layers = [l for l in layers if l['type'] == 'shared']

    print(f"Independent: {len(independent_layers)}层, Shared: {len(shared_layers)}层")

    if layers:
        importances = [l['importance'] for l in layers]
        print(f"Importance范围: [{min(importances):.6f}, {max(importances):.6f}]")
        print(f"平均importance: {sum(importances)/len(importances):.6f}")

    print(f"\n{'Layer Name':<70} {'Importance':<15} {'Type':<12}")
    print("-" * 97)

    for layer in layers:
        # 简化层名显示
        short_name = layer['name'].replace('base_model.model.model.layers.', 'L')
        print(f"{short_name:<70} {layer['importance']:<15.6f} {layer['type']:<12}")

# 归一化importance并显示
print(f"\n{'='*80}")
print("归一化后的Importance (用于自适应剪枝)")
print(f"{'='*80}")

# 只对independent层归一化
independent_layers_all = {name: imp for name, imp in layer_importance.items()
                          if layer_lora_type.get(name) == 'independent'}

if independent_layers_all:
    max_imp = max(independent_layers_all.values())
    min_imp = min(independent_layers_all.values())
    imp_range = max_imp - min_imp

    print(f"\nIndependent层数: {len(independent_layers_all)}")
    print(f"原始importance范围: [{min_imp:.6f}, {max_imp:.6f}]")

    if imp_range > 0:
        normalized = {}
        for name, imp in independent_layers_all.items():
            normalized[name] = (imp - min_imp) / imp_range

        # 按投影类型分组显示归一化后的值
        for proj_type in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
            proj_layers = [(name, norm_imp) for name, norm_imp in normalized.items() if proj_type in name]
            if not proj_layers:
                continue

            proj_layers.sort(key=lambda x: x[1], reverse=True)
            avg_norm = sum(x[1] for x in proj_layers) / len(proj_layers)

            print(f"\n{proj_type}: {len(proj_layers)}层, 平均归一化importance={avg_norm:.3f}")

            # 显示前3和后3
            print("  Top 3:")
            for name, norm_imp in proj_layers[:3]:
                short_name = name.replace('base_model.model.model.layers.', 'L')
                # 计算自适应保留比例 (base=0.1)
                keep_ratio = 0.1 + 0.9 * norm_imp
                print(f"    {short_name:<60} norm={norm_imp:.3f} → keep={keep_ratio:.1%}")

            if len(proj_layers) > 6:
                print("  ...")

            print("  Bottom 3:")
            for name, norm_imp in proj_layers[-3:]:
                short_name = name.replace('base_model.model.model.layers.', 'L')
                keep_ratio = 0.1 + 0.9 * norm_imp
                print(f"    {short_name:<60} norm={norm_imp:.3f} → keep={keep_ratio:.1%}")
    else:
        print("所有层importance相同，归一化后都是0.5")

print("\n" + "="*80)