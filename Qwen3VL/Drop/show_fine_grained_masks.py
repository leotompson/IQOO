#!/usr/bin/env python3
"""读取Task2的pruner状态，显示independent层的细粒度剪枝信息"""

import torch

pruner_path = "/data/taosen/code/MCITlib/checkpoint/UCIT/LLaVA-1.5/Drop/Task2_llava_lora/lora_drop_pruner.pt"

print("=" * 80)
print("Task2 - Independent层的细粒度剪枝信息")
print("=" * 80)

state = torch.load(pruner_path, map_location='cpu', weights_only=False)

layer_lora_type = state.get('layer_lora_type', {})
fine_grained_masks = state.get('fine_grained_masks', {})

print(f"\n总层数: {len(layer_lora_type)}")
print(f"有细粒度mask的层数: {len(fine_grained_masks)}")

# 统计independent层
independent_layers = [name for name, lora_type in layer_lora_type.items()
                      if lora_type == 'independent']
print(f"Independent层数: {len(independent_layers)}")

# 检查是否所有independent层都有mask
independent_with_mask = [name for name in independent_layers if name in fine_grained_masks]
print(f"有mask的independent层: {len(independent_with_mask)}")

if len(independent_with_mask) < len(independent_layers):
    print(f"\n⚠ 警告: {len(independent_layers) - len(independent_with_mask)}个independent层没有mask!")

# 按投影类型分组显示
proj_groups = {
    'q_proj': [],
    'k_proj': [],
    'v_proj': [],
    'o_proj': [],
    'gate_proj': [],
    'up_proj': [],
    'down_proj': [],
}

for layer_name in independent_with_mask:
    masks = fine_grained_masks[layer_name]

    # 提取投影类型
    proj_type = None
    for pt in proj_groups.keys():
        if pt in layer_name:
            proj_type = pt
            break

    if proj_type:
        # 计算保留比例
        mask_A = masks.get('lora_A')
        mask_B = masks.get('lora_B')

        if mask_A is not None and mask_B is not None:
            total_A = mask_A.numel()
            kept_A = (mask_A > 0).sum().item()
            keep_ratio_A = kept_A / total_A

            total_B = mask_B.numel()
            kept_B = (mask_B > 0).sum().item()
            keep_ratio_B = kept_B / total_B

            # 获取元数据（如果有）
            importance = masks.get('importance', 'N/A')
            keep_ratio = masks.get('keep_ratio', 'N/A')
            kept_ranks = masks.get('kept_ranks', 'N/A')

            proj_groups[proj_type].append({
                'name': layer_name,
                'keep_ratio_A': keep_ratio_A,
                'keep_ratio_B': keep_ratio_B,
                'importance': importance,
                'adaptive_keep_ratio': keep_ratio,
                'kept_ranks': kept_ranks,
                'total_ranks': mask_A.shape[0] if len(mask_A.shape) > 1 else 'N/A'
            })

# 打印每个投影类型的统计
for proj_type in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
    layers = proj_groups[proj_type]
    if not layers:
        continue

    print(f"\n{'='*80}")
    print(f"{proj_type.upper()} - {len(layers)}个independent层")
    print(f"{'='*80}")

    # 统计
    avg_keep_A = sum(l['keep_ratio_A'] for l in layers) / len(layers)
    avg_keep_B = sum(l['keep_ratio_B'] for l in layers) / len(layers)

    print(f"平均保留比例: A={avg_keep_A:.1%}, B={avg_keep_B:.1%}")

    # 检查是否有元数据
    has_metadata = any(isinstance(l['importance'], float) for l in layers)

    if has_metadata:
        print("\n自适应剪枝详细信息:")
        print(f"{'Layer':<60} {'Importance':<12} {'Keep Ratio':<12} {'Kept Ranks':<12}")
        print("-" * 96)

        # 按importance排序
        layers.sort(key=lambda x: x['importance'] if isinstance(x['importance'], float) else 0, reverse=True)

        for layer in layers:  # 显示所有层
            short_name = layer['name'].replace('base_model.model.model.layers.', 'L')
            imp = f"{layer['importance']:.3f}" if isinstance(layer['importance'], float) else "N/A"
            keep = f"{layer['adaptive_keep_ratio']:.1%}" if isinstance(layer['adaptive_keep_ratio'], float) else "N/A"
            ranks = f"{layer['kept_ranks']}/{layer['total_ranks']}" if layer['kept_ranks'] != 'N/A' else "N/A"
            print(f"{short_name:<60} {imp:<12} {keep:<12} {ranks:<12}")
    else:
        print("\n⚠ 没有自适应剪枝元数据 (旧版本的mask)")
        print(f"{'Layer':<60} {'Keep A':<12} {'Keep B':<12}")
        print("-" * 84)

        for layer in layers:  # 显示所有层
            short_name = layer['name'].replace('base_model.model.model.layers.', 'L')
            print(f"{short_name:<60} {layer['keep_ratio_A']:.1%}      {layer['keep_ratio_B']:.1%}")

print("\n" + "="*80)