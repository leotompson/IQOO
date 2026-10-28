#!/usr/bin/env python3
"""
计算每个任务的LoRA-Drop参数量统计

从lora_drop_pruner.pt中提取：
1. Original: 原始可学习参数量（所有层的LoRA参数）
2. After_Drop: Drop后的参数量（independent层的参数）
3. After_Structured: 结构化剪枝后的参数量（应用fine_grained_masks后的有效参数）
4. Drop_Reduction_%: Drop剪枝的减少百分比
5. Structured_Reduction_%: 结构化剪枝的减少百分比
"""

import torch
import os
import csv
from pathlib import Path


def calculate_lora_params(rank, in_features, out_features):
    """计算一个LoRA层的参数量: A(rank, in_features) + B(out_features, rank)"""
    return rank * in_features + out_features * rank


def analyze_pruner_state(pruner_path, lora_rank=None):
    """
    分析pruner状态文件，计算参数量

    Args:
        pruner_path: lora_drop_pruner.pt文件路径
        lora_rank: LoRA的rank值（如果为None，从mask中自动检测）

    Returns:
        dict: 包含original, after_drop, after_structured的参数量
    """
    if not os.path.exists(pruner_path):
        print(f"⚠ 文件不存在: {pruner_path}")
        return None

    state = torch.load(pruner_path, map_location='cpu', weights_only=False)

    layer_lora_type = state.get('layer_lora_type', {})
    fine_grained_masks = state.get('fine_grained_masks', {})

    # 自动检测rank
    if lora_rank is None:
        if fine_grained_masks:
            # 从mask中获取rank
            first_mask = next(iter(fine_grained_masks.values()))
            if 'lora_A' in first_mask:
                lora_rank = first_mask['lora_A'].shape[0]
                print(f"  检测到的rank: {lora_rank}")
        else:
            # 默认值
            lora_rank = 32
            print(f"  使用默认rank: {lora_rank}")

    if not layer_lora_type:
        print(f"⚠ {pruner_path} 中没有layer_lora_type信息")
        return None

    # 统计层数
    total_layers = len(layer_lora_type)
    independent_layers = [name for name, lora_type in layer_lora_type.items()
                          if lora_type == 'independent']
    shared_layers = [name for name, lora_type in layer_lora_type.items()
                     if lora_type == 'shared']

    print(f"\n层统计:")
    print(f"  总层数: {total_layers}")
    print(f"  Independent层: {len(independent_layers)}")
    print(f"  Shared层: {len(shared_layers)}")

    # 假设标准的LLaMA-7B维度
    # q_proj, k_proj, v_proj, o_proj: 4096 -> 4096
    # gate_proj, up_proj: 4096 -> 11008
    # down_proj: 11008 -> 4096

    layer_dims = {
        'q_proj': (4096, 4096),
        'k_proj': (4096, 4096),
        'v_proj': (4096, 4096),
        'o_proj': (4096, 4096),
        'gate_proj': (4096, 11008),
        'up_proj': (4096, 11008),
        'down_proj': (11008, 4096),
    }

    def get_layer_dims(layer_name):
        """根据层名推断维度"""
        for proj_type, dims in layer_dims.items():
            if proj_type in layer_name:
                return dims
        # 默认维度
        return (4096, 4096)

    # 1. 计算原始参数量（所有层都有独立LoRA）
    original_params = 0
    for layer_name in layer_lora_type.keys():
        in_features, out_features = get_layer_dims(layer_name)
        original_params += calculate_lora_params(lora_rank, in_features, out_features)

    # 2. 计算Drop后的参数量（independent层 + 7个共享LoRA）
    after_drop_params = 0

    # Independent层的参数
    for layer_name in independent_layers:
        in_features, out_features = get_layer_dims(layer_name)
        after_drop_params += calculate_lora_params(lora_rank, in_features, out_features)

    # 7个共享LoRA的参数（每个投影类型一个）
    shared_lora_params = 0
    for proj_type, dims in layer_dims.items():
        in_features, out_features = dims
        shared_lora_params += calculate_lora_params(lora_rank, in_features, out_features)

    after_drop_params += shared_lora_params

    # 3. 计算结构化剪枝后的参数量
    after_structured_params = after_drop_params  # 先假设等于after_drop

    if fine_grained_masks:
        print(f"\n细粒度剪枝信息:")
        print(f"  有mask的层数: {len(fine_grained_masks)}")

        # 对于有mask的层，重新计算有效参数
        for layer_name in independent_layers:
            if layer_name in fine_grained_masks:
                masks = fine_grained_masks[layer_name]

                # 获取kept_ranks（保留的rank数量）
                kept_ranks = masks.get('kept_ranks', lora_rank)
                in_features, out_features = get_layer_dims(layer_name)

                # 减去原始参数，加上剪枝后的参数
                original_layer_params = calculate_lora_params(lora_rank, in_features, out_features)
                pruned_layer_params = calculate_lora_params(kept_ranks, in_features, out_features)

                after_structured_params -= original_layer_params
                after_structured_params += pruned_layer_params

        # 打印一些示例
        sample_layers = list(fine_grained_masks.keys())[:3]
        for layer_name in sample_layers:
            masks = fine_grained_masks[layer_name]
            kept_ranks = masks.get('kept_ranks', lora_rank)
            keep_ratio = masks.get('keep_ratio', 1.0)
            importance = masks.get('importance', 0.0)
            print(f"    {layer_name}:")
            print(f"      kept_ranks: {kept_ranks}/{lora_rank} ({keep_ratio:.1%})")
            print(f"      importance: {importance:.3f}")

    return {
        'original': original_params,
        'after_drop': after_drop_params,
        'after_structured': after_structured_params,
        'drop_reduction': (original_params - after_drop_params) / original_params * 100,
        'structured_reduction': (original_params - after_structured_params) / original_params * 100,
    }


def main():
    base_dir = Path("/data/taosen/code/MCITlib_local/checkpoint/MLLM-DCL/LLaVA-1.5/Drop")
    output_csv = base_dir / "Drop_params_summary.csv"

    # 查找所有任务
    task_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("Task")])

    print("=" * 80)
    print("LoRA-Drop 参数量统计")
    print("=" * 80)

    results = []

    for task_dir in task_dirs:
        task_name = task_dir.name.replace("_llava_lora", "")
        pruner_path = task_dir / "lora_drop_pruner.pt"

        print(f"\n{'=' * 80}")
        print(f"分析 {task_name}")
        print(f"{'=' * 80}")
        print(f"Pruner文件: {pruner_path}")

        stats = analyze_pruner_state(pruner_path)

        if stats:
            results.append({
                'Task': task_name,
                'Original': stats['original'],
                'After_Drop': stats['after_drop'],
                'After_Structured': stats['after_structured'],
                'Drop_Reduction_%': f"{stats['drop_reduction']:.2f}",
                'Structured_Reduction_%': f"{stats['structured_reduction']:.2f}",
            })

            print(f"\n参数量统计:")
            print(f"  Original:            {stats['original']:>15,}")
            print(f"  After_Drop:          {stats['after_drop']:>15,}")
            print(f"  After_Structured:    {stats['after_structured']:>15,}")
            print(f"  Drop_Reduction:      {stats['drop_reduction']:>14.2f}%")
            print(f"  Structured_Reduction:{stats['structured_reduction']:>14.2f}%")

    # 写入CSV
    if results:
        print(f"\n{'=' * 80}")
        print(f"写入CSV文件: {output_csv}")
        print(f"{'=' * 80}")

        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'Task', 'Original', 'After_Drop', 'After_Structured',
                'Drop_Reduction_%', 'Structured_Reduction_%'
            ])
            writer.writeheader()
            writer.writerows(results)

        print(f"✓ 成功写入 {len(results)} 个任务的统计数据")

        # 打印汇总表格
        print(f"\n{'=' * 80}")
        print("汇总表格")
        print(f"{'=' * 80}")
        print(f"{'Task':<10} {'Original':>15} {'After_Drop':>15} {'After_Struct':>15} {'Drop%':>8} {'Struct%':>8}")
        print("-" * 80)
        for r in results:
            print(f"{r['Task']:<10} {r['Original']:>15,} {r['After_Drop']:>15,} "
                  f"{r['After_Structured']:>15,} {r['Drop_Reduction_%']:>7}% {r['Structured_Reduction_%']:>7}%")
    else:
        print("\n⚠ 没有找到有效的统计数据")


if __name__ == "__main__":
    main()
