#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取LoRA-Drop pruner状态文件，打印Q投影层的importance
"""

import torch
import sys
import os

def print_q_importance(pruner_path):
    """读取pruner文件并打印Q层的importance"""

    if not os.path.exists(pruner_path):
        print(f"错误: 文件不存在 {pruner_path}")
        return

    print(f"正在读取: {pruner_path}\n")

    # 加载pruner状态
    state = torch.load(pruner_path, map_location='cpu')

    print("=" * 80)
    print("Pruner状态信息:")
    print("=" * 80)
    print(f"Keys: {list(state.keys())}\n")

    # 获取layer_importance
    if 'layer_importance' in state:
        layer_importance = state['layer_importance']
        print(f"总层数: {len(layer_importance)}")
        print(f"层名称示例: {list(layer_importance.keys())[:3]}\n")

        # 筛选Q投影层
        q_layers = {}
        for layer_name, importance in layer_importance.items():
            if 'q_proj' in layer_name:
                q_layers[layer_name] = importance

        print("=" * 80)
        print(f"Q投影层 (q_proj) 的Importance (共{len(q_layers)}层):")
        print("=" * 80)

        # 按层序号排序
        sorted_q_layers = sorted(q_layers.items(), key=lambda x: x[0])

        for layer_name, importance in sorted_q_layers:
            # 提取层号
            if 'layers.' in layer_name:
                layer_idx = layer_name.split('layers.')[1].split('.')[0]
                print(f"Layer {layer_idx:>2s} | {layer_name:<80s} | Importance: {importance:.6e}")
            else:
                print(f"         | {layer_name:<80s} | Importance: {importance:.6e}")

        # 统计信息
        if q_layers:
            importances = list(q_layers.values())
            print("\n" + "=" * 80)
            print("统计信息:")
            print("=" * 80)
            print(f"最大值: {max(importances):.6e}")
            print(f"最小值: {min(importances):.6e}")
            print(f"平均值: {sum(importances)/len(importances):.6e}")
            print(f"中位数: {sorted(importances)[len(importances)//2]:.6e}")

    # 打印layer_decisions（如果有）
    if 'layer_decisions' in state:
        layer_decisions = state['layer_decisions']
        print("\n" + "=" * 80)
        print("Layer Decisions (剪枝决策):")
        print("=" * 80)

        q_decisions = {k: v for k, v in layer_decisions.items() if 'q_proj' in k}

        independent_count = sum(1 for v in q_decisions.values() if v == 'independent')
        shared_count = sum(1 for v in q_decisions.values() if v == 'shared')

        print(f"Q层决策统计:")
        print(f"  Independent (独立LoRA): {independent_count}")
        print(f"  Shared (共享LoRA):      {shared_count}")
        print(f"  总计:                   {len(q_decisions)}")

        print("\n详细决策:")
        sorted_q_decisions = sorted(q_decisions.items(), key=lambda x: x[0])
        for layer_name, decision in sorted_q_decisions:
            if 'layers.' in layer_name:
                layer_idx = layer_name.split('layers.')[1].split('.')[0]
                print(f"Layer {layer_idx:>2s} | {decision}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python print_q_importance.py <pruner_path>")
        print("示例: python print_q_importance.py /data/taosen/code/MCITlib/checkpoint/UCIT/LLaVA-1.5/Drop/Task1_llava_lora/lora_drop_pruner.pt")
        sys.exit(1)

    pruner_path = sys.argv[1]
    print_q_importance(pruner_path)