#!/usr/bin/env python3
"""
在GPU 0上初始化带LoRA的LLaVA模型，提取七个投影层的形状和可学习参数量

功能：
1. 按照Drop的方式加载LLaVA基础模型到GPU 0
2. 初始化秩为16的LoRA到七个投影层
3. 打印每个投影类型的LoRA_A和LoRA_B形状
4. 计算总的可学习参数量
"""

import torch
import os
import sys
from collections import defaultdict
from transformers import AutoTokenizer

# 添加Drop路径
sys.path.append('/data/taosen/code/MCITlib/LLaVA/Drop')

from llava.model import LlavaLlamaForCausalLM
from CoIN.peft import get_peft_model, CoINMOELoraConfig, TaskType


def init_lora_model_on_gpu0(model_base, lora_rank=16, lora_alpha=32, num_task=1):
    """
    在GPU 0上初始化带LoRA的LLaVA模型

    Args:
        model_base: 基础模型路径（llava-v1.5-7b）
        lora_rank: LoRA的秩（默认16）
        lora_alpha: LoRA的alpha（默认32）
        num_task: 任务数量（默认1）

    Returns:
        model: 初始化了LoRA的模型
    """
    device = "cuda:0"
    kwargs = {
        "device_map": {"": 0},  # 强制所有层到GPU 0
        "torch_dtype": torch.float16
    }

    print(f"加载基础模型: {model_base}")
    print(f"LoRA rank: {lora_rank}, alpha: {lora_alpha}")

    # 1. 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)

    # 2. 加载LLaVA基础模型
    print('Loading LLaVA base model...')
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_base,
        low_cpu_mem_usage=True,
        **kwargs
    )

    # 3. 设置tokenizer
    model.set_tokenizer(tokenizer)

    # 4. 配置LoRA - 只对七个投影层添加LoRA
    print('Initializing LoRA for 7 projection layers...')
    lora_config = CoINMOELoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        inference_mode=False,
    )

    # 5. 应用LoRA
    model = get_peft_model(model, lora_config)

    print('Model with LoRA initialized successfully!')
    return model


def extract_lora_layer_info(model):
    """
    提取LoRA层的形状和参数量信息

    Args:
        model: 嵌入LoRA的模型

    Returns:
        dict: 包含每个投影类型的形状和参数统计
    """
    # 七个投影类型
    proj_types = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

    # 按投影类型分组存储层信息
    proj_layers = defaultdict(list)

    # 遍历所有模块
    for name, module in model.named_modules():
        # 检查是否是LoRA层
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            # 判断属于哪个投影类型
            for proj_type in proj_types:
                if proj_type in name:
                    proj_layers[proj_type].append({
                        'name': name,
                        'module': module
                    })
                    break

    # 统计信息
    results = {
        'proj_info': {},
        'total_params': 0,
        'total_layers': 0
    }

    print("\n" + "=" * 80)
    print("LoRA层形状信息")
    print("=" * 80)

    for proj_type in proj_types:
        if proj_type not in proj_layers:
            print(f"\n{proj_type}: 未找到")
            continue

        layers = proj_layers[proj_type]
        print(f"\n{proj_type}: 找到 {len(layers)} 层")

        # 取第一层作为代表，打印形状
        first_layer = layers[0]
        module = first_layer['module']

        # 获取LoRA_A和LoRA_B的形状
        adapter_name = module.active_adapter if hasattr(module, 'active_adapter') else 'default'

        if adapter_name in module.lora_A:
            lora_A = module.lora_A[adapter_name]
            lora_B = module.lora_B[adapter_name]

            # 获取权重形状
            if hasattr(lora_A, 'weight'):
                # 标准LoRA
                shape_A = lora_A.weight.shape
                shape_B = lora_B.weight.shape
                params_per_layer = lora_A.weight.numel() + lora_B.weight.numel()
            elif hasattr(lora_A, 'loraA'):
                # MOE-LoRA (多个expert)
                num_experts = len(lora_A.loraA)
                first_expert_A = lora_A.loraA[0]
                first_expert_B = lora_B.loraB[0]

                if hasattr(first_expert_A, 'mlp'):
                    # 有MLP包装
                    shape_A = first_expert_A.mlp.weight.shape
                    shape_B = first_expert_B.mlp.weight.shape
                    params_per_expert = first_expert_A.mlp.weight.numel() + first_expert_B.mlp.weight.numel()
                    params_per_layer = params_per_expert * num_experts
                else:
                    shape_A = first_expert_A.weight.shape
                    shape_B = first_expert_B.weight.shape
                    params_per_expert = first_expert_A.weight.numel() + first_expert_B.weight.numel()
                    params_per_layer = params_per_expert * num_experts

                print(f"  形状 (每个expert):")
                print(f"    LoRA_A: {shape_A}")
                print(f"    LoRA_B: {shape_B}")
                print(f"  Experts数量: {num_experts}")
            else:
                print(f"  ⚠ 未知的LoRA结构")
                continue

            print(f"  形状:")
            print(f"    LoRA_A: {shape_A}")
            print(f"    LoRA_B: {shape_B}")
            print(f"  每层参数量: {params_per_layer:,}")

            # 计算该投影类型的总参数量
            total_params_this_proj = params_per_layer * len(layers)
            print(f"  该投影类型总参数量: {total_params_this_proj:,}")

            results['proj_info'][proj_type] = {
                'num_layers': len(layers),
                'shape_A': shape_A,
                'shape_B': shape_B,
                'params_per_layer': params_per_layer,
                'total_params': total_params_this_proj
            }

            results['total_params'] += total_params_this_proj
            results['total_layers'] += len(layers)

    return results


def count_all_learnable_params(model):
    """
    计算模型中所有可学习参数（包括LoRA和其他可训练参数）

    Args:
        model: 模型

    Returns:
        dict: 参数统计
    """
    total_params = 0
    lora_params = 0
    other_trainable_params = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            total_params += num_params

            if 'lora' in name.lower():
                lora_params += num_params
            else:
                other_trainable_params += num_params

    return {
        'total_learnable': total_params,
        'lora_params': lora_params,
        'other_trainable': other_trainable_params
    }


def load_drop_lists_from_pruner(pruner_path):
    """
    从lora_drop_pruner.pt加载七个投影层的drop列表

    Args:
        pruner_path: lora_drop_pruner.pt文件路径

    Returns:
        dict: 包含七个投影层的drop列表
            {
                'q_proj': [被drop的层索引列表],
                'k_proj': [...],
                ...
            }
    """
    if not os.path.exists(pruner_path):
        print(f"⚠ 文件不存在: {pruner_path}")
        return None

    state = torch.load(pruner_path, map_location='cpu', weights_only=False)
    layer_lora_type = state.get('layer_lora_type', {})

    if not layer_lora_type:
        print(f"⚠ {pruner_path} 中没有layer_lora_type信息")
        return None

    # 七个投影类型
    proj_types = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

    # 初始化drop列表
    drop_lists = {proj_type: [] for proj_type in proj_types}

    # 遍历所有层，找出被drop的层（shared层）
    for layer_name, lora_type in layer_lora_type.items():
        if lora_type == 'shared':
            # 提取层索引和投影类型
            # 例如: "base_model.model.model.layers.28.self_attn.q_proj"
            for proj_type in proj_types:
                if proj_type in layer_name:
                    # 提取层索引
                    parts = layer_name.split('.')
                    for i, part in enumerate(parts):
                        if part == 'layers' and i + 1 < len(parts):
                            layer_idx = int(parts[i + 1])
                            drop_lists[proj_type].append(layer_idx)
                            break
                    break

    # 排序
    for proj_type in proj_types:
        drop_lists[proj_type].sort()

    return drop_lists


def load_rank_lists_from_pruner(pruner_path, original_rank=32):
    """
    从lora_drop_pruner.pt加载七个投影层的rank列表（每层保留的rank数量）

    Args:
        pruner_path: lora_drop_pruner.pt文件路径
        original_rank: 原始rank值（默认32）

    Returns:
        dict: 包含七个投影层的rank列表
            {
                'q_proj': [每层保留的rank数量],
                'k_proj': [...],
                ...
            }
    """
    if not os.path.exists(pruner_path):
        print(f"⚠ 文件不存在: {pruner_path}")
        return None

    state = torch.load(pruner_path, map_location='cpu', weights_only=False)
    layer_lora_type = state.get('layer_lora_type', {})
    fine_grained_masks = state.get('fine_grained_masks', {})

    if not layer_lora_type:
        print(f"⚠ {pruner_path} 中没有layer_lora_type信息")
        return None

    # 自动检测rank
    detected_rank = original_rank
    if fine_grained_masks:
        first_mask = next(iter(fine_grained_masks.values()))
        if 'lora_A' in first_mask:
            detected_rank = first_mask['lora_A'].shape[0]

    # 七个投影类型
    proj_types = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

    # 初始化rank列表（假设32层，每层初始为original_rank）
    num_layers = 32
    rank_lists = {proj_type: [detected_rank] * num_layers for proj_type in proj_types}

    # 更新有fine_grained_masks的层的rank
    for layer_name, masks in fine_grained_masks.items():
        kept_ranks = masks.get('kept_ranks', detected_rank)

        # 提取层索引和投影类型
        for proj_type in proj_types:
            if proj_type in layer_name:
                # 提取层索引
                parts = layer_name.split('.')
                for i, part in enumerate(parts):
                    if part == 'layers' and i + 1 < len(parts):
                        layer_idx = int(parts[i + 1])
                        rank_lists[proj_type][layer_idx] = kept_ranks
                        break
                break

    return rank_lists


def calculate_params_from_drop_and_rank(drop_lists, rank_lists, original_rank=None):
    """
    根据drop_lists和rank_lists计算参数量统计

    Args:
        drop_lists: 七个投影层的drop列表
        rank_lists: 七个投影层的rank列表
        original_rank: 原始rank值（如果为None，从rank_lists中自动检测）

    Returns:
        dict: 包含Original, After_Drop, After_Structured等统计信息
    """
    # LLaMA-7B的层维度
    layer_dims = {
        'q_proj': (4096, 4096),
        'k_proj': (4096, 4096),
        'v_proj': (4096, 4096),
        'o_proj': (4096, 4096),
        'gate_proj': (4096, 11008),
        'up_proj': (4096, 11008),
        'down_proj': (11008, 4096),
    }

    num_layers = 32
    proj_types = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

    # 自动检测original_rank（取所有rank_lists中的最大值）
    if original_rank is None:
        original_rank = 0
        for proj_type in proj_types:
            if proj_type in rank_lists:
                original_rank = max(original_rank, max(rank_lists[proj_type]))
        print(f"  自动检测到的original_rank: {original_rank}")

    # 1. 计算Original参数量（所有层都有独立LoRA）
    original_params = 0
    for proj_type in proj_types:
        in_features, out_features = layer_dims[proj_type]
        params_per_layer = original_rank * in_features + out_features * original_rank
        original_params += params_per_layer * num_layers

    # 2. 计算After_Drop参数量（independent层 + 7个共享LoRA）
    after_drop_params = 0

    # Independent层的参数
    for proj_type in proj_types:
        in_features, out_features = layer_dims[proj_type]
        params_per_layer = original_rank * in_features + out_features * original_rank

        # 计算independent层数量（总层数 - drop层数）
        num_independent = num_layers - len(drop_lists[proj_type])
        after_drop_params += params_per_layer * num_independent

    # 7个共享LoRA的参数
    for proj_type in proj_types:
        in_features, out_features = layer_dims[proj_type]
        params_per_shared = original_rank * in_features + out_features * original_rank
        after_drop_params += params_per_shared

    # 3. 计算After_Structured参数量（应用fine_grained_masks后）
    after_structured_params = 0

    # Independent层的参数（使用实际的kept_ranks）
    for proj_type in proj_types:
        in_features, out_features = layer_dims[proj_type]
        drop_set = set(drop_lists[proj_type])

        for layer_idx in range(num_layers):
            if layer_idx not in drop_set:
                # Independent层，使用实际的rank
                kept_rank = rank_lists[proj_type][layer_idx]
                params = kept_rank * in_features + out_features * kept_rank
                after_structured_params += params

    # 7个共享LoRA的参数（使用original_rank）
    for proj_type in proj_types:
        in_features, out_features = layer_dims[proj_type]
        params_per_shared = original_rank * in_features + out_features * original_rank
        after_structured_params += params_per_shared

    return {
        'original': original_params,
        'after_drop': after_drop_params,
        'after_structured': after_structured_params,
        'drop_reduction': (original_params - after_drop_params) / original_params * 100,
        'structured_reduction': (original_params - after_structured_params) / original_params * 100,
    }


def analyze_all_tasks_and_print_table(base_dir):
    """
    分析所有任务并打印参数量统计表格

    Args:
        base_dir: Drop目录路径

    Returns:
        tuple: (rank_lists_dict, drop_lists_dict)
            rank_lists_dict: {task_name: rank_lists, ...}
            drop_lists_dict: {task_name: drop_lists, ...}
    """
    from pathlib import Path

    base_path = Path(base_dir)

    # 查找所有任务
    task_dirs = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("Task")])

    print("\n" + "=" * 100)
    print("LoRA-Drop 参数量统计表格")
    print("=" * 100)

    results = []
    rank_lists_dict = {}
    drop_lists_dict = {}

    for task_dir in task_dirs:
        task_name = task_dir.name.replace("_llava_lora", "")
        pruner_path = task_dir / "lora_drop_pruner.pt"

        if not pruner_path.exists():
            print(f"⚠ {task_name}: pruner文件不存在")
            continue

        # 加载drop列表和rank列表
        drop_lists = load_drop_lists_from_pruner(str(pruner_path))
        rank_lists = load_rank_lists_from_pruner(str(pruner_path))

        if drop_lists is None or rank_lists is None:
            print(f"⚠ {task_name}: 无法加载数据")
            continue

        # 存储到字典中
        rank_lists_dict[task_name] = rank_lists
        drop_lists_dict[task_name] = drop_lists

        # 计算参数量
        stats = calculate_params_from_drop_and_rank(drop_lists, rank_lists)

        results.append({
            'Task': task_name,
            'Original': stats['original'],
            'After_Drop': stats['after_drop'],
            'After_Structured': stats['after_structured'],
            'Drop_Reduction_%': f"{stats['drop_reduction']:.2f}",
            'Structured_Reduction_%': f"{stats['structured_reduction']:.2f}",
        })

    # 打印表格
    if results:
        print(f"\n{'Task':<10} {'Original':>15} {'After_Drop':>15} {'After_Struct':>15} {'Drop%':>8} {'Struct%':>8}")
        print("-" * 100)
        for r in results:
            print(f"{r['Task']:<10} {r['Original']:>15,} {r['After_Drop']:>15,} "
                  f"{r['After_Structured']:>15,} {r['Drop_Reduction_%']:>7}% {r['Structured_Reduction_%']:>7}%")
        print("=" * 100)
    else:
        print("\n⚠ 没有找到有效的统计数据")

    return rank_lists_dict, drop_lists_dict


def main():
    # 基础模型路径（llava-v1.5-7b）
    model_base = "/data/taosen/pretrain/llava-v1.5-7b"

    # LoRA配置
    lora_rank = 32
    lora_alpha = 32
    num_task = 1

    print("=" * 80)
    print("在GPU 0上初始化LoRA模型并提取信息")
    print("=" * 80)

    # 检查路径是否存在
    if not os.path.exists(model_base):
        print(f"\n⚠ 基础模型路径不存在: {model_base}")
        return

    # 初始化模型
    model = init_lora_model_on_gpu0(model_base, lora_rank, lora_alpha, num_task)

    import pdb;pdb.set_trace()

    # 提取LoRA层信息
    lora_info = extract_lora_layer_info(model)

    # 计算所有可学习参数
    param_stats = count_all_learnable_params(model)

    # 打印汇总
    print("\n" + "=" * 80)
    print("参数量汇总")
    print("=" * 80)
    print(f"LoRA总层数: {lora_info['total_layers']}")
    print(f"LoRA总参数量: {lora_info['total_params']:,}")
    print(f"\n所有可学习参数:")
    print(f"  LoRA参数: {param_stats['lora_params']:,}")
    print(f"  其他可训练参数: {param_stats['other_trainable']:,}")
    print(f"  总计: {param_stats['total_learnable']:,}")

    # 打印每个投影类型的详细信息
    print("\n" + "=" * 80)
    print("各投影类型详细信息")
    print("=" * 80)
    for proj_type, info in lora_info['proj_info'].items():
        print(f"\n{proj_type}:")
        print(f"  层数: {info['num_layers']}")
        print(f"  LoRA_A形状: {info['shape_A']}")
        print(f"  LoRA_B形状: {info['shape_B']}")
        print(f"  每层参数: {info['params_per_layer']:,}")
        print(f"  总参数: {info['total_params']:,}")

def mine_yanzheng(rank_lists_dict):
    # 粗算，没有考虑MLP那边的形状区别,32/16要切换
    scores = {}

    for task_name, module_dict in rank_lists_dict.items():
        total_rank = 0
        total_max = 0

        for module_name, rank_list in module_dict.items():
            total_rank += sum(rank_list)
            total_max += len(rank_list) * 16   # 全16时的最大值

        scores[task_name] = total_rank / total_max

    print(scores)



if __name__ == "__main__":

    # base_dir = "/data/taosen/code/MCITlib_local/checkpoint/MLLM-DCL/LLaVA-1.5/Drop"
    # base_dir = "/data/taosen/code/MCITlib_local/checkpoint/UCIT/LLaVA-1.5/Drop-relative-regular"
    base_dir = "/data/taosen/code/MCITlib/checkpoint/UCIT/LLaVA-1.5/Drop"
    rank_lists_dict, drop_lists_dict = analyze_all_tasks_and_print_table(base_dir)

    print(f"weight dir {base_dir},\nRank_lists_dict {rank_lists_dict}, \nDrop_lists_dict {drop_lists_dict}")

    # mine_yanzheng(rank_lists_dict)
    
    # 初始化模式：初始化LoRA模型
    # main()