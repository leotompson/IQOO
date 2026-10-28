# -*- encoding: utf-8 -*-
"""
LoRA-Drop: Efficient LoRA Parameter Pruning for Continual Learning

在持续学习场景下，通过评估新任务 LoRA 的重要性来决定哪些层需要独立 LoRA，
哪些层可以共享一个跨层 LoRA，从而减少参数增长。

核心逻辑：
1. 随机抽取 10% 训练数据，预热 warmup_steps 后计算 importance
2. importance = mean(||ΔW × x||²) 对每层
3. 每个投影类型（q/k/v/o/gate/up/down）独立处理：
   - 层级归一化 importance
   - 按 importance 降序排列
   - 累积求和 < 0.9 的层：保留独立 LoRA
   - 累积求和 ≥ 0.9 的层：共享一个跨层 LoRA
4. 最终每个投影类型有 1 个共享 LoRA，共 7 个
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import numpy as np


class LoRADropPruner:
    """
    LoRA-Drop pruning for continual learning with MoE-LoRA.

    支持两级剪枝：
    1. 粗粒度剪枝（层级）：将LoRA层分为independent和shared
    2. 细粒度剪枝（参数级）：对independent的LoRA进行参数级剪枝
    """

    def __init__(
        self,
        warmup_steps: int = 50,
        eval_ratio: float = 0.1,
        importance_threshold: float = 0.9,
        metric: str = "relative_l2_squared",
        fine_grained_prune_ratio: float = 0.1,
        fine_grained_start_task: int = 2,
        fine_grained_min_keep_ratio: float = 0.4,
        fine_grained_max_keep_ratio: float = 0.9,
        pretrain_reg_lambda: float = 1e-3,
        pretrain_reg_omega: float = 1.0
    ):
        """
        Args:
            warmup_steps: 预热步数（让 B 从零初始化更新）
            eval_ratio: 用于评估 importance 的数据比例
            importance_threshold: 累积重要性阈值（0.9 表示前 90% 重要的层独立）
            metric: 重要性度量方式
                - "l2_norm_squared": ||ΔW × x||²
                - "l2_norm": ||ΔW × x||
                - "mean_squared": mean((ΔW × x)²)
                - "relative_l2_squared": ||ΔW × x||² / ||W_0 × x||²
            fine_grained_prune_ratio: 细粒度剪枝保留参数比例（默认0.1，保留10%参数）【已废弃，保留用于兼容性】
            fine_grained_start_task: 从哪个任务开始进行细粒度剪枝（默认2，即task2开始）
            fine_grained_min_keep_ratio: 细粒度剪枝最小保留比例（默认0.4，最不重要的层保留40% rank）
            fine_grained_max_keep_ratio: 细粒度剪枝最大保留比例（默认0.9，最重要的层保留90% rank）
            pretrain_reg_lambda: 预训练权重指导正则化系数（默认1e-3）
            pretrain_reg_omega: 预训练权重重要性mask缩放因子（默认1.0）
        """
        self.warmup_steps = warmup_steps
        self.eval_ratio = eval_ratio
        self.importance_threshold = importance_threshold
        self.metric = metric
        self.fine_grained_prune_ratio = fine_grained_prune_ratio  # 保留用于兼容性
        self.fine_grained_start_task = fine_grained_start_task
        self.fine_grained_min_keep_ratio = fine_grained_min_keep_ratio
        self.fine_grained_max_keep_ratio = fine_grained_max_keep_ratio
        self.pretrain_reg_lambda = pretrain_reg_lambda
        self.pretrain_reg_omega = pretrain_reg_omega

        # {layer_name: importance_score}
        self.layer_importance = {}

        # {layer_name: "independent" | "shared"}
        self.layer_lora_type = {}

        # {proj_type: shared_lora_params} - 7 个共享 LoRA
        self.shared_lora_params = {}

        # 细粒度剪枝的mask: {layer_name: {'lora_A': mask_tensor, 'lora_B': mask_tensor}}
        self.fine_grained_masks = {}

        # 预训练权重重要性mask: {layer_name: importance_mask_tensor}
        self.pretrain_masks = {}

        # 收集 importance 的临时存储
        self._importance_accumulator = {}
        self._importance_count = {}

    def register_layer(self, layer_name: str):
        """注册一个层用于 importance 追踪"""
        self.layer_importance[layer_name] = 0.0
        self._importance_accumulator[layer_name] = 0.0
        self._importance_count[layer_name] = 0

    def update_importance(
        self,
        layer_name: str,
        lora_output: torch.Tensor,
        step: int,
        base_output: torch.Tensor = None
    ):
        """
        更新层的 importance 分数

        Args:
            layer_name: 层名称
            lora_output: LoRA 的输出 ΔW × x
            step: 当前训练步数
            base_output: 预训练权重的输出 W_0 × x (仅在 metric="relative_l2_squared" 时需要)
        """
        # 只在预热后收集 importance
        if step < self.warmup_steps:
            return

        if layer_name not in self._importance_accumulator:
            self.register_layer(layer_name)

        # 计算 importance
        if self.metric == "l2_norm_squared":
            importance = torch.norm(lora_output, p=2) ** 2
        elif self.metric == "l2_norm":
            importance = torch.norm(lora_output, p=2)
        elif self.metric == "mean_squared":
            importance = (lora_output ** 2).mean()
        elif self.metric == "relative_l2_squared":
            # ||ΔW × x||² / ||W_0 × x||²
            if base_output is None:
                raise ValueError("base_output is required for metric='relative_l2_squared'")
            lora_norm_sq = torch.norm(lora_output, p=2) ** 2
            base_norm_sq = torch.norm(base_output, p=2) ** 2
            # 避免除零
            importance = lora_norm_sq / (base_norm_sq + 1e-8)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        # 累积
        self._importance_accumulator[layer_name] += importance.item()
        self._importance_count[layer_name] += 1

    def compute_layer_decisions(self) -> Dict[str, str]:
        """
        计算每层的 LoRA 类型（独立 vs 共享）

        Returns:
            {layer_name: "independent" | "shared"}
        """
        # 1. 计算平均 importance
        for layer_name in self._importance_accumulator:
            count = self._importance_count[layer_name]
            if count > 0:
                self.layer_importance[layer_name] = \
                    self._importance_accumulator[layer_name] / count

        # 2. 按投影类型分组（q/k/v/o/gate/up/down 各自独立处理）
        projection_groups = self._group_by_projection()

        # 3. 对每个投影组独立计算
        for proj_type, layer_names in projection_groups.items():
            self._compute_group_decisions(proj_type, layer_names)

        return self.layer_lora_type

    def _group_by_projection(self) -> Dict[str, List[str]]:
        """按投影类型分组层"""
        groups = {
            'q_proj': [],
            'k_proj': [],
            'v_proj': [],
            'o_proj': [],
            'gate_proj': [],
            'up_proj': [],
            'down_proj': [],
        }

        for layer_name in self.layer_importance.keys():
            for proj_type in groups.keys():
                if proj_type in layer_name:
                    groups[proj_type].append(layer_name)
                    break

        return groups

    def _compute_group_decisions(self, proj_type: str, layer_names: List[str]):
        """
        对一组层（同一投影类型）计算剪枝决策

        步骤：
        1. 提取该组的 importance 值
        2. 归一化到 [0, 1]
        3. 按 importance 降序排列
        4. 累积求和 < threshold 的层：独立 LoRA
        5. 累积求和 ≥ threshold 的层：共享一个跨层 LoRA
        """
        if not layer_names:
            return

        # 提取 importance
        importances = np.array([
            self.layer_importance.get(name, 0.0) for name in layer_names
        ])

        # 归一化
        total = importances.sum()
        if total > 0:
            importances = importances / total
        else:
            # 如果全为 0，均匀分配
            importances = np.ones_like(importances) / len(importances)

        # 排序（降序）
        sorted_indices = np.argsort(importances)[::-1]

        # 累积求和，找到分界点
        cumsum = 0.0
        independent_layers = []
        shared_layers = []

        for idx in sorted_indices:
            layer_name = layer_names[idx]
            cumsum += importances[idx]

            # 累积未超过阈值，保留独立 LoRA
            if cumsum < self.importance_threshold:
                self.layer_lora_type[layer_name] = "independent"
                independent_layers.append(layer_name)
            else:
                # 累积超过阈值，共享 LoRA
                self.layer_lora_type[layer_name] = "shared"
                shared_layers.append(layer_name)

        print(f"  {proj_type}: {len(independent_layers)} independent, "
              f"{len(shared_layers)} shared")

    def compute_shared_layer_fsa_masks(self, model, cur_task: int, target_rank: int = 10):
        """
        对共享LoRA层进行FSA剪枝 - 随机剪枝秩从16到目标秩

        策略：
        1. 对每个投影类型的共享LoRA，随机选择要保留的rank维度
        2. 只剪枝一次，之后所有share layer共享同一个mask
        3. 剪枝整个rank维度（同时剪A的行和B的列）

        Args:
            model: 包含 CoINMOELoraLinear 层的模型
            cur_task: 当前任务 ID
            target_rank: 目标秩（默认10）
        """
        if cur_task < self.fine_grained_start_task:
            return

        from torch.distributed import get_rank
        is_rank0 = get_rank() == 0 if torch.distributed.is_initialized() else True

        if is_rank0:
            print(f"\n开始对共享LoRA层进行FSA剪枝")
            print(f"  目标秩: {target_rank}")
            print(f"  策略: 随机选择保留的rank维度")

        # 存储每个投影类型的共享mask
        shared_fsa_masks = {}

        # 定义投影类型的固定顺序，确保所有rank以相同顺序处理
        proj_types_order = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

        # 第一步：收集每个投影类型的第一个共享层的信息
        proj_type_first_layer = {}
        for name, module in model.named_modules():
            if not hasattr(module, 'lora_A') or not hasattr(module, 'lora_B'):
                continue

            lora_type = self.layer_lora_type.get(name, "independent")
            if lora_type != "shared":
                continue

            proj_type = self._get_projection_type(name)
            if proj_type is None or proj_type in proj_type_first_layer:
                continue

            adapter_name = module.active_adapter
            if adapter_name not in module.lora_A:
                continue

            lora_A = module.lora_A[adapter_name]
            lora_B = module.lora_B[adapter_name]

            if not hasattr(lora_A, 'loraA') or not hasattr(lora_B, 'loraB'):
                continue

            if len(lora_A.loraA) == 0 or len(lora_B.loraB) == 0:
                continue

            if cur_task >= len(lora_A.loraA) or cur_task >= len(lora_B.loraB):
                continue

            # 记录这个投影类型的第一个共享层
            proj_type_first_layer[proj_type] = (name, module)

        # 第二步：按固定顺序为每个投影类型创建mask
        for proj_type in proj_types_order:
            if proj_type not in proj_type_first_layer:
                continue

            name, module = proj_type_first_layer[proj_type]
            adapter_name = module.active_adapter
            lora_A = module.lora_A[adapter_name]
            lora_B = module.lora_B[adapter_name]

            # 获取当前任务的A和B矩阵
            weight_A = lora_A.loraA[cur_task].mlp.weight.data  # shape: [rank, in_features]
            weight_B = lora_B.loraB[cur_task].mlp.weight.data  # shape: [out_features, rank]

            current_rank = weight_A.shape[0]

            if is_rank0:
                print(f"  {proj_type}: 当前秩={current_rank}, 目标秩={target_rank}")

            # 随机选择要保留的rank维度
            keep_num_ranks = min(target_rank, current_rank)

            # 使用稳定的投影序号；Python hash 在不同进程中会随机化，
            # 会导致多 GPU rank 生成不同的剪枝 mask。
            torch.manual_seed(42 + proj_types_order.index(proj_type))
            selected_ranks = torch.randperm(current_rank)[:keep_num_ranks]
            selected_ranks = selected_ranks.sort()[0]  # 排序以保持顺序

            # 创建mask
            mask_A = torch.zeros_like(weight_A)
            mask_B = torch.zeros_like(weight_B)

            for r in selected_ranks:
                mask_A[r, :] = 1.0  # 保留A的第r行
                mask_B[:, r] = 1.0  # 保留B的第r列

            # 保存这个投影类型的共享mask
            shared_fsa_masks[proj_type] = {
                'mask_A': mask_A,
                'mask_B': mask_B,
                'selected_ranks': selected_ranks.tolist(),
                'current_rank': current_rank,
                'target_rank': keep_num_ranks
            }

            if is_rank0:
                print(f"    保留的rank索引: {selected_ranks.tolist()}")

        # 应用mask到所有共享层
        shared_layer_count = 0
        for name, module in model.named_modules():
            if not hasattr(module, 'lora_A') or not hasattr(module, 'lora_B'):
                continue

            lora_type = self.layer_lora_type.get(name, "independent")
            if lora_type != "shared":
                continue

            proj_type = self._get_projection_type(name)
            if proj_type is None or proj_type not in shared_fsa_masks:
                continue

            adapter_name = module.active_adapter
            if adapter_name not in module.lora_A:
                continue

            lora_A = module.lora_A[adapter_name]
            lora_B = module.lora_B[adapter_name]

            if not hasattr(lora_A, 'loraA') or cur_task >= len(lora_A.loraA):
                continue

            # 应用共享mask
            mask_info = shared_fsa_masks[proj_type]
            mask_A = mask_info['mask_A'].to(lora_A.loraA[cur_task].mlp.weight.device)
            mask_B = mask_info['mask_B'].to(lora_B.loraB[cur_task].mlp.weight.device)

            lora_A.loraA[cur_task].mlp.weight.data.mul_(mask_A)
            lora_B.loraB[cur_task].mlp.weight.data.mul_(mask_B)

            # 保存mask到fine_grained_masks中（用于后续训练时应用）
            if name not in self.fine_grained_masks:
                self.fine_grained_masks[name] = {}
            self.fine_grained_masks[name]['lora_A'] = mask_A
            self.fine_grained_masks[name]['lora_B'] = mask_B
            self.fine_grained_masks[name]['is_shared'] = True
            self.fine_grained_masks[name]['proj_type'] = proj_type

            shared_layer_count += 1

        if is_rank0:
            print(f"\n共享层FSA剪枝完成:")
            print(f"  处理了 {len(shared_fsa_masks)} 个投影类型")
            print(f"  应用到 {shared_layer_count} 个共享层")
            for proj_type, mask_info in shared_fsa_masks.items():
                sparsity = 1.0 - (mask_info['target_rank'] / mask_info['current_rank'])
                print(f"  {proj_type}: {mask_info['current_rank']} -> {mask_info['target_rank']} (稀疏度: {sparsity:.1%})")

    def compute_fine_grained_masks(self, model, cur_task: int):
        """
        计算细粒度参数剪枝的mask - 基于ΔW的结构化剪枝

        策略：
        1. 计算每个rank维度对ΔW的贡献
        2. 根据layer-wise importance自适应确定保留比例
        3. 剪枝整个rank维度（同时剪A的行和B的列）

        Args:
            model: 包含 CoINMOELoraLinear 层的模型
            cur_task: 当前任务 ID
        """
        if cur_task < self.fine_grained_start_task:
            return

        from torch.distributed import get_rank
        is_rank0 = get_rank() == 0 if torch.distributed.is_initialized() else True

        if is_rank0:
            print(f"\n开始计算细粒度参数剪枝mask (基于ΔW结构化剪枝)")
            print(f"  保留比例范围: [{self.fine_grained_min_keep_ratio:.1%}, {self.fine_grained_max_keep_ratio:.1%}]")
            print(f"  按投影类型分组，根据layer importance自适应调整")

        # 第一步：按投影类型分组收集independent层的importance
        proj_independent_layers = {
            'q_proj': {},
            'k_proj': {},
            'v_proj': {},
            'o_proj': {},
            'gate_proj': {},
            'up_proj': {},
            'down_proj': {},
        }

        for name in self.layer_lora_type:
            if self.layer_lora_type[name] == "independent":
                importance = self.layer_importance.get(name, 0.0)

                # 确定投影类型
                proj_type = self._get_projection_type(name)
                if proj_type and proj_type in proj_independent_layers:
                    proj_independent_layers[proj_type][name] = importance

        # 第二步：每个投影类型内部独立归一化importance到[0, 1]
        normalized_importance = {}

        for proj_type, layers in proj_independent_layers.items():
            if not layers:
                continue

            importances = list(layers.values())
            max_importance = max(importances)
            min_importance = min(importances)
            importance_range = max_importance - min_importance

            if importance_range > 0:
                # 归一化到[0, 1]
                for name, importance in layers.items():
                    normalized_importance[name] = (importance - min_importance) / importance_range
            else:
                # 所有层importance相同，设为0.5
                for name in layers.keys():
                    normalized_importance[name] = 0.5

        if is_rank0:
            total_independent = sum(len(layers) for layers in proj_independent_layers.values())
            print(f"  Independent层数: {total_independent}")
            for proj_type, layers in proj_independent_layers.items():
                if layers:
                    print(f"    {proj_type}: {len(layers)}层")

        total_params_before = 0
        total_params_after = 0
        layer_stats = []  # 保存每层的统计信息

        for name, module in model.named_modules():
            if not hasattr(module, 'lora_A') or not hasattr(module, 'lora_B'):
                continue

            # 只对independent的层进行细粒度剪枝
            lora_type = self.layer_lora_type.get(name, "independent")
            if lora_type != "independent":
                continue

            adapter_name = module.active_adapter
            if adapter_name not in module.lora_A:
                continue

            lora_A = module.lora_A[adapter_name]
            lora_B = module.lora_B[adapter_name]

            if not hasattr(lora_A, 'loraA') or not hasattr(lora_B, 'loraB'):
                continue

            if len(lora_A.loraA) == 0 or len(lora_B.loraB) == 0:
                continue

            # 检查cur_task是否在有效范围内
            if cur_task >= len(lora_A.loraA) or cur_task >= len(lora_B.loraB):
                continue

            # 获取当前任务的A和B矩阵
            weight_A = lora_A.loraA[cur_task].mlp.weight.data  # shape: [rank, in_features]
            weight_B = lora_B.loraB[cur_task].mlp.weight.data  # shape: [out_features, rank]

            rank = weight_A.shape[0]

            # 根据layer importance自适应确定保留比例
            # importance高的层保留更多参数
            layer_importance = normalized_importance.get(name, 0.5)
            # 保留比例 = min_ratio + (max_ratio - min_ratio) * importance
            # 例如：min=0.4, max=0.9, importance=0 -> keep=0.4 (40%)
            #       min=0.4, max=0.9, importance=1 -> keep=0.9 (90%)
            adaptive_keep_ratio = self.fine_grained_min_keep_ratio + \
                                  (self.fine_grained_max_keep_ratio - self.fine_grained_min_keep_ratio) * layer_importance
            adaptive_keep_ratio = max(self.fine_grained_min_keep_ratio,
                                     min(self.fine_grained_max_keep_ratio, adaptive_keep_ratio))
            adaptive_keep_ratio = self.fine_grained_min_keep_ratio + \
                                  (self.fine_grained_max_keep_ratio - self.fine_grained_min_keep_ratio) * layer_importance
            adaptive_keep_ratio = max(0.1, min(1.0, adaptive_keep_ratio))  # 限制在[0.1, 1.0]

            # 计算每个rank维度对ΔW的贡献
            # contribution[r] = ||B[:, r]|| * ||A[r, :]||
            rank_contributions = torch.zeros(rank, device=weight_A.device)
            for r in range(rank):
                norm_B = torch.norm(weight_B[:, r], p=2)
                norm_A = torch.norm(weight_A[r, :], p=2)
                rank_contributions[r] = norm_B * norm_A

                # 策略L1范数
                # norm_B = torch.norm(weight_B[:, r], p=1)
                # norm_A = torch.norm(weight_A[r, :], p=1)
                # rank_contributions[r] = norm_B * norm_A

                # 策略：范数相加而非相乘
                # norm_B = torch.norm(weight_B[:, r], p=2)
                # norm_A = torch.norm(weight_A[r, :], p=2)
                # rank_contributions[r] =(norm_B / norm_B.max()) + (norm_A / norm_A.max())

            # 根据adaptive_keep_ratio选择top-k个rank
            # keep_num_ranks = max(1, int(rank * 0.5))
            keep_num_ranks = max(1, int(rank * adaptive_keep_ratio))
            top_ranks = torch.topk(rank_contributions, keep_num_ranks, largest=True).indices

            # 创建mask：保留选中的rank维度
            mask_A = torch.zeros_like(weight_A)
            mask_B = torch.zeros_like(weight_B)

            for r in top_ranks:
                mask_A[r, :] = 1.0  # 保留A的第r行
                mask_B[:, r] = 1.0  # 保留B的第r列

            # 保存mask和importance
            if name not in self.fine_grained_masks:
                self.fine_grained_masks[name] = {}
            self.fine_grained_masks[name]['lora_A'] = mask_A
            self.fine_grained_masks[name]['lora_B'] = mask_B
            self.fine_grained_masks[name]['importance'] = layer_importance
            self.fine_grained_masks[name]['keep_ratio'] = adaptive_keep_ratio
            self.fine_grained_masks[name]['kept_ranks'] = keep_num_ranks

            # 应用mask到权重
            weight_A.mul_(mask_A)
            weight_B.mul_(mask_B)

            # 统计：计算ΔW = B @ A的有效参数
            delta_W = torch.matmul(weight_B, weight_A)
            num_params_delta_W = delta_W.numel()
            nonzero_delta_W = (torch.abs(delta_W) > 1e-8).sum().item()

            total_params_before += num_params_delta_W
            total_params_after += nonzero_delta_W

            # 保存统计信息
            layer_stats.append({
                'name': name,
                'importance': layer_importance,
                'keep_ratio': adaptive_keep_ratio,
                'kept_ranks': keep_num_ranks,
                'total_ranks': rank,
                'delta_W_sparsity': 1.0 - (nonzero_delta_W / num_params_delta_W)
            })

        if is_rank0 and total_params_before > 0:
            reduction_ratio = 1.0 - (total_params_after / total_params_before)
            print(f"\n细粒度剪枝完成:")
            print(f"  ΔW参数总数: {total_params_before:,} -> {int(total_params_after):,}")
            print(f"  ΔW参数减少: {reduction_ratio:.2%}")
            print(f"  ΔW实际保留: {total_params_after/total_params_before:.2%}")

            # 打印每层的详细信息（按importance排序）
            print(f"\n各层剪枝详情 (按importance排序):")
            layer_stats.sort(key=lambda x: x['importance'], reverse=True)

            # 按投影类型分组统计
            proj_stats = {}
            for stat in layer_stats:
                proj_type = self._get_projection_type(stat['name'])
                if proj_type not in proj_stats:
                    proj_stats[proj_type] = []
                proj_stats[proj_type].append(stat)

            for proj_type in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
                if proj_type not in proj_stats:
                    continue

                stats = proj_stats[proj_type]
                avg_importance = sum(s['importance'] for s in stats) / len(stats)
                avg_keep_ratio = sum(s['keep_ratio'] for s in stats) / len(stats)
                avg_sparsity = sum(s['delta_W_sparsity'] for s in stats) / len(stats)

                print(f"\n  {proj_type}: {len(stats)}层")
                print(f"    平均importance: {avg_importance:.3f}")
                print(f"    平均保留比例: {avg_keep_ratio:.1%}")
                print(f"    平均ΔW稀疏度: {avg_sparsity:.1%}")

        # 计算预训练权重重要性mask
        self.compute_pretrain_importance_masks(model)

    def compute_pretrain_importance_masks(self, model):
        """
        计算预训练权重的重要性mask

        对每个LoRA层：
        1. 获取预训练权重 W_0
        2. 计算重要性: M = tanh(ω * |1/log(|W_0/||W_0||_2| + ε)|)
        3. 存储mask供训练时使用

        Args:
            model: 包含 CoINMOELoraLinear 层的模型
        """
        from torch.distributed import get_rank
        is_rank0 = get_rank() == 0 if torch.distributed.is_initialized() else True

        if is_rank0:
            print(f"\n开始计算预训练权重重要性mask (omega={self.pretrain_reg_omega})")

        mask_count = 0

        for name, module in model.named_modules():
            # 只对有LoRA的层计算
            if not hasattr(module, 'lora_A') or not hasattr(module, 'lora_B'):
                continue

            # 只对independent层计算
            lora_type = self.layer_lora_type.get(name, "independent")
            if lora_type != "independent":
                continue

            # CoINMOELoraLinear继承自nn.Linear，预训练权重在self.weight中
            if not hasattr(module, 'weight'):
                continue

            # 获取预训练权重 W_0
            W_0 = module.weight.data  # [out_features, in_features]

            # L2归一化
            W_norm = W_0 / (torch.norm(W_0, p=2) + 1e-15)

            # 计算重要性分数: S = |1 / log(|W_norm| + ε)|
            S = torch.abs(1.0 / torch.log(torch.abs(W_norm) + 1e-15))

            # 应用tanh缩放: M = tanh(ω * S)
            M = torch.tanh(self.pretrain_reg_omega * S)

            # 存储mask（detach以避免梯度计算）
            self.pretrain_masks[name] = M.detach()
            mask_count += 1

        if is_rank0:
            print(f"  计算了 {mask_count} 个预训练权重重要性mask")

    def compute_pretrain_guided_regularization(self, model, cur_task: int):
        """
        计算预训练权重指导的正则化损失

        Loss = λ * Σ ||M ⊙ ΔW||_F / N

        其中：
        - M: 预训练权重重要性mask
        - ΔW = B @ A: LoRA的增量权重
        - ⊙: 逐元素乘法（Hadamard product）
        - ||·||_F: Frobenius范数
        - λ: 正则化系数
        - N: 层数

        Args:
            model: 包含 CoINMOELoraLinear 层的模型
            cur_task: 当前任务 ID

        Returns:
            reg_loss: 正则化损失值
        """
        if cur_task < self.fine_grained_start_task:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        if not self.pretrain_masks:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        reg_loss = 0.0
        layer_count = 0

        for name, module in model.named_modules():
            # 去除DeepSpeed/DDP包装的"module."前缀
            clean_name = name.replace('module.', '') if name.startswith('module.') else name

            # 只对有细粒度mask的层计算正则化
            if clean_name not in self.fine_grained_masks:
                continue

            # 只对有预训练mask的层计算正则化
            if clean_name not in self.pretrain_masks:
                continue

            # 只对independent层计算正则化
            lora_type = self.layer_lora_type.get(clean_name, "independent")
            if lora_type != "independent":
                continue

            if not hasattr(module, 'lora_A') or not hasattr(module, 'lora_B'):
                continue

            adapter_name = module.active_adapter
            if adapter_name not in module.lora_A:
                continue

            lora_A = module.lora_A[adapter_name]
            lora_B = module.lora_B[adapter_name]

            if not hasattr(lora_A, 'loraA') or not hasattr(lora_B, 'loraB'):
                continue

            if cur_task >= len(lora_A.loraA) or cur_task >= len(lora_B.loraB):
                continue

            # 获取当前任务的A和B矩阵
            weight_A = lora_A.loraA[cur_task].mlp.weight  # [rank, in_features]
            weight_B = lora_B.loraB[cur_task].mlp.weight  # [out_features, rank]

            # 计算ΔW = B @ A
            delta_W = torch.matmul(weight_B, weight_A)  # [out_features, in_features]

            # 获取预训练权重重要性mask
            M = self.pretrain_masks[clean_name].to(delta_W.device)

            # 计算正则化项: ||M ⊙ ΔW||_F
            masked_delta_W = M * delta_W
            reg_loss += torch.norm(masked_delta_W, p='fro')
            layer_count += 1

        # 应用正则化系数（不平均，直接求和）
        if layer_count > 0:
            reg_loss = self.pretrain_reg_lambda * reg_loss
            # 注意：不除以layer_count，保持与LoRASculpt一致（他们也是直接求和）

        return reg_loss

    def apply_fine_grained_masks(self, model, cur_task: int):
        """
        在训练过程中应用细粒度剪枝mask（包括共享层的FSA mask）

        在每次梯度更新后调用，确保被剪枝的参数保持为0

        Args:
            model: 包含 CoINMOELoraLinear 层的模型
            cur_task: 当前任务 ID
        """
        if cur_task < self.fine_grained_start_task:
            return

        if not self.fine_grained_masks:
            return

        for name, module in model.named_modules():
            # 去除DeepSpeed/DDP包装的"module."前缀
            clean_name = name.replace('module.', '') if name.startswith('module.') else name

            if clean_name not in self.fine_grained_masks:
                continue

            if not hasattr(module, 'lora_A') or not hasattr(module, 'lora_B'):
                continue

            adapter_name = module.active_adapter
            if adapter_name not in module.lora_A:
                continue

            lora_A = module.lora_A[adapter_name]
            lora_B = module.lora_B[adapter_name]

            if not hasattr(lora_A, 'loraA') or cur_task >= len(lora_A.loraA):
                continue

            # 应用mask（对independent层和shared层都适用）
            masks = self.fine_grained_masks[clean_name]
            if 'lora_A' in masks:
                mask_A = masks['lora_A'].to(lora_A.loraA[cur_task].mlp.weight.device)
                lora_A.loraA[cur_task].mlp.weight.data.mul_(mask_A)
            if 'lora_B' in masks:
                mask_B = masks['lora_B'].to(lora_B.loraB[cur_task].mlp.weight.device)
                lora_B.loraB[cur_task].mlp.weight.data.mul_(mask_B)

    def apply_pruning_to_model(self, model, cur_task: int):
        """
        应用剪枝决策到模型

        对于 lora_type="shared" 的层：
        - 将它们的 LoRA 权重指向同一个共享参数（按投影类型）
        - 共享参数在第一次遇到时创建，后续层复用

        Args:
            model: 包含 CoINMOELoraLinear 层的模型
            cur_task: 当前任务 ID
        """
        # 收集每个投影类型的 independent 和 shared 层列表
        proj_layers = {
            'q_proj': {'independent': [], 'shared': []},
            'k_proj': {'independent': [], 'shared': []},
            'v_proj': {'independent': [], 'shared': []},
            'o_proj': {'independent': [], 'shared': []},
            'gate_proj': {'independent': [], 'shared': []},
            'up_proj': {'independent': [], 'shared': []},
            'down_proj': {'independent': [], 'shared': []},
        }

        # 初始化共享 LoRA 参数存储
        shared_lora_A = {}  # {proj_type: shared_weight}
        shared_lora_B = {}

        for name, module in model.named_modules():
            if not hasattr(module, 'lora_A') or not hasattr(module, 'lora_B'):
                continue

            lora_type = self.layer_lora_type.get(name, "independent")
            proj_type = self._get_projection_type(name)

            if proj_type is None:
                continue

            # 提取层号（例如从 base_model.model.model.layers.0.self_attn.q_proj 提取 0）
            import re
            match = re.search(r'layers\.(\d+)\.', name)
            layer_idx = int(match.group(1)) if match else -1

            # 收集层信息
            if lora_type == "independent":
                proj_layers[proj_type]['independent'].append(layer_idx)
                continue
            else:
                proj_layers[proj_type]['shared'].append(layer_idx)

            # 共享 LoRA
            adapter_name = module.active_adapter
            if adapter_name not in module.lora_A:
                continue

            lora_A = module.lora_A[adapter_name]
            lora_B = module.lora_B[adapter_name]

            if not hasattr(lora_A, 'loraA') or cur_task >= len(lora_A.loraA):
                continue

            # 第一次遇到该投影类型，创建共享参数
            if proj_type not in shared_lora_A:
                # 使用第一个遇到的层的参数作为共享参数
                shared_lora_A[proj_type] = lora_A.loraA[cur_task].mlp.weight
                shared_lora_B[proj_type] = lora_B.loraB[cur_task].mlp.weight
            else:
                # 后续层指向共享参数
                lora_A.loraA[cur_task].mlp.weight = shared_lora_A[proj_type]
                lora_B.loraB[cur_task].mlp.weight = shared_lora_B[proj_type]

        # 保存共享参数引用
        self.shared_lora_params = {
            'lora_A': shared_lora_A,
            'lora_B': shared_lora_B,
        }

        # 打印统计信息（只在rank 0打印）
        from torch.distributed import get_rank
        is_rank0 = get_rank() == 0 if torch.distributed.is_initialized() else True

        if is_rank0:
            print(f"\nLoRA-Drop Summary:")
            for proj_type in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
                layers = proj_layers[proj_type]
                independent_layers = sorted(layers['independent'])
                shared_layers = sorted(layers['shared'])

                print(f"\n  {proj_type}:")
                if independent_layers:
                    print(f"    Independent: layers {independent_layers}")
                if shared_layers:
                    print(f"    Shared:      layers {shared_layers}")

        total_independent = sum(len(l['independent']) for l in proj_layers.values())
        total_shared = sum(len(l['shared']) for l in proj_layers.values())
        print(f"\n  Total: {total_independent} independent, {total_shared} shared")

        # 在粗粒度剪枝后，进行细粒度参数剪枝
        self.compute_fine_grained_masks(model, cur_task)

        return total_independent, total_shared

    def _get_projection_type(self, layer_name: str) -> str:
        """从层名称提取投影类型"""
        proj_types = ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                      'gate_proj', 'up_proj', 'down_proj']
        for proj_type in proj_types:
            if proj_type in layer_name:
                return proj_type
        return None

    def reset_for_new_task(self):
        """为新任务重置累积器"""
        self._importance_accumulator = {k: 0.0 for k in self.layer_importance}
        self._importance_count = {k: 0 for k in self.layer_importance}

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        total_layers = len(self.layer_lora_type)
        independent_layers = sum(1 for t in self.layer_lora_type.values() if t == "independent")
        shared_layers = total_layers - independent_layers

        # 统计细粒度剪枝信息
        fine_grained_stats = {}
        if self.fine_grained_masks:
            total_params = 0
            pruned_params = 0
            for layer_name, masks in self.fine_grained_masks.items():
                for matrix_name, mask in masks.items():
                    # 只统计tensor类型的mask（lora_A和lora_B），跳过标量值
                    if isinstance(mask, torch.Tensor):
                        total_params += mask.numel()
                        pruned_params += (mask == 0).sum().item()

            fine_grained_stats = {
                "fine_grained_num_layers": len(self.fine_grained_masks),
                "fine_grained_total_params": total_params,
                "fine_grained_pruned_params": pruned_params,
                "fine_grained_kept_params": total_params - pruned_params,
                "fine_grained_prune_ratio": pruned_params / max(1, total_params),
            }

        return {
            "total_layers": total_layers,
            "independent_lora_layers": independent_layers,
            "shared_lora_layers": shared_layers,
            "num_shared_lora_groups": len(self.shared_lora_params.get('lora_A', {})),
            "parameter_reduction_ratio": shared_layers / max(1, total_layers),
            "importance_threshold": self.importance_threshold,
            "fine_grained_prune_ratio": self.fine_grained_prune_ratio,
            "fine_grained_start_task": self.fine_grained_start_task,
            **fine_grained_stats,
        }

    def save_state(self, path: str):
        """保存状态"""
        # 确保父目录存在
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # 将mask转换为CPU tensor以便保存
        # 同时保存importance, keep_ratio等元数据
        fine_grained_masks_cpu = {}
        for layer_name, masks in self.fine_grained_masks.items():
            fine_grained_masks_cpu[layer_name] = {}
            for k, v in masks.items():
                if isinstance(v, torch.Tensor):
                    fine_grained_masks_cpu[layer_name][k] = v.cpu()
                else:
                    # 保存标量值（importance, keep_ratio, kept_ranks）
                    fine_grained_masks_cpu[layer_name][k] = v

        state = {
            "layer_importance": self.layer_importance,
            "layer_lora_type": self.layer_lora_type,
            "warmup_steps": self.warmup_steps,
            "eval_ratio": self.eval_ratio,
            "importance_threshold": self.importance_threshold,
            "fine_grained_prune_ratio": self.fine_grained_prune_ratio,
            "fine_grained_start_task": self.fine_grained_start_task,
            "fine_grained_min_keep_ratio": self.fine_grained_min_keep_ratio,
            "fine_grained_max_keep_ratio": self.fine_grained_max_keep_ratio,
            "fine_grained_masks": fine_grained_masks_cpu,
        }
        torch.save(state, path)

    def load_state(self, path: str):
        """加载状态"""
        state = torch.load(path)
        self.layer_importance = state["layer_importance"]
        self.layer_lora_type = state["layer_lora_type"]
        self.warmup_steps = state["warmup_steps"]
        self.eval_ratio = state["eval_ratio"]
        self.importance_threshold = state["importance_threshold"]
        self.fine_grained_prune_ratio = state.get("fine_grained_prune_ratio", 0.1)
        self.fine_grained_start_task = state.get("fine_grained_start_task", 2)
        self.fine_grained_min_keep_ratio = state.get("fine_grained_min_keep_ratio", 0.4)
        self.fine_grained_max_keep_ratio = state.get("fine_grained_max_keep_ratio", 0.9)
        self.fine_grained_masks = state.get("fine_grained_masks", {})
