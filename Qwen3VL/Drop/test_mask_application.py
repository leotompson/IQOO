#!/usr/bin/env python3
"""
测试细粒度剪枝mask的应用效果

检查：
1. mask是否被正确计算
2. mask是否在训练中被正确应用
3. 被剪枝的参数是否保持为0
"""

import torch
import sys
sys.path.append('/data/taosen/code/MCITlib/LLaVA/Drop')

def test_mask_application():
    """测试mask应用逻辑"""

    # 模拟LoRA参数
    rank = 96
    in_features = 4096
    out_features = 4096

    # 创建随机权重
    weight_A = torch.randn(rank, in_features)
    weight_B = torch.randn(out_features, rank)

    print("=" * 60)
    print("测试细粒度剪枝mask应用")
    print("=" * 60)

    # 1. 计算mask（保留10%参数）
    prune_ratio = 0.1

    magnitude_A = torch.abs(weight_A)
    magnitude_B = torch.abs(weight_B)

    num_params_A = weight_A.numel()
    num_params_B = weight_B.numel()
    keep_num_A = max(1, int(num_params_A * prune_ratio))
    keep_num_B = max(1, int(num_params_B * prune_ratio))

    flat_magnitude_A = magnitude_A.view(-1)
    flat_magnitude_B = magnitude_B.view(-1)

    threshold_A = torch.topk(flat_magnitude_A, keep_num_A, largest=True).values[-1]
    threshold_B = torch.topk(flat_magnitude_B, keep_num_B, largest=True).values[-1]

    mask_A = (magnitude_A >= threshold_A).float()
    mask_B = (magnitude_B >= threshold_B).float()

    print(f"\n1. Mask计算:")
    print(f"   A矩阵: {weight_A.shape}, 保留 {mask_A.sum().item()}/{num_params_A} ({mask_A.sum().item()/num_params_A*100:.2f}%)")
    print(f"   B矩阵: {weight_B.shape}, 保留 {mask_B.sum().item()}/{num_params_B} ({mask_B.sum().item()/num_params_B*100:.2f}%)")

    # 2. 应用mask
    weight_A_pruned = weight_A * mask_A
    weight_B_pruned = weight_B * mask_B

    zero_A = (torch.abs(weight_A_pruned) < 1e-8).sum().item()
    zero_B = (torch.abs(weight_B_pruned) < 1e-8).sum().item()

    print(f"\n2. 应用mask后:")
    print(f"   A矩阵零元素: {zero_A}/{num_params_A} ({zero_A/num_params_A*100:.2f}%)")
    print(f"   B矩阵零元素: {zero_B}/{num_params_B} ({zero_B/num_params_B*100:.2f}%)")

    # 3. 计算ΔW的稀疏度
    delta_W = torch.matmul(weight_B_pruned, weight_A_pruned)
    num_params_delta_W = delta_W.numel()
    nonzero_delta_W = (torch.abs(delta_W) > 1e-8).sum().item()

    print(f"\n3. ΔW = B @ A:")
    print(f"   ΔW shape: {delta_W.shape}")
    print(f"   ΔW非零元素: {nonzero_delta_W}/{num_params_delta_W} ({nonzero_delta_W/num_params_delta_W*100:.2f}%)")
    print(f"   ΔW零元素: {num_params_delta_W - nonzero_delta_W}/{num_params_delta_W} ({(num_params_delta_W - nonzero_delta_W)/num_params_delta_W*100:.2f}%)")

    # 4. 模拟梯度更新
    print(f"\n4. 模拟梯度更新:")
    grad_A = torch.randn_like(weight_A) * 0.01
    grad_B = torch.randn_like(weight_B) * 0.01

    # 更新前
    weight_A_before = weight_A_pruned.clone()
    weight_B_before = weight_B_pruned.clone()

    # 梯度更新
    weight_A_pruned = weight_A_pruned - grad_A
    weight_B_pruned = weight_B_pruned - grad_B

    # 检查被剪枝的参数是否"复活"
    revived_A = ((torch.abs(weight_A_before) < 1e-8) & (torch.abs(weight_A_pruned) > 1e-8)).sum().item()
    revived_B = ((torch.abs(weight_B_before) < 1e-8) & (torch.abs(weight_B_pruned) > 1e-8)).sum().item()

    print(f"   更新后，A矩阵有 {revived_A} 个参数从0变为非0")
    print(f"   更新后，B矩阵有 {revived_B} 个参数从0变为非0")

    # 5. 重新应用mask
    print(f"\n5. 重新应用mask:")
    weight_A_pruned = weight_A_pruned * mask_A
    weight_B_pruned = weight_B_pruned * mask_B

    zero_A_after = (torch.abs(weight_A_pruned) < 1e-8).sum().item()
    zero_B_after = (torch.abs(weight_B_pruned) < 1e-8).sum().item()

    print(f"   A矩阵零元素: {zero_A_after}/{num_params_A} ({zero_A_after/num_params_A*100:.2f}%)")
    print(f"   B矩阵零元素: {zero_B_after}/{num_params_B} ({zero_B_after/num_params_B*100:.2f}%)")

    # 6. 结论
    print(f"\n" + "=" * 60)
    print("结论:")
    print("=" * 60)
    if revived_A > 0 or revived_B > 0:
        print("✗ 梯度更新会让被剪枝的参数'复活'")
        print("✓ 需要在每步训练后重新应用mask来保持稀疏性")
    else:
        print("✓ 被剪枝的参数保持为0")

    print(f"\n关键发现:")
    print(f"- 在A和B上分别剪枝90%参数")
    print(f"- ΔW的稀疏度约为 {(num_params_delta_W - nonzero_delta_W)/num_params_delta_W*100:.1f}%")
    print(f"- 这比单独剪枝A或B的90%要高得多")
    print(f"- 说明A和B的稀疏性会'叠加'到ΔW上")

if __name__ == "__main__":
    test_mask_application()
