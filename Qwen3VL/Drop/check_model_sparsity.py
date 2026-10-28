#!/usr/bin/env python3
"""检查训练后的模型参数稀疏度"""

import torch
import sys
sys.path.append('/data/taosen/code/MCITlib/LLaVA/Drop')

# 加载Task2的pruner状态
pruner_path = "/data/taosen/code/MCITlib/checkpoint/UCIT/LLaVA-1.5/Drop/Task2_llava_lora/lora_drop_pruner.pt"
pruner_state = torch.load(pruner_path, map_location='cpu', weights_only=False)

# 加载Task2的模型权重
model_path = "/data/taosen/code/MCITlib/checkpoint/UCIT/LLaVA-1.5/Drop/Task2_llava_lora/adapter_model.bin"
model_weights = torch.load(model_path, map_location='cpu', weights_only=False)

print("=" * 60)
print("检查Task2训练后的模型参数稀疏度")
print("=" * 60)

fine_grained_masks = pruner_state.get('fine_grained_masks', {})
print(f"\n有mask的层数: {len(fine_grained_masks)}")

# 检查前5个有mask的层
print(f"\n检查前5个层的参数稀疏度:")
print("=" * 60)

checked = 0
for layer_name, masks in fine_grained_masks.items():
    if checked >= 5:
        break

    # 构造权重的key
    # mask中的name: base_model.model.model.layers.0.self_attn.k_proj
    # 权重中的key: base_model.model.model.layers.0.self_attn.k_proj.lora_A.default.loraA.0.mlp.weight

    # 查找对应的权重
    lora_A_key = None
    lora_B_key = None
    for key in model_weights.keys():
        if layer_name in key and 'lora_A' in key and 'loraA.1.weight' in key:  # task2 = cur_task=1
            lora_A_key = key
        if layer_name in key and 'lora_B' in key and 'loraB.1.weight' in key:
            lora_B_key = key

    if lora_A_key and lora_B_key:
        weight_A = model_weights[lora_A_key]
        weight_B = model_weights[lora_B_key]
        mask_A = masks['lora_A']
        mask_B = masks['lora_B']

        # 检查稀疏度
        zero_A = (torch.abs(weight_A) < 1e-8).sum().item()
        zero_B = (torch.abs(weight_B) < 1e-8).sum().item()
        total_A = weight_A.numel()
        total_B = weight_B.numel()

        # 检查mask期望的稀疏度
        expected_zero_A = (mask_A == 0).sum().item()
        expected_zero_B = (mask_B == 0).sum().item()

        print(f"\n{checked+1}. {layer_name}")
        print(f"   lora_A:")
        print(f"     实际零元素: {zero_A}/{total_A} ({zero_A/total_A*100:.2f}%)")
        print(f"     期望零元素: {expected_zero_A}/{total_A} ({expected_zero_A/total_A*100:.2f}%)")
        print(f"     匹配: {'✓' if abs(zero_A - expected_zero_A) < 10 else '✗'}")

        print(f"   lora_B:")
        print(f"     实际零元素: {zero_B}/{total_B} ({zero_B/total_B*100:.2f}%)")
        print(f"     期望零元素: {expected_zero_B}/{total_B} ({expected_zero_B/total_B*100:.2f}%)")
        print(f"     匹配: {'✓' if abs(zero_B - expected_zero_B) < 10 else '✗'}")

        checked += 1

print("\n" + "=" * 60)
print("结论:")
print("=" * 60)
print("如果实际零元素数量远小于期望，说明mask没有被正确应用")
print("如果实际零元素数量接近期望，说明mask被正确应用了")