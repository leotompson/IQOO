# Global Identity Token 获取与 Expert 选择机制详解

本文档详细说明如何获得 global identity token 以及如何利用这个 token 来选择 experts 进行推理。

---

## 一、Global Identity Token 的初始化与存储

### 1.1 初始化位置
**文件**: [llava/model/language_model/llava_llama.py:57-68](llava/model/language_model/llava_llama.py#L57-L68)

```python
class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        
        # ... 其他初始化代码 ...
        
        self.cur_task = 0
        self.lora_id = self.cur_task
        self.expert_num = 8
        
        # Global identity tokens for text and image modalities
        self.global_text_feature = nn.ParameterList(
            [nn.Parameter(torch.zeros(768, dtype=torch.bfloat16)) for _ in range(8)]
        )
        self.global_image_feature = nn.ParameterList(
            [nn.Parameter(torch.zeros(768, dtype=torch.bfloat16)) for _ in range(8)]
        )
        
        # Boundary parameters for incremental averaging
        self.image_boundary = nn.ParameterList(
            [nn.Parameter(torch.ones(1, dtype=torch.bfloat16)) for _ in range(8)]
        )
        self.text_boundary = nn.ParameterList(
            [nn.Parameter(torch.ones(1, dtype=torch.bfloat16)) for _ in range(8)]
        )
```

**说明**:
- `global_text_feature`: 存储每个任务的全局文本特征 (8个任务 × 768维)
- `global_image_feature`: 存储每个任务的全局图像特征 (8个任务 × 768维)
- `image_boundary` 和 `text_boundary`: 用于增量平均计算的边界参数

---

## 二、训练阶段：Global Identity Token 的更新

### 2.1 更新位置
**文件**: [llava/model/llava_arch.py:185-197](llava/model/llava_arch.py#L185-L197)

```python
def prepare_inputs_labels_for_multimodal(
    self, input_ids, position_ids, attention_mask, past_key_values, labels, images
):
    # ... 前面的代码：提取图像和文本特征 ...
    
    if self.training:
        current_image_features = image_guide_features  # [batch_size, feature_dim]
        current_text_features = text_guide_features    # [batch_size, feature_dim]
        task_id = self.cur_task
        
        # 增量更新全局图像特征 (Incremental averaging)
        image_sum = self.global_image_feature[task_id] * self.image_boundary[task_id] + \
                    current_image_features.sum(dim=0)
        text_sum = self.global_text_feature[task_id] * self.text_boundary[task_id] + \
                   current_text_features.sum(dim=0)
        
        # 更新样本计数
        self.image_boundary[task_id].data += current_image_features.shape[0]
        self.text_boundary[task_id].data += current_text_features.shape[0]
        
        # 计算新的平均特征
        self.global_image_feature[task_id] = image_sum / self.image_boundary[task_id]
        self.global_text_feature[task_id] = text_sum / self.text_boundary[task_id]
```

**更新机制**:
1. 对于当前任务 `task_id`，累积该任务所有样本的图像和文本特征
2. 使用增量平均公式更新全局特征：
   ```
   new_mean = (old_mean * old_count + new_samples_sum) / (old_count + new_samples_count)
   ```
3. 这样每个任务都会有一个代表性的全局特征向量（identity token）

---

## 三、推理阶段：利用 Global Identity Token 选择 Experts

### 3.1 特征提取
**文件**: [llava/model/llava_arch.py:156-173](llava/model/llava_arch.py#L156-L173)

```python
def prepare_inputs_labels_for_multimodal(...):
    # 1. 编码当前输入的图像特征
    image_guide_features, image_features = self.encode_images(images)
    
    # 2. 提取文本特征
    text_tower = self.get_text_tower()
    
    # 解码输入文本
    input_pad = np.where(input_ids.cpu().detach().numpy()!=-200,
                         input_ids.cpu().detach().numpy(),
                         self.tokenizer.pad_token_id)
    decoded_inputs = self.tokenizer.batch_decode(input_pad, skip_special_tokens=True)
    decoded_hidden_inputs = ['\n'.join(decode_input.split('\n')[1:]) 
                             for decode_input in decoded_inputs]
    decoded_clip_inputs = [decode_input.split(' ASSISTANT')[0] 
                          for decode_input in decoded_hidden_inputs]
    
    # 使用 CLIP tokenizer 编码文本
    clip_text_inputs = self.clip_tokenizer(
        decoded_clip_inputs,
        padding="longest",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    
    # 获取文本特征
    text_guide_features = text_tower(clip_text_inputs)
```

### 3.2 计算相似度并选择 Experts
**文件**: [llava/model/llava_arch.py:175-182](llava/model/llava_arch.py#L175-L182)

```python
# 计算当前输入与所有任务 global identity token 的相似度
text_guide_features_mean = text_guide_features.mean(dim=0, keepdim=True)
image_guide_features_mean = image_guide_features.mean(dim=0, keepdim=True)

# 获取所有任务的 global identity tokens
global_text_feature = torch.stack([param for param in self.global_text_feature])
global_image_feature = torch.stack([param for param in self.global_image_feature]) 

# 计算余弦相似度
cos_sim_text = F.cosine_similarity(global_text_feature, text_guide_features_mean, dim=1)
cos_sim_img = F.cosine_similarity(global_image_feature, image_guide_features_mean, dim=1)

# 使用文本相似度作为主要依据
cos_sim = cos_sim_text[:self.expert_num]  # 可以组合: 0.5 * cos_sim_text + 0.5 * cos_sim_img

# Softmax 归一化得到每个 expert 的权重
cos_sim_softmax = F.softmax(cos_sim / 0.05)  # 温度系数 0.05
```

**相似度计算公式**:
```
cos_sim[i] = cosine_similarity(global_text_feature[i], current_text_feature)
           = (global_text_feature[i] · current_text_feature) / 
             (||global_text_feature[i]|| * ||current_text_feature||)
```

### 3.3 将 Expert 权重应用到所有 LoRA 层
**文件**: [llava/model/llava_arch.py:211-226](llava/model/llava_arch.py#L211-L226)

```python
else:  # 推理模式
    cos_sim_list = cos_sim_softmax.tolist()
    
    # 遍历所有 Transformer 层
    num_layers = 32
    proj_names = [
        'q_proj', 'k_proj', 'v_proj', 'o_proj',  # self_attn 投影
        'gate_proj', 'up_proj', 'down_proj'      # mlp 投影
    ]
    
    for layer_idx in range(num_layers):
        for proj_name in proj_names:
            # 获取对应的投影层
            if proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                proj_layer = getattr(self.model.layers[layer_idx].self_attn, proj_name)
            else:
                proj_layer = getattr(self.model.layers[layer_idx].mlp, proj_name)
            
            # 设置 mask_signal，即每个 expert 的权重
            proj_layer.mask_signal = cos_sim_list
```

**说明**:
- `mask_signal` 是一个长度为 `expert_num` 的列表，包含每个 expert 的权重
- 这个权重会在 LoRA 层的 forward 中使用

---

## 四、LoRA 层中使用 Expert 权重进行推理

### 4.1 LoRA Forward 函数
**文件**: [CoIN/peft/tuners/clitmoelora.py:408-458](CoIN/peft/tuners/clitmoelora.py#L408-L458)

```python
class CoINMOELoraLinear(nn.Linear, CoINMOELoraLayer):
    def forward(self, x: torch.Tensor, **kwargs):
        previous_dtype = x.dtype
        
        if self.active_adapter not in self.lora_A.keys():
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        
        if self.r[self.active_adapter] > 0:
            # 基础线性变换
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            
            x = x.to(self.lora_A[self.active_adapter].loraA[0].weight.dtype)
            
            # 构建对角 mask 矩阵，用于加权不同的 experts
            for i, flag in enumerate(self.mask_signal):
                if flag != 0:
                    start = i * self.diag_size
                    end = start + self.diag_size
                    if end > start and self.diag_size > 0:
                        # 设置对角块为 flag * I
                        self.lora_AB[self.active_adapter].weight.data[start:end, start:end] = \
                            torch.eye(self.diag_size) * flag
            
            self.lora_AB = self.lora_AB.to(x.device)
            
            if not self.training:  # 推理模式
                # LoRA A: 所有 experts 并行计算
                lora_a_output = self.lora_A[self.active_adapter](
                    self.lora_dropout[self.active_adapter](x)
                )
                
                # 通过 lora_AB 加权不同 experts 的输出
                lora_ab_output = self.lora_AB[self.active_adapter](lora_a_output)
                
                # LoRA B: 加权后的特征通过 B 矩阵
                lora_b_output = self.lora_B[self.active_adapter](lora_ab_output)
                
                # 添加到原始输出
                result += lora_b_output * self.scaling[self.active_adapter]
        
        result = result.to(previous_dtype)
        return result
```

### 4.2 CoINMOELinearA 的推理过程
**文件**: [CoIN/peft/tuners/clitmoelora.py:481-502](CoIN/peft/tuners/clitmoelora.py#L481-L502)

```python
class CoINMOELinearA(nn.Module):
    def forward(self, x):
        if self.training:
            # 训练模式：只使用当前任务的 expert
            assert 0 <= self.lora_id < self.expert_num
            output = self.loraA[self.lora_id](x)
            return output
        else:
            # 推理模式：拼接所有 experts 的权重
            total_out_features = self.expert_num * self.r
            temp_mlp = nn.Linear(self.in_features, total_out_features, bias=False).to(x.device)
            
            # 拼接所有 expert 的权重矩阵
            concatenated_weight = []
            for i in range(self.expert_num):
                concatenated_weight.append(self.loraA[i].mlp.weight)
            
            concatenated_weight = torch.cat(concatenated_weight, dim=0)
            
            with torch.no_grad(): 
                temp_mlp.weight.copy_(concatenated_weight)
            
            # 一次性计算所有 experts 的输出
            output = temp_mlp(x)
            return output
```

### 4.3 CoINMOELinearB 的推理过程
**文件**: [CoIN/peft/tuners/clitmoelora.py:526-547](CoIN/peft/tuners/clitmoelora.py#L526-L547)

```python
class CoINMOELinearB(nn.Module):
    def forward(self, x):
        if self.training:
            # 训练模式：只使用当前任务的 expert
            assert 0 <= self.lora_id < self.expert_num
            output = self.loraB[self.lora_id](x)
            return output
        else:
            # 推理模式：拼接所有 experts 的权重
            total_in_features = self.expert_num * self.r
            temp_mlp = nn.Linear(total_in_features, self.out_features, bias=False).to(x.device)
            
            # 拼接所有 expert 的权重矩阵（按列拼接）
            concatenated_weight = []
            for i in range(self.expert_num):
                concatenated_weight.append(self.loraB[i].mlp.weight)
            
            concatenated_weight = torch.cat(concatenated_weight, dim=1)
            
            with torch.no_grad(): 
                temp_mlp.weight.copy_(concatenated_weight)
            
            # 一次性计算所有 experts 的输出
            output = temp_mlp(x)
            return output
```

---

## 五、完整推理流程总结

### 5.1 流程图

```
输入 (图像 + 文本)
    ↓
[1] 特征提取
    ├─ 图像特征: image_guide_features (通过 Vision Tower)
    └─ 文本特征: text_guide_features (通过 Text Tower)
    ↓
[2] 计算与 Global Identity Tokens 的相似度
    ├─ cos_sim_text = cosine_similarity(global_text_feature, current_text)
    ├─ cos_sim_img = cosine_similarity(global_image_feature, current_image)
    └─ cos_sim = cos_sim_text (或组合)
    ↓
[3] Softmax 归一化得到 Expert 权重
    cos_sim_softmax = softmax(cos_sim / temperature)
    ↓
[4] 将权重分发到所有 LoRA 层
    for each layer:
        proj_layer.mask_signal = cos_sim_softmax.tolist()
    ↓
[5] LoRA 层推理
    ├─ LoRA A: 并行计算所有 experts → [batch, expert_num * r]
    ├─ lora_AB: 通过对角矩阵加权 → [batch, expert_num * r]
    └─ LoRA B: 加权求和 → [batch, out_features]
    ↓
输出
```

### 5.2 关键代码位置汇总

| 步骤 | 文件 | 行号 | 功能 |
|------|------|------|------|
| Global Token 初始化 | [llava/model/language_model/llava_llama.py](llava/model/language_model/llava_llama.py#L57-L68) | 57-68 | 初始化 global_text_feature 和 global_image_feature |
| Global Token 更新 | [llava/model/llava_arch.py](llava/model/llava_arch.py#L185-L197) | 185-197 | 训练时增量更新全局特征 |
| 特征提取 | [llava/model/llava_arch.py](llava/model/llava_arch.py#L156-L173) | 156-173 | 提取当前输入的图像和文本特征 |
| 相似度计算 | [llava/model/llava_arch.py](llava/model/llava_arch.py#L175-L182) | 175-182 | 计算余弦相似度并 Softmax |
| 权重分发 | [llava/model/llava_arch.py](llava/model/llava_arch.py#L211-L226) | 211-226 | 将 expert 权重设置到所有层 |
| LoRA Forward | [CoIN/peft/tuners/clitmoelora.py](CoIN/peft/tuners/clitmoelora.py#L408-L458) | 408-458 | 使用 mask_signal 加权 experts |
| LoRA A 推理 | [CoIN/peft/tuners/clitmoelora.py](CoIN/peft/tuners/clitmoelora.py#L481-L502) | 481-502 | 并行计算所有 experts |
| LoRA B 推理 | [CoIN/peft/tuners/clitmoelora.py](CoIN/peft/tuners/clitmoelora.py#L526-L547) | 526-547 | 加权求和所有 experts |

---

## 六、数学原理

### 6.1 Global Identity Token 更新（增量平均）

对于任务 $t$，在第 $n$ 个 batch 后：

$$
\text{global\_feature}_t^{(n)} = \frac{\text{global\_feature}_t^{(n-1)} \cdot N^{(n-1)} + \sum_{i=1}^{B} \text{feature}_i}{N^{(n-1)} + B}
$$

其中：
- $N^{(n-1)}$: 前 $n-1$ 个 batch 的总样本数
- $B$: 当前 batch 的样本数
- $\text{feature}_i$: 当前 batch 中第 $i$ 个样本的特征

### 6.2 Expert 选择（Softmax 加权）

给定当前输入特征 $\mathbf{f}_{\text{current}}$ 和 $K$ 个任务的 global identity tokens $\{\mathbf{g}_1, \mathbf{g}_2, ..., \mathbf{g}_K\}$：

$$
\text{sim}_k = \frac{\mathbf{g}_k \cdot \mathbf{f}_{\text{current}}}{\|\mathbf{g}_k\| \cdot \|\mathbf{f}_{\text{current}}\|}
$$

$$
w_k = \frac{\exp(\text{sim}_k / \tau)}{\sum_{j=1}^{K} \exp(\text{sim}_j / \tau)}
$$

其中 $\tau$ 是温度系数（代码中为 0.05）。

### 6.3 LoRA 加权输出

对于输入 $\mathbf{x}$，LoRA 的输出为：

$$
\mathbf{y} = \mathbf{W}\mathbf{x} + \sum_{k=1}^{K} w_k \cdot \mathbf{B}_k \mathbf{A}_k \mathbf{x} \cdot \alpha
$$

其中：
- $\mathbf{W}$: 预训练权重
- $\mathbf{A}_k, \mathbf{B}_k$: 第 $k$ 个 expert 的 LoRA 矩阵
- $w_k$: 第 $k$ 个 expert 的权重
- $\alpha$: scaling 系数

---

## 七、调试和验证

### 7.1 查看 Global Identity Token

```python
# 在推理时打印 global identity tokens
print("Global Text Features:")
for i, feat in enumerate(model.global_text_feature):
    print(f"Task {i}: {feat[:10]}...")  # 打印前10维

print("\nGlobal Image Features:")
for i, feat in enumerate(model.global_image_feature):
    print(f"Task {i}: {feat[:10]}...")
```

### 7.2 查看 Expert 权重

```python
# 在 llava_arch.py 的 prepare_inputs_labels_for_multimodal 中添加
print(f"Cosine similarities: {cos_sim_softmax}")
print(f"Selected expert weights: {cos_sim_list}")
```

### 7.3 验证 LoRA 层的 mask_signal

```python
# 在 clitmoelora.py 的 forward 中添加
if not self.training:
    print(f"Layer {self.layer_name} mask_signal: {self.mask_signal}")
```

---

## 八、相关配置参数

### 8.1 训练参数
**文件**: [llava/train/train_MOE.py](llava/train/train_MOE.py#L92-L93)

```python
task_embedding_dim: Optional[int] = field(default=64)
expert_num: Optional[int] = field(default=None)
```

### 8.2 设置当前任务
**文件**: [llava/train/train_MOE.py](llava/train/train_MOE.py#L1457)

```python
model.set_cur_task(model_args.cur_task, model_args.expert_num)
```

---

**文档生成时间**: 2026-05-30
**代码库**: MCITlib/LLaVA/Drop
