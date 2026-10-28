# LoRA-Drop for Qwen3-VL

The active training path follows LLaVA/Drop:
`scripts/MCITlib/Train/Task*.sh` -> `llava/train/train_mem_MOE.py` ->
`llava/train/train_MOE.py::train` -> `CoIN.peft.get_peft_model` ->
`CoINMOELoraLinear.forward` -> `LoRADropPruner` -> `LLaVATrainer`.
Qwen3-VL supplies the base model and multimodal data processing.
`train_qwen.py` remains only as a compatibility entry point.

The Qwen version preserves the two-stage Drop procedure:

1. warm up the current CoIN-MoE LoRA expert;
2. measure relative LoRA-output importance on a data subset;
3. choose independent/shared adapters per Qwen attention/MLP projection;
4. construct fine-grained rank masks and pretrained-weight regularization;
5. apply masks after every optimizer step during full Qwen3-VL SFT.

As in the original Drop branch, DISCO's text-prototype router is retained so
evaluation can mix the learned task experts without a supplied task label.

## Training

Annotations use the upstream Qwen conversation format with `<image>` and
`<video>` placeholders. Use `--dataset_use`, or pass an MCIT annotation and
media root with `--data_path` and `--image_folder`.

```bash
torchrun --nproc_per_node=8 llava/train/train_mem_MOE.py \
  --deepspeed scripts/zero1_lora_drop.json \
  --model_name_or_path Qwen/Qwen3-VL-4B-Instruct \
  --data_path /data/task1.json --image_folder /data/images \
  --output_dir outputs/drop-task1 \
  --cur_task 0 --expert_num 8 \
  --lora_enable True --lora_r 128 --lora_alpha 256 \
  --lora_drop_enable True --lora_drop_warmup_steps 2 \
  --lora_drop_eval_ratio 0.1 --lora_drop_threshold 0.9 \
  --bf16 True --gradient_checkpointing True \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 --num_train_epochs 1 --report_to none
```

For task N, add `--previous_task_model_path outputs/drop-taskN-1` and increment
the zero-based `--cur_task`. `lora_r` must be divisible by `expert_num`.
The method saves `adapter_model.bin`, `lora_drop_pruner.pt`, `disco_router.pt`,
the Qwen processor, and `continual_config.json`.

The existing JSON-driven launchers were converted to Qwen3-VL and remain at
`scripts/MCITlib/Train/Task1.sh` and `Taskn.sh`.

For evaluation, `from model_loader import load_model` returns the Qwen model,
processor, and live task router for a saved checkpoint.

## UCIT

The six Qwen3-VL train/eval configurations are under
`configs/train_configs/Drop/Qwen3VL/UCIT`. The top-level launcher preserves the
explicit LLaVA/Drop blocks rather than looping over tasks:

```bash
cd Qwen3VL/Drop
bash scripts/MCITlib/Train/train_UCIT.sh
```

Task 1 keeps Drop disabled, matching the original configuration. Tasks 2-6
enable warm-up, importance collection, layer sharing, fine-grained masks, and
pretrained-weight-guided regularization. Enable or comment individual train
and evaluation commands directly in `train_UCIT.sh`.

