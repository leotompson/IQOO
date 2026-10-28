# DISCO for Qwen3-VL

The active training call path follows the original LLaVA/DISCO layout:
`scripts/MCITlib/Train/Task*.sh` -> `llava/train/train_mem_MOE.py` ->
`llava/train/train_MOE.py::train` -> `CoIN.peft.get_peft_model` ->
`CoINMOELoraLinear.forward`. Qwen3-VL supplies the base model, processor, chat
template, multimodal RoPE, image/video inputs, and supervised data module.
`train_qwen.py` is only a compatibility entry point for direct invocations.

DISCO keeps one CoIN-MoE LoRA expert per continual task. During training the
current expert is selected by `--cur_task`. A normalized prototype of the
frozen Qwen prompt embeddings is accumulated for each task. During evaluation,
cosine similarity to the saved prototypes produces the soft expert mixture.
This follows the original implementation, whose final routing score uses only
the text similarity (`cos_sim_text`).

## Data

The supported annotation format is the upstream Qwen format:

```json
{
  "image": "images/example.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nWhat is shown?"},
    {"from": "gpt", "value": "A cat."}
  ]
}
```

Use either an upstream `--dataset_use` registry name or the existing MCIT pair
`--data_path /annotations.json --image_folder /images`.

## Training

Task ids are zero-based and `lora_r` must be divisible by `expert_num`.

```bash
torchrun --nproc_per_node=8 llava/train/train_mem_MOE.py \
  --deepspeed scripts/zero2.json \
  --model_name_or_path Qwen/Qwen3-VL-4B-Instruct \
  --data_path /data/task1.json --image_folder /data/images \
  --output_dir outputs/disco-task1 \
  --cur_task 0 --expert_num 8 \
  --lora_enable True --lora_r 128 --lora_alpha 256 \
  --bf16 True --gradient_checkpointing True \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 --num_train_epochs 1 --report_to none
```

For later tasks add `--previous_task_model_path outputs/disco-taskN`. The
checkpoint contains `adapter_model.bin`, `adapter_config.json`,
`disco_router.pt`, the Qwen processor, and `continual_config.json`.

The config-driven launchers remain available at
`scripts/MCITlib/Train/Task1.sh` and `Taskn.sh`. The model JSON only needs a
Qwen3-VL `model_name`; data/train JSON keys keep their existing names.

For evaluation, `from model_loader import load_model` returns the Qwen model,
processor, and live DISCO router for a saved checkpoint.

## UCIT six-task pipeline

The Qwen3-VL UCIT configs live under
`configs/train_configs/DISCO/Qwen3VL/UCIT`. The launcher keeps the original
LLaVA/DISCO layout: every task has an explicit training command followed by a
separate cumulative evaluation command. Enable or disable the individual
commands directly in `train_UCIT.sh`, just as in the original LLaVA script.

```bash
cd Qwen3VL/DISCO
bash scripts/MCITlib/Train/train_UCIT.sh
```

Training and evaluation can also be run independently:

```bash
bash scripts/MCITlib/Train/Taskn.sh \
  ../../configs/modal_configs/qwen3vl.json \
  ../../configs/data_configs/UCIT/ArxivQA.json \
  ../../configs/train_configs/DISCO/Qwen3VL/UCIT/train/task2.json
bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh 2
```

Exact-answer tasks use the existing MCIT evaluator. VizWiz and Flickr30k use
the caption evaluator. Install the optional scoring dependencies with
`pip install -e '.[eval]'` before running the complete evaluation flow.

