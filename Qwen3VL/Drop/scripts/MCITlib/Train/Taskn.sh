#!/usr/bin/env bash
set -euo pipefail

METHOD_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
MODEL_CONFIG=$1
DATA_CONFIG=$2
TRAIN_CONFIG=$3
read_config() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"; }
read_optional() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], sys.argv[3]))' "$1" "$2" "$3"; }

GPU_NUM=$(read_config "$TRAIN_CONFIG" gpu_num)
RANK=$(read_config "$TRAIN_CONFIG" rank)
EXPERT=$(read_config "$TRAIN_CONFIG" expert_num)
MODEL_NAME=$(read_config "$MODEL_CONFIG" model_name)
PREVIOUS=$(read_config "$TRAIN_CONFIG" previous_model)
DATA_PATH=$(read_config "$DATA_CONFIG" train_path)
IMAGE_FOLDER=$(read_config "$DATA_CONFIG" train_folder)
OUTPUT_DIR=$(read_config "$TRAIN_CONFIG" output_dir)
CUR_TASK=$(read_config "$TRAIN_CONFIG" cur_task)
EPOCH=$(read_config "$TRAIN_CONFIG" epoch)
BATCH_SIZE=$(read_config "$TRAIN_CONFIG" batch_size)
GRAD_ACC=$(read_config "$TRAIN_CONFIG" grad_acc)
LR=$(read_config "$TRAIN_CONFIG" lr)
DROP_ENABLE=$(read_optional "$TRAIN_CONFIG" lora_drop_enable True)

torchrun --nproc_per_node="$GPU_NUM" --master_port="${MASTER_PORT:-9001}" \
    "$METHOD_ROOT/train_qwen.py" \
    --deepspeed "$METHOD_ROOT/scripts/zero2_lora_drop.json" \
    --model_name_or_path "$MODEL_NAME" \
    --previous_task_model_path "$PREVIOUS" \
    --data_path "$DATA_PATH" --image_folder "$IMAGE_FOLDER" \
    --output_dir "$OUTPUT_DIR" --cur_task "$CUR_TASK" --expert_num "$EXPERT" \
    --lora_enable True --lora_drop_enable "$DROP_ENABLE" \
    --lora_r "$RANK" --lora_alpha "$((RANK * 2))" \
    --bf16 True --num_train_epochs "$EPOCH" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACC" --learning_rate "$LR" \
    --eval_strategy no --save_strategy steps --save_steps 50000 \
    --weight_decay 0 --warmup_ratio 0.03 --lr_scheduler_type cosine \
    --logging_steps 1 --model_max_length 8192 --gradient_checkpointing True \
    --dataloader_num_workers 4 --report_to none

