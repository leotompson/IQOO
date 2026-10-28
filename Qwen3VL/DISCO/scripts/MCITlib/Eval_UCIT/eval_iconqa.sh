#!/usr/bin/env bash

MODEL_CONFIG=$1
DATA_CONFIG=$2
TRAIN_CONFIG=$3
METHOD_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
REPO_ROOT=$(cd "$METHOD_ROOT/../.." && pwd)

read_config() {
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"
}
resolve_repo_path() {
    [[ "$1" = /* ]] && printf '%s\n' "$1" || printf '%s/%s\n' "$REPO_ROOT" "$1"
}

TASK="IconQA"
GPU_NUM=$(read_config "$TRAIN_CONFIG" gpu_num)
STAGE=$(read_config "$TRAIN_CONFIG" stage)
MODELPATH=$(resolve_repo_path "$(read_config "$TRAIN_CONFIG" model_path)")
DATA_PATH=$(read_config "$DATA_CONFIG" test_path)
IMAGE=$(read_config "$DATA_CONFIG" test_folder)
RESULT_PATH=$(resolve_repo_path "$(read_config "$TRAIN_CONFIG" result_path)")

gpu_list=""
for ((i=0; i<GPU_NUM; i++)); do
    gpu_list+="$i,"
done
gpu_list=${gpu_list%,}
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$gpu_list}"
IFS=',' read -ra GPULIST <<< "$CUDA_VISIBLE_DEVICES"
CHUNKS=${#GPULIST[@]}
RESULT_DIR="$RESULT_PATH/$TASK/$STAGE"
mkdir -p "$RESULT_DIR"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 "$METHOD_ROOT/eval_qwen.py" \
        --model-path "$MODELPATH" \
        --question-file "$DATA_PATH" \
        --image-folder "$IMAGE" \
        --answers-file "$RESULT_DIR/${CHUNKS}_${IDX}.jsonl" \
        --num-chunks "$CHUNKS" --chunk-idx "$IDX" --temperature 0 &
done
wait

output_file="$RESULT_DIR/merge.jsonl"
> "$output_file"
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat "$RESULT_DIR/${CHUNKS}_${IDX}.jsonl" >> "$output_file"
done

PYTHONPATH="$METHOD_ROOT:${PYTHONPATH:-}" python3 -m llava.eval.CoIN.eval_deepseek_r1 \
    --annotation-file "$DATA_PATH" \
    --result-file "$output_file" \
    --output-dir "$RESULT_DIR"
