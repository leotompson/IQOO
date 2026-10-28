#!/usr/bin/env bash

TASK_ID=$1
HARD_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)
MODEL_CONFIG="$HARD_PATH/configs/modal_configs/qwen3vl.json"
CONFIG_ROOT="$HARD_PATH/configs/train_configs/DISCO/Qwen3VL/UCIT/eval"
DATA_ROOT="$HARD_PATH/configs/data_configs/UCIT"

if [ "$TASK_ID" == "1" ]; then
    bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh "$MODEL_CONFIG" "$DATA_ROOT/ImageNet-R.json" "$CONFIG_ROOT/task1.json"
elif [ "$TASK_ID" == "2" ]; then
    bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh "$MODEL_CONFIG" "$DATA_ROOT/ImageNet-R.json" "$CONFIG_ROOT/task2.json"
    bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh "$MODEL_CONFIG" "$DATA_ROOT/ArxivQA.json" "$CONFIG_ROOT/task2.json"
elif [ "$TASK_ID" == "3" ]; then
    bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh "$MODEL_CONFIG" "$DATA_ROOT/ImageNet-R.json" "$CONFIG_ROOT/task3.json"
    bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh "$MODEL_CONFIG" "$DATA_ROOT/ArxivQA.json" "$CONFIG_ROOT/task3.json"
    bash scripts/MCITlib/Eval_UCIT/eval_vizwiz.sh "$MODEL_CONFIG" "$DATA_ROOT/VizWiz.json" "$CONFIG_ROOT/task3.json"
elif [ "$TASK_ID" == "4" ]; then
    bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh "$MODEL_CONFIG" "$DATA_ROOT/ImageNet-R.json" "$CONFIG_ROOT/task4.json"
    bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh "$MODEL_CONFIG" "$DATA_ROOT/ArxivQA.json" "$CONFIG_ROOT/task4.json"
    bash scripts/MCITlib/Eval_UCIT/eval_vizwiz.sh "$MODEL_CONFIG" "$DATA_ROOT/VizWiz.json" "$CONFIG_ROOT/task4.json"
    bash scripts/MCITlib/Eval_UCIT/eval_iconqa.sh "$MODEL_CONFIG" "$DATA_ROOT/IconQA.json" "$CONFIG_ROOT/task4.json"
elif [ "$TASK_ID" == "5" ]; then
    bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh "$MODEL_CONFIG" "$DATA_ROOT/ArxivQA.json" "$CONFIG_ROOT/task5.json"
    bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh "$MODEL_CONFIG" "$DATA_ROOT/ImageNet-R.json" "$CONFIG_ROOT/task5.json"
    bash scripts/MCITlib/Eval_UCIT/eval_vizwiz.sh "$MODEL_CONFIG" "$DATA_ROOT/VizWiz.json" "$CONFIG_ROOT/task5.json"
    bash scripts/MCITlib/Eval_UCIT/eval_iconqa.sh "$MODEL_CONFIG" "$DATA_ROOT/IconQA.json" "$CONFIG_ROOT/task5.json"
    bash scripts/MCITlib/Eval_UCIT/eval_clevr.sh "$MODEL_CONFIG" "$DATA_ROOT/CLEVR-Math.json" "$CONFIG_ROOT/task5.json"
else
    bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh "$MODEL_CONFIG" "$DATA_ROOT/ImageNet-R.json" "$CONFIG_ROOT/task6.json"
    bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh "$MODEL_CONFIG" "$DATA_ROOT/ArxivQA.json" "$CONFIG_ROOT/task6.json"
    bash scripts/MCITlib/Eval_UCIT/eval_vizwiz.sh "$MODEL_CONFIG" "$DATA_ROOT/VizWiz.json" "$CONFIG_ROOT/task6.json"
    bash scripts/MCITlib/Eval_UCIT/eval_iconqa.sh "$MODEL_CONFIG" "$DATA_ROOT/IconQA.json" "$CONFIG_ROOT/task6.json"
    bash scripts/MCITlib/Eval_UCIT/eval_clevr.sh "$MODEL_CONFIG" "$DATA_ROOT/CLEVR-Math.json" "$CONFIG_ROOT/task6.json"
    bash scripts/MCITlib/Eval_UCIT/eval_flickr30k.sh "$MODEL_CONFIG" "$DATA_ROOT/Flickr30k.json" "$CONFIG_ROOT/task6.json"
fi
