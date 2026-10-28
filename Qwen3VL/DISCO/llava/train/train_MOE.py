"""Qwen3-VL training entry for DISCO's CoIN-MoE LoRA.

This file intentionally keeps the original LLaVA/DISCO training call path:
``train_mem_MOE.py -> train_MOE.train -> CoIN.peft.get_peft_model ->
CoINMOELoraLinear.forward``.  Only model loading and multimodal data processing
are delegated to the upstream Qwen3-VL finetuning implementation.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import torch
import transformers
from transformers import AutoProcessor, Trainer


METHOD_ROOT = Path(__file__).resolve().parents[2]
COMMON_ROOT = METHOD_ROOT.parent / "common"
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))
if str(COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(COMMON_ROOT))

from CoIN.peft import CoINMOELoraConfig, TaskType, get_peft_model  # noqa: E402
from qwen3vl_continual import (  # noqa: E402
    DataArguments,
    DiscoRouter,
    ModelArguments,
    TrainingArguments,
    _enable_flatten_attention,
    _load_qwen3vl,
    _register_explicit_dataset,
    _target_modules,
    _validate_args,
    _write_run_metadata,
    is_main_process,
    load_previous_adapter,
    make_supervised_data_module,
    save_adapter,
    set_training_task,
)


LOG = logging.getLogger("qwen3vl_disco.train_MOE")


def _add_disco_lora(model, model_args: ModelArguments, training_args: TrainingArguments):
    """Mirror LLaVA/DISCO's explicit CoIN-MoE adapter construction."""
    for parameter in model.parameters():
        parameter.requires_grad = False

    lora_config = CoINMOELoraConfig(
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        target_modules=_target_modules(model, model_args.lora_target_modules),
        lora_dropout=training_args.lora_dropout,
        bias=training_args.lora_bias,
        task_type=TaskType.CAUSAL_LM_CoIN,
        task_embedding_dim=model_args.task_embedding_dim,
        expert_num=model_args.expert_num,
        cur_task=model_args.cur_task,
    )
    print("Adding DISCO CoIN-MoE LoRA adapters...", flush=True)
    return get_peft_model(model, lora_config)


def train(attn_implementation: str = "flash_attention_2") -> None:
    """Train one continual task with Qwen3-VL and DISCO's LoRA forward path."""
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args.attn_implementation = attn_implementation
    _validate_args(model_args, data_args, training_args)
    training_args.cur_task = model_args.cur_task
    os.makedirs(training_args.output_dir, exist_ok=True)

    # Same position as LLaVA/DISCO: register data, load the base model, then
    # inject CoIN-MoE LoRA before loading a previous task checkpoint.
    _register_explicit_dataset(data_args)
    _enable_flatten_attention(data_args)
    model = _load_qwen3vl(model_args, training_args)
    model.config.use_cache = False
    model.train()

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    model = _add_disco_lora(model, model_args, training_args)
    set_training_task(model, model_args.cur_task)
    if model_args.previous_task_model_path is not None:
        load_previous_adapter(model, model_args.previous_task_model_path, METHOD_ROOT)

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    tokenizer = processor.tokenizer
    tokenizer.model_max_length = training_args.model_max_length
    tokenizer.padding_side = "right"
    data_module = make_supervised_data_module(processor, data_args=data_args)

    # DISCO's task prototype router is installed before Trainer, matching the
    # routing step performed inside LLaVA's multimodal forward preparation.
    controller = DiscoRouter(
        model,
        task_id=model_args.cur_task,
        expert_num=model_args.expert_num,
        temperature=model_args.disco_temperature,
    )
    if model_args.previous_task_model_path is not None:
        controller.load(model_args.previous_task_model_path)

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        **data_module,
    )
    checkpoints = list(Path(training_args.output_dir).glob("checkpoint-*"))
    if checkpoints:
        LOG.info("Checkpoint found; resuming DISCO training")
    trainer.train(resume_from_checkpoint=True if checkpoints else None)
    trainer.save_state()

    model.config.use_cache = True
    save_adapter(model, training_args.output_dir, training_args)
    if is_main_process(training_args):
        processor.save_pretrained(training_args.output_dir)
    controller.save(training_args.output_dir, training_args)
    _write_run_metadata("disco", model_args, data_args, training_args)


if __name__ == "__main__":
    train()
