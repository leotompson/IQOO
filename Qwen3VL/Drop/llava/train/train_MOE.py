"""Qwen3-VL training entry for DISCO + LoRA-Drop.

The call order mirrors LLaVA/Drop:
``train_mem_MOE.py -> train_MOE.train -> CoIN-MoE LoRA injection ->
LoRA-Drop warm-up/importance/pruning -> LLaVATrainer full training``.
Qwen3-VL replaces only the base model and multimodal data implementation.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import transformers
from transformers import AutoProcessor


METHOD_ROOT = Path(__file__).resolve().parents[2]
COMMON_ROOT = METHOD_ROOT.parent / "common"
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))
if str(COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(COMMON_ROOT))

from CoIN.peft import CoINMOELoraConfig, TaskType, get_peft_model  # noqa: E402
from CoIN.peft.tuners.clitmoelora import set_lora_drop_pruner  # noqa: E402
from llava.train.llava_trainer import LLaVATrainer  # noqa: E402
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
    prepare_lora_drop,
    save_adapter,
    set_training_task,
)


def _add_drop_lora(model, model_args: ModelArguments, training_args: TrainingArguments):
    """Construct the same CoIN-MoE adapter used by LLaVA/Drop."""
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
    print("Adding Drop CoIN-MoE LoRA adapters...", flush=True)
    return get_peft_model(model, lora_config)


def run_lora_drop_warmup_phase(
    model,
    data_module,
    training_args: TrainingArguments,
    model_args: ModelArguments,
):
    """Run LLaVA/Drop's phase-1 warm-up, importance and pruning sequence."""
    return prepare_lora_drop(
        model,
        data_module,
        model_args,
        training_args,
        METHOD_ROOT,
    )


def _prepare_phase_two(model) -> None:
    """Match LLaVA/Drop's cleanup before DeepSpeed takes over phase two."""
    model.zero_grad(set_to_none=True)
    for parameter in model.parameters():
        parameter.grad = None
        if not parameter.is_contiguous():
            parameter.data = parameter.data.contiguous()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def train(attn_implementation: str = "flash_attention_2") -> None:
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args.attn_implementation = attn_implementation
    _validate_args(model_args, data_args, training_args)
    training_args.cur_task = model_args.cur_task
    os.makedirs(training_args.output_dir, exist_ok=True)

    _register_explicit_dataset(data_args)
    _enable_flatten_attention(data_args)
    model = _load_qwen3vl(model_args, training_args)
    model.config.use_cache = False
    model.train()
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    model = _add_drop_lora(model, model_args, training_args)
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

    # Drop keeps DISCO's task router. It must be active during warm-up and the
    # complete phase-2 training, as in the LLaVA multimodal forward path.
    controller = DiscoRouter(
        model,
        task_id=model_args.cur_task,
        expert_num=model_args.expert_num,
        temperature=model_args.disco_temperature,
    )
    if model_args.previous_task_model_path is not None:
        controller.load(model_args.previous_task_model_path)

    pruner = None
    if training_args.lora_drop_enable:
        pruner = run_lora_drop_warmup_phase(
            model, data_module, training_args, model_args
        )
        _prepare_phase_two(model)
    else:
        set_lora_drop_pruner(None)

    # Reuse the original Drop trainer: compute_loss adds pretrain-guided
    # regularization and training_step reapplies fine-grained masks.
    trainer = LLaVATrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        **data_module,
    )
    trainer.train()
    trainer.save_state()

    model.config.use_cache = True
    save_adapter(model, training_args.output_dir, training_args)
    if is_main_process(training_args):
        processor.save_pretrained(training_args.output_dir)
    controller.save(training_args.output_dir, training_args)
    if pruner is not None and is_main_process(training_args):
        pruner.save_state(str(Path(training_args.output_dir) / "lora_drop_pruner.pt"))
    _write_run_metadata("drop", model_args, data_args, training_args)


if __name__ == "__main__":
    train()
