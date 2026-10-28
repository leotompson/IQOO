"""Shared Qwen3-VL continual instruction-tuning utilities.

The data path and multimodal forward pass intentionally come from the upstream
``Qwen3-VL/qwen-vl-finetune`` implementation.  This module only adds the
continual-learning adapter lifecycle used by DISCO and LoRA-Drop.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import sys
import uuid
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import DataLoader, Subset
from transformers import AutoConfig, AutoProcessor, Trainer, TrainerCallback


REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_FINETUNE_ROOT = REPO_ROOT / "Qwen3-VL" / "qwen-vl-finetune"
if not UPSTREAM_FINETUNE_ROOT.is_dir():
    raise RuntimeError(f"Qwen3-VL finetuning code was not found: {UPSTREAM_FINETUNE_ROOT}")
if str(UPSTREAM_FINETUNE_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_FINETUNE_ROOT))

from qwenvl.data.data_processor import make_supervised_data_module  # noqa: E402
import qwenvl.data as qwen_data_registry  # noqa: E402


LOG = logging.getLogger("qwen3vl_continual")
DEFAULT_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="Qwen/Qwen3-VL-4B-Instruct")
    previous_task_model_path: Optional[str] = field(default=None)
    cur_task: int = field(default=0)
    expert_num: int = field(default=8)
    task_embedding_dim: int = field(default=64)
    lora_target_modules: str = field(default=DEFAULT_TARGET_MODULES)
    attn_implementation: str = field(default="flash_attention_2")
    disco_temperature: float = field(default=0.05)


@dataclass
class DataArguments:
    # The upstream registry interface.
    dataset_use: str = field(default="")
    # Compatibility with the existing MCIT data JSON files and shell scripts.
    data_path: Optional[str] = field(default=None)
    image_folder: Optional[str] = field(default=None)
    data_flatten: bool = field(default=False)
    data_packing: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    video_max_pixels: int = field(default=1024 * 28 * 28)
    video_min_pixels: int = field(default=256 * 28 * 28)
    video_fps: float = field(default=2.0)
    model_type: str = field(default="qwen3vl", init=False)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=8192)
    remove_unused_columns: bool = field(default=False)
    mm_projector_lr: Optional[float] = field(default=None)
    group_by_modality_length: bool = field(default=False)

    lora_enable: bool = field(default=True)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.05)
    lora_bias: str = field(default="none")

    # LoRA-Drop options. They are ignored by DISCO.
    lora_drop_enable: bool = field(default=False)
    lora_drop_warmup_steps: int = field(default=2)
    lora_drop_eval_ratio: float = field(default=0.1)
    lora_drop_threshold: float = field(default=0.9)
    fine_grained_prune_ratio: float = field(default=0.1)
    fine_grained_start_task: int = field(default=2)
    fine_grained_min_keep_ratio: float = field(default=0.4)
    fine_grained_max_keep_ratio: float = field(default=0.9)
    shared_target_rank: int = field(default=12)
    pretrain_reg_lambda: float = field(default=1e-3)
    pretrain_reg_omega: float = field(default=1.0)
    cur_task: int = field(default=0, init=False)


def is_main_process(args: TrainingArguments) -> bool:
    return args.local_rank in (-1, 0)


def rank0_print(args: TrainingArguments, *values) -> None:
    if is_main_process(args):
        print(*values, flush=True)


def _validate_args(model_args: ModelArguments, data_args: DataArguments, args: TrainingArguments) -> None:
    if not args.lora_enable:
        raise ValueError("DISCO/Drop require --lora_enable True")
    if model_args.expert_num <= 0:
        raise ValueError("expert_num must be positive")
    if not 0 <= model_args.cur_task < model_args.expert_num:
        raise ValueError("cur_task must satisfy 0 <= cur_task < expert_num")
    if args.lora_r <= 0 or args.lora_r % model_args.expert_num:
        raise ValueError("lora_r must be positive and divisible by expert_num")
    if not data_args.dataset_use and not data_args.data_path:
        raise ValueError("Set either --dataset_use or --data_path")
    if data_args.data_flatten and data_args.data_packing:
        raise ValueError("data_flatten and data_packing are mutually exclusive")


def _register_explicit_dataset(data_args: DataArguments) -> None:
    if not data_args.data_path:
        return
    name = "__mcit_continual_task__"
    qwen_data_registry.data_dict[name] = {
        "annotation_path": str(Path(data_args.data_path).expanduser()),
        "data_path": str(Path(data_args.image_folder or "").expanduser()),
    }
    data_args.dataset_use = name


def _load_qwen3vl(model_args: ModelArguments, args: TrainingArguments):
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=args.cache_dir,
    )
    model_type = getattr(config, "model_type", "")
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
    load_kwargs = {
        "cache_dir": args.cache_dir,
        "attn_implementation": model_args.attn_implementation,
        "dtype": dtype,
    }
    if model_type == "qwen3_vl_moe":
        from transformers import Qwen3VLMoeForConditionalGeneration

        cls = Qwen3VLMoeForConditionalGeneration
    elif model_type == "qwen3_vl":
        from transformers import Qwen3VLForConditionalGeneration

        cls = Qwen3VLForConditionalGeneration
    else:
        raise ValueError(
            f"Expected a Qwen3-VL checkpoint, but config.model_type is {model_type!r}"
        )
    return cls.from_pretrained(model_args.model_name_or_path, **load_kwargs)


def _enable_flatten_attention(data_args: DataArguments) -> None:
    if not (data_args.data_flatten or data_args.data_packing):
        return
    # Imported lazily because the upstream trainer requires flash-attn.
    from qwenvl.train.trainer import replace_qwen2_vl_attention_class

    replace_qwen2_vl_attention_class()


def _target_modules(model, requested: str) -> list[str]:
    candidates = [item.strip() for item in requested.split(",") if item.strip()]
    language_linear_suffixes = {
        name.rsplit(".", 1)[-1]
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and "language_model" in name
    }
    selected = [name for name in candidates if name in language_linear_suffixes]
    if not selected:
        raise ValueError(
            f"None of the requested LoRA targets occur in the Qwen language model: {candidates}"
        )
    return selected


def _add_coin_adapter(model, method_root: Path, model_args: ModelArguments, args: TrainingArguments):
    if str(method_root) not in sys.path:
        sys.path.insert(0, str(method_root))
    from CoIN.peft import CoINMOELoraConfig, TaskType, get_peft_model

    for parameter in model.parameters():
        parameter.requires_grad = False
    config = CoINMOELoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=_target_modules(model, model_args.lora_target_modules),
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        task_type=TaskType.CAUSAL_LM_CoIN,
        task_embedding_dim=model_args.task_embedding_dim,
        expert_num=model_args.expert_num,
        cur_task=model_args.cur_task,
    )
    return get_peft_model(model, config)


def iter_coin_layers(model) -> Iterable[torch.nn.Module]:
    for module in model.modules():
        if (
            hasattr(module, "lora_A")
            and hasattr(module, "lora_B")
            and hasattr(module, "lora_id")
            and hasattr(module, "mask_signal")
        ):
            yield module


def set_training_task(model, task_id: int) -> None:
    for module in iter_coin_layers(model):
        module.lora_id = task_id


def _load_tensor_file(path: Path) -> Dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")
    return torch.load(path, map_location="cpu")


def load_previous_adapter(model, checkpoint: Optional[str], method_root: Path) -> None:
    if not checkpoint:
        return
    checkpoint_path = Path(checkpoint)
    candidates = [checkpoint_path / "adapter_model.bin", checkpoint_path / "adapter_model.safetensors"]
    weight_path = next((path for path in candidates if path.is_file()), None)
    if weight_path is None:
        raise FileNotFoundError(
            f"No adapter_model.bin or adapter_model.safetensors in {checkpoint_path}"
        )
    if str(method_root) not in sys.path:
        sys.path.insert(0, str(method_root))
    from CoIN.peft import set_peft_model_state_dict

    result = set_peft_model_state_dict(model, _load_tensor_file(weight_path), adapter_name="default")
    missing = getattr(result, "missing_keys", [])
    unexpected = getattr(result, "unexpected_keys", [])
    if missing:
        LOG.warning("Missing keys while loading previous adapter: %s", missing[:20])
    if unexpected:
        LOG.warning("Unexpected keys while loading previous adapter: %s", unexpected[:20])


def _gather_parameter(parameter: torch.nn.Parameter) -> torch.Tensor:
    if hasattr(parameter, "ds_id"):
        from deepspeed import zero

        with zero.GatheredParameters([parameter]):
            return parameter.detach().cpu().clone()
    return parameter.detach().cpu().clone()


def save_adapter(model, output_dir: str, args: TrainingArguments) -> None:
    # ZeRO-3 gathering is collective, so every rank must enter the contexts.
    state = {
        name: _gather_parameter(parameter)
        # Keep duplicate names: LoRA-Drop aliases shared parameters across
        # layers, while a freshly constructed next-task model is not aliased.
        for name, parameter in model.named_parameters(remove_duplicate=False)
        if "lora_" in name
    }
    if is_main_process(args):
        model.save_pretrained(output_dir, state_dict=state, safe_serialization=False)


class DiscoRouter:
    """DISCO task router based on frozen Qwen text-embedding prototypes.

    The LLaVA implementation computes image and text prototypes but routes with
    ``cos_sim_text`` only.  Here the same decision is made with Qwen3-VL's own
    frozen token embeddings, avoiding a second CLIP tower and tokenizer.
    """

    STATE_NAME = "disco_router.pt"

    def __init__(self, model, task_id: int, expert_num: int, temperature: float):
        self.model = model
        self.task_id = task_id
        self.expert_num = expert_num
        self.temperature = temperature
        embeddings = model.get_input_embeddings()
        hidden_size = embeddings.weight.shape[1]
        self.prototypes = torch.zeros(expert_num, hidden_size, dtype=torch.float32)
        self.counts = torch.zeros(expert_num, dtype=torch.float64)
        # PEFT's generate() delegates directly to the base model, so the hook
        # must live on that model rather than only on the outer PEFT wrapper.
        self.hook_model = model.get_base_model() if hasattr(model, "get_base_model") else model
        self.handles = [model.register_forward_pre_hook(self._route, with_kwargs=True)]
        if self.hook_model is not model:
            self.handles.append(
                self.hook_model.register_forward_pre_hook(self._route, with_kwargs=True)
            )

    def _features(self, kwargs) -> Optional[torch.Tensor]:
        input_ids = kwargs.get("input_ids")
        if input_ids is None or input_ids.ndim != 2:
            return None
        with torch.no_grad():
            embeddings = self.model.get_input_embeddings()(input_ids).float()
            mask = torch.ones_like(input_ids, dtype=torch.bool)
            attention_mask = kwargs.get("attention_mask")
            if attention_mask is not None and attention_mask.shape == input_ids.shape:
                mask &= attention_mask.bool()
            labels = kwargs.get("labels")
            if labels is not None and labels.shape == input_ids.shape:
                # Prompt tokens are ignored by the language-model loss.
                mask &= labels.eq(-100)
            denominator = mask.sum(dim=1, keepdim=True).clamp_min(1)
            features = (embeddings * mask.unsqueeze(-1)).sum(dim=1) / denominator
            return F.normalize(features, dim=-1).detach()

    def _route(self, module, args, kwargs):
        input_ids = kwargs.get("input_ids")
        # Generation calls forward once per decoded token. Keep the route
        # chosen from the full prompt, matching the original LLaVA branch.
        if not module.training and input_ids is not None and input_ids.shape[-1] == 1:
            return args, kwargs
        features = self._features(kwargs)
        if features is None:
            set_training_task(self.model, self.task_id)
            return args, kwargs

        if module.training:
            set_training_task(self.model, self.task_id)
            batch_sum = features.sum(dim=0).cpu()
            batch_count = features.shape[0]
            old_count = self.counts[self.task_id].item()
            total = old_count + batch_count
            self.prototypes[self.task_id] = (
                self.prototypes[self.task_id] * old_count + batch_sum
            ) / total
            self.counts[self.task_id] = total
        else:
            valid = self.counts > 0
            if not valid.any():
                set_training_task(self.model, self.task_id)
                return args, kwargs
            query = F.normalize(features.mean(dim=0), dim=0).cpu()
            similarities = torch.full((self.expert_num,), -torch.inf)
            similarities[valid] = F.cosine_similarity(
                self.prototypes[valid], query.unsqueeze(0), dim=1
            )
            weights = F.softmax(similarities / self.temperature, dim=0).tolist()
            for coin_layer in iter_coin_layers(self.model):
                coin_layer.mask_signal = weights
        return args, kwargs

    def load(self, checkpoint: Optional[str]) -> None:
        if not checkpoint:
            return
        path = Path(checkpoint) / self.STATE_NAME
        if not path.is_file():
            raise FileNotFoundError(f"Missing DISCO router state: {path}")
        state = torch.load(path, map_location="cpu")
        prototypes = state["prototypes"]
        counts = state["counts"]
        if prototypes.shape != self.prototypes.shape or counts.shape != self.counts.shape:
            raise ValueError("The previous DISCO checkpoint uses a different expert_num/hidden_size")
        self.prototypes.copy_(prototypes.float())
        self.counts.copy_(counts.double())

    def save(self, output_dir: str, args: TrainingArguments) -> None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            device = next(self.model.parameters()).device
            weighted_sum = (
                self.prototypes[self.task_id] * self.counts[self.task_id].float()
            ).to(device)
            count = self.counts[self.task_id].to(device)
            torch.distributed.all_reduce(weighted_sum)
            torch.distributed.all_reduce(count)
            if count.item() > 0:
                self.prototypes[self.task_id].copy_((weighted_sum / count.float()).cpu())
            self.counts[self.task_id] = count.cpu()
        if is_main_process(args):
            torch.save(
                {"prototypes": self.prototypes, "counts": self.counts},
                Path(output_dir) / self.STATE_NAME,
            )


class DropTrainer(Trainer):
    def __init__(self, *args, drop_pruner=None, drop_task_id=0, **kwargs):
        self.drop_pruner = drop_pruner
        self.drop_task_id = drop_task_id
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        call_kwargs = {"return_outputs": return_outputs}
        if num_items_in_batch is not None:
            call_kwargs["num_items_in_batch"] = num_items_in_batch
        try:
            result = super().compute_loss(model, inputs, **call_kwargs)
        except TypeError:
            call_kwargs.pop("num_items_in_batch", None)
            result = super().compute_loss(model, inputs, **call_kwargs)
        loss, outputs = result if return_outputs else (result, None)
        if self.drop_pruner is not None:
            regularization = self.drop_pruner.compute_pretrain_guided_regularization(
                model, self.drop_task_id
            )
            loss = loss + regularization
        return (loss, outputs) if return_outputs else loss


class DropMaskCallback(TrainerCallback):
    def __init__(self, pruner, task_id: int):
        self.pruner = pruner
        self.task_id = task_id

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is not None:
            self.pruner.apply_fine_grained_masks(model, self.task_id)
        return control


def _move_batch(batch: Dict, device: torch.device, dtype: torch.dtype) -> Dict:
    result = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            result[key] = (
                value.to(device=device, dtype=dtype)
                if value.is_floating_point()
                else value.to(device=device)
            )
        else:
            result[key] = value
    return result


def prepare_lora_drop(model, data_module, model_args, args, method_root: Path):
    if str(method_root) not in sys.path:
        sys.path.insert(0, str(method_root))
    from CoIN.peft.tuners.clitmoelora import set_lora_drop_pruner
    from CoIN.peft.tuners.lora_drop import LoRADropPruner

    pruner = LoRADropPruner(
        warmup_steps=args.lora_drop_warmup_steps,
        eval_ratio=args.lora_drop_eval_ratio,
        importance_threshold=args.lora_drop_threshold,
        fine_grained_prune_ratio=args.fine_grained_prune_ratio,
        fine_grained_start_task=args.fine_grained_start_task,
        fine_grained_min_keep_ratio=args.fine_grained_min_keep_ratio,
        fine_grained_max_keep_ratio=args.fine_grained_max_keep_ratio,
        pretrain_reg_lambda=args.pretrain_reg_lambda,
        pretrain_reg_omega=args.pretrain_reg_omega,
    )
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            pruner.register_layer(name)

    previous_state = (
        Path(model_args.previous_task_model_path) / "lora_drop_pruner.pt"
        if model_args.previous_task_model_path
        else None
    )
    if previous_state and previous_state.is_file():
        pruner.load_state(str(previous_state))
    set_lora_drop_pruner(pruner)
    transformers.set_seed(args.seed)

    dataset = data_module["train_dataset"]
    eval_size = max(1, min(len(dataset), math.ceil(len(dataset) * args.lora_drop_eval_ratio)))
    generator = random.Random(args.seed)
    indices = generator.sample(range(len(dataset)), eval_size)
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=args.per_device_train_batch_size,
        shuffle=False,
        collate_fn=data_module["data_collator"],
        num_workers=0,
    )
    device = args.device
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    rank0_print(args, f"LoRA-Drop warm-up: {args.lora_drop_warmup_steps} steps; importance subset: {eval_size}")

    iterator = iter(loader)
    for step in range(args.lora_drop_warmup_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = _move_batch(batch, device, dtype)
        autocast = (
            torch.autocast(device_type="cuda", dtype=dtype)
            if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )
        with autocast:
            loss = model(**batch).loss
        loss.backward()
        if args.max_grad_norm and args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        rank0_print(args, f"LoRA-Drop warm-up {step + 1}/{args.lora_drop_warmup_steps}: loss={loss.item():.4f}")

    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device, dtype)
            autocast = (
                torch.autocast(device_type="cuda", dtype=dtype)
                if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
                else nullcontext()
            )
            with autocast:
                model(**batch)

    optimizer.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)
    del optimizer
    pruner.compute_layer_decisions()
    pruner.apply_pruning_to_model(model, model_args.cur_task)
    pruner.compute_fine_grained_masks(model, model_args.cur_task)
    pruner.compute_shared_layer_fsa_masks(
        model, model_args.cur_task, target_rank=args.shared_target_rank
    )
    stats = pruner.get_statistics()
    rank0_print(args, "LoRA-Drop decisions:", json.dumps(stats, ensure_ascii=False, indent=2))
    return pruner


def _write_run_metadata(method: str, model_args, data_args, args) -> None:
    if not is_main_process(args):
        return
    metadata = {
        "method": method,
        "base_model": model_args.model_name_or_path,
        "previous_task_model_path": model_args.previous_task_model_path,
        "cur_task": model_args.cur_task,
        "expert_num": model_args.expert_num,
        "disco_temperature": model_args.disco_temperature,
        "dataset_use": data_args.dataset_use,
        "data_path": data_args.data_path,
    }
    with open(Path(args.output_dir) / "continual_config.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


def run(method: str, method_root: Path) -> None:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, args = parser.parse_args_into_dataclasses()
    _validate_args(model_args, data_args, args)
    args.cur_task = model_args.cur_task
    os.makedirs(args.output_dir, exist_ok=True)
    _register_explicit_dataset(data_args)
    _enable_flatten_attention(data_args)

    model = _load_qwen3vl(model_args, args)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.enable_input_require_grads()
    model = _add_coin_adapter(model, method_root, model_args, args)
    set_training_task(model, model_args.cur_task)
    load_previous_adapter(model, model_args.previous_task_model_path, method_root)

    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, cache_dir=args.cache_dir)
    tokenizer = processor.tokenizer
    tokenizer.model_max_length = args.model_max_length
    tokenizer.padding_side = "right"
    data_module = make_supervised_data_module(processor, data_args=data_args)

    controller = None
    pruner = None
    trainer_class = Trainer
    trainer_kwargs = {}
    callbacks = []
    if method == "disco":
        controller = DiscoRouter(
            model,
            task_id=model_args.cur_task,
            expert_num=model_args.expert_num,
            temperature=model_args.disco_temperature,
        )
        controller.load(model_args.previous_task_model_path)
    elif method == "drop":
        # The original Drop branch is built on DISCO's task-aware CoIN-MoE
        # adapter and therefore retains its prototype router at evaluation.
        controller = DiscoRouter(
            model,
            task_id=model_args.cur_task,
            expert_num=model_args.expert_num,
            temperature=model_args.disco_temperature,
        )
        controller.load(model_args.previous_task_model_path)
        if args.lora_drop_enable:
            pruner = prepare_lora_drop(model, data_module, model_args, args, method_root)
            callbacks.append(DropMaskCallback(pruner, model_args.cur_task))
        trainer_class = DropTrainer
        trainer_kwargs.update(drop_pruner=pruner, drop_task_id=model_args.cur_task)
    else:
        raise ValueError(f"Unknown continual method: {method}")

    trainer = trainer_class(
        model=model,
        processing_class=tokenizer,
        args=args,
        callbacks=callbacks,
        **trainer_kwargs,
        **data_module,
    )
    checkpoints = list(Path(args.output_dir).glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=True if checkpoints else None)
    trainer.save_state()
    model.config.use_cache = True
    save_adapter(model, args.output_dir, args)
    if is_main_process(args):
        processor.save_pretrained(args.output_dir)
    if controller is not None:
        controller.save(args.output_dir, args)
    if pruner is not None and is_main_process(args):
        pruner.save_state(str(Path(args.output_dir) / "lora_drop_pruner.pt"))
    _write_run_metadata(method, model_args, data_args, args)


def load_continual_model(
    checkpoint: str,
    method_root: Path,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "flash_attention_2",
):
    """Load a saved DISCO/Drop Qwen3-VL adapter for generation/evaluation."""
    checkpoint_path = Path(checkpoint)
    with open(checkpoint_path / "continual_config.json", encoding="utf-8") as handle:
        metadata = json.load(handle)
    base_model = metadata["base_model"]
    config = AutoConfig.from_pretrained(base_model)
    if config.model_type == "qwen3_vl_moe":
        from transformers import Qwen3VLMoeForConditionalGeneration

        model_cls = Qwen3VLMoeForConditionalGeneration
    elif config.model_type == "qwen3_vl":
        from transformers import Qwen3VLForConditionalGeneration

        model_cls = Qwen3VLForConditionalGeneration
    else:
        raise ValueError(f"Not a Qwen3-VL checkpoint: {config.model_type!r}")
    raw_model = model_cls.from_pretrained(
        base_model,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    if str(method_root) not in sys.path:
        sys.path.insert(0, str(method_root))
    from CoIN.peft import PeftModel

    model = PeftModel.from_pretrained(raw_model, str(checkpoint_path), is_trainable=False)
    router = DiscoRouter(
        model,
        task_id=int(metadata["cur_task"]),
        expert_num=int(metadata["expert_num"]),
        temperature=float(metadata.get("disco_temperature", 0.05)),
    )
    router.load(str(checkpoint_path))
    model.eval().to(device)
    processor = AutoProcessor.from_pretrained(str(checkpoint_path))
    return model, processor, router


def _get_eval_chunk(items: list, num_chunks: int, chunk_idx: int) -> list:
    if num_chunks <= 0 or not 0 <= chunk_idx < num_chunks:
        raise ValueError("chunk_idx must satisfy 0 <= chunk_idx < num_chunks")
    chunk_size = math.ceil(len(items) / num_chunks)
    start = chunk_idx * chunk_size
    return items[start : start + chunk_size]


def eval_main(method_root: Path) -> None:
    """Generate Qwen3-VL answers in the JSONL format used by MCIT evaluators."""
    import argparse
    from PIL import Image
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--question-file", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--answers-file", required=True)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    eval_args = parser.parse_args()

    model, processor, router = load_continual_model(
        eval_args.model_path,
        method_root,
        device=eval_args.device,
        attn_implementation=eval_args.attn_implementation,
    )
    del router  # The forward hooks retain the router; this documents intentional use.

    with open(eval_args.question_file, encoding="utf-8") as handle:
        questions = json.load(handle)
    questions = _get_eval_chunk(questions, eval_args.num_chunks, eval_args.chunk_idx)
    output_path = Path(eval_args.answers_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as answer_file:
        for sample in tqdm(questions, desc=f"chunk {eval_args.chunk_idx}"):
            question_id = sample.get("question_id", sample.get("id"))
            question = sample.get("text") or sample.get("question")
            image_value = sample.get("image")
            if question_id is None or question is None or image_value is None:
                raise ValueError(
                    "Each UCIT test sample needs question_id/id, text/question, and image"
                )
            image_values = image_value if isinstance(image_value, list) else [image_value]
            content = []
            opened_images = []
            for image_name in image_values:
                image_path = Path(image_name)
                if not image_path.is_absolute():
                    image_path = Path(eval_args.image_folder) / image_path
                image = Image.open(image_path).convert("RGB")
                opened_images.append(image)
                content.append({"type": "image", "image": image})
            content.append({"type": "text", "text": question})
            messages = [{"role": "user", "content": content}]
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = {
                key: value.to(eval_args.device) if isinstance(value, torch.Tensor) else value
                for key, value in inputs.items()
            }
            generation_kwargs = {
                "do_sample": eval_args.temperature > 0,
                "temperature": eval_args.temperature if eval_args.temperature > 0 else None,
                "top_p": eval_args.top_p,
                "num_beams": eval_args.num_beams,
                "max_new_tokens": eval_args.max_new_tokens,
                "use_cache": True,
            }
            generation_kwargs = {
                key: value for key, value in generation_kwargs.items() if value is not None
            }
            with torch.inference_mode():
                generated_ids = model.generate(**inputs, **generation_kwargs)
            input_length = inputs["input_ids"].shape[1]
            generated_text = processor.batch_decode(
                generated_ids[:, input_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            for image in opened_images:
                image.close()
            answer_file.write(
                json.dumps(
                    {
                        "question_id": question_id,
                        "prompt": question,
                        "text": generated_text,
                        "answer_id": uuid.uuid4().hex,
                        "model_id": Path(eval_args.model_path).name,
                        "metadata": {},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            answer_file.flush()

