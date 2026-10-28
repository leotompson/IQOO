# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy

# =========================================================================
# 动态补丁：解决 transformers 向 accelerate 1.0+ 传递废弃参数的问题
# 保持环境不变的专属修复方案
# =========================================================================
import accelerate

# 保存原始的初始化函数
_original_accelerator_init = accelerate.Accelerator.__init__

# 定义一个拦截函数
def _patched_accelerator_init(self, *args, **kwargs):
    # 如果 kwargs 里有 dispatch_batches，直接悄悄删掉
    kwargs.pop('dispatch_batches', None)
    # 调用原本的初始化函数
    _original_accelerator_init(self, *args, **kwargs)

# 替换回库中
accelerate.Accelerator.__init__ = _patched_accelerator_init
# =========================================================================

from dataclasses import dataclass, field
import json, deepspeed
import logging
import pathlib, random
from typing import Dict, Optional, Sequence, List

import torch
import sys
sys.path.append('/data/taosen/code/MCITlib/LLaVA/Drop')
import transformers
import subprocess

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token

from CoIN.peft import PeftModel, TaskType, get_peft_model, CoINMOELoraConfig, WEIGHTS_NAME, set_peft_model_state_dict

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS=None

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    previous_task_model_path: Optional[str] = field(default=None)
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    text_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    mm_text_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    cur_task: Optional[int] = field(default=0)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")

    task_embedding_dim: Optional[int] = field(default=64)
    expert_num: Optional[int] = field(default=None)


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    memory_data_path: str = field(default=None,
                           metadata={"help": "Path to the memory data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)

    # LoRA-Drop parameters
    lora_drop_enable: bool = field(default=False, metadata={"help": "Enable LoRA-Drop pruning"})
    lora_drop_warmup_steps: int = field(default=2, metadata={"help": "Warmup steps before importance evaluation"})
    lora_drop_eval_ratio: float = field(default=0.1, metadata={"help": "Ratio of data for importance evaluation"})
    lora_drop_threshold: float = field(default=0.9, metadata={"help": "Cumulative importance threshold"})

    # Fine-grained pruning parameters
    fine_grained_prune_ratio: float = field(default=0.1, metadata={"help": "Fine-grained pruning keep ratio (deprecated, use min/max instead)"})
    fine_grained_start_task: int = field(default=2, metadata={"help": "Start fine-grained pruning from this task (0-indexed, so 2 means task2)"})
    fine_grained_min_keep_ratio: float = field(default=0.4, metadata={"help": "Minimum keep ratio for least important layers (e.g., 0.4 = keep 40% ranks)"})
    fine_grained_max_keep_ratio: float = field(default=0.9, metadata={"help": "Maximum keep ratio for most important layers (e.g., 0.9 = keep 90% ranks)"})

    # Pretrain-guided regularization parameters
    pretrain_reg_lambda: float = field(default=1e-3, metadata={"help": "Regularization coefficient for pretrain-guided loss (default: 1e-3)"})
    pretrain_reg_omega: float = field(default=1.0, metadata={"help": "Scaling factor for pretrain importance mask (default: 1.0)"})


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer_image_token(rou, tokenizer)) + len(tokenizer_image_token(conv.sep, tokenizer))
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        if data_args.memory_data_path is not None:
            list_memory_data_dict = json.load(open(data_args.memory_data_path, "r"))

            list_data_dict = list_data_dict + list_memory_data_dict
            
            random.shuffle(list_data_dict)

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # 兼容int和numpy整数类型
        import numpy as np
        if isinstance(i, (int, np.integer)):
            sources = [self.list_data_dict[i]]
        else:
            # 如果是list/array，说明数据格式有问题，取第一个元素
            sources = [self.list_data_dict[i[0] if hasattr(i, '__getitem__') else i]]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        import numpy as np
        if isinstance(i, (int, np.integer)):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def run_lora_drop_warmup_phase(model, data_module, training_args, model_args):
    """
    LoRA-Drop Phase 1: 使用DDP进行多GPU warmup训练和重要性评估

    上策方案：Warmup多GPU + Importance评估多GPU
    - Warmup阶段：参数未共享，DDP正常工作，backward自动同步梯度
    - Importance评估阶段：只forward不backward，DDP不触发梯度同步
    - 剪枝阶段：unwrap DDP后执行，避免共享参数冲突
    """
    from CoIN.peft.tuners.clitmoelora import get_lora_drop_pruner
    from torch.utils.data import Subset, DataLoader, DistributedSampler
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers.trainer_pt_utils import get_parameter_names
    import numpy as np

    pruner = get_lora_drop_pruner()
    if pruner is None:
        return

    # ============ 调试模式：直接加载pruner状态，跳过warm-up ============
    debug_pruner_path = os.getenv('DEBUG_PRUNER_PATH', None)
    if debug_pruner_path and os.path.exists(debug_pruner_path):
        rank0_print(f"\n{'='*80}")
        rank0_print(f"🔧 调试模式：直接加载pruner状态")
        rank0_print(f"{'='*80}")
        rank0_print(f"路径: {debug_pruner_path}\n")

        # 加载pruner状态
        pruner.load_state(debug_pruner_path)

        # 打印统计信息
        stats = pruner.get_statistics()
        rank0_print(f"✓ 加载成功:")
        rank0_print(f"  Independent层: {stats['independent_lora_layers']}, Shared层: {stats['shared_lora_layers']}")
        if 'fine_grained_num_layers' in stats:
            rank0_print(f"  细粒度剪枝层: {stats['fine_grained_num_layers']}")

        # 打印按投影类型分组的importance
        rank0_print(f"\n按投影类型的Importance:")
        for proj_type in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
            proj_layers = {n: i for n, i in pruner.layer_importance.items()
                          if proj_type in n and pruner.layer_lora_type.get(n) == 'independent'}
            if proj_layers:
                imps = list(proj_layers.values())
                rank0_print(f"  {proj_type}: {len(proj_layers)}层, 范围=[{min(imps):.6f}, {max(imps):.6f}]")

        # 应用剪枝决策到模型（重要！）
        rank0_print(f"\n应用剪枝决策到模型...")
        pruner.apply_pruning_to_model(model, model_args.cur_task)
        rank0_print(f"✓ 剪枝决策已应用")

        # 计算预训练权重mask
        rank0_print(f"\n计算预训练权重重要性mask...")
        pruner.compute_pretrain_importance_masks(model)

        # 打印剪枝后的模型参数统计
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        rank0_print(f"\n剪枝后模型参数统计:")
        rank0_print(f"  总参数: {total_params / 1e6:.2f}M")
        rank0_print(f"  可训练参数: {trainable_params / 1e6:.2f}M ({trainable_params / total_params * 100:.2f}%)")

        rank0_print(f"\n✓ 跳过warm-up，直接进入Phase 2训练")
        rank0_print(f"{'='*80}\n")

        # 返回剪枝后的模型
        return model

    rank0_print("\n" + "="*60)
    rank0_print("LoRA-Drop Phase 1: Warmup and Importance Evaluation (多GPU)")
    rank0_print("="*60)

    # 创建10%数据子集用于评估
    train_dataset = data_module['train_dataset']
    eval_size = int(len(train_dataset) * training_args.lora_drop_eval_ratio)

    # 使用固定seed确保所有GPU选择相同的样本
    np.random.seed(42)
    eval_indices = np.random.choice(len(train_dataset), eval_size, replace=False)
    eval_subset = Subset(train_dataset, eval_indices)

    rank0_print(f"使用 {eval_size} 个样本 ({training_args.lora_drop_eval_ratio:.1%}) 进行重要性评估")

    # 计算评估步数
    eval_steps = training_args.lora_drop_warmup_steps

    rank0_print(f"中策方案：Warmup单GPU + Importance评估多GPU")
    rank0_print(f"  - Warmup: {eval_steps}步，仅rank 0训练，避免DDP冲突")
    rank0_print(f"  - Importance评估: 多GPU并行，充分利用资源\n")

    # ============ 优化 1: 启用梯度检查点（节省显存）============
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        rank0_print("✓ 已启用梯度检查点")

    # ============ 优化 2: 设备和数据类型 ============
    device = torch.device(f"cuda:{training_args.local_rank}" if training_args.local_rank != -1 else "cuda")
    model_dtype = next(model.parameters()).dtype

    # 确保整个模型都在GPU上
    model = model.to(device)

    rank0_print(f"✓ 设备: {device}, 数据类型: {model_dtype}")

    # ============ 不使用DDP，手动控制多GPU ============
    is_distributed = training_args.local_rank != -1
    if is_distributed:
        import torch.distributed as dist
        rank0_print(f"✓ 多GPU模式，当前GPU: {training_args.local_rank}")

    # ============ 优化 3: Batch size配置 ============
    micro_batch_size = 16
    rank0_print(f"✓ Batch size: {micro_batch_size}")

    # ============ 优化 4: 创建优化器（按LLaVATrainer方式分组参数）============
    ALL_LAYERNORM_LAYERS = [torch.nn.LayerNorm]
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
            "weight_decay": 0.0,
        },
    ]

    warmup_optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=training_args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rank0_print(f"✓ 可训练参数: {trainable_params / 1e6:.2f}M")

    # ============ 优化 5: 创建DataLoader ============
    # Warmup阶段：只在rank 0创建dataloader
    # Importance评估阶段：所有GPU都创建dataloader（使用DistributedSampler）
    if is_distributed:
        # 为importance评估准备DistributedSampler
        importance_sampler = DistributedSampler(
            eval_subset,
            num_replicas=training_args.world_size,
            rank=training_args.local_rank,
            shuffle=False,
            seed=42
        )
        importance_dataloader = DataLoader(
            eval_subset,
            batch_size=micro_batch_size,
            sampler=importance_sampler,
            collate_fn=data_module['data_collator'],
            drop_last=False
        )

        # Warmup阶段只在rank 0创建dataloader
        if training_args.local_rank == 0:
            warmup_dataloader = DataLoader(
                eval_subset,
                batch_size=micro_batch_size,
                collate_fn=data_module['data_collator'],
                shuffle=False,
                drop_last=False
            )
        else:
            warmup_dataloader = None
    else:
        # 单GPU模式
        warmup_dataloader = DataLoader(
            eval_subset,
            batch_size=micro_batch_size,
            collate_fn=data_module['data_collator'],
            shuffle=False,
            drop_last=False
        )
        importance_dataloader = warmup_dataloader

    # ============ 训练循环 ============
    model.train()
    total_samples = len(eval_subset)

    rank0_print(f"开始两阶段训练：")
    rank0_print(f"  阶段1 - Warmup: {eval_steps}步 (仅rank 0)")
    rank0_print(f"  阶段2 - Importance评估: 遍历所有{total_samples}个样本 (多GPU)\n")

    # ============ 阶段1：Warmup训练（仅rank 0）============
    if training_args.local_rank in [-1, 0]:
        rank0_print("[阶段1] Warmup训练开始...")
        for step, batch in enumerate(warmup_dataloader):
            if step >= eval_steps:
                break

            # 数据移到GPU并转换数据类型
            for k, v in batch.items():
                if hasattr(v, 'to'):
                    if v.is_floating_point():
                        batch[k] = v.to(device=device, dtype=model_dtype)
                    else:
                        batch[k] = v.to(device=device)

            # 前向传播（使用autocast保持混合精度）
            with torch.autocast(device_type="cuda", dtype=model_dtype):
                outputs = model(**batch)
                loss = outputs.loss

            # 反向传播和参数更新
            loss.backward()

            # 梯度裁剪
            if training_args.max_grad_norm is not None and training_args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)

            # 更新参数
            warmup_optimizer.step()
            warmup_optimizer.zero_grad(set_to_none=True)

            print(f"[Warmup] Step {step+1}/{eval_steps}, Loss: {loss.item():.4f}")

            # 定期清理显存
            if step % 5 == 0:
                torch.cuda.empty_cache()

        rank0_print(f"[阶段1] Warmup训练完成\n")

    # ============ 同步：Broadcast参数到所有GPU ============
    if is_distributed:
        rank0_print("Broadcasting warmup参数到所有GPU...")
        for param in model.parameters():
            # 确保tensor是连续的
            if not param.data.is_contiguous():
                param.data = param.data.contiguous()
            dist.broadcast(param.data, src=0)
        dist.barrier()
        rank0_print("✓ 参数已同步\n")

    # ============ 阶段2：Importance评估（多GPU）============
    rank0_print(f"[阶段2] Importance评估开始 (GPU {training_args.local_rank})...")

    for step, batch in enumerate(importance_dataloader):
        # 数据移到GPU并转换数据类型
        for k, v in batch.items():
            if hasattr(v, 'to'):
                if v.is_floating_point():
                    batch[k] = v.to(device=device, dtype=model_dtype)
                else:
                    batch[k] = v.to(device=device)

        # 前向传播（只forward，不backward）
        with torch.autocast(device_type="cuda", dtype=model_dtype):
            with torch.no_grad():  # 不需要梯度
                outputs = model(**batch)

        # importance会在forward过程中自动通过hook收集
        if training_args.local_rank in [-1, 0] and step % 10 == 0:
            processed_samples = step * micro_batch_size * (training_args.world_size if is_distributed else 1)
            print(f"[Importance评估] 已处理 {processed_samples}/{total_samples} 样本")

        # 定期清理显存
        if step % 5 == 0:
            torch.cuda.empty_cache()

    rank0_print(f"[阶段2] Importance评估完成\n")

    # 同步所有GPU
    if is_distributed:
        dist.barrier()

    rank0_print(f"✓ 两阶段完成：Warmup {eval_steps}步 + Importance评估 {total_samples}样本")

    # ============ 清理资源 ============
    # 1. 确保所有梯度都被清零
    warmup_optimizer.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)

    # 2. 清理优化器和dataloader
    del warmup_optimizer
    if warmup_dataloader is not None:
        del warmup_dataloader
    del importance_dataloader

    # 3. 清空CUDA缓存
    torch.cuda.empty_cache()

    # 4. 同步所有GPU
    if is_distributed:
        dist.barrier()

    rank0_print("✓ Warmup阶段完成，准备剪枝\n")

    # ============ 计算剪枝决策 ============
    rank0_print("="*60)
    rank0_print("LoRA-Drop: 计算层重要性并应用剪枝...")
    rank0_print("="*60)

    # 在多GPU环境下，所有GPU都需要计算剪枝决策（因为importance是分布式收集的）
    pruner.compute_layer_decisions()

    # 打印统计信息（只在rank 0打印）
    if training_args.local_rank in [-1, 0]:
        stats = pruner.get_statistics()
        print(f"\nLoRA-Drop 统计:")
        print(f"  总层数: {stats['total_layers']}")
        print(f"  独立LoRA层: {stats['independent_lora_layers']}")
        print(f"  共享LoRA层: {stats['shared_lora_layers']}")
        print(f"  参数减少率: {stats['parameter_reduction_ratio']:.2%}")

    # 应用剪枝
    rank0_print("\n应用LoRA-Drop剪枝...")
    pruner.apply_pruning_to_model(model, model_args.cur_task)

    # ============ 细粒度参数剪枝 ============
    # 在粗粒度剪枝后，对independent层进行细粒度参数剪枝
    rank0_print("\n计算细粒度参数剪枝mask...")
    pruner.compute_fine_grained_masks(model, model_args.cur_task)

    # ============ 共享层FSA剪枝 ============
    # 对共享LoRA层进行FSA剪枝（随机剪枝秩从16到10）
    rank0_print("\n对共享LoRA层进行FSA剪枝...")
    pruner.compute_shared_layer_fsa_masks(model, model_args.cur_task, target_rank=12)

    # ============ 同步剪枝后的模型参数到所有GPU ============
    if is_distributed:
        rank0_print("Broadcasting pruned model parameters to all GPUs...")
        if dist.is_initialized():
            # 将rank 0的参数broadcast到所有GPU
            for param in model.parameters():
                # 确保tensor是连续的
                if not param.data.is_contiguous():
                    param.data = param.data.contiguous()
                dist.broadcast(param.data, src=0)
            dist.barrier()
        rank0_print("✓ 剪枝后的模型参数已同步到所有GPU")

    # 保存pruner状态
    if training_args.local_rank == 0 or training_args.local_rank == -1:
        pruner_path = os.path.join(training_args.output_dir, 'lora_drop_pruner.pt')
        pruner.save_state(pruner_path)
        rank0_print(f"已保存LoRA-Drop pruner状态到 {pruner_path}")

    rank0_print("="*60)
    rank0_print("LoRA-Drop Phase 2: 使用剪枝后的LoRA结构进行完整训练")
    rank0_print("="*60 + "\n")

    # 打印剪枝后的模型参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rank0_print(f"剪枝后模型参数统计:")
    rank0_print(f"  总参数: {total_params / 1e6:.2f}M")
    rank0_print(f"  可训练参数: {trainable_params / 1e6:.2f}M ({trainable_params / total_params * 100:.2f}%)")
    rank0_print(f"✓ 模型保持在 {next(model.parameters()).device} 上，等待DeepSpeed接管\n")

    # 返回更新后的模型
    return model


def load_model_from_previous_task(model, previous_task_model_path):
    token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
    # if model.lm_head.weight.shape[0] != token_num:
    #     model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
    #     model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

    print('Loading additional LLaVA weights...')
    if os.path.exists(os.path.join(previous_task_model_path, 'non_lora_trainables.bin')):
        non_lora_trainables = torch.load(os.path.join(previous_task_model_path, 'non_lora_trainables.bin'), map_location='cpu')
    else:
        # this is probably from HF Hub
        from huggingface_hub import hf_hub_download
        def load_from_hf(repo_id, filename, subfolder=None):
            cache_file = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                subfolder=subfolder)
            return torch.load(cache_file, map_location='cpu')
        non_lora_trainables = load_from_hf(previous_task_model_path, 'non_lora_trainables.bin')
    non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
    if any(k.startswith('model.model.') for k in non_lora_trainables):
        non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}

    model.base_model.model.load_state_dict(non_lora_trainables, strict=False)

    from peft import PeftModel
    print('Loading LoRA weights...')
    filename = os.path.join(previous_task_model_path, WEIGHTS_NAME)
    adapters_weights = torch.load(filename, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    load_result = set_peft_model_state_dict(model, adapters_weights, adapter_name="default")
    print('Model is loaded...')

def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMPTForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args,
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False
    model.training = True

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        kwargs = { 
                "task_embedding_dim": model_args.task_embedding_dim,
                "expert_num": model_args.expert_num,
                "cur_task": model_args.cur_task,
            }
        lora_config = CoINMOELoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type=TaskType.CAUSAL_LM_CoIN,
            **kwargs
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

        # LoRA-Drop: Initialize pruner if enabled
        # 注意：通常剪枝只在 cur_task > 0 时执行，但可以通过配置强制启用
        if training_args.lora_drop_enable:
            if model_args.cur_task == 0:
                rank0_print("LoRA-Drop is adapted in task 1.")

            from CoIN.peft.tuners.lora_drop import LoRADropPruner
            from CoIN.peft.tuners.clitmoelora import set_lora_drop_pruner

            rank0_print("Initializing LoRA-Drop pruner...")
            pruner = LoRADropPruner(
                warmup_steps=training_args.lora_drop_warmup_steps,
                eval_ratio=training_args.lora_drop_eval_ratio,
                importance_threshold=training_args.lora_drop_threshold,
                fine_grained_prune_ratio=training_args.fine_grained_prune_ratio,
                fine_grained_start_task=training_args.fine_grained_start_task,
                fine_grained_min_keep_ratio=training_args.fine_grained_min_keep_ratio,
                fine_grained_max_keep_ratio=training_args.fine_grained_max_keep_ratio,
                pretrain_reg_lambda=training_args.pretrain_reg_lambda,
                pretrain_reg_omega=training_args.pretrain_reg_omega,
            )

            # Register all LoRA layers
            for name, module in model.named_modules():
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    pruner.register_layer(name)

            # 如果有previous task，加载其pruner状态（包含fine_grained_masks）
            # 注意：只有task3及以后才需要加载，因为task2是第一个做细粒度剪枝的任务
            if model_args.previous_task_model_path is not None and model_args.cur_task >= 2:
                pruner_path = os.path.join(model_args.previous_task_model_path, 'lora_drop_pruner.pt')
                if os.path.exists(pruner_path):
                    rank0_print(f"\n加载previous task的pruner状态: {pruner_path}")
                    pruner.load_state(pruner_path)
                    rank0_print(f"✓ 成功加载pruner状态")

                    # 打印加载的状态信息
                    stats = pruner.get_statistics()
                    rank0_print(f"  - Independent层: {stats['independent_lora_layers']}")
                    rank0_print(f"  - Shared层: {stats['shared_lora_layers']}")
                    if 'fine_grained_num_layers' in stats:
                        rank0_print(f"  - 已有细粒度剪枝层: {stats['fine_grained_num_layers']}")
                        rank0_print(f"  - 已有mask保留参数比例: {(1-stats.get('fine_grained_prune_ratio', 0))*100:.1f}%")
                    rank0_print("")
                else:
                    rank0_print(f"⚠ 未找到previous task的pruner状态: {pruner_path}")
                    rank0_print(f"  将从头开始计算剪枝决策\n")

            # Set global pruner
            set_lora_drop_pruner(pruner)
            rank0_print(f"LoRA-Drop enabled: warmup={training_args.lora_drop_warmup_steps}, "
                       f"threshold={training_args.lora_drop_threshold}")
            rank0_print(f"Fine-grained pruning: keep_ratio={training_args.fine_grained_prune_ratio}, "
                       f"start_task={training_args.fine_grained_start_task}")
        else:
            pruner = None

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        model.get_model().initialize_text_modules(
                model_args=model_args,
                fsdp=training_args.fsdp
            )
        text_tower = model.get_text_tower()
        text_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    clip_tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.text_tower,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
        )

    model.set_clip_tokenizer(clip_tokenizer)
    model.set_tokenizer(tokenizer)
    model.set_cur_task(model_args.cur_task, model_args.expert_num)

    if model_args.previous_task_model_path is not None:
        # load model from previous task
        load_model_from_previous_task(model, model_args.previous_task_model_path)

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)

    # LoRA-Drop: Phase 1 - Warmup and importance evaluation with 10% data
    if training_args.lora_enable and training_args.lora_drop_enable:
    # if False: # debug跳过
        model = run_lora_drop_warmup_phase(model, data_module, training_args, model_args)

        # 在Phase 2之前，确保模型参数连续且没有残留状态
        rank0_print("准备Phase 2训练：最终清理和验证...")

        # 1. 确保所有参数连续
        for param in model.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()

        # 2. 再次确保所有梯度为None
        model.zero_grad(set_to_none=True)
        for param in model.parameters():
            param.grad = None

        # 3. 清除所有可能的DeepSpeed残留
        for param in model.parameters():
            for attr in ['ds_tensor', 'ds_id', 'ds_status', 'ds_numel', 'ds_shape',
                         'ds_param_alias', 'ds_active_sub_modules', 'ds_summary']:
                if hasattr(param, attr):
                    delattr(param, attr)

        # 4. 最后一次清理显存
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        rank0_print("✓ Phase 2准备完成，模型状态已完全清理\n")

    # Phase 2: Full training with 100% data
    # 将cur_task从model_args复制到training_args，以便trainer可以访问
    training_args.cur_task = model_args.cur_task

    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    # Ensure contiguous parameters (再次确保)
    for param in model.parameters():
        if not param.is_contiguous():
            param.data = param.data.contiguous()

    trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        model.set_boundary_for_save()
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)

    remove_dir = training_args.output_dir
    subprocess.run(f"find {remove_dir} -maxdepth 1 -type d -name 'checkpoint-*' -exec rm -rf {{}} +", shell=True)

if __name__ == "__main__":
    train()
