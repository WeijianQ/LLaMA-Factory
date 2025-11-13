# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import TYPE_CHECKING, Any, Optional, TypedDict

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSeq2SeqLM,
    AutoModelForTextToWaveform,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
)
from trl import AutoModelForCausalLMWithValueHead

from ..extras import logging
from ..extras.misc import count_parameters, skip_check_imports, try_download_model_from_other_hub
from .adapter import init_adapter
from .model_utils.liger_kernel import apply_liger_kernel
from .model_utils.misc import register_autoclass
from .model_utils.mod import convert_pretrained_model_to_mod, load_mod_pretrained_model
from .model_utils.unsloth import load_unsloth_pretrained_model
from .model_utils.valuehead import load_valuehead_params
from .patcher import patch_config, patch_model, patch_processor, patch_tokenizer, patch_valuehead_model


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer, ProcessorMixin

    from ..hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


# Cache for dynamically imported memory model classes
_memory_classes_cache = None
_memory_suffix_classes_cache = None


def _import_memory_model_classes():
    """
    Dynamically import memory model classes from external Qwen25_1p5B_Memory module.

    This function handles the import of custom memory model classes that are located
    outside the LLaMA-Factory directory structure. It adds the parent verl-agent
    directory to sys.path and imports the necessary classes.

    Returns:
        dict: A dictionary containing the imported classes with keys:
            - 'Qwen2_5_MemoryForCausalLMForFreezeTraining'
            - 'Qwen2_5_MemoryForCausalLM'
            - 'Qwen2_5_MemoryConfig'
            - 'Qwen2_5_MemoryProcessor'
    """
    global _memory_classes_cache

    # Return cached classes if already imported
    if _memory_classes_cache is not None:
        return _memory_classes_cache

    import sys
    from pathlib import Path

    # Add verl-agent directory to sys.path
    # Path structure: .../verl-agent/LLaMA-Factory/src/llamafactory/model/loader.py
    # We need to go up 4 levels to reach verl-agent/
    verl_agent_path = Path(__file__).resolve().parents[4]
    if str(verl_agent_path) not in sys.path:
        sys.path.insert(0, str(verl_agent_path))

    # Import the classes from Qwen25_1p5B_Memory
    from Qwen25_1p5B_Memory.modeling_qwen2_5_memory import (
        Qwen2_5_MemoryDualMode,
        Qwen2_5_MemoryForCausalLM,
    )
    from Qwen25_1p5B_Memory.configuration_qwen2_5_memory import Qwen2_5_MemoryConfig
    from Qwen25_1p5B_Memory.processing_qwen2_5_memory import Qwen2_5_MemoryProcessor

    # Cache the imported classes
    _memory_classes_cache = {
        "Qwen2_5_MemoryDualMode": Qwen2_5_MemoryDualMode,
        "Qwen2_5_MemoryForCausalLM": Qwen2_5_MemoryForCausalLM,
        "Qwen2_5_MemoryConfig": Qwen2_5_MemoryConfig,
        "Qwen2_5_MemoryProcessor": Qwen2_5_MemoryProcessor,
    }

    return _memory_classes_cache

def _import_memory_suffix_model_classes():
    global _memory_suffix_classes_cache
    if _memory_suffix_classes_cache is not None:
        return _memory_suffix_classes_cache

    import sys
    from pathlib import Path

    verl_agent_path = Path(__file__).resolve().parents[4]
    if str(verl_agent_path) not in sys.path:
        sys.path.insert(0, str(verl_agent_path))

    from Qwen25_1p5B_Memory_suffix.modeling_qwen2_5_memory import Qwen2_5_MemorySuffixForCausalLM
    from Qwen25_1p5B_Memory_suffix.configuration_qwen2_5_memory import Qwen2_5_MemorySuffixConfig
    from Qwen25_1p5B_Memory_suffix.processing_qwen2_5_memory import Qwen2_5_MemorySuffixProcessor

    _memory_suffix_classes_cache = {
        "Qwen2_5_MemorySuffixForCausalLM": Qwen2_5_MemorySuffixForCausalLM,
        "Qwen2_5_MemorySuffixConfig": Qwen2_5_MemorySuffixConfig,
        "Qwen2_5_MemorySuffixProcessor": Qwen2_5_MemorySuffixProcessor,
    }

    return _memory_suffix_classes_cache

class TokenizerModule(TypedDict):
    tokenizer: "PreTrainedTokenizer"
    processor: Optional["ProcessorMixin"]


def _get_init_kwargs(model_args: "ModelArguments") -> dict[str, Any]:
    r"""Get arguments to load config/tokenizer/model.

    Note: including inplace operation of model_args.
    """
    skip_check_imports()
    model_args.model_name_or_path = try_download_model_from_other_hub(model_args)
    return {
        "trust_remote_code": model_args.trust_remote_code,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }


def load_tokenizer(model_args: "ModelArguments") -> "TokenizerModule":
    r"""Load pretrained tokenizer and optionally loads processor.

    Note: including inplace operation of model_args.
    """
    init_kwargs = _get_init_kwargs(model_args)
    tokenizer_name_or_path = model_args.tokenizer_name_or_path
    if tokenizer_name_or_path is None:
        tokenizer_name_or_path = model_args.model_name_or_path
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            split_special_tokens=model_args.split_special_tokens,
            padding_side="right",
            **init_kwargs,
        )
    except ValueError:  # try another one
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            use_fast=not model_args.use_fast_tokenizer,
            padding_side="right",
            **init_kwargs,
        )
    except Exception as e:
        raise OSError("Failed to load tokenizer.") from e

    patch_tokenizer(tokenizer, model_args)

    if "memory" in model_args.model_name_or_path.lower():
        if 'suffix' in model_args.model_name_or_path.lower():
            memory_classes = _import_memory_suffix_model_classes()
            Qwen2_5_MemorySuffixProcessor = memory_classes["Qwen2_5_MemorySuffixProcessor"]
            processor = Qwen2_5_MemorySuffixProcessor(tokenizer=tokenizer)
        else:
            memory_classes = _import_memory_model_classes()
            Qwen2_5_MemoryProcessor = memory_classes["Qwen2_5_MemoryProcessor"]
            processor = Qwen2_5_MemoryProcessor(tokenizer=tokenizer)
    else:
        try:
            processor = AutoProcessor.from_pretrained(
                model_args.model_name_or_path,
                use_fast=model_args.use_fast_tokenizer,
                **init_kwargs,
            )
        except ValueError:  # try another one
            processor = AutoProcessor.from_pretrained(
                model_args.model_name_or_path,
                use_fast=not model_args.use_fast_tokenizer,
                **init_kwargs,
            )
        except Exception as e:
            logger.info_rank0(f"Failed to load processor: {e}.")
            processor = None

    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/auto/processing_auto.py#L324
    if processor is not None and "Processor" not in processor.__class__.__name__:
        logger.debug("The loaded processor is not an instance of Processor. Dropping it.")
        processor = None

    if processor is not None:
        patch_processor(processor, tokenizer, model_args)

    return {"tokenizer": tokenizer, "processor": processor}


def load_config(model_args: "ModelArguments") -> "PretrainedConfig":
    r"""Load model config."""
    init_kwargs = _get_init_kwargs(model_args)

    # If it's a Memory model, reload with the correct config class to ensure _no_split_modules is set
    if model_args.is_memory_model:
        logger.info("Detected Qwen2_5_Memory model, loading with Qwen2_5_MemoryConfig to set FSDP policy.")
        memory_classes = _import_memory_model_classes()
        init_kwargs['skip_embed_head'] = model_args.skip_embed_head
        Qwen2_5_MemoryConfig = memory_classes["Qwen2_5_MemoryConfig"]
        return Qwen2_5_MemoryConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)
    elif model_args.is_memory_suffix_model:
        logger.info("Detected Qwen2_5_MemorySuffix model, loading with Qwen2_5_MemorySuffixConfig to set FSDP policy.")
        memory_classes = _import_memory_suffix_model_classes()
        Qwen2_5_MemorySuffixConfig = memory_classes["Qwen2_5_MemorySuffixConfig"]
        return Qwen2_5_MemorySuffixConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)
    # First load with AutoConfig to check the model type
    temp_config = AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)
    return temp_config


def load_model(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool = False,
    add_valuehead: bool = False,
) -> "PreTrainedModel":
    r"""Load pretrained model."""
    init_kwargs = _get_init_kwargs(model_args)
    is_memory_model = model_args.is_memory_model
    config = load_config(model_args)
    patch_config(config, tokenizer, model_args, init_kwargs, is_trainable)
    apply_liger_kernel(config, model_args, is_trainable, require_logits=(finetuning_args.stage not in ["pt", "sft"]))

    model = None
    lazy_load = False
    if model_args.use_unsloth:
        if model_args.adapter_name_or_path is not None:
            lazy_load = True
        elif is_trainable:
            model = load_unsloth_pretrained_model(config, model_args, finetuning_args)

    if model is None and not lazy_load:
        init_kwargs["config"] = config
        init_kwargs["pretrained_model_name_or_path"] = model_args.model_name_or_path

        if model_args.mixture_of_depths == "load":
            model = load_mod_pretrained_model(**init_kwargs)
        else:
            if type(config) in AutoModelForImageTextToText._model_mapping.keys():  # image-text
                load_class = AutoModelForImageTextToText
            elif type(config) in AutoModelForVision2Seq._model_mapping.keys():  # image-text
                load_class = AutoModelForVision2Seq
            elif type(config) in AutoModelForSeq2SeqLM._model_mapping.keys():  # audio-text
                load_class = AutoModelForSeq2SeqLM
            elif type(config) in AutoModelForTextToWaveform._model_mapping.keys():  # audio hack for qwen omni
                load_class = AutoModelForTextToWaveform
            elif is_memory_model:
                memory_classes = _import_memory_model_classes()
                if finetuning_args.finetuning_type == "freeze_llm_for_memory":
                    load_class = memory_classes["Qwen2_5_MemoryDualMode"]
                else:
                    load_class = memory_classes["Qwen2_5_MemoryForCausalLM"]
                print(f"Loading model class: {load_class}")
            elif model_args.is_memory_suffix_model:
                memory_classes = _import_memory_suffix_model_classes()
                load_class = memory_classes["Qwen2_5_MemorySuffixForCausalLM"]
                print(f"Loading model class: {load_class}")
            else:
                load_class = AutoModelForCausalLM

            if model_args.train_from_scratch:
                model = load_class.from_config(config, trust_remote_code=model_args.trust_remote_code)
            elif is_memory_model or model_args.is_memory_suffix_model:
                init_kwargs['config'] = config
                model = load_class.from_pretrained(**init_kwargs)
            else:
                model = load_class.from_pretrained(**init_kwargs)
                if getattr(model.config, "model_type", None) in ["qwen2_5_omni", "qwen3_omni_moe"]:
                    model = getattr(model, "thinker")

            # Check if model is Qwen2_5_MemoryForCausalLMForFreezeTraining type
            if type(model).__name__ == "Qwen2_5_MemoryForCausalLMForFreezeTraining":
                # reapply weight tying
                model.special_lm_head.weight = model.special_embed_tokens.weight

        if model_args.mixture_of_depths == "convert":
            model = convert_pretrained_model_to_mod(model, config, model_args)

    if not lazy_load:
        patch_model(model, tokenizer, model_args, is_trainable, add_valuehead)
        register_autoclass(config, model, tokenizer)

    model = init_adapter(config, model, model_args, finetuning_args, is_trainable)

    if add_valuehead:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        patch_valuehead_model(model)

        if model_args.adapter_name_or_path is not None:
            vhead_path = model_args.adapter_name_or_path[-1]
        else:
            vhead_path = model_args.model_name_or_path

        vhead_params = load_valuehead_params(vhead_path, model_args)
        if vhead_params is not None:
            model.load_state_dict(vhead_params, strict=False)
            logger.info_rank0(f"Loaded valuehead from checkpoint: {vhead_path}")

    if not is_trainable:
        model.requires_grad_(False)
        for param in model.parameters():
            if param.data.dtype == torch.float32 and model_args.compute_dtype != torch.float32:
                param.data = param.data.to(model_args.compute_dtype)

        model.eval()
    else:
        model.train()

    trainable_params, all_param = count_parameters(model)
    if is_trainable:
        param_stats = (
            f"trainable params: {trainable_params:,} || "
            f"all params: {all_param:,} || trainable%: {100 * trainable_params / all_param:.4f}"
        )
    else:
        param_stats = f"all params: {all_param:,}"

    logger.info_rank0(param_stats)

    if model_args.print_param_status and int(os.getenv("LOCAL_RANK", "0")) == 0:
        for name, param in model.named_parameters():
            print(f"name: {name}, dtype: {param.dtype}, device: {param.device}, trainable: {param.requires_grad}")

    return model
