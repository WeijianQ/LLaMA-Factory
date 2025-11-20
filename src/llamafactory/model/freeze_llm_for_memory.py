from typing import TYPE_CHECKING
import torch
from transformers import PreTrainedModel
from ..extras import logging
from ..hparams import FinetuningArguments

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

logger = logging.get_logger(__name__)


def _setup_freeze_tuning_llm_for_memory(
    model: "PreTrainedModel",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
    cast_trainable_params_to_fp32: bool,
) -> None:
    if not is_trainable:
        return

    logger.info_rank0("Fine-tuning method: Freeze LLM for Memory")
    logger.info_rank0("Freezing causal LLM, only training override_table and embed_head")

    # Define trainable module names
    trainable_module_names = ["memory_projector"]
    
    # First, freeze all parameters
    for name, param in model.named_parameters():
        param.requires_grad_(False)
    
    # Then, unfreeze only override_table and embed_head
    trainable_params = []
    seen_param_ids = set()  # Track parameter tensor IDs to avoid counting tied weights twice
    total_trainable_params = 0

    for name, param in model.named_parameters():
        # Check if this parameter belongs to override_table or embed_head
        if any(module_name in name for module_name in trainable_module_names):
            param.requires_grad_(True)
            if cast_trainable_params_to_fp32:
                param.data = param.data.to(torch.float32)

            # Only count unique parameters (for tied weights)
            param_id = id(param)
            if param_id not in seen_param_ids:
                seen_param_ids.add(param_id)
                total_trainable_params += param.numel()

            trainable_params.append(name)

    logger.info_rank0(f"Set trainable modules: {', '.join(trainable_module_names)}")
    logger.info_rank0(f"Number of trainable parameter names: {len(trainable_params)}")
    logger.info_rank0(f"Number of unique trainable parameters: {total_trainable_params:,}")

    if trainable_params:
        logger.info_rank0(f"Trainable parameter names: {', '.join(trainable_params)}")
    # Store trainable parameter info as model attributes for FSDP compatibility
    # This helps Trainer correctly count trainable params after FSDP wrapping
    model._trainable_param_names = trainable_params
    model._num_trainable_params = total_trainable_params
        