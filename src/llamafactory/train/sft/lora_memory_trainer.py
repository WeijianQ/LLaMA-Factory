# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
LoraMemoryTrainer: Trainer for lora_memory models.

LoRA state is controlled by the model via enable_lora parameter:
    - encode(): enable_lora=True (LoRA ON)
    - forward(): enable_lora=False by default (LoRA OFF)

This is GC-safe because enable_lora is preserved during recomputation.
"""

from typing import TYPE_CHECKING, Optional
from peft import PeftModel
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging

if TYPE_CHECKING:
    from transformers import ProcessorMixin
    from ...hparams import FinetuningArguments, ModelArguments

logger = logging.get_logger(__name__)


class LoraMemoryTrainer(Seq2SeqTrainer):
    """
    Trainer for lora_memory models.

    Training flow:
        1. Extract memory_input_ids, memory_attention_mask
        2. Call model(do_encoding=True) -> memory_embeddings (LoRA ON via enable_lora=True)
        3. Call model(memory_embeddings=...) -> loss (LoRA OFF via enable_lora=False default)
    """

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"] = None,
        model_args: Optional["ModelArguments"] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        self.model_args = model_args
        logger.info_rank0("LoraMemoryTrainer initialized")

    def _set_adapters_enabled(self, model, enabled: bool):
        """Enable or disable LoRA adapters."""
        for module in model.modules():
            if hasattr(module, '_disable_adapters'):
                module._disable_adapters = not enabled

    @override
    def compute_loss(self, model: PeftModel, inputs, return_outputs=False, **kwargs):
        # Step 1: Encode with LoRA ON (default state)
        memory_input_ids = inputs.pop("memory_input_ids")
        memory_attention_mask = inputs.pop("memory_attention_mask")

        # Flatten: (B, num_memories, mem_seq_len) -> (B * num_memories, mem_seq_len)
        B, num_memories, mem_seq_len = memory_input_ids.shape
        flat_memory_ids = memory_input_ids.view(B * num_memories, mem_seq_len)
        flat_memory_mask = memory_attention_mask.view(B * num_memories, mem_seq_len)

        # Filter valid memories (non-padding)
        valid_mask = flat_memory_mask.sum(dim=1) > 0
        valid_memory_ids = flat_memory_ids[valid_mask]
        valid_memory_mask = flat_memory_mask[valid_mask]

        # Encode: (num_valid, mem_seq_len) -> (num_valid, Q, D)
        # from torch.distributed import get_rank
        # if get_rank() == 0:
        #     from ...debug_utils import wait_for_debugger
        #     wait_for_debugger()
        # else:
        #     torch.distributed.barrier()
        model.set_adapter("default")
        memory_embeddings = model(
            input_ids=valid_memory_ids,
            attention_mask=valid_memory_mask,
            do_encoding=True
        )
        inputs["memory_embeddings"] = memory_embeddings

        model.disable_adapters()
        outputs = model(**inputs)

        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
