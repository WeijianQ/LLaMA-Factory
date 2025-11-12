#!/bin/bash
set -euo pipefail

# Enable OmegaConf to accept extra CLI overrides for dataset files.
export ALLOW_EXTRA_ARGS=1

# Stage 2: Full fine-tuning after stage 1 projector training
MODEL_PATH="saves/freeze_llm_for_memory/stage_1_sft/checkpoint-300_converted"

# Set wandb notes with proper escaping
# WANDB_NOTES="stage 2 full fine-tuning with DeepSpeed ZeRO-3 - policy tasks only"
WANDB_PROJECT="webshop_sft_using_llamafactory"

# Launch DeepSpeed training with 4 GPUs
# DeepSpeed handles distributed training automatically
export WANDB_PROJECT=${WANDB_PROJECT}
export TOKENIZERS_PARALLELISM=false

# Use accelerate launch with FSDP for proper GPU mapping

