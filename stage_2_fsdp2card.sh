#!/bin/bash
set -euo pipefail

# Enable OmegaConf to accept extra CLI overrides for dataset files.
export ALLOW_EXTRA_ARGS=1

# Set wandb project
WANDB_PROJECT="webshop_sft_using_llamafactory"

export WANDB_PROJECT=${WANDB_PROJECT}
export TOKENIZERS_PARALLELISM=false

# Stage 2: Full fine-tuning with FSDP on 2 GPUs
# Use accelerate launch with FSDP config
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file examples/accelerate/osc_2cards_fsdp.yaml \
    src/train.py examples/webshop/stage_2_fsdp2card.yaml