#!/bin/bash
set -euo pipefail

# Enable OmegaConf to accept extra CLI overrides for dataset files.
export ALLOW_EXTRA_ARGS=1

MODEL_PATH="WeijianQi1999/Qwen25-1p5B-Memory"

# Set wandb notes with proper escaping
# Use single quotes in the command line to avoid YAML parsing issues
# For newlines in wandb, use \n in the string
WANDB_NOTES="stage 1 freeze llm only train the projector and special tokens"
WANDB_PROJECT="webshop_sft_using_llamafactory"

# Launch single-process training with device_map="auto" for model parallelism
# The frozen LLM will be automatically sharded across GPUs 0,1,2,3
# Use accelerate launch with --num_processes=1 for single-process training
export WANDB_PROJECT=${WANDB_PROJECT}
NPROC_PER_NODE=1 CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes=1 \
    src/train.py examples/webshop/freeze_stage_1.yaml \
    model_name_or_path=${MODEL_PATH} \
    report_to="wandb" \
    wandb_notes="${WANDB_NOTES}" \
    run_name="stage_1_freeze_llm_proxy_tasks_only"
    # resume_from_checkpoint="saves/qwen25-1p5b-memory/full/webshop_sft/checkpoint-53" \
    # num_train_epochs=3
