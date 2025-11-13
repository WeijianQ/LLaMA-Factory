#!/bin/bash
set -euo pipefail

# Enable OmegaConf to accept extra CLI overrides for dataset files.
export ALLOW_EXTRA_ARGS=1

# MODEL_PATH="WeijianQi1999/Qwen25-1p5B-Memory"
MODEL_PATH="/fs/ess/PAS1576/qwjian/verl-s-for-codex/verl-agent/Qwen25_1p5B_Memory_suffix"
# Set wandb notes with proper escaping
# Use single quotes in the command line to avoid YAML parsing issues
# For newlines in wandb, use \n in the string
WANDB_NOTES="stage 1 very few data collected from trial run. This is the suffix version of the model."
WANDB_PROJECT="webshop_sft_using_llamafactory"


# Use accelerate launch with --num_processes=1 for single-process training
# deepspeed: examples/deepspeed/ds_z2_config.json
export WANDB_PROJECT=${WANDB_PROJECT}
# deepspeed: examples/deepspeed/ds_z2_config.json
# NPROC_PER_NODE=1 CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file examples/accelerate/deepspeed_config_osc_2cards.yaml \
#     src/train.py examples/webshop/freeze_stage_1.yaml \
#     model_name_or_path=${MODEL_PATH} \
#     deepspeed=examples/deepspeed/ds_z2_config.json \
#     report_to="wandb" \
#     wandb_notes="${WANDB_NOTES}" \
#     is_memory_suffix_model=True \
#     skip_embed_head=false \
#     run_name="stage_1_freeze_llm_proxy_tasks_suffix_version" logging_steps=1 num_train_epochs=5 \
#     gradient_accumulation_steps=16 per_device_train_batch_size=2 \
#     learning_rate=1e-1 save_steps=100


CUDA_VISIBLE_DEVICES=0 python3 src/train.py examples/webshop/freeze_stage_1.yaml model_name_or_path=${MODEL_PATH} report_to="none" is_memory_suffix_model=True run_name="stage_1_freeze_llm_proxy_tasks_suffix_version" logging_steps=1 num_train_epochs=5 gradient_accumulation_steps=8 learning_rate=1e-1 save_steps=100 per_device_train_batch_size=2