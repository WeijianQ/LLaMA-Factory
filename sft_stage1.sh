#!/bin/bash
set -euo pipefail

# Enable OmegaConf to accept extra CLI overrides for dataset files.
export ALLOW_EXTRA_ARGS=1
# Set wandb notes with proper escaping
# Use single quotes in the command line to avoid YAML parsing issues
# For newlines in wandb, use \n in the string
WANDB_NOTES="stage 1 freeze llm only train the projector, mean pool, no special tokens, only observation recognition tasks"
WANDB_PROJECT="webshop_sft_using_llamafactory_new"

# Launch single-process training with device_map="auto" for model parallelism
# The frozen LLM will be automatically sharded across GPUs 0,1,2,3
# Use accelerate launch with --num_processes=1 for single-process training
export WANDB_PROJECT=${WANDB_PROJECT}
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file examples/accelerate/osc_2cards_fsdp.yaml \
    src/train.py examples/webshop/freeze_stage_1.yaml \
    report_to="wandb" \
    wandb_notes="${WANDB_NOTES}" \
    num_train_epochs=10.0 \
    run_name="MIDDLE_LAYER_stage_1_denaturalized_output_observation_recognition" \
    dataset="webshop_train_denaturalized_output_observation_recognition" \
    eval_dataset="webshop_val_denaturalized_output_observation_recognition" \
    output_dir="saves/qwen25-1p5b-memory/freeze_llm_for_memory/MIDDLE_LAYER_stage_1_sft_denaturalized_output_observation_recognition" \
    per_device_train_batch_size=16 gradient_accumulation_steps=8 \
    save_steps=50
    # resume_from_checkpoint="saves/qwen25-1p5b-memory/full/webshop_sft/checkpoint-53" \
    # num_train_epochs=3
