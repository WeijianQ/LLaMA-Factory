#!/bin/bash
set -euo pipefail

# Enable OmegaConf to accept extra CLI overrides for dataset files.
export ALLOW_EXTRA_ARGS=1

MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct"
TRAIN_FILE="webshop_sft_data/converted_from_mixup_for_sft_only_TRAIN_13342.parquet"
VAL_FILE="webshop_sft_data/converted_from_mixup_for_sft_only_VAL_1736.parquet"

# Set wandb notes with proper escaping
# Use single quotes in the command line to avoid YAML parsing issues
# For newlines in wandb, use \n in the string
WANDB_NOTES="resume_from_epoch_1 train for another 2 epochs, total 3 epochs. Training configuration Qwen2.5-1.5B on webshop data."

# Launch 2-way distributed SFT training with the specified data files.
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/webshop/qwen25_full_sft.yaml \
    model_name_or_path=${MODEL_PATH} \
    report_to="wandb" \
    wandb_notes="${WANDB_NOTES}" \
    run_name="sft_baseline_resume_from_epoch_1_total_3_epochs" \
    resume_from_checkpoint="saves/qwen25-1p5b-memory/full/webshop_sft/checkpoint-53" \
    num_train_epochs=3
