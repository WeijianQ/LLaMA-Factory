#!/bin/bash
#SBATCH --account=PAS1576
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH -p quad
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=80
#SBATCH --mem=500G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=qi.658@osu.edu
#SBATCH --job-name=qwen3_8b_webarena_baseline_4gpu
#SBATCH --output=logs/qwen3_8b_webarena_baseline_4gpu_%j.out
#SBATCH --error=logs/qwen3_8b_webarena_baseline_4gpu_%j.err

set -euo pipefail

# Enable OmegaConf to accept extra CLI overrides for dataset files.
export ALLOW_EXTRA_ARGS=1

WANDB_NOTES="baseline qwen3 webarena, no memory module, fsdp"
WANDB_PROJECT="webarena_sft_using_llamafactory"

# Launch 4-GPU FSDP training
export WANDB_PROJECT=${WANDB_PROJECT}
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3

accelerate launch \
    --config_file examples/accelerate/osc_4cards_fsdp_baseline.yaml \
    src/train.py \
    --use_reentrant_gc false \
    --stage sft \
    --model_name_or_path Qwen/Qwen3-8B \
    --do_train \
    --dataset webarena_sft_baseline_train \
    --template qwen \
    --finetuning_type full \
    --output_dir saves/qwen3_8b/webarena_baseline_4gpu \
    --overwrite_cache \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 4 \
    --cutoff_len 8192 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 64 \
    --learning_rate 1e-5 \
    --num_train_epochs 3.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.0 \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --save_steps 50 \
    --plot_loss \
    --bf16 \
    --flash_attn fa2 \
    --trust_remote_code \
    --report_to wandb \
    --run_name qwen3_8b_webarena_baseline_4gpu \
    --overwrite_output_dir \
    --save_only_model false
