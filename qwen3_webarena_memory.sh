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
#SBATCH --job-name=qwen3_8b_webarena_memory_4gpu
#SBATCH --output=logs/qwen3_8b_webarena_memory_4gpu_%j.out
#SBATCH --error=logs/qwen3_8b_webarena_memory_4gpu_%j.err

set -euo pipefail

# Enable OmegaConf to accept extra CLI overrides for dataset files.
export ALLOW_EXTRA_ARGS=1

WANDB_NOTES="freeze llm, train memory module, webarena repeat obs, fsdp"
WANDB_PROJECT="webarena_sft_using_llamafactory"

# Launch 4-GPU FSDP training
export WANDB_PROJECT=${WANDB_PROJECT}
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3

accelerate launch \
    --config_file examples/accelerate/osc_4cards_fsdp.yaml \
    src/train.py \
    --use_reentrant_gc false \
    --stage sft \
    --model_name_or_path /fs/ess/PAS1576/qwjian/agent-memory-lab/hf_models/Qwen3_memory_8B_instruct \
    --do_train \
    --dataset webarena_sft_memory_repeat_obs_train \
    --template qwen \
    --finetuning_type full \
    --output_dir saves/qwen3_8b-memory/webarena_repeat_obs_4gpu_lr1e-5 \
    --overwrite_cache \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 4 \
    --is_memory_model \
    --has_memory \
    --cutoff_len 16384 \
    --memory_truncate_length 8192 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 64 \
    --learning_rate 1e-5 \
    --memory_lr 1e-3 \
    --num_train_epochs 3.0 \
    --max_memory_num 15 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.0 \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --save_steps 50 \
    --plot_loss \
    --bf16 \
    --flash_attn fa2 \
    --trust_remote_code \
    --report_to none \
    --run_name qwen3_8b_webarena_repeat_obs_4gpu \
    --overwrite_output_dir \
    --save_only_model false
