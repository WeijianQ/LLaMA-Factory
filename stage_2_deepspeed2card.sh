#!/bin/bash
set -euo pipefail

# Enable OmegaConf to accept extra CLI overrides for dataset files.
export ALLOW_EXTRA_ARGS=1

# Stage 2: Full fine-tuning with DeepSpeed ZeRO-2 on 2 GPUs
MODEL_PATH="saves/freeze_llm_for_memory/stage_1_sft/checkpoint-460_converted"

# Set wandb project
WANDB_PROJECT="webshop_sft_using_llamafactory"

export WANDB_PROJECT=${WANDB_PROJECT}
export TOKENIZERS_PARALLELISM=false

# DeepSpeed ZeRO-2 training with explicit launcher (2 GPUs)
# Use deepspeed command directly (not llamafactory-cli)
export CUDA_VISIBLE_DEVICES=0,1
# --include localhost:0,1 specifies to use GPU 0 and 1 on this node
deepspeed --include localhost:0,1 src/train.py \
    --deepspeed examples/deepspeed/ds_z2_config.json \
    --use_reentrant_gc false \
    --stage sft \
    --model_name_or_path ${MODEL_PATH} \
    --do_train \
    --dataset webshop_train_keep_action_with_cm_policy_only \
    --eval_dataset webshop_val_keep_action_with_cm_policy_only \
    --template qwen \
    --finetuning_type full \
    --output_dir saves/qwen25-1p5b-memory/stage_2_deepspeed2card_sft \
    --overwrite_cache \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 4 \
    --has_memory \
    --cutoff_len 2048 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-5 \
    --num_train_epochs 3.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --logging_steps 5 \
    --save_steps 40 \
    --eval_strategy steps \
    --eval_steps 40 \
    --plot_loss \
    --bf16 \
    --pure_bf16 \
    --gradient_checkpointing \
    --flash_attn fa2 \
    --trust_remote_code \
    --skip_embed_head false \
    --report_to wandb \
    --run_name stage_2_deepspeed_z2_2card_full_ft \
    --overwrite_output_dir \
    --save_only_model false 
