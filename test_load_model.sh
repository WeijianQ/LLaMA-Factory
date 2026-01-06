#!/bin/bash
# Test script to verify memory model loading

set -euo pipefail

# export CUDA_VISIBLE_DEVICES=0

# Test lite model
# echo "Testing memory_model_type=lite"
python src/test_load_model.py \
    --stage sft \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --memory_model_type lite \
    --num_query_tokens 16 \
    --do_train \
    --dataset alpaca_en_demo \
    --template qwen \
    --finetuning_type full \
    --output_dir saves/test_load

# Uncomment to test qformer model
# echo "Testing memory_model_type=qformer"
# accelerate launch \
#     --config_file examples/accelerate/osc_4cards_fsdp_lite.yaml \
#     src/test_load_model.py \
#     --model_name_or_path Qwen/Qwen3-8B \
#     --memory_model_type lite \
#     --num_query_tokens 7 \
#     --do_train \
#     --dataset webarena_inverse_dynamics_mixed_100 \
#     --template qwen \
#     --finetuning_type full \
#     --output_dir saves/test_load
