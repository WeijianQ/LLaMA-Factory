#!/bin/bash

# Model and data paths
export MODEL_PATH=WeijianQi1999/Qwen25-1p5B-Memory
export TRAIN_FILES=webshop_sft_data/webshop_KEEP_ACTION_proxy_tasks_only_12000.parquet
export VAL_FILES=webshop_sft_data/webshop_KEEP_ACTION_proxy_tasks_only_1600.parquet

# Training configuration
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    src/llamafactory/cli.py train \
    examples/webshop/qwen25_full_sft.yaml
