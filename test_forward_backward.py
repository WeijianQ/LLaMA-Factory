#!/usr/bin/env python3
"""
Test forward and backward pass with Qwen25-1p5B-Memory model.
Uses 4 GPUs with device_map="auto".
"""

import pickle
import torch
from transformers import AutoTokenizer, AutoConfig
from src.llamafactory.hf_memory_qwen25.modeling_qwen2_5_memory import (
    Qwen2_5_MemoryForCausalLMForFreezeTraining
)
from src.llamafactory.hf_memory_qwen25.configuration_qwen2_5_memory import Qwen2_5_MemoryConfig


def load_sample_data(pkl_path):
    """Load sample data from pickle file."""
    print(f"Loading sample data from {pkl_path}...")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded data keys: {data.keys() if isinstance(data, dict) else type(data)}")
    return data


def main():
    # Configuration
    model_name = "WeijianQi1999/Qwen25-1p5B-Memory"
    pkl_path = "/local/scratch/qi.658/LLaMA-Factory/batch_sample_webshop_val_keep_action_with_cm_proxy_tasks_only.pkl"

    print("=" * 80)
    print("Testing Forward and Backward Pass")
    print("=" * 80)

    # Load sample data
    sample_data = load_sample_data(pkl_path)

    # # Load tokenizer
    # print(f"\nLoading tokenizer from {model_name}...")
    # tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load config
    print(f"\nLoading config from {model_name}...")
    config = Qwen2_5_MemoryConfig.from_pretrained(model_name, trust_remote_code=True)

    # Create custom device_map:
    # - GPU 0: embeddings, lm_head, special layers
    # - GPU 1,2,3: model.layers
    num_layers = config.num_hidden_layers
    device_map = {
        "model.embed_tokens": 0,
        "lm_head": 0,
        "embed_head": 0,
        "special_embed_tokens": 0,
        "special_lm_head": 0,
        "model.norm": 3,  # Final norm on last GPU
        "model.rotary_emb": 0,
    }

    # Distribute model.layers across GPUs 1, 2, 3
    layers_per_gpu = num_layers // 3
    for i in range(num_layers):
        if i < layers_per_gpu:
            gpu = 1
        elif i < 2 * layers_per_gpu:
            gpu = 2
        else:
            gpu = 3
        device_map[f"model.layers.{i}"] = gpu

    print(f"\nCustom device map created:")
    print(f"  GPU 0: embeddings, lm_head, special layers")
    print(f"  GPU 1: layers 0-{layers_per_gpu-1}")
    print(f"  GPU 2: layers {layers_per_gpu}-{2*layers_per_gpu-1}")
    print(f"  GPU 3: layers {2*layers_per_gpu}-{num_layers-1}, norm")

    # Load model with custom device_map
    print(f"\nLoading model from {model_name} with custom device_map...")
    model = Qwen2_5_MemoryForCausalLMForFreezeTraining.from_pretrained(
        model_name,
        config=config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.train()
    # src/llamafactory/model/freeze_llm_for_memory.py
    from src.llamafactory.model.freeze_llm_for_memory import _setup_freeze_tuning_llm_for_memory
    _setup_freeze_tuning_llm_for_memory(model, None, True, False)

    print(f"\nModel device map:")
    if hasattr(model, 'hf_device_map'):
        for name, device in model.hf_device_map.items():
            print(f"  {name}: {device}")

    # Prepare input data
    print("\n" + "=" * 80)
    print("Preparing input data...")
    print("=" * 80)

    # Check what's in the sample data
    print(f"Sample data structure:")
    for key, value in sample_data.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        elif isinstance(value, list):
            print(f"  {key}: list of length {len(value)}")
        else:
            print(f"  {key}: {type(value)}")

    # Extract tensors and move to appropriate device
    input_ids = sample_data.get("input_ids")
    attention_mask = sample_data.get("attention_mask")
    labels = sample_data.get("labels")
    memory_input_ids = sample_data.get("memory_input_ids")
    memory_attention_mask = sample_data.get("memory_attention_mask")
    # Ensure inputs are on the correct device (first device in the map)
    first_device = next(iter(model.hf_device_map.values())) if hasattr(model, 'hf_device_map') else 'cuda:0'

    if input_ids is not None:
        input_ids = input_ids.to(first_device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(first_device)
    if labels is not None:
        labels = labels.to(first_device)
    if memory_input_ids is not None:
        memory_input_ids = memory_input_ids.to(first_device)
    if memory_attention_mask is not None:
        memory_attention_mask = memory_attention_mask.to(first_device)

    print(f"\nInput shapes:")
    if input_ids is not None:
        print(f"  input_ids: {input_ids.shape}")
    if attention_mask is not None:
        print(f"  attention_mask: {attention_mask.shape}")
    if labels is not None:
        print(f"  labels: {labels.shape}")
    if memory_input_ids is not None:
        print(f"  memory_input_ids: {memory_input_ids.shape}")
    if memory_attention_mask is not None:
        print(f"  memory_attention_mask: {memory_attention_mask.shape}")

    # Calculate and print expected valid memory shapes
    if input_ids is not None and memory_input_ids is not None:
        print(f"\nCalculating valid memory shapes:")
        B, L = input_ids.shape
        memory_pad_mask = (input_ids == config.memory_pad_token_id)
        num_memory_positions = memory_pad_mask.sum().item()
        print(f"  Number of memory pad positions: {num_memory_positions}")
        if memory_input_ids is not None:
            max_mem_len = memory_input_ids.shape[-1]
            print(f"  Expected valid_memory_input_ids shape: ({num_memory_positions}, {max_mem_len})")
            print(f"  Expected valid_memory_attention_mask shape: ({num_memory_positions}, {max_mem_len})")

    # Forward pass
    print("\n" + "=" * 80)
    print("Running forward pass...")
    print("=" * 80)

    # slice to 4 batch size
    # input_ids = input_ids[:4]
    # attention_mask = attention_mask[:4]
    # labels = labels[:4]
    # memory_input_ids = memory_input_ids[:4]
    # memory_attention_mask = memory_attention_mask[:4]

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        memory_input_ids=memory_input_ids,
        memory_attention_mask=memory_attention_mask,
    )

    print(f"\nForward pass successful!")
    print(f"  Loss: {outputs.loss.item() if outputs.loss is not None else 'N/A'}")
    print(f"  Logits shape: {outputs.logits.shape}")

    # Backward pass
    print("\n" + "=" * 80)
    print("Running backward pass...")
    print("=" * 80)

    if outputs.loss is not None:
        outputs.loss.backward()
        print(f"\nBackward pass successful!")

        # Check gradients
        print("\nChecking gradients for special parameters:")
        if hasattr(model, 'special_embed_tokens'):
            if model.special_embed_tokens.weight.grad is not None:
                print(f"  special_embed_tokens.weight.grad: shape={model.special_embed_tokens.weight.grad.shape}, "
                      f"norm={model.special_embed_tokens.weight.grad.norm().item():.6f}")
            else:
                print(f"  special_embed_tokens.weight.grad: None")

        if hasattr(model, 'special_lm_head'):
            if model.special_lm_head.weight.grad is not None:
                print(f"  special_lm_head.weight.grad: shape={model.special_lm_head.weight.grad.shape}, "
                      f"norm={model.special_lm_head.weight.grad.norm().item():.6f}")
            else:
                print(f"  special_lm_head.weight.grad: None")

        # Verify weight tying
        if hasattr(model, 'special_embed_tokens') and hasattr(model, 'special_lm_head'):
            print("\nVerifying weight tying:")
            weights_same = model.special_embed_tokens.weight is model.special_lm_head.weight
            print(f"  special_embed_tokens.weight is special_lm_head.weight: {weights_same}")

            if model.special_embed_tokens.weight.grad is not None and model.special_lm_head.weight.grad is not None:
                grads_same = model.special_embed_tokens.weight.grad is model.special_lm_head.weight.grad
                print(f"  Gradients are the same object: {grads_same}")
    else:
        print("Warning: Loss is None, skipping backward pass")

    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
