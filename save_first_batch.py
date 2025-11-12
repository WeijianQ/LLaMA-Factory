#!/usr/bin/env python3
"""
Script to extract and save the first batch from the dataloader.
"""
import os
import pickle
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from llamafactory.data import get_dataset
from llamafactory.hparams import get_train_args
from llamafactory.model import load_model, load_tokenizer


def main():
    # Parse arguments from the training config
    # You can modify the path to your YAML config file
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(
        [
            "--stage", "sft",
            "--model_name_or_path", "saves/freeze_llm_for_memory/stage_1_sft/checkpoint-300_converted",
            "--do_train",
            "--dataset", "webshop_train_keep_action_with_cm_policy_only",
            "--template", "qwen",
            "--finetuning_type", "full",
            "--output_dir", "saves/temp_output",
            "--has_memory",
            "--cutoff_len", "2048",
            "--per_device_train_batch_size", "2",
            "--preprocessing_num_workers", "16",
            "--overwrite_cache",
            "--flash_attn", "fa2",
            "--bf16",
        ]
    )

    print("Loading tokenizer...")
    tokenizer = load_tokenizer(model_args)

    print("Loading dataset...")
    dataset_module = get_dataset(
        model_args,
        data_args,
        training_args,
        stage="sft",
        tokenizer=tokenizer,
    )

    train_dataset = dataset_module.train_dataset
    data_collator = dataset_module.data_collator

    print(f"Dataset size: {len(train_dataset)}")

    # Create a simple dataloader
    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        collate_fn=data_collator,
        shuffle=False,  # Don't shuffle to get consistent first batch
        num_workers=0,  # Use 0 to avoid multiprocessing issues
    )

    print("Extracting first batch...")
    first_batch = next(iter(train_dataloader))

    # Save to pickle
    output_file = "first_batch.pkl"
    print(f"Saving first batch to {output_file}...")

    with open(output_file, "wb") as f:
        pickle.dump(first_batch, f)

    print(f"✓ Successfully saved first batch to {output_file}")

    # Print some info about the batch
    print("\nBatch information:")
    for key, value in first_batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: type={type(value)}")

    # Optionally save a human-readable version
    print("\nSaving human-readable version...")
    output_txt = "first_batch_info.txt"
    with open(output_txt, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("FIRST BATCH INFORMATION\n")
        f.write("=" * 80 + "\n\n")

        for key, value in first_batch.items():
            f.write(f"\n{key}:\n")
            f.write("-" * 40 + "\n")
            if isinstance(value, torch.Tensor):
                f.write(f"Shape: {value.shape}\n")
                f.write(f"Dtype: {value.dtype}\n")
                f.write(f"Device: {value.device}\n")
                if value.numel() < 100:  # Only print small tensors
                    f.write(f"Values:\n{value}\n")
            else:
                f.write(f"Type: {type(value)}\n")
                f.write(f"Value: {value}\n")

        # Try to decode the input_ids back to text
        if "input_ids" in first_batch:
            f.write("\n" + "=" * 80 + "\n")
            f.write("DECODED TEXT (first example in batch):\n")
            f.write("=" * 80 + "\n")
            input_ids = first_batch["input_ids"][0]  # First example
            decoded_text = tokenizer.decode(input_ids, skip_special_tokens=False)
            f.write(decoded_text)
            f.write("\n")

    print(f"✓ Successfully saved human-readable info to {output_txt}")


if __name__ == "__main__":
    main()
