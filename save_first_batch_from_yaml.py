#!/usr/bin/env python3
"""
Script to extract and save the first batch from the dataloader using YAML config.

Usage:
    python save_first_batch_from_yaml.py examples/webshop/stage_2_deepspeed2card.yaml
    python save_first_batch_from_yaml.py examples/webshop/stage_2_deepspeed2card.yaml --output first_batch_stage2.pkl
"""
import argparse
import pickle
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from omegaconf import OmegaConf
from llamafactory.data import get_dataset
from llamafactory.hparams import get_train_args
from llamafactory.model import load_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Save first batch from dataloader")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument("--output", type=str, default="first_batch.pkl", help="Output pickle file name")
    parser.add_argument("--no-shuffle", action="store_true", help="Don't shuffle the dataloader")
    parser.add_argument("--num-batches", type=int, default=1, help="Number of batches to save")
    args = parser.parse_args()

    print(f"Loading config from: {args.config}")

    # Load YAML config and convert to dict for get_train_args
    yaml_config = OmegaConf.load(Path(args.config).absolute())
    dict_config = OmegaConf.to_container(yaml_config)

    # Parse arguments from YAML config dict
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(
        dict_config
    )

    print("Configuration loaded:")
    print(f"  Model: {model_args.model_name_or_path}")
    print(f"  Dataset: {data_args.dataset}")
    print(f"  Template: {data_args.template}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Has memory: {getattr(data_args, 'has_memory', False)}")

    print("\nLoading tokenizer...")
    tokenizer = load_tokenizer(model_args)
    print(f"  Tokenizer vocab size: {len(tokenizer)}")

    print("\nLoading dataset...")
    dataset_module = get_dataset(
        model_args,
        data_args,
        training_args,
        stage=training_args.stage,
        tokenizer=tokenizer,
    )

    train_dataset = dataset_module.train_dataset
    data_collator = dataset_module.data_collator

    print(f"  Dataset size: {len(train_dataset)}")

    # Create a simple dataloader
    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        collate_fn=data_collator,
        shuffle=not args.no_shuffle,
        num_workers=0,  # Use 0 to avoid multiprocessing issues
    )

    print(f"\nExtracting {args.num_batches} batch(es)...")
    batches = []
    for i, batch in enumerate(train_dataloader):
        if i >= args.num_batches:
            break
        batches.append(batch)
        print(f"  Batch {i+1}/{args.num_batches} extracted")

    # Save to pickle
    output_file = args.output
    print(f"\nSaving batch(es) to {output_file}...")

    data_to_save = {
        "batches": batches,
        "num_batches": len(batches),
        "batch_size": training_args.per_device_train_batch_size,
        "dataset": data_args.dataset,
        "model": model_args.model_name_or_path,
        "template": data_args.template,
    }

    with open(output_file, "wb") as f:
        pickle.dump(data_to_save, f)

    print(f"✓ Successfully saved {len(batches)} batch(es) to {output_file}")

    # Print info about the first batch
    first_batch = batches[0]
    print("\nFirst batch information:")
    for key, value in first_batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: type={type(value)}")

    # Save a human-readable version
    output_txt = output_file.replace(".pkl", "_info.txt")
    print(f"\nSaving human-readable version to {output_txt}...")

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("BATCH INFORMATION\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Number of batches: {len(batches)}\n")
        f.write(f"Batch size: {training_args.per_device_train_batch_size}\n")
        f.write(f"Dataset: {data_args.dataset}\n")
        f.write(f"Model: {model_args.model_name_or_path}\n")
        f.write(f"Template: {data_args.template}\n\n")

        for batch_idx, batch in enumerate(batches):
            f.write("=" * 80 + "\n")
            f.write(f"BATCH {batch_idx + 1}\n")
            f.write("=" * 80 + "\n\n")

            for key, value in batch.items():
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
            if "input_ids" in batch:
                f.write("\n" + "=" * 80 + "\n")
                f.write("DECODED TEXT (all examples in batch):\n")
                f.write("=" * 80 + "\n\n")

                for idx in range(batch["input_ids"].shape[0]):
                    f.write(f"\n--- Example {idx + 1} ---\n\n")
                    input_ids = batch["input_ids"][idx]
                    decoded_text = tokenizer.decode(input_ids, skip_special_tokens=False)
                    f.write(decoded_text)
                    f.write("\n")

                    # Also show with special tokens removed
                    f.write("\n[Without special tokens]:\n")
                    decoded_text_clean = tokenizer.decode(input_ids, skip_special_tokens=True)
                    f.write(decoded_text_clean)
                    f.write("\n\n")

    print(f"✓ Successfully saved human-readable info to {output_txt}")

    # Print a sample of the decoded text
    if "input_ids" in first_batch:
        print("\n" + "=" * 80)
        print("SAMPLE - First example decoded text:")
        print("=" * 80)
        sample_text = tokenizer.decode(first_batch["input_ids"][0], skip_special_tokens=False)
        # Print first 500 characters
        print(sample_text[:500])
        if len(sample_text) > 500:
            print(f"\n... (truncated, total length: {len(sample_text)} chars)")
        print("=" * 80)


if __name__ == "__main__":
    main()
