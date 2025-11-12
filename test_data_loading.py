#!/usr/bin/env python3
"""
Lightweight test script to verify data loading and batching from parquet files.
"""

import sys
sys.path.insert(0, "src")

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.model import load_tokenizer
from llamafactory.hparams import get_train_args
from torch.utils.data import DataLoader
import pickle

def test_single_dataset(dataset_name, tokenizer_module, template, model_args, data_args, training_args):
    """Test loading a single dataset and creating batches."""
    
    print("\n" + "=" * 80)
    print(f"Testing Dataset: {dataset_name}")
    print("=" * 80)
    
    # Update data_args with the new dataset name
    data_args.dataset = [dataset_name]
    
    print(f"\nLoading dataset: {dataset_name}...")
    
    # Load dataset
    if "keep_action_with_cm" in dataset_name:
        data_args.has_memory = True
    else:
        data_args.has_memory = False

    dataset_module = get_dataset(
        template=template,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        stage="sft",
        **tokenizer_module
    )

    eval_dataset = dataset_module["train_dataset"]  # since we only passed one dataset

    print(f"\nDataset size: {len(eval_dataset)}")
    print(f"Dataset features: {eval_dataset.features if hasattr(eval_dataset, 'features') else 'N/A'}")

    print("\n" + "-" * 80)
    print("Inspecting first example...")
    print("-" * 80)

    # Get first example
    first_example = eval_dataset[0]
    print(f"\nKeys in example: {list(first_example.keys())}")
    print(f"Input IDs shape: {len(first_example['input_ids'])}")
    print(f"Labels shape: {len(first_example['labels'])}")
    print(f"Attention mask shape: {len(first_example['attention_mask'])}")

    # Check for memory fields
    import torch
    has_memory = "memory_input_ids" in first_example

    # Count non-ignored labels
    from llamafactory.extras.constants import IGNORE_INDEX
    non_ignored = sum(1 for label_id in first_example['labels'] if label_id != IGNORE_INDEX)
    print(f"\nNon-ignored label tokens: {non_ignored} / {len(first_example['labels'])} ({non_ignored/len(first_example['labels'])*100:.1f}%)")

    print("\n" + "-" * 80)
    print("Creating DataLoader with batch_size=16...")
    print("-" * 80)

    # Select appropriate collator based on memory presence
    if has_memory:
        from llamafactory.data import MemoryDataCollator
        print("Using MemoryDataCollator for memory-augmented data")
        data_collator = MemoryDataCollator(
            padding='longest',  # Use 'longest' to pad to batch max, not model max (131072)
            memory_truncate_length=data_args.memory_truncate_length,
            pad_to_multiple_of=8,  # Changed from 64 to match workflow.py
            label_pad_token_id=IGNORE_INDEX,
            **tokenizer_module,
        )
    else:
        from llamafactory.data import SFTDataCollatorWith4DAttentionMask
        print("Using SFTDataCollatorWith4DAttentionMask")
        data_collator = SFTDataCollatorWith4DAttentionMask(
            template=template,
            model=None,
            pad_to_multiple_of=None,
            label_pad_token_id=IGNORE_INDEX,
            block_diag_attn=False,
            attn_implementation="eager",
            compute_dtype=None,
            **tokenizer_module,
        )

    # Create DataLoader
    dataloader = DataLoader(
        eval_dataset,
        batch_size=16,
        collate_fn=data_collator,
        shuffle=False,
        num_workers=0,
    )

    print(f"\nDataLoader created successfully!")
    print(f"Number of batches: {len(dataloader)}")

    print("\n" + "-" * 80)
    print("Getting first batch...")
    print("-" * 80)

    # Get first batch
    batch = next(iter(dataloader))
    
    # Print batch information
    print_batch_info(batch, tokenizer_module["tokenizer"], dataset_name)

    # save batch to a pickle
    with open(f"batch_sample_{dataset_name}.pkl", "wb") as f:
        pickle.dump(batch, f)
    print(f"Batch saved to batch_sample_{dataset_name}.pkl")

    return batch


def print_batch_info(batch, tokenizer, dataset_name):
    """Print detailed information about a batch."""
    
    print("\n" + "=" * 80)
    print(f"FIRST BATCH for {dataset_name}:")
    print("=" * 80)
    
    print(f"\nBatch keys: {list(batch.keys())}")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Attention mask shape: {batch['attention_mask'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")

    # Check memory fields in batch
    if "memory_input_ids" in batch:
        print(f"\n✓ Memory in batch!")
        print(f"Memory input IDs shape: {batch['memory_input_ids'].shape}")
        print(f"Memory attention mask shape: {batch['memory_attention_mask'].shape}")
        print(f"  - Batch size: {batch['memory_input_ids'].shape[0]}")
        print(f"  - Max memory num: {batch['memory_input_ids'].shape[1]}")
        print(f"  - Max memory len: {batch['memory_input_ids'].shape[2]}")

        # Check first sample's memories
        first_sample_memories = batch['memory_input_ids'][0]
        first_sample_mask = batch['memory_attention_mask'][0]
        non_empty_memories = (first_sample_mask.sum(dim=1) > 0).sum().item()
        print(f"\nFirst sample has {non_empty_memories} non-empty memories")

        if non_empty_memories > 0:
            print("\nFirst memory of first sample:")
            first_memory = first_sample_memories[0]
            first_memory_mask = first_sample_mask[0]
            actual_len = first_memory_mask.sum().item()
            print(f"  Actual length: {actual_len}")
            if actual_len > 0:
                # Decode first memory
                valid_tokens = first_memory[first_memory_mask.bool()]
                decoded_memory = tokenizer.decode(valid_tokens, skip_special_tokens=False)
                print(f"  Decoded: {decoded_memory[:200]}")

    # Check padding
    print(f"\nPad token ID: {tokenizer.pad_token_id}")
    print(f"EOS token ID: {tokenizer.eos_token_id}")

    # Count padding in first example of batch
    first_in_batch = batch['input_ids'][0]
    pad_count = (first_in_batch == tokenizer.pad_token_id).sum().item()
    print(f"Padding tokens in first example: {pad_count} / {len(first_in_batch)}")
    
    # Print batch tensor details
    print(f"\nBatch details:")
    print(f"  - input_ids dtype: {batch['input_ids'].dtype}")
    print(f"  - input_ids device: {batch['input_ids'].device}")
    print(f"  - attention_mask dtype: {batch['attention_mask'].dtype}")
    print(f"  - labels dtype: {batch['labels'].dtype}")
    
    # Print first few values of first sample's input_ids
    print(f"\nFirst sample input_ids (first 20 tokens): {batch['input_ids'][0][:20].tolist()}")
    
    # print the real labels
    IGNORE_INDEX = -100
    # non label_start_ba
    real_labels = [label for label in batch['labels'][0].tolist() if label != IGNORE_INDEX]
    print(f"First sample labels (first 20 tokens): {real_labels[:20]}")
    print(f"decoded labels: {[tokenizer.convert_ids_to_tokens(label) for label in real_labels[:20]]}")
    
    print("\n" + "=" * 80)


def test_data_loading():
    """Test loading webshop validation data and creating batches."""

    # Minimal arguments for testing
    args = {
        "model_name_or_path": "WeijianQi1999/Qwen25-1p5B-Memory",
        "dataset": "webshop_val_baseline",  # Default, will be overridden
        "template": "qwen",
        "cutoff_len": 4096,
        "stage": "sft",
        "do_train": False,
        "output_dir": "test_output",
        "overwrite_cache": True,
        "preprocessing_num_workers": 1,
        "trust_remote_code": True,
    }

    # Convert to command line args format
    cmd_args = []
    for key, value in args.items():
        cmd_args.append(f"--{key}")
        cmd_args.append(str(value))

    print("=" * 80)
    print("Loading tokenizer and template...")
    print("=" * 80)

    # Parse arguments
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(cmd_args)

    # Load tokenizer
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]

    # Get template
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    print(f"\nTokenizer: {tokenizer.__class__.__name__}")
    print(f"Template: {template.__class__.__name__}")
    print(f"Vocab size: {len(tokenizer)}")

    # Test both datasets
    datasets_to_test = [
        "webshop_train_keep_action_with_cm_policy_only"
    ]
    
    batches = {}
    
    for dataset_name in datasets_to_test:
        try:
            batch = test_single_dataset(
                dataset_name=dataset_name,
                tokenizer_module=tokenizer_module,
                template=template,
                model_args=model_args,
                data_args=data_args,
                training_args=training_args
            )
            batches[dataset_name] = batch
            print(f"\n✓ SUCCESS! Dataset '{dataset_name}' loaded and batched correctly.")
        except Exception as e:
            print(f"\n✗ FAILED! Dataset '{dataset_name}' failed with error:")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            batches[dataset_name] = None

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for dataset_name, batch in batches.items():
        if batch is not None:
            print(f"✓ {dataset_name}: SUCCESS")
        else:
            print(f"✗ {dataset_name}: FAILED")
    print("=" * 80)

    return batches


if __name__ == "__main__":
    try:
        batches = test_data_loading()
        print("\n✓ All tests completed!")
    except Exception as e:
        print("\n✗ Test failed with error:")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)