#!/usr/bin/env python3
"""
Lightweight test script to verify data loading and batching from parquet files.
"""

import sys
sys.path.insert(0, "src")

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.model import load_tokenizer
from llamafactory.hparams import get_train_args

def test_single_dataset(dataset_name, tokenizer_module, template, model_args, data_args, training_args):
    """Test loading a single dataset and creating batches."""
    
    print("\n" + "=" * 80)
    print(f"Testing Dataset: {dataset_name}")
    print("=" * 80)
    
    # Update data_args with the new dataset name
    data_args.dataset = [dataset_name]
    
    print(f"\nLoading dataset: {dataset_name}...")
    
    # Load dataset
    data_args.has_memory = True

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
    has_memory = "memory_input_ids" in first_example

    # Count non-ignored labels
    from llamafactory.extras.constants import IGNORE_INDEX
    non_ignored = sum(1 for label_id in first_example['labels'] if label_id != IGNORE_INDEX)
    print(f"\nNon-ignored label tokens: {non_ignored} / {len(first_example['labels'])} ({non_ignored/len(first_example['labels'])*100:.1f}%)")

    print("\n" + "-" * 80)
    print("Creating DataLoader and analyzing collator output for first 200 samples...")
    print("-" * 80)

    import torch
    from torch.utils.data import DataLoader
    from llamafactory.data import MemoryDataCollator

    # Create MemoryDataCollator
    data_collator = MemoryDataCollator(
        tokenizer=tokenizer_module["tokenizer"],
        padding='longest',
        memory_truncate_length=data_args.memory_truncate_length,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX,
    )

    # Create DataLoader with batch_size=1 to inspect each sample's collator output
    dataloader = DataLoader(
        eval_dataset,
        batch_size=4,
        collate_fn=data_collator,
        shuffle=False,
        num_workers=0,
    )

    print(f"\nDataLoader created with batch_size=1")
    print(f"Total batches: {len(dataloader)}")

    # Print info for first 200 samples after collation
    num_samples = min(200, len(dataloader))
    print(f"\n{'='*100}")
    print(f"{'Idx':<6} {'Main Seq Len':<15} {'Num Memories':<15} {'Memory Shape':<25} {'Non-Empty Mems'}")
    print(f"{'='*100}")

    for i, batch in enumerate(dataloader):
        if i >= num_samples:
            break

        main_seq_len = batch['input_ids'].shape[1]
        if main_seq_len > 512:
            from utils import wait_for_debugger
            wait_for_debugger()

        tokenizer = tokenizer_module["tokenizer"]
        if "memory_input_ids" in batch:
            mem_shape = tuple(batch['memory_input_ids'].shape)  # [1, num_mem, mem_len]
            mem_attn = batch['memory_attention_mask'][0]  # [num_mem, mem_len]
            non_empty_count = (mem_attn.sum(dim=-1) > 0).sum().item()
            memory_shape_str = f"{mem_shape}"
            if non_empty_count == 1:
                # decode the main sequence
                main_seq = batch['input_ids'][0]
                main_seq_text = tokenizer.decode(main_seq, skip_special_tokens=False)
                print(f"Main sequence text: {main_seq_text}")
        else:
            non_empty_count = 0
            memory_shape_str = "N/A"

        print(f"{i:<6} {main_seq_len:<15} {mem_shape[1] if 'memory_input_ids' in batch else 0:<15} {memory_shape_str:<25} {non_empty_count}")

    print(f"{'='*100}")
    print(f"Printed collator output for {num_samples} samples.")

    return None


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

    # Minimal arguments for testing - based on qwen3_hard_code_obs.sh
    args = {
        "model_name_or_path": "/fs/ess/PAS1576/qwjian/agent-memory-lab/hf_models/Qwen3_memory_8B_instruct",
        "dataset": "new_webshop_hard_coded_obs_train_debug",  # From qwen3_hard_code_obs.sh
        "template": "qwen",
        "cutoff_len": 512,
        "stage": "sft",
        "do_train": False,
        "output_dir": "test_output",
        "overwrite_cache": True,
        "preprocessing_num_workers": 1,
        "trust_remote_code": True,
        "is_memory_model": True,
        "has_memory": True,
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
        "new_webshop_hard_coded_obs_train_debug"
    ]
    
    results = {}

    for dataset_name in datasets_to_test:
        try:
            test_single_dataset(
                dataset_name=dataset_name,
                tokenizer_module=tokenizer_module,
                template=template,
                model_args=model_args,
                data_args=data_args,
                training_args=training_args
            )
            results[dataset_name] = True
            print(f"\nSUCCESS: Dataset '{dataset_name}' loaded correctly.")
        except Exception as e:
            print(f"\nFAILED: Dataset '{dataset_name}' failed with error:")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            results[dataset_name] = False

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for dataset_name, success in results.items():
        if success:
            print(f"SUCCESS: {dataset_name}")
        else:
            print(f"FAILED: {dataset_name}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    try:
        results = test_data_loading()
        print("\nAll tests completed!")
    except Exception as e:
        print("\nTest failed with error:")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)