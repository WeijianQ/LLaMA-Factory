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

    # Minimal arguments for testing - matching qwen3_webarena_memory_debug.sh
    args = {
        "model_name_or_path": "/fs/ess/PAS1576/qwjian/agent-memory-lab/external/LLaMA-Factory/hf_models/Qwen3_memory_8B_16q_tokens",
        "dataset": "webarena_stress_test_top200",  # From qwen3_webarena_memory_debug.sh
        "template": "qwen",
        "cutoff_len": 16384,  # From qwen3_webarena_memory_debug.sh
        "memory_truncate_length": 8192,  # From qwen3_webarena_memory_debug.sh
        "max_memory_num": 15,  # From qwen3_webarena_memory_debug.sh
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

    # Test dataset matching qwen3_webarena_memory.sh
    datasets_to_test = [
        "webarena_stress_test_top200"
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


def simulate_rank_step(target_rank=1, target_optimizer_step=35, verbose=False):
    """
    Simulate what a given rank sees at a given optimizer step during 4-GPU FSDP training.

    Config from qwen3_webarena_memory.sh:
    - 4 GPUs, batch_size=1, gradient_accumulation_steps=64
    - DistributedSampler distributes samples:
      Rank 0: samples 0, 4, 8, ...
      Rank 1: samples 1, 5, 9, ...
      Rank 2: samples 2, 6, 10, ...
      Rank 3: samples 3, 7, 11, ...
    """
    import os
    # Fake single-GPU environment to bypass distributed check
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    if verbose:
        print("=" * 80)
        print(f"Simulating Rank {target_rank}, Step {target_optimizer_step} (matching qwen3_webarena_memory.sh config)")
        print("=" * 80)

    args = {
        "model_name_or_path": "/fs/ess/PAS1576/qwjian/agent-memory-lab/hf_models/Qwen3_memory_8B_instruct",
        "dataset": "webarena_sft_memory_repeat_obs_train",
        "template": "qwen",
        "cutoff_len": 16384,
        "memory_truncate_length": 8192,
        "max_memory_num": 15,
        "stage": "sft",
        "do_train": True,  # Need this to pass validation
        "output_dir": "test_output",
        "overwrite_cache": False,  # Use cache for speed
        "preprocessing_num_workers": 16,
        "trust_remote_code": True,
        "is_memory_model": True,
        "has_memory": True,
        "per_device_train_batch_size": 1,
    }

    cmd_args = []
    for key, value in args.items():
        cmd_args.append(f"--{key}")
        cmd_args.append(str(value))

    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(cmd_args)
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    if verbose:
        print(f"Loading dataset...")
    data_args.has_memory = True
    dataset_module = get_dataset(
        template=template,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        stage="sft",
        **tokenizer_module
    )
    dataset = dataset_module["train_dataset"]
    if verbose:
        print(f"Dataset size: {len(dataset)}")

    # Simulate DistributedSampler for 4 GPUs
    # Config: 4 GPUs, batch_size=1, gradient_accumulation_steps=64
    num_gpus = 4
    ga_steps = 64

    # Each optimizer step = 64 micro-batches per rank
    # DistributedSampler distributes:
    # Global micro-batch i -> Rank (i % num_gpus)
    # Rank N gets global indices: N, N+4, N+8, ... (where idx % 4 == N)
    # Rank N's micro-batch M corresponds to global index: N + M * num_gpus

    if verbose:
        print(f"\nOptimizer step {target_optimizer_step}:")
        print(f"  Rank {target_rank} processes micro-batches {(target_optimizer_step-1)*ga_steps} to {target_optimizer_step*ga_steps-1}")

    # Check all 64 micro-batches in this optimizer step
    start_micro = (target_optimizer_step - 1) * ga_steps
    end_micro = target_optimizer_step * ga_steps

    if verbose:
        print(f"\nChecking all {ga_steps} micro-batches in optimizer step {target_optimizer_step}:")
        print(f"{'Micro':>8} {'Global Idx':>12} {'Input Len':>12} {'Mem Count':>12} {'Max Mem Len':>12}")
        print("-" * 60)

    max_input_len = 0
    max_mem_len = 0
    total_mem_tokens = 0  # Sum of all memory tokens
    problematic_samples = []

    for micro_batch in range(start_micro, end_micro):
        global_idx = target_rank + micro_batch * num_gpus

        if global_idx >= len(dataset):
            if verbose:
                print(f"{micro_batch:>8} {global_idx:>12} OUT OF RANGE")
            continue

        sample = dataset[global_idx]
        input_len = len(sample['input_ids'])

        if 'memory_input_ids' in sample:
            mem_count = len(sample['memory_input_ids'])
            mem_lens = [len(m) for m in sample['memory_input_ids']]
            sample_max_mem_len = max(mem_lens) if mem_lens else 0
            sample_total_mem = sum(mem_lens)
        else:
            mem_count = 0
            sample_max_mem_len = 0
            sample_total_mem = 0

        if verbose:
            print(f"{micro_batch:>8} {global_idx:>12} {input_len:>12} {mem_count:>12} {sample_max_mem_len:>12}")

        if input_len > max_input_len:
            max_input_len = input_len
        if sample_max_mem_len > max_mem_len:
            max_mem_len = sample_max_mem_len
        total_mem_tokens += sample_total_mem

        # Flag potentially problematic samples
        if input_len > 10000 or sample_max_mem_len > 6000:
            problematic_samples.append((micro_batch, global_idx, input_len, mem_count, sample_max_mem_len))

    if verbose:
        print("-" * 60)
        print(f"Max input_len in this step: {max_input_len}")
        print(f"Max memory_len in this step: {max_mem_len}")
        print(f"Total memory tokens in this step: {total_mem_tokens}")

        if problematic_samples:
            print(f"\n⚠️  Potentially problematic samples:")
            for micro, gidx, ilen, mcnt, mlen in problematic_samples:
                print(f"  Micro {micro}, Global {gidx}: input={ilen}, mem_count={mcnt}, max_mem_len={mlen}")

        print(f"\n{'='*80}")
        print("Analysis complete.")
        print(f"{'='*80}")

    return {
        "step": target_optimizer_step,
        "rank": target_rank,
        "max_input_len": max_input_len,
        "max_mem_len": max_mem_len,
        "total_mem_tokens": total_mem_tokens,
        "problematic_count": len(problematic_samples),
        "problematic_samples": problematic_samples,
    }


def compare_steps():
    """Compare memory statistics across multiple steps to find what's special about the OOM step."""
    print("=" * 100)
    print("Comparing Steps 33-40 for Rank 1 (OOM occurred at pbar=35, likely during step 36)")
    print("=" * 100)

    steps_to_check = [33, 34, 35, 36, 37, 38, 39, 40]
    results = []

    for step in steps_to_check:
        result = simulate_rank_step(target_rank=1, target_optimizer_step=step, verbose=False)
        results.append(result)

    # Print comparison table
    print(f"\n{'Step':>6} {'Max Input':>12} {'Max Mem':>12} {'Total Mem':>14} {'Problematic':>12}")
    print("-" * 60)
    for r in results:
        print(f"{r['step']:>6} {r['max_input_len']:>12} {r['max_mem_len']:>12} {r['total_mem_tokens']:>14} {r['problematic_count']:>12}")
    print("-" * 60)

    # # Find the step with max input length
    # max_input_step = max(results, key=lambda x: x['max_input_len'])
    # max_mem_step = max(results, key=lambda x: x['max_mem_len'])
    # max_total_mem_step = max(results, key=lambda x: x['total_mem_tokens'])

    # print(f"\nStep with max input_len: Step {max_input_step['step']} ({max_input_step['max_input_len']} tokens)")
    # print(f"Step with max memory_len: Step {max_mem_step['step']} ({max_mem_step['max_mem_len']} tokens)")
    # print(f"Step with max total_mem: Step {max_total_mem_step['step']} ({max_total_mem_step['total_mem_tokens']} tokens)")

    # # Show problematic samples for the likely OOM step (36)
    # print("\n" + "=" * 100)
    # print("Details for Step 36 (likely OOM step):")
    # print("=" * 100)
    # step36 = simulate_rank_step(target_rank=1, target_optimizer_step=36, verbose=True)

    # # Also check all ranks at step 36 to see which rank has the biggest samples
    # print("\n" + "=" * 100)
    # print("Comparing all ranks at Step 36:")
    # print("=" * 100)
    # print(f"{'Rank':>6} {'Max Input':>12} {'Max Mem':>12} {'Total Mem':>14} {'Problematic':>12}")
    # print("-" * 60)
    # for rank in range(4):
    #     r = simulate_rank_step(target_rank=rank, target_optimizer_step=36, verbose=False)
    #     print(f"{rank:>6} {r['max_input_len']:>12} {r['max_mem_len']:>12} {r['total_mem_tokens']:>14} {r['problematic_count']:>12}")
    # print("-" * 60)


def use_real_dataloader(target_rank=1, target_step=35):
    """
    Use the actual DistributedSampler (like HuggingFace Trainer) to see what each rank gets.
    """
    import os
    import torch
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    # Set environment to simulate the target rank
    os.environ["RANK"] = str(target_rank)
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "4"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    print("=" * 100)
    print(f"Using real DistributedSampler to check what Rank {target_rank} sees at step {target_step}")
    print("=" * 100)

    args = {
        "model_name_or_path": "/fs/ess/PAS1576/qwjian/agent-memory-lab/hf_models/Qwen3_memory_8B_instruct",
        "dataset": "webarena_sft_memory_repeat_obs_train",
        "template": "qwen",
        "cutoff_len": 16384,
        "memory_truncate_length": 8192,
        "max_memory_num": 15,
        "stage": "sft",
        "do_train": True,
        "output_dir": "test_output",
        "overwrite_cache": False,
        "preprocessing_num_workers": 16,
        "trust_remote_code": True,
        "is_memory_model": True,
        "has_memory": True,
        "per_device_train_batch_size": 1,
    }

    cmd_args = []
    for key, value in args.items():
        cmd_args.append(f"--{key}")
        cmd_args.append(str(value))

    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(cmd_args)
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    print(f"Loading dataset...")
    data_args.has_memory = True
    dataset_module = get_dataset(
        template=template,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        stage="sft",
        **tokenizer_module
    )
    dataset = dataset_module["train_dataset"]
    print(f"Dataset size: {len(dataset)}")

    # Create DistributedSampler like HuggingFace Trainer does
    # shuffle=True (default), seed=42 (default in Trainer)
    sampler = DistributedSampler(
        dataset,
        num_replicas=4,
        rank=target_rank,
        shuffle=True,
        seed=42,
    )

    from llamafactory.extras.constants import IGNORE_INDEX
    from llamafactory.data import MemoryDataCollator

    data_collator = MemoryDataCollator(
        tokenizer=tokenizer,
        padding='longest',
        memory_truncate_length=data_args.memory_truncate_length,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        collate_fn=data_collator,
        num_workers=0,
    )

    print(f"\nDataLoader created with DistributedSampler (rank={target_rank}, world_size=4)")
    print(f"Total batches for this rank: {len(dataloader)}")

    # GA = 64, so optimizer step N covers batches [(N-1)*64, N*64)
    ga_steps = 64
    start_batch = (target_step - 1) * ga_steps
    end_batch = target_step * ga_steps

    print(f"\nChecking optimizer step {target_step} (batches {start_batch} to {end_batch-1}):")
    print(f"{'Batch':>8} {'Input Len':>12} {'Mem Count':>12} {'Max Mem Len':>12}")
    print("-" * 50)

    max_input_len = 0
    max_mem_len = 0
    problematic = []

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx < start_batch:
            continue
        if batch_idx >= end_batch:
            break

        input_len = batch['input_ids'].shape[1]

        if 'memory_input_ids' in batch:
            mem_count = batch['memory_input_ids'].shape[1]
            mem_len = batch['memory_input_ids'].shape[2]
            mem_attn = batch['memory_attention_mask'][0]
            non_empty = (mem_attn.sum(dim=-1) > 0).sum().item()
        else:
            mem_count = 0
            mem_len = 0
            non_empty = 0

        print(f"{batch_idx:>8} {input_len:>12} {non_empty:>12} {mem_len:>12}")

        if input_len > max_input_len:
            max_input_len = input_len
        if mem_len > max_mem_len:
            max_mem_len = mem_len

        if input_len > 10000 or mem_len > 6000:
            problematic.append((batch_idx, input_len, non_empty, mem_len))

    print("-" * 50)
    print(f"Max input_len: {max_input_len}")
    print(f"Max mem_len: {max_mem_len}")

    if problematic:
        print(f"\n⚠️  Problematic batches:")
        for b, ilen, mcnt, mlen in problematic:
            print(f"  Batch {b}: input={ilen}, mem_count={mcnt}, mem_len={mlen}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--step", type=int, default=35)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--real", action="store_true", help="Use real DistributedSampler")
    args = parser.parse_args()
    if args.compare:
        print("Comparing steps...")
        compare_steps()
    elif args.test:
        print("Testing data loading...")
        test_data_loading()
    elif args.real:
        use_real_dataloader(target_rank=args.rank, target_step=args.step)
    else:
        simulate_rank_step(target_rank=args.rank, target_optimizer_step=args.step, verbose=True)