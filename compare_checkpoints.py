#!/usr/bin/env python3
"""
Compare two FSDP checkpoints to see which parameters differ.

Usage:
    python compare_checkpoints.py \
        --original rl_ckpts/rl_on_a_stage_1_trained_memory_model_NOMEMGRAD/step-100/model_world_size_2_rank_0.pt \
        --new saves/aligned_step100/model_world_size_2_rank_0.pt
"""

import torch
from torch.distributed.tensor import DTensor
import argparse
from pathlib import Path


def unwrap_tensor(t):
    """Extract actual tensor from DTensor or regular tensor."""
    if isinstance(t, DTensor):
        return t.to_local()
    return t


def compare_state_dicts(state_dict_1, state_dict_2, name1="Original", name2="New"):
    """Compare two state dicts and report differences."""

    keys1 = set(state_dict_1.keys())
    keys2 = set(state_dict_2.keys())

    # Check for missing/extra keys
    only_in_1 = keys1 - keys2
    only_in_2 = keys2 - keys1
    common_keys = keys1 & keys2

    print(f"\n{'='*80}")
    print(f"Key Comparison:")
    print(f"{'='*80}")
    print(f"Total keys in {name1}: {len(keys1)}")
    print(f"Total keys in {name2}: {len(keys2)}")
    print(f"Common keys: {len(common_keys)}")

    if only_in_1:
        print(f"\n❌ Keys only in {name1} ({len(only_in_1)}):")
        for key in sorted(only_in_1)[:10]:
            print(f"  - {key}")
        if len(only_in_1) > 10:
            print(f"  ... and {len(only_in_1) - 10} more")

    if only_in_2:
        print(f"\n❌ Keys only in {name2} ({len(only_in_2)}):")
        for key in sorted(only_in_2)[:10]:
            print(f"  - {key}")
        if len(only_in_2) > 10:
            print(f"  ... and {len(only_in_2) - 10} more")

    # Compare common keys
    print(f"\n{'='*80}")
    print(f"Value Comparison for Common Keys:")
    print(f"{'='*80}")

    shape_mismatches = []
    dtype_mismatches = []
    value_differences = []
    identical_params = []
    close_params = []
    non_tensor_diffs = []

    for key in sorted(common_keys):
        v1 = state_dict_1[key]
        v2 = state_dict_2[key]

        # Handle non-tensor values
        if not isinstance(v1, (torch.Tensor, DTensor)) or not isinstance(v2, (torch.Tensor, DTensor)):
            if v1 != v2:
                non_tensor_diffs.append((key, type(v1), type(v2), v1, v2))
            continue

        # Unwrap tensors first
        t1 = unwrap_tensor(v1)
        t2 = unwrap_tensor(v2)

        # Special row-by-row comparison for embedding tables
        if key in ['lm_head.weight', 'model.embed_tokens.weight']:
            if t1.shape == t2.shape and len(t1.shape) == 2:
                R, _ = t1.shape
                mismatch_rows = []
                for r in range(R):
                    if not torch.allclose(t1[r], t2[r], rtol=1e-3, atol=1e-3):
                        mismatch_rows.append(r)
                print(f"  {key}: {len(mismatch_rows)}/{R} rows mismatch")
                if len(mismatch_rows) > 0 and len(mismatch_rows) <= 10:
                    print(f"    Mismatch rows: {mismatch_rows}")
                elif len(mismatch_rows) > 10:
                    print(f"    First 10 mismatch rows: {mismatch_rows[:10]}")
            else:
                print(f"  {key}: shapes differ or not 2D, skipping row comparison")

        # Check shapes
        if t1.shape != t2.shape:
            shape_mismatches.append((key, t1.shape, t2.shape))
            continue

        # Check dtypes
        if t1.dtype != t2.dtype:
            dtype_mismatches.append((key, t1.dtype, t2.dtype))
            # Try to compare values anyway (converting to same dtype)
            t1 = t1.float()
            t2 = t2.float()

        # Check values
        if torch.equal(t1, t2):
            identical_params.append(key)
        elif torch.allclose(t1, t2, rtol=1e-3, atol=1e-3):
            close_params.append(key)
        else:
            max_diff = (t1 - t2).abs().max().item()
            mean_diff = (t1 - t2).abs().mean().item()
            rel_diff = ((t1 - t2).abs() / (t1.abs() + 1e-8)).max().item()

            value_differences.append({
                'key': key,
                'shape': t1.shape,
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'rel_diff': rel_diff,
                'norm_1': t1.norm().item(),
                'norm_2': t2.norm().item(),
            })

    # Print results
    print(f"\n✓ Identical parameters: {len(identical_params)}")
    print(f"\n✓ Close parameters: {len(close_params)}")
    if shape_mismatches:
        print(f"\n❌ Shape mismatches ({len(shape_mismatches)}):")
        for key, shape1, shape2 in shape_mismatches[:10]:
            print(f"  {key}:")
            print(f"    {name1}: {shape1}")
            print(f"    {name2}: {shape2}")
        if len(shape_mismatches) > 10:
            print(f"  ... and {len(shape_mismatches) - 10} more")

    if dtype_mismatches:
        print(f"\n⚠️  Dtype mismatches ({len(dtype_mismatches)}):")
        for key, dtype1, dtype2 in dtype_mismatches[:10]:
            print(f"  {key}: {dtype1} vs {dtype2}")
        if len(dtype_mismatches) > 10:
            print(f"  ... and {len(dtype_mismatches) - 10} more")

    if non_tensor_diffs:
        print(f"\n⚠️  Non-tensor differences ({len(non_tensor_diffs)}):")
        for key, type1, type2, val1, val2 in non_tensor_diffs:
            print(f"  {key}:")
            print(f"    Type: {type1} vs {type2}")
            print(f"    {name1}: {val1}")
            print(f"    {name2}: {val2}")

    if value_differences:
        print(f"\n⚠️  Value differences ({len(value_differences)}):")

        # Sort by max_diff descending
        value_differences.sort(key=lambda x: x['max_diff'], reverse=True)

        print(f"\n  Top 10 largest differences:")
        for i, diff in enumerate(value_differences[:10], 1):
            print(f"\n  {i}. {diff['key']}")
            print(f"     Shape: {diff['shape']}")
            print(f"     Max absolute diff: {diff['max_diff']:.6e}")
            print(f"     Mean absolute diff: {diff['mean_diff']:.6e}")
            print(f"     Max relative diff: {diff['rel_diff']:.6e}")
            print(f"     Norm ({name1}): {diff['norm_1']:.6e}")
            print(f"     Norm ({name2}): {diff['norm_2']:.6e}")

        if len(value_differences) > 10:
            print(f"\n  ... and {len(value_differences) - 10} more parameters with differences")

        # Summary statistics
        all_max_diffs = [d['max_diff'] for d in value_differences]
        all_mean_diffs = [d['mean_diff'] for d in value_differences]

        print(f"\n  Overall statistics:")
        print(f"    Max of max diffs: {max(all_max_diffs):.6e}")
        print(f"    Mean of max diffs: {sum(all_max_diffs)/len(all_max_diffs):.6e}")
        print(f"    Max of mean diffs: {max(all_mean_diffs):.6e}")
        print(f"    Mean of mean diffs: {sum(all_mean_diffs)/len(all_mean_diffs):.6e}")

    # Final summary
    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"{'='*80}")
    total_compared = len(common_keys)
    num_issues = len(shape_mismatches) + len(value_differences)

    if num_issues == 0:
        print(f"✅ All {total_compared} common parameters are identical!")
    else:
        print(f"⚠️  {num_issues} / {total_compared} parameters have differences")
        print(f"   - Shape mismatches: {len(shape_mismatches)}")
        print(f"   - Value differences: {len(value_differences)}")
        print(f"   - Identical: {len(identical_params)}")

    print(f"{'='*80}\n")

    return {
        'identical': identical_params,
        'shape_mismatches': shape_mismatches,
        'dtype_mismatches': dtype_mismatches,
        'value_differences': value_differences,
        'non_tensor_diffs': non_tensor_diffs,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare two FSDP checkpoints")
    parser.add_argument(
        "--original",
        type=str,
        required=True,
        help="Path to original checkpoint"
    )
    parser.add_argument(
        "--new",
        type=str,
        required=True,
        help="Path to new checkpoint"
    )
    parser.add_argument(
        "--name1",
        type=str,
        default="Original",
        help="Name for original checkpoint in output"
    )
    parser.add_argument(
        "--name2",
        type=str,
        default="New",
        help="Name for new checkpoint in output"
    )

    args = parser.parse_args()

    print("="*80)
    print("FSDP Checkpoint Comparison")
    print("="*80)
    print(f"{args.name1}: {args.original}")
    print(f"{args.name2}: {args.new}")

    # Load checkpoints
    print(f"\nLoading {args.name1}...")
    original = torch.load(args.original, map_location='cpu', weights_only=False)
    print(f"Loaded {len(original)} keys")

    print(f"\nLoading {args.name2}...")
    new = torch.load(args.new, map_location='cpu', weights_only=False)
    print(f"Loaded {len(new)} keys")

    # Compare
    compare_state_dicts(original, new, args.name1, args.name2)


if __name__ == "__main__":
    main()
