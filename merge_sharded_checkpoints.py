import argparse
import torch
from torch.distributed.tensor import DTensor
from torch.distributed._tensor import Shard, Replicate


def main():
    parser = argparse.ArgumentParser(
        description="Merge sharded checkpoints from distributed training into a single checkpoint"
    )
    parser.add_argument(
        "--rank-0-path",
        type=str,
        required=True,
        help="Path to the checkpoint file for rank 0"
    )
    parser.add_argument(
        "--rank-1-path",
        type=str,
        required=True,
        help="Path to the checkpoint file for rank 1"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path where the merged checkpoint will be saved"
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=True,
        help="Convert all tensors to bfloat16 (BF16) format (default: True)"
    )
    parser.add_argument(
        "--no-bf16",
        dest="bf16",
        action="store_false",
        help="Disable BF16 conversion (keep original dtype)"
    )
    
    args = parser.parse_args()
    
    # Load both checkpoints
    print("Loading checkpoints...")
    model_0 = torch.load(args.rank_0_path, map_location="cpu", weights_only=False)
    model_1 = torch.load(args.rank_1_path, map_location="cpu", weights_only=False)
    
    print(f"Checkpoint 0 keys: {len(model_0)}")
    print(f"Checkpoint 1 keys: {len(model_1)}")
    
    # Check if keys are the same
    keys_same = set(model_0.keys()) == set(model_1.keys())
    print(f"Keys are the same: {keys_same}")
    
    if not keys_same:
        print("ERROR: Different keys found!")
        print("Only in rank 0:", set(model_0.keys()) - set(model_1.keys()))
        print("Only in rank 1:", set(model_1.keys()) - set(model_0.keys()))
        exit(1)
    
    # Convert to normal state_dict by merging shards
    merged_state_dict = {}
    num_sharded = 0
    num_replicated = 0
    num_regular = 0
    errors = []
    
    for key in model_0.keys():
        v0 = model_0[key]
        v1 = model_1[key]
    
        if isinstance(v0, DTensor):
            placement = v0.placements[0]  # Assuming single placement
            global_shape = v0.size()
    
            # Get local tensors
            local_0 = v0.to_local()
            local_1 = v1.to_local()
    
            if placement.is_shard():
                # Concatenate along the sharded dimension
                shard_dim = placement.dim
                merged_tensor = torch.cat([local_0, local_1], dim=shard_dim).contiguous()
    
                # Verify the merged shape matches the global shape
                if merged_tensor.shape != global_shape:
                    error_msg = f"ERROR: {key} shape mismatch! Expected {global_shape}, got {merged_tensor.shape}"
                    print(error_msg)
                    errors.append(error_msg)
                else:
                    # Convert to BF16 if requested
                    if args.bf16 and merged_tensor.dtype.is_floating_point:
                        merged_tensor = merged_tensor.to(torch.bfloat16)
                    merged_state_dict[key] = merged_tensor
                    num_sharded += 1
                    if num_sharded <= 3:  # Only print first 3 to avoid clutter
                        print(f"✓ Merged {key}: {local_0.shape} + {local_1.shape} -> {merged_tensor.shape}")
    
            elif placement.is_replicate():
                # Both should be the same, verify it
                if not torch.allclose(local_0, local_1, rtol=1e-5, atol=1e-8):
                    error_msg = f"WARNING: {key} is replicated but values differ!"
                    print(error_msg)
                    errors.append(error_msg)
                # Convert to BF16 if requested
                result_tensor = local_0
                if args.bf16 and result_tensor.dtype.is_floating_point:
                    result_tensor = result_tensor.to(torch.bfloat16)
                merged_state_dict[key] = result_tensor
                num_replicated += 1
                if num_replicated <= 3:
                    print(f"✓ Replicated {key}: {local_0.shape}")
            else:
                print(f"WARNING: Unknown placement for {key}: {placement}")
                result_tensor = local_0
                if args.bf16 and isinstance(result_tensor, torch.Tensor) and result_tensor.dtype.is_floating_point:
                    result_tensor = result_tensor.to(torch.bfloat16)
                merged_state_dict[key] = result_tensor
        else:
            # Not a DTensor, just use rank 0
            result_value = v0
            if args.bf16 and isinstance(result_value, torch.Tensor) and result_value.dtype.is_floating_point:
                result_value = result_value.to(torch.bfloat16)
            merged_state_dict[key] = result_value
            num_regular += 1
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Sharded tensors merged: {num_sharded}")
    print(f"  Replicated tensors: {num_replicated}")
    print(f"  Regular (non-DTensor) values: {num_regular}")
    print(f"  Total keys: {len(merged_state_dict)}")
    if args.bf16:
        print(f"  Format: BF16 (bfloat16)")
    
    if errors:
        print(f"\n⚠ Warnings/Errors found: {len(errors)}")
        for error in errors[:5]:  # Show first 5 errors
            print(f"  {error}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")
    
    # Save the merged state_dict
    torch.save(merged_state_dict, args.output_path)
    print(f"\n✓ Merged state_dict saved to: {args.output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

