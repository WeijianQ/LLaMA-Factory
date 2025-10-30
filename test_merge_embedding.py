#!/usr/bin/env python3
"""
Sanity check test for embedding merging logic.

Test flow:
1. Load Qwen2_5_MemoryForCausalLMForFreezeTraining model
2. Randomly initialize special_embed_tokens
3. Extract state_dict and merge special embeddings into normal embed_tokens
4. Initialize Qwen2_5_MemoryForCausalLM with merged weights
5. Forward both models with same input
6. Assert logits are identical
"""

import pickle
import torch
from transformers import AutoTokenizer
from src.llamafactory.hf_memory_qwen25.modeling_qwen2_5_memory import (
    Qwen2_5_MemoryForCausalLM,
    Qwen2_5_MemoryForCausalLMForFreezeTraining
)
from src.llamafactory.hf_memory_qwen25.configuration_qwen2_5_memory import Qwen2_5_MemoryConfig
from copy import deepcopy

def convert_state_dict_from_freeze_to_normal(freeze_state_dict, config):
    """
    Convert state dict from freeze training model to normal model.
    """
    # copy
    normal_state_dict = deepcopy(freeze_state_dict)

    # Pop special layers (not needed in normal model)
    special_embed_tokens_weight = normal_state_dict.pop('special_embed_tokens.weight')
    normal_state_dict.pop('special_lm_head.weight', None)  # Tied to special_embed_tokens
    normal_state_dict.pop('id2override', None)  # Only used in freeze model

    # Pop lm_head.weight so it will be tied to embed_tokens.weight
    normal_state_dict.pop('lm_head.weight')

    # Merge special embeddings into embed_tokens
    current_embed_tokens_weight = normal_state_dict['model.embed_tokens.weight']

    for idx, token_id in enumerate(config.special_token_ids):
        current_embed_tokens_weight[token_id] = special_embed_tokens_weight[idx]
        print(f"Merged special token {token_id} into embed_tokens.weight")
    print(f"Merged {len(config.special_token_ids)} special tokens into embed_tokens.weight")

    return normal_state_dict

def main():
    # Configuration
    model_name = "WeijianQi1999/Qwen25-1p5B-Memory"
    pkl_path = "/local/scratch/qi.658/LLaMA-Factory/batch_sample_webshop_val_keep_action_with_cm_proxy_tasks_only.pkl"

    print("=" * 80)
    print("Sanity Check: Embedding Merge Test")
    print("=" * 80)

    # Load config
    print(f"\nLoading config from {model_name}...")
    config = Qwen2_5_MemoryConfig.from_pretrained(model_name, trust_remote_code=True)

    # Step 1: Load freeze training model
    print(f"\n[Step 1] Loading Qwen2_5_MemoryForCausalLMForFreezeTraining...")
    freeze_model = Qwen2_5_MemoryForCausalLMForFreezeTraining.from_pretrained(
        model_name,
        config=config,
        device_map="cpu",  # Use CPU for this test
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )

    # Re-establish weight tying (from_pretrained may have broken it)
    freeze_model.special_lm_head.weight = freeze_model.special_embed_tokens.weight
    print(f"  Re-established weight tying: special_lm_head.weight = special_embed_tokens.weight")

    freeze_model.eval()

    # Step 2: Randomly initialize special_embed_tokens
    print(f"\n[Step 2] Randomly initializing special_embed_tokens...")
    torch.manual_seed(42)
    with torch.no_grad():
        freeze_model.special_embed_tokens.weight.normal_(mean=0.0, std=0.02)
    print(f"  Initialized shape: {freeze_model.special_embed_tokens.weight.shape}")
    print(f"  Weight stats: mean={freeze_model.special_embed_tokens.weight.mean().item():.6f}, "
          f"std={freeze_model.special_embed_tokens.weight.std().item():.6f}")

    # Step 3: Extract state_dict and convert (pure state_dict operations)
    print(f"\n[Step 3] Extracting state_dict and converting to normal model format...")
    print(f"  (All operations on state_dict, no modification to original freeze_model)")
    freeze_state_dict_original = freeze_model.state_dict()

    # Create a merged freeze state dict (for freeze_model to use in forward)
    # This merges special_embed_tokens into embed_tokens and lm_head
    freeze_state_dict = deepcopy(freeze_state_dict_original)

    # Convert to normal model format
    merged_state_dict = convert_state_dict_from_freeze_to_normal(freeze_state_dict, config)

    # Assert lm_head.weight is NOT in merged_state_dict (it should be tied)
    assert 'lm_head.weight' not in merged_state_dict, "ERROR: lm_head.weight should not be in merged state_dict!"
    print(f"  ✓ Verified: lm_head.weight is NOT in state_dict (will be tied to embed_tokens.weight)")

    # Step 4: Initialize normal model with merged weights
    print(f"\n[Step 4] Loading Qwen2_5_MemoryForCausalLM with merged weights...")
    normal_model = Qwen2_5_MemoryForCausalLM.from_pretrained(
        model_name,
        config=config,
        device_map="cpu",
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )

    # Load merged state dict
    missing_keys, unexpected_keys = normal_model.load_state_dict(merged_state_dict, strict=False)
    print(f"  Missing keys: {missing_keys}")
    print(f"  Unexpected keys: {unexpected_keys}")
    normal_model.eval()

    # Verify weight tying
    print(f"\n  Verifying weight tying...")
    is_tied = normal_model.lm_head.weight is normal_model.model.embed_tokens.weight
    print(f"    lm_head.weight is embed_tokens.weight: {is_tied}")
    assert is_tied, "ERROR: lm_head and embed_tokens should be tied!"
    print(f"    ✓ Verified: Weight tying is correctly established")

    # Verify embeddings were merged correctly
    print(f"\n  Verifying merged embeddings...")
    special_token_ids = torch.tensor(config.special_token_ids)
    for idx, token_id in enumerate(special_token_ids):
        freeze_emb = freeze_model.special_embed_tokens.weight[idx]
        normal_emb = normal_model.model.embed_tokens.weight[token_id]
        diff = (freeze_emb - normal_emb).abs().max().item()
        print(f"    Token {token_id}: max diff = {diff:.6e}")
        assert diff < 1e-6, f"ERROR: Embeddings don't match for token {token_id}!"
    print(f"    ✓ Verified: All special token embeddings merged correctly")

    # Step 5: Load sample data
    print(f"\n[Step 5] Loading sample data from {pkl_path}...")
    with open(pkl_path, "rb") as f:
        sample_data = pickle.load(f)

    # Take first sample only
    input_ids = sample_data["input_ids"][:1].to(torch.long)  # (1, seq_len)
    attention_mask = sample_data["attention_mask"][:1].to(torch.long)
    # For this test, we disable memories to isolate embedding differences
    memory_input_ids = None
    memory_attention_mask = None

    print(f"  Input shapes:")
    print(f"    input_ids: {input_ids.shape}")
    print(f"    attention_mask: {attention_mask.shape}")
    if memory_input_ids is not None:
        print(f"    memory_input_ids: {memory_input_ids.shape}")
    if memory_attention_mask is not None:
        print(f"    memory_attention_mask: {memory_attention_mask.shape}")

    # Check for special tokens in input
    special_token_ids = torch.tensor(config.special_token_ids)
    special_token_mask = torch.isin(input_ids, special_token_ids)
    num_special_tokens = special_token_mask.sum().item()
    print(f"  Number of special tokens in input: {num_special_tokens}")


    # Step 6: do nn parameter check
    # normal embed_token should be the same as freeze embed_token with special tokens merged
    print(f"\n[Step 6] Verifying parameter consistency...")
    normal_embed_tokens_weight = normal_model.model.embed_tokens.weight
    freeze_embed_tokens_weight = freeze_model.model.embed_tokens.weight
    freeze_special_embed_tokens_weight = freeze_model.special_embed_tokens.weight
    special_token_ids_map = {token_id.item(): idx for idx, token_id in enumerate(special_token_ids)}

    for token_id in range(config.vocab_size):
        if token_id in special_token_ids_map:
            # For special tokens, compare with special_embed_tokens
            expected = freeze_special_embed_tokens_weight[special_token_ids_map[token_id]]
            actual = normal_embed_tokens_weight[token_id]
            assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6), \
                f"Embedding mismatch for special token {token_id}"
        else:
            # For normal tokens, compare with normal embed_tokens
            expected = freeze_embed_tokens_weight[token_id]
            actual = normal_embed_tokens_weight[token_id]
            assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6), \
                f"Embedding mismatch for token {token_id}"
    print(f"  ✓ Verified: All embed_tokens weights match")

    normal_lm_head_weight = normal_model.lm_head.weight
    freeze_lm_head_weight = freeze_model.lm_head.weight
    freeze_special_lm_head_weight = freeze_model.special_lm_head.weight

    # assert freeze_special_lm_head_weight is the same as freeze_special_embed_tokens_weight
    assert torch.allclose(freeze_special_lm_head_weight, freeze_special_embed_tokens_weight, rtol=1e-5, atol=1e-6), \
        "Special lm_head weight is not the same as special embed_tokens weight"
    print(f"  ✓ Verified: Special lm_head weight is the same as special embed_tokens weight")

    # assert freeze_lm_head_weight is the same as freeze_embed_tokens_weight
    assert torch.allclose(freeze_lm_head_weight, freeze_embed_tokens_weight, rtol=1e-5, atol=1e-6), \
        "LM head weight is not the same as embed_tokens weight"
    print(f"  ✓ Verified: LM head weight is the same as embed_tokens weight")

    for token_id in range(config.vocab_size):
        if token_id in special_token_ids_map:
            # For special tokens, compare with special_lm_head
            expected = freeze_special_lm_head_weight[special_token_ids_map[token_id]]
            actual = normal_lm_head_weight[token_id]
            assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6), \
                f"LM head mismatch for special token {token_id}"
        else:
            # For normal tokens, compare with normal lm_head
            expected = freeze_lm_head_weight[token_id]
            actual = normal_lm_head_weight[token_id]
            assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6), \
                f"LM head mismatch for token {token_id}"
    print(f"  ✓ Verified: All lm_head weights match")

    # Step 6: Forward both models
    print(f"\n[Step 6] Running forward pass on both models...")



    with torch.no_grad():
        # Forward freeze model
        print("  Forward freeze_model...")
        freeze_outputs = freeze_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            memory_input_ids=memory_input_ids,
            memory_attention_mask=memory_attention_mask,
        )
        freeze_logits = freeze_outputs.logits

        # Forward normal model
        print("  Forward normal_model...")
        normal_outputs = normal_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            memory_input_ids=memory_input_ids,
            memory_attention_mask=memory_attention_mask,
        )
        normal_logits = normal_outputs.logits

    print(f"  Freeze model logits shape: {freeze_logits.shape}")
    print(f"  Normal model logits shape: {normal_logits.shape}")

    # Step 7: Compare logits
    print(f"\n[Step 7] Comparing logits...")

    # Compute differences
    abs_diff = (freeze_logits - normal_logits).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()

    print(f"  Max absolute difference: {max_diff:.6e}")
    print(f"  Mean absolute difference: {mean_diff:.6e}")

    # Check if logits are close
    tolerance = 1e-5
    are_close = torch.allclose(freeze_logits, normal_logits, rtol=1e-5, atol=tolerance)

    print(f"\n{'='*80}")
    if are_close:
        print("✅ PASS: Logits are identical (within tolerance)!")
        print(f"   Tolerance: {tolerance:.6e}")
    else:
        print("❌ FAIL: Logits differ!")
        print(f"   Max diff: {max_diff:.6e} > tolerance {tolerance:.6e}")

        # Print top differences
        print(f"\n  Top 5 differences:")
        flat_diff = abs_diff.view(-1)
        top_k = min(5, flat_diff.numel())
        top_diffs, top_indices = torch.topk(flat_diff, top_k)
        for i, (diff, idx) in enumerate(zip(top_diffs, top_indices)):
            pos = idx.item()
            batch_idx = pos // (freeze_logits.shape[1] * freeze_logits.shape[2])
            seq_idx = (pos % (freeze_logits.shape[1] * freeze_logits.shape[2])) // freeze_logits.shape[2]
            vocab_idx = pos % freeze_logits.shape[2]
            print(f"    {i+1}. Position [batch={batch_idx}, seq={seq_idx}, vocab={vocab_idx}]: "
                  f"diff={diff.item():.6e}, "
                  f"freeze={freeze_logits[batch_idx, seq_idx, vocab_idx].item():.6f}, "
                  f"normal={normal_logits[batch_idx, seq_idx, vocab_idx].item():.6f}")

    print("=" * 80)


if __name__ == "__main__":
    main()
