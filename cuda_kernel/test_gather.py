"""Test CUDA paged gather kernel against PyTorch baseline."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "baseline_pytorch"))

import torch
from model import PagedKVCacheConfig, generate_paged_kv_cache, generate_block_table
from build import load_paged_gather

print("Compiling CUDA kernel (first run may take a minute)...")
cuda_module = load_paged_gather()
print("Compilation done.\n")


def test_gather_correctness():
    """Verify CUDA gather matches PyTorch gather."""
    print("=== Test: gather correctness ===")
    torch.manual_seed(42)

    config = PagedKVCacheConfig(
        num_blocks=16,
        block_size=16,
        num_heads=4,
        head_dim=64,
        num_seqs=2,
        max_context_len=48,
        device="cuda",
    )
    k_cache, v_cache, query, _, _ = generate_paged_kv_cache(config)
    context_lengths = torch.tensor([48, 32], dtype=torch.int32)
    max_blocks_per_seq = (48 + 16 - 1) // 16
    block_table = generate_block_table(
        2, max_blocks_per_seq, 16, context_lengths, 16, fragmentation=0.0
    )

    max_ctx_len = context_lengths.max().item()

    # CUDA gather
    k_gathered_cuda, v_gathered_cuda = cuda_module.paged_gather(
        k_cache, v_cache, block_table, context_lengths, max_ctx_len
    )

    # PyTorch gather (manual)
    for seq_id in range(2):
        ctx_len = context_lengths[seq_id].item()
        num_blocks_needed = (ctx_len + 16 - 1) // 16

        k_blocks = []
        for block_idx in range(num_blocks_needed):
            phys_id = block_table[seq_id, block_idx].item()
            k_blocks.append(k_cache[phys_id])  # [num_heads, block_size, head_dim]
        k_ref = torch.cat(k_blocks, dim=1)[:, :ctx_len, :]  # [num_heads, ctx_len, head_dim]

        k_cuda_seq = k_gathered_cuda[seq_id, :, :ctx_len, :]

        if not torch.allclose(k_cuda_seq, k_ref, atol=1e-6):
            print(f"  FAILED: seq {seq_id} K mismatch")
            print(f"  max diff: {(k_cuda_seq - k_ref).abs().max().item()}")
            return
        print(f"  seq {seq_id}: K match OK")

    print("  All sequences match OK\n")


def test_gather_fragmented():
    """Test with fragmented block table."""
    print("=== Test: fragmented gather ===")
    torch.manual_seed(123)

    config = PagedKVCacheConfig(
        num_blocks=32,
        block_size=16,
        num_heads=4,
        head_dim=64,
        num_seqs=1,
        max_context_len=64,
        device="cuda",
    )
    k_cache, v_cache, query, _, _ = generate_paged_kv_cache(config)
    context_lengths = torch.tensor([64], dtype=torch.int32)
    max_blocks_per_seq = 4

    block_table = generate_block_table(
        1, max_blocks_per_seq, 32, context_lengths, 16, fragmentation=1.0
    )
    print(f"  Block table (fragmented): {block_table[0].tolist()}")

    k_gathered, v_gathered = cuda_module.paged_gather(
        k_cache, v_cache, block_table, context_lengths, 64
    )

    # Verify each block was gathered from the right physical location
    for block_idx in range(4):
        phys_id = block_table[0, block_idx].item()
        start = block_idx * 16
        end = start + 16
        k_ref = k_cache[phys_id]  # [num_heads, block_size, head_dim]
        k_got = k_gathered[0, :, start:end, :]
        assert torch.allclose(k_got, k_ref, atol=1e-6), f"Block {block_idx} mismatch"

    print("  Fragmented gather correct OK\n")


def test_gather_partial_block():
    """Test with partial last block."""
    print("=== Test: partial block gather ===")
    torch.manual_seed(456)

    config = PagedKVCacheConfig(
        num_blocks=8,
        block_size=16,
        num_heads=2,
        head_dim=32,
        num_seqs=1,
        max_context_len=20,
        device="cuda",
    )
    k_cache, v_cache, query, _, _ = generate_paged_kv_cache(config)
    context_lengths = torch.tensor([20], dtype=torch.int32)
    block_table = generate_block_table(1, 2, 8, context_lengths, 16, fragmentation=0.0)

    k_gathered, v_gathered = cuda_module.paged_gather(
        k_cache, v_cache, block_table, context_lengths, 20
    )

    # Check first block (full)
    phys0 = block_table[0, 0].item()
    assert torch.allclose(k_gathered[0, :, :16, :], k_cache[phys0], atol=1e-6)

    # Check second block (partial: only 4 tokens)
    phys1 = block_table[0, 1].item()
    assert torch.allclose(k_gathered[0, :, 16:20, :], k_cache[phys1][:, :4, :], atol=1e-6)

    print("  Partial block gather correct OK\n")


if __name__ == "__main__":
    test_gather_correctness()
    test_gather_fragmented()
    test_gather_partial_block()
    print("All gather tests passed!")
