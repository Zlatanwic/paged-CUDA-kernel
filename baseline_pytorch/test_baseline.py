"""
Test the PyTorch baseline paged attention implementation.
"""

import torch
import time
from model import PagedKVCacheConfig, generate_paged_kv_cache, generate_block_table, paged_attention_naive


def test_single_block():
    """Test with context fitting in exactly one block."""
    print("=== Test: single block ===")
    config = PagedKVCacheConfig(
        num_blocks=4,
        block_size=16,
        num_heads=4,
        head_dim=64,
        num_seqs=1,
        max_context_len=16,
    )
    k_cache, v_cache, query, block_table, _ = generate_paged_kv_cache(config)
    context_lengths = torch.tensor([16], dtype=torch.int32)
    block_table = generate_block_table(1, 1, 4, context_lengths, 16, fragmentation=0.0)

    output = paged_attention_naive(query, k_cache, v_cache, block_table, context_lengths)
    assert output.shape == (1, 4, 64), f"Expected (1,4,64), got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"
    print(f"  Output shape: {output.shape} OK")
    print(f"  No NaN: OK")


def test_partial_block():
    """Test with context that doesn't fill the last block."""
    print("\n=== Test: partial block ===")
    config = PagedKVCacheConfig(
        num_blocks=4,
        block_size=16,
        num_heads=4,
        head_dim=64,
        num_seqs=1,
        max_context_len=20,
    )
    k_cache, v_cache, query, _, _ = generate_paged_kv_cache(config)
    context_lengths = torch.tensor([20], dtype=torch.int32)
    block_table = generate_block_table(1, 2, 4, context_lengths, 16, fragmentation=0.0)

    output = paged_attention_naive(query, k_cache, v_cache, block_table, context_lengths)
    assert output.shape == (1, 4, 64), f"Expected (1,4,64), got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"
    print(f"  Output shape: {output.shape} OK")
    print(f"  Context length 20, block_size 16 -> 2 blocks, last block partial OK")


def test_multi_seq():
    """Test with multiple sequences of different lengths."""
    print("\n=== Test: multi-sequence ===")
    config = PagedKVCacheConfig(
        num_blocks=20,
        block_size=16,
        num_heads=4,
        head_dim=64,
        num_seqs=3,
        max_context_len=64,
    )
    k_cache, v_cache, query, _, _ = generate_paged_kv_cache(config)
    context_lengths = torch.tensor([10, 32, 50], dtype=torch.int32)
    max_blocks_per_seq = (64 + 16 - 1) // 16  # = 4
    block_table = generate_block_table(3, max_blocks_per_seq, 20, context_lengths, 16, fragmentation=0.0)

    output = paged_attention_naive(query, k_cache, v_cache, block_table, context_lengths)
    assert output.shape == (3, 4, 64), f"Expected (3,4,64), got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"
    print(f"  Output shape: {output.shape} OK")
    print(f"  Context lengths: [10, 32, 50] OK")


def test_fragmentation():
    """Test that fragmented block table runs correctly."""
    print("\n=== Test: fragmentation consistency ===")
    config = PagedKVCacheConfig(
        num_blocks=32,
        block_size=16,
        num_heads=4,
        head_dim=64,
        num_seqs=2,
        max_context_len=64,
    )
    k_cache, v_cache, query, _, _ = generate_paged_kv_cache(config)
    context_lengths = torch.tensor([48, 32], dtype=torch.int32)
    max_blocks_per_seq = 4

    bt_contig = generate_block_table(2, max_blocks_per_seq, 32, context_lengths, 16, fragmentation=0.0)
    out_contig = paged_attention_naive(query, k_cache, v_cache, bt_contig, context_lengths)

    # Same block table should give same result
    out_contig2 = paged_attention_naive(query, k_cache, v_cache, bt_contig, context_lengths)
    assert torch.allclose(out_contig, out_contig2, atol=1e-6), "Same input gives different output"
    print(f"  Deterministic output: OK")

    # Fragmented block table (different physical blocks -> different data -> different result, but no crash)
    bt_frag = generate_block_table(2, max_blocks_per_seq, 32, context_lengths, 16, fragmentation=1.0)
    out_frag = paged_attention_naive(query, k_cache, v_cache, bt_frag, context_lengths)
    assert out_frag.shape == out_contig.shape, "Shape mismatch"
    assert not torch.isnan(out_frag).any(), "Fragmented output contains NaN"
    print(f"  Fragmented block table runs without error: OK")


def test_attention_correctness():
    """Verify against manual attention computation for a simple case."""
    print("\n=== Test: attention correctness ===")
    torch.manual_seed(42)

    num_heads = 2
    head_dim = 4
    block_size = 4
    ctx_len = 4  # exactly one block

    k_cache = torch.randn(1, num_heads, block_size, head_dim)
    v_cache = torch.randn(1, num_heads, block_size, head_dim)
    query = torch.randn(1, num_heads, head_dim)
    block_table = torch.tensor([[0]], dtype=torch.int32)
    context_lengths = torch.tensor([ctx_len], dtype=torch.int32)

    # Our implementation
    output = paged_attention_naive(query, k_cache, v_cache, block_table, context_lengths)

    # Manual computation
    scale = head_dim ** -0.5
    q = query[0].unsqueeze(1)  # [num_heads, 1, head_dim]
    k = k_cache[0]  # [num_heads, block_size, head_dim]
    v = v_cache[0]
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    weights = torch.softmax(scores, dim=-1)
    expected = torch.matmul(weights, v).squeeze(1)

    assert torch.allclose(output[0], expected, atol=1e-6), \
        f"Mismatch:\n  output={output[0]}\n  expected={expected}"
    print(f"  Manual verification: OK")


def test_timing():
    """Basic timing for the naive implementation."""
    print("\n=== Timing: naive PyTorch paged attention ===")
    config = PagedKVCacheConfig(
        num_blocks=1024,
        block_size=16,
        num_heads=16,
        head_dim=128,
        num_seqs=1,
        max_context_len=4096,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    k_cache, v_cache, query, _, _ = generate_paged_kv_cache(config)
    context_lengths = torch.tensor([4096], dtype=torch.int32)
    max_blocks_per_seq = 4096 // 16
    block_table = generate_block_table(
        1, max_blocks_per_seq, 1024, context_lengths, 16, fragmentation=0.0
    )

    k_cache = k_cache.to(config.device)
    v_cache = v_cache.to(config.device)
    query = query.to(config.device)

    # Warmup
    for _ in range(3):
        paged_attention_naive(query, k_cache, v_cache, block_table, context_lengths)

    if config.device == "cuda":
        torch.cuda.synchronize()

    # Timed runs
    num_runs = 10
    start = time.perf_counter()
    for _ in range(num_runs):
        paged_attention_naive(query, k_cache, v_cache, block_table, context_lengths)
        if config.device == "cuda":
            torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / num_runs * 1000

    print(f"  Config: ctx_len=4096, block_size=16, heads=16, head_dim=128")
    print(f"  Device: {config.device}")
    print(f"  Avg latency: {elapsed:.3f} ms")


if __name__ == "__main__":
    test_single_block()
    test_partial_block()
    test_multi_seq()
    test_fragmentation()
    test_attention_correctness()
    test_timing()
    print("\nAll tests passed!")
