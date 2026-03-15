"""Test end-to-end V1 (gather + naive attention) against PyTorch baseline."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "baseline_pytorch"))

import torch
import time
from model import (
    PagedKVCacheConfig, generate_paged_kv_cache,
    generate_block_table, paged_attention_naive,
)
from build import load_paged_gather

print("Compiling CUDA kernels...")
cuda_module = load_paged_gather()
print("Done.\n")


def test_v1_correctness(num_seqs, ctx_len, block_size, num_heads, head_dim, frag=0.0):
    """Compare CUDA V1 output with PyTorch baseline."""
    torch.manual_seed(42)
    num_blocks = (ctx_len // block_size + 1) * num_seqs + 16
    config = PagedKVCacheConfig(
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_dim=head_dim,
        num_seqs=num_seqs,
        max_context_len=ctx_len,
        device="cuda",
    )
    k_cache, v_cache, query, _, _ = generate_paged_kv_cache(config)
    context_lengths = torch.full((num_seqs,), ctx_len, dtype=torch.int32)
    max_blocks_per_seq = (ctx_len + block_size - 1) // block_size
    block_table = generate_block_table(
        num_seqs, max_blocks_per_seq, num_blocks, context_lengths, block_size,
        fragmentation=frag,
    )

    # PyTorch baseline
    ref = paged_attention_naive(query, k_cache, v_cache, block_table, context_lengths)

    # CUDA V1
    out = cuda_module.paged_attention_v1(
        query, k_cache, v_cache, block_table, context_lengths, ctx_len
    )

    max_diff = (out - ref).abs().max().item()
    # float32 accumulation over ctx_len tokens can have meaningful error
    atol = 1e-3 if ctx_len > 1024 else 1e-4
    ok = max_diff < atol
    tag = "OK" if ok else "FAIL"
    desc = f"seqs={num_seqs}, ctx={ctx_len}, bs={block_size}, h={num_heads}, d={head_dim}, frag={frag}"
    print(f"  [{tag}] {desc}  (max_diff={max_diff:.2e})")
    return ok


def test_v1_timing():
    """Benchmark V1 vs PyTorch baseline."""
    print("\n=== Timing: V1 vs PyTorch baseline ===")
    torch.manual_seed(0)

    ctx_len = 4096
    block_size = 16
    num_heads = 16
    head_dim = 128
    num_seqs = 1
    num_blocks = (ctx_len // block_size) * num_seqs + 16

    config = PagedKVCacheConfig(
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_dim=head_dim,
        num_seqs=num_seqs,
        max_context_len=ctx_len,
        device="cuda",
    )
    k_cache, v_cache, query, _, _ = generate_paged_kv_cache(config)
    context_lengths = torch.full((num_seqs,), ctx_len, dtype=torch.int32)
    max_blocks_per_seq = ctx_len // block_size
    block_table = generate_block_table(
        num_seqs, max_blocks_per_seq, num_blocks, context_lengths, block_size,
    )

    # Warmup
    for _ in range(5):
        paged_attention_naive(query, k_cache, v_cache, block_table, context_lengths)
        cuda_module.paged_attention_v1(
            query, k_cache, v_cache, block_table, context_lengths, ctx_len
        )
    torch.cuda.synchronize()

    # Benchmark PyTorch baseline
    num_runs = 20
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        paged_attention_naive(query, k_cache, v_cache, block_table, context_lengths)
        torch.cuda.synchronize()
    pytorch_ms = (time.perf_counter() - start) / num_runs * 1000

    # Benchmark CUDA V1
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        cuda_module.paged_attention_v1(
            query, k_cache, v_cache, block_table, context_lengths, ctx_len
        )
        torch.cuda.synchronize()
    cuda_v1_ms = (time.perf_counter() - start) / num_runs * 1000

    print(f"  Config: ctx_len={ctx_len}, block_size={block_size}, heads={num_heads}, head_dim={head_dim}")
    print(f"  PyTorch baseline: {pytorch_ms:.3f} ms")
    print(f"  CUDA V1:          {cuda_v1_ms:.3f} ms")
    print(f"  Speedup:          {pytorch_ms / cuda_v1_ms:.2f}x")


if __name__ == "__main__":
    print("=== V1 Correctness Tests ===")
    all_ok = True
    # Small cases
    all_ok &= test_v1_correctness(1, 32, 16, 4, 64)
    all_ok &= test_v1_correctness(1, 48, 16, 4, 64)  # partial block
    all_ok &= test_v1_correctness(2, 64, 16, 4, 64)   # multi-seq
    all_ok &= test_v1_correctness(1, 64, 16, 4, 64, frag=1.0)  # fragmented
    # Larger, more realistic
    all_ok &= test_v1_correctness(1, 1024, 16, 16, 128)
    all_ok &= test_v1_correctness(1, 4096, 16, 16, 128)
    all_ok &= test_v1_correctness(1, 4096, 32, 16, 128)

    if all_ok:
        print("\nAll correctness tests passed!")
    else:
        print("\nSome tests FAILED!")
        sys.exit(1)

    test_v1_timing()
