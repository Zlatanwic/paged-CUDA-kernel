"""Test V2-B (Layout B) and V3 (shared memory K), then benchmark all versions."""

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


def make_test_data(num_seqs, ctx_len, block_size, num_heads, head_dim, frag=0.0, seed=42):
    torch.manual_seed(seed)
    num_blocks = (ctx_len // block_size + 1) * num_seqs + 16
    config = PagedKVCacheConfig(
        num_blocks=num_blocks, block_size=block_size, num_heads=num_heads,
        head_dim=head_dim, num_seqs=num_seqs, max_context_len=ctx_len, device="cuda",
    )
    k_cache, v_cache, query, _, _ = generate_paged_kv_cache(config)
    context_lengths = torch.full((num_seqs,), ctx_len, dtype=torch.int32)
    max_blocks_per_seq = (ctx_len + block_size - 1) // block_size
    block_table = generate_block_table(
        num_seqs, max_blocks_per_seq, num_blocks, context_lengths, block_size,
        fragmentation=frag,
    )
    return k_cache, v_cache, query, block_table, context_lengths


def convert_layout_b(k_cache, v_cache):
    """Convert Layout A [B, H, T, D] to Layout B [B, T, H, D]."""
    return k_cache.permute(0, 2, 1, 3).contiguous(), v_cache.permute(0, 2, 1, 3).contiguous()


# --- Correctness tests ---

def test_correctness(name, fn, num_seqs, ctx_len, block_size, num_heads, head_dim, frag=0.0):
    k_cache, v_cache, query, block_table, ctx_lens = make_test_data(
        num_seqs, ctx_len, block_size, num_heads, head_dim, frag)
    ref = paged_attention_naive(query, k_cache, v_cache, block_table, ctx_lens)
    out = fn(query, k_cache, v_cache, block_table, ctx_lens)
    max_diff = (out - ref).abs().max().item()
    atol = 1e-3 if ctx_len > 1024 else 1e-4
    ok = max_diff < atol
    tag = "OK" if ok else "FAIL"
    print(f"  [{tag}] {name}: ctx={ctx_len}, bs={block_size}, frag={frag}  (max_diff={max_diff:.2e})")
    return ok


def test_v2b_correctness(num_seqs, ctx_len, block_size, num_heads, head_dim, frag=0.0):
    k_a, v_a, query, block_table, ctx_lens = make_test_data(
        num_seqs, ctx_len, block_size, num_heads, head_dim, frag)
    ref = paged_attention_naive(query, k_a, v_a, block_table, ctx_lens)
    k_b, v_b = convert_layout_b(k_a, v_a)
    out = cuda_module.paged_attention_v2_layout_b(query, k_b, v_b, block_table, ctx_lens)
    max_diff = (out - ref).abs().max().item()
    atol = 1e-3 if ctx_len > 1024 else 1e-4
    ok = max_diff < atol
    tag = "OK" if ok else "FAIL"
    print(f"  [{tag}] V2-B: ctx={ctx_len}, bs={block_size}, frag={frag}  (max_diff={max_diff:.2e})")
    return ok


print("=== V3 Correctness ===")
all_ok = True
for ctx, bs, frag in [(32,16,0), (48,16,0), (64,16,1.0), (1024,16,0), (4096,16,0), (4096,32,0), (4096,64,0)]:
    all_ok &= test_correctness("V3", cuda_module.paged_attention_v3, 1, ctx, bs, 16, 128, frag)

print("\n=== V2-B (Layout B) Correctness ===")
for ctx, bs, frag in [(32,16,0), (48,16,0), (64,16,1.0), (1024,16,0), (4096,16,0), (4096,32,0)]:
    all_ok &= test_v2b_correctness(1, ctx, bs, 16, 128, frag)

if not all_ok:
    print("\nSome tests FAILED!")
    sys.exit(1)
print("\nAll correctness tests passed!")


# --- Benchmarks ---

def bench(fn, *args, warmup=5, runs=20):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn(*args)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs * 1000


print("\n" + "="*70)
print("BENCHMARK: All versions, Layout A vs B, block sizes")
print("="*70)

configs = [
    # (ctx_len, block_size)
    (1024, 16),
    (4096, 16),
    (4096, 32),
    (4096, 64),
    (16384, 16),
    (16384, 32),
]

num_heads = 16
head_dim = 128

for ctx_len, block_size in configs:
    k_a, v_a, query, bt, ctx_lens = make_test_data(1, ctx_len, block_size, num_heads, head_dim)
    k_b, v_b = convert_layout_b(k_a, v_a)

    ms_pytorch = bench(paged_attention_naive, query, k_a, v_a, bt, ctx_lens)
    ms_v1 = bench(cuda_module.paged_attention_v1, query, k_a, v_a, bt, ctx_lens, ctx_len)
    ms_v2 = bench(cuda_module.paged_attention_v2, query, k_a, v_a, bt, ctx_lens)
    ms_v2b = bench(cuda_module.paged_attention_v2_layout_b, query, k_b, v_b, bt, ctx_lens)
    ms_v3 = bench(cuda_module.paged_attention_v3, query, k_a, v_a, bt, ctx_lens)

    print(f"\n  ctx={ctx_len}, bs={block_size}, h={num_heads}, d={head_dim}")
    print(f"    PyTorch:       {ms_pytorch:.3f} ms")
    print(f"    V1 (gather):   {ms_v1:.3f} ms  ({ms_pytorch/ms_v1:.2f}x)")
    print(f"    V2 (fused A):  {ms_v2:.3f} ms  ({ms_pytorch/ms_v2:.2f}x)")
    print(f"    V2-B (fused B):{ms_v2b:.3f} ms  ({ms_pytorch/ms_v2b:.2f}x)")
    print(f"    V3 (smem K):   {ms_v3:.3f} ms  ({ms_pytorch/ms_v3:.2f}x)")
