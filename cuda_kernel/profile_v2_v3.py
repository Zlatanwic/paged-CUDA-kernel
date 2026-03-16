"""Minimal script for ncu profiling: runs V2 and V3 once each.
   Uses larger batch to saturate GPU SMs.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "baseline_pytorch"))

import torch
from model import PagedKVCacheConfig, generate_paged_kv_cache, generate_block_table
from build import load_paged_gather

cuda_module = load_paged_gather()

# Config: batch=32 to fill GPU, ctx=4096, bs=16, h=16, d=128
ctx_len, block_size, num_heads, head_dim = 4096, 16, 16, 128
num_seqs = 32
num_blocks = (ctx_len // block_size + 1) * num_seqs + 64

torch.manual_seed(42)
config = PagedKVCacheConfig(
    num_blocks=num_blocks, block_size=block_size, num_heads=num_heads,
    head_dim=head_dim, num_seqs=num_seqs, max_context_len=ctx_len, device="cuda",
)
k_cache, v_cache, query, _, _ = generate_paged_kv_cache(config)
context_lengths = torch.full((num_seqs,), ctx_len, dtype=torch.int32)
max_blocks_per_seq = (ctx_len + block_size - 1) // block_size
block_table = generate_block_table(
    num_seqs, max_blocks_per_seq, num_blocks, context_lengths, block_size,
    fragmentation=0.0,
)

# Warmup
for _ in range(3):
    cuda_module.paged_attention_v2(query, k_cache, v_cache, block_table, context_lengths)
    cuda_module.paged_attention_v3(query, k_cache, v_cache, block_table, context_lengths)
torch.cuda.synchronize()

# Profiled runs
cuda_module.paged_attention_v2(query, k_cache, v_cache, block_table, context_lengths)
torch.cuda.synchronize()

cuda_module.paged_attention_v3(query, k_cache, v_cache, block_table, context_lengths)
torch.cuda.synchronize()

print(f"Done. grid=({num_seqs}, {num_heads}) = {num_seqs * num_heads} blocks launched.")
