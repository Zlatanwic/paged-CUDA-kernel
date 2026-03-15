"""
PyTorch baseline for paged KV cache attention.

Provides data generation and naive paged attention implementation
as correctness reference for CUDA kernels.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class PagedKVCacheConfig:
    num_blocks: int        # total physical blocks in the cache
    block_size: int        # tokens per block
    num_heads: int         # number of attention heads
    head_dim: int          # dimension per head
    num_seqs: int          # number of sequences in the batch
    max_context_len: int   # maximum context length across sequences
    dtype: torch.dtype = torch.float32
    device: str = "cuda"


def generate_block_table(
    num_seqs: int,
    max_blocks_per_seq: int,
    num_blocks: int,
    context_lengths: torch.Tensor,
    block_size: int,
    fragmentation: float = 0.0,
) -> torch.Tensor:
    """
    Generate a block table mapping logical blocks to physical blocks.

    Args:
        fragmentation: 0.0 = contiguous allocation, 1.0 = fully random.
    """
    block_table = torch.full(
        (num_seqs, max_blocks_per_seq), -1, dtype=torch.int32
    )

    if fragmentation == 0.0:
        # Contiguous allocation: seq0 gets blocks 0,1,2,..., seq1 continues
        offset = 0
        for seq_id in range(num_seqs):
            num_tokens = context_lengths[seq_id].item()
            num_needed = (num_tokens + block_size - 1) // block_size
            for i in range(num_needed):
                block_table[seq_id, i] = offset + i
            offset += num_needed
    else:
        # Allocate then shuffle based on fragmentation level
        all_block_ids = list(range(num_blocks))

        # First do contiguous allocation to know which blocks each seq needs
        assignments = []
        offset = 0
        for seq_id in range(num_seqs):
            num_tokens = context_lengths[seq_id].item()
            num_needed = (num_tokens + block_size - 1) // block_size
            assignments.append(num_needed)
            offset += num_needed

        total_needed = sum(assignments)
        pool = all_block_ids[:total_needed]

        # Shuffle a fraction of the pool based on fragmentation
        import random
        num_to_shuffle = int(len(pool) * fragmentation)
        if num_to_shuffle > 1:
            indices_to_shuffle = random.sample(range(len(pool)), num_to_shuffle)
            values = [pool[i] for i in indices_to_shuffle]
            random.shuffle(values)
            for idx, val in zip(indices_to_shuffle, values):
                pool[idx] = val

        # Assign from shuffled pool
        pos = 0
        for seq_id in range(num_seqs):
            for i in range(assignments[seq_id]):
                block_table[seq_id, i] = pool[pos]
                pos += 1

    return block_table


def generate_paged_kv_cache(config: PagedKVCacheConfig):
    """
    Generate random paged KV cache data for testing.

    Returns:
        k_cache: [num_blocks, num_heads, block_size, head_dim]
        v_cache: [num_blocks, num_heads, block_size, head_dim]
        query:   [num_seqs, num_heads, head_dim]
        block_table: [num_seqs, max_blocks_per_seq]
        context_lengths: [num_seqs]
    """
    k_cache = torch.randn(
        config.num_blocks, config.num_heads, config.block_size, config.head_dim,
        dtype=config.dtype, device=config.device
    )
    v_cache = torch.randn(
        config.num_blocks, config.num_heads, config.block_size, config.head_dim,
        dtype=config.dtype, device=config.device
    )
    query = torch.randn(
        config.num_seqs, config.num_heads, config.head_dim,
        dtype=config.dtype, device=config.device
    )

    # Generate random context lengths up to max_context_len
    # Ensure at least 1 token per sequence
    context_lengths = torch.randint(
        1, config.max_context_len + 1, (config.num_seqs,),
        dtype=torch.int32
    )

    max_blocks_per_seq = (config.max_context_len + config.block_size - 1) // config.block_size

    block_table = generate_block_table(
        num_seqs=config.num_seqs,
        max_blocks_per_seq=max_blocks_per_seq,
        num_blocks=config.num_blocks,
        context_lengths=context_lengths,
        block_size=config.block_size,
        fragmentation=0.0,
    )

    return k_cache, v_cache, query, block_table, context_lengths


def paged_attention_naive(
    query: torch.Tensor,         # [num_seqs, num_heads, head_dim]
    k_cache: torch.Tensor,       # [num_blocks, num_heads, block_size, head_dim]
    v_cache: torch.Tensor,       # [num_blocks, num_heads, block_size, head_dim]
    block_table: torch.Tensor,   # [num_seqs, max_blocks_per_seq]
    context_lengths: torch.Tensor,  # [num_seqs]
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Naive paged attention: gather K/V from paged cache, then compute attention.

    For each sequence:
      1. Use block_table to gather K/V blocks into contiguous tensors
      2. Compute scaled dot-product attention: softmax(Q @ K^T / sqrt(d)) @ V

    Returns:
        output: [num_seqs, num_heads, head_dim]
    """
    num_seqs, num_heads, head_dim = query.shape
    block_size = k_cache.shape[2]

    if scale is None:
        scale = head_dim ** -0.5

    outputs = []

    for seq_id in range(num_seqs):
        ctx_len = context_lengths[seq_id].item()
        num_blocks_needed = (ctx_len + block_size - 1) // block_size

        # Gather K and V blocks into contiguous tensors
        k_blocks = []
        v_blocks = []
        for block_idx in range(num_blocks_needed):
            physical_block_id = block_table[seq_id, block_idx].item()
            k_blocks.append(k_cache[physical_block_id])  # [num_heads, block_size, head_dim]
            v_blocks.append(v_cache[physical_block_id])

        # Concatenate along the token dimension
        # k_gathered: [num_heads, total_tokens_in_blocks, head_dim]
        k_gathered = torch.cat(k_blocks, dim=1)
        v_gathered = torch.cat(v_blocks, dim=1)

        # Trim to actual context length (handle partial last block)
        k_gathered = k_gathered[:, :ctx_len, :]  # [num_heads, ctx_len, head_dim]
        v_gathered = v_gathered[:, :ctx_len, :]

        # Query for this sequence: [num_heads, head_dim]
        q = query[seq_id]  # [num_heads, head_dim]
        q = q.unsqueeze(1)  # [num_heads, 1, head_dim]

        # Attention scores: [num_heads, 1, ctx_len]
        scores = torch.matmul(q, k_gathered.transpose(-2, -1)) * scale

        # Softmax over context dimension
        attn_weights = F.softmax(scores, dim=-1)

        # Weighted sum of values: [num_heads, 1, head_dim]
        out = torch.matmul(attn_weights, v_gathered)
        out = out.squeeze(1)  # [num_heads, head_dim]

        outputs.append(out)

    # Stack: [num_seqs, num_heads, head_dim]
    return torch.stack(outputs, dim=0)
