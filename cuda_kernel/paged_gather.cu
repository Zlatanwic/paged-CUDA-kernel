/*
 * Paged KV Cache Gather Kernel (V1 - Naive)
 *
 * KV cache layout: [num_blocks, num_heads, block_size, head_dim]
 * Block table:     [num_seqs, max_blocks_per_seq]
 * Output K/V:      [num_seqs, num_heads, max_context_len, head_dim]
 */

#include <cuda_runtime.h>

// Naive gather kernel: each thread copies one element
__global__ void paged_gather_kernel(
    const float* __restrict__ kv_cache,
    float* __restrict__ output,
    const int* __restrict__ block_table,
    const int* __restrict__ context_lengths,
    int num_heads,
    int block_size,
    int head_dim,
    int max_blocks_per_seq,
    int max_ctx_len
) {
    int seq_id = blockIdx.x;
    int head_id = blockIdx.y;
    int ctx_len = context_lengths[seq_id];
    int total_elements = ctx_len * head_dim;

    for (int idx = threadIdx.x; idx < total_elements; idx += blockDim.x) {
        int token_pos = idx / head_dim;
        int d = idx % head_dim;

        int logical_block = token_pos / block_size;
        int offset_in_block = token_pos % block_size;

        int physical_block = block_table[seq_id * max_blocks_per_seq + logical_block];

        int cache_idx = physical_block * (num_heads * block_size * head_dim)
                      + head_id * (block_size * head_dim)
                      + offset_in_block * head_dim
                      + d;
        float val = kv_cache[cache_idx];

        int out_idx = seq_id * (num_heads * max_ctx_len * head_dim)
                    + head_id * (max_ctx_len * head_dim)
                    + token_pos * head_dim
                    + d;
        output[out_idx] = val;
    }
}

// C interface for the kernel launch
extern "C" void launch_paged_gather(
    const float* kv_cache,
    float* output,
    const int* block_table,
    const int* context_lengths,
    int num_seqs,
    int num_heads,
    int block_size,
    int head_dim,
    int max_blocks_per_seq,
    int max_ctx_len
) {
    dim3 grid(num_seqs, num_heads);
    int threads = 256;
    paged_gather_kernel<<<grid, threads>>>(
        kv_cache, output, block_table, context_lengths,
        num_heads, block_size, head_dim, max_blocks_per_seq, max_ctx_len
    );
}
