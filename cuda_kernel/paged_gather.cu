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

/*
 * Naive Attention Kernel
 *
 * Computes single-query attention on contiguous (already gathered) K/V.
 * query:  [num_seqs, num_heads, head_dim]
 * k:      [num_seqs, num_heads, max_ctx_len, head_dim]
 * v:      [num_seqs, num_heads, max_ctx_len, head_dim]
 * output: [num_seqs, num_heads, head_dim]
 *
 * Each CUDA block handles one (seq, head) pair.
 * Step 1: compute scores = q . k[t] * scale for all t
 * Step 2: softmax over scores
 * Step 3: output = sum(scores[t] * v[t])
 */
__global__ void naive_attention_kernel(
    const float* __restrict__ query,   // [num_seqs, num_heads, head_dim]
    const float* __restrict__ k,       // [num_seqs, num_heads, max_ctx_len, head_dim]
    const float* __restrict__ v,       // [num_seqs, num_heads, max_ctx_len, head_dim]
    float* __restrict__ output,        // [num_seqs, num_heads, head_dim]
    const int* __restrict__ context_lengths,
    int num_heads,
    int head_dim,
    int max_ctx_len,
    float scale
) {
    int seq_id = blockIdx.x;
    int head_id = blockIdx.y;
    int tid = threadIdx.x;
    int ctx_len = context_lengths[seq_id];

    // Pointers for this (seq, head)
    const float* q_ptr = query + seq_id * (num_heads * head_dim)
                               + head_id * head_dim;
    const float* k_ptr = k + seq_id * (num_heads * max_ctx_len * head_dim)
                           + head_id * (max_ctx_len * head_dim);
    const float* v_ptr = v + seq_id * (num_heads * max_ctx_len * head_dim)
                           + head_id * (max_ctx_len * head_dim);
    float* out_ptr = output + seq_id * (num_heads * head_dim)
                            + head_id * head_dim;

    // --- Step 1: Compute attention scores ---
    // Use shared memory for scores (one per token position)
    extern __shared__ float shared[];
    float* scores = shared;  // [ctx_len]

    for (int t = tid; t < ctx_len; t += blockDim.x) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += q_ptr[d] * k_ptr[t * head_dim + d];
        }
        scores[t] = dot * scale;
    }
    __syncthreads();

    // --- Step 2: Softmax ---
    // Find max (parallel reduction)
    float local_max = -1e20f;
    for (int t = tid; t < ctx_len; t += blockDim.x) {
        local_max = fmaxf(local_max, scores[t]);
    }
    // Warp-level reduction for max
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    // Block-level reduction: use first element of each warp
    __shared__ float warp_maxes[32];  // max 32 warps per block (1024 threads)
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) warp_maxes[warp_id] = local_max;
    __syncthreads();
    if (tid == 0) {
        float block_max = -1e20f;
        int num_warps = (blockDim.x + 31) / 32;
        for (int w = 0; w < num_warps; w++) {
            block_max = fmaxf(block_max, warp_maxes[w]);
        }
        warp_maxes[0] = block_max;  // store global max in warp_maxes[0]
    }
    __syncthreads();
    float global_max = warp_maxes[0];

    // Compute exp and sum
    float local_sum = 0.0f;
    for (int t = tid; t < ctx_len; t += blockDim.x) {
        scores[t] = expf(scores[t] - global_max);
        local_sum += scores[t];
    }
    // Warp-level reduction for sum
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    __shared__ float warp_sums[32];
    if (lane_id == 0) warp_sums[warp_id] = local_sum;
    __syncthreads();
    if (tid == 0) {
        float block_sum = 0.0f;
        int num_warps = (blockDim.x + 31) / 32;
        for (int w = 0; w < num_warps; w++) {
            block_sum += warp_sums[w];
        }
        warp_sums[0] = block_sum;
    }
    __syncthreads();
    float global_sum = warp_sums[0];

    // Normalize scores
    for (int t = tid; t < ctx_len; t += blockDim.x) {
        scores[t] /= global_sum;
    }
    __syncthreads();

    // --- Step 3: Weighted sum of V ---
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < ctx_len; t++) {
            acc += scores[t] * v_ptr[t * head_dim + d];
        }
        out_ptr[d] = acc;
    }
}

// C interface for attention kernel launch
extern "C" void launch_naive_attention(
    const float* query,
    const float* k,
    const float* v,
    float* output,
    const int* context_lengths,
    int num_seqs,
    int num_heads,
    int head_dim,
    int max_ctx_len,
    float scale
) {
    dim3 grid(num_seqs, num_heads);
    int threads = 256;
    // Shared memory: scores array sized to max_ctx_len
    int shared_mem_size = max_ctx_len * sizeof(float);
    naive_attention_kernel<<<grid, threads, shared_mem_size>>>(
        query, k, v, output, context_lengths,
        num_heads, head_dim, max_ctx_len, scale
    );
}

// C interface for the gather kernel launch
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
