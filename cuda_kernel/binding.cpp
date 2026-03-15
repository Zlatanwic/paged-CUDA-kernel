/*
 * PyTorch binding for paged gather CUDA kernel.
 */

#include <torch/extension.h>
#include <vector>

// Declared in paged_gather.cu
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
);

std::vector<torch::Tensor> paged_gather(
    torch::Tensor k_cache,          // [num_blocks, num_heads, block_size, head_dim]
    torch::Tensor v_cache,          // [num_blocks, num_heads, block_size, head_dim]
    torch::Tensor block_table,      // [num_seqs, max_blocks_per_seq]
    torch::Tensor context_lengths,  // [num_seqs]
    int max_ctx_len
) {
    int num_seqs = block_table.size(0);
    int max_blocks_per_seq = block_table.size(1);
    int num_heads = k_cache.size(1);
    int block_size = k_cache.size(2);
    int head_dim = k_cache.size(3);

    auto options = k_cache.options();
    auto k_out = torch::zeros({num_seqs, num_heads, max_ctx_len, head_dim}, options);
    auto v_out = torch::zeros({num_seqs, num_heads, max_ctx_len, head_dim}, options);

    auto block_table_cuda = block_table.to(k_cache.device()).to(torch::kInt32).contiguous();
    auto ctx_lens_cuda = context_lengths.to(k_cache.device()).to(torch::kInt32).contiguous();

    launch_paged_gather(
        k_cache.data_ptr<float>(),
        k_out.data_ptr<float>(),
        block_table_cuda.data_ptr<int>(),
        ctx_lens_cuda.data_ptr<int>(),
        num_seqs, num_heads, block_size, head_dim, max_blocks_per_seq, max_ctx_len
    );

    launch_paged_gather(
        v_cache.data_ptr<float>(),
        v_out.data_ptr<float>(),
        block_table_cuda.data_ptr<int>(),
        ctx_lens_cuda.data_ptr<int>(),
        num_seqs, num_heads, block_size, head_dim, max_blocks_per_seq, max_ctx_len
    );

    return {k_out, v_out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("paged_gather", &paged_gather, "Gather K/V from paged cache into contiguous tensors");
}
