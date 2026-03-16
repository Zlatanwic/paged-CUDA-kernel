# Profiling & Benchmark Results

## Hardware

- GPU: NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM, Blackwell, CC 12.0)
- DRAM Bandwidth (theoretical peak): ~352 GB/s
- CUDA 13.2, Nsight Compute 2026.1.0

## Kernel Versions

| Version | Description | Layout |
|---------|-------------|--------|
| PyTorch | Naive baseline: gather via block_table + `torch` scaled dot-product | A |
| V1 | CUDA gather kernel → separate naive attention kernel (two-step) | A |
| V2 | Fused gather + attention, online softmax, query in shared memory | A |
| V2-B | Same as V2, but Layout B | B |
| V3 | Fused + cooperatively load K block into shared memory before scoring | A |

**Layout A:** `[num_blocks, num_heads, block_size, head_dim]`
**Layout B:** `[num_blocks, block_size, num_heads, head_dim]`

---

## 1. Benchmark: All Versions (num_seqs=1, num_heads=16, head_dim=128)

| ctx_len | block_size | PyTorch (ms) | V1 (ms) | V2 (ms) | V2-B (ms) | V3 (ms) |
|---------|-----------|-------------|---------|---------|-----------|---------|
| 1024 | 16 | 0.438 | 0.408 | 0.119 | 0.127 | 0.183 |
| 4096 | 16 | 1.700 | 2.398 | 0.568 | 0.568 | 1.181 |
| 4096 | 32 | 1.197 | 2.384 | 0.586 | 0.553 | 1.079 |
| 4096 | 64 | 0.851 | 2.414 | 0.551 | 0.578 | 1.078 |
| 16384 | 16 | 6.448 | 7.629 | 2.129 | 2.120 | 4.377 |
| 16384 | 32 | 4.307 | 7.661 | 2.128 | 2.058 | 4.196 |

### Observations

- **V2 is the fastest across all configs**, achieving 2-3.7x speedup over PyTorch baseline.
- **V1 is slower than PyTorch** for long contexts — the two-step approach (gather into
  contiguous buffer → attention) creates extra memory traffic that outweighs any kernel
  efficiency gain.
- **V2 vs V2-B (Layout A vs B):** Within 5% of each other. Layout difference has minimal
  impact at these configurations, likely because the access pattern is already
  head-partitioned (one CUDA block per (seq, head) pair).
- **V3 is ~2x slower than V2.** Shared memory K loading adds overhead without reducing
  total DRAM traffic (see Nsight analysis below).

---

## 2. Nsight Compute Profiling: V2 vs V3

### Run 1: Small Grid (num_seqs=1, grid=(1,16)=16 blocks)

| Metric | V2 | V3 |
|--------|----|----|
| Duration | 760 μs | 1530 μs |
| DRAM Throughput | 25.78% | 12.81% |
| Memory Throughput | 90.65 GB/s | 45.05 GB/s |
| Compute Throughput | 9.13% | 6.24% |
| Achieved Occupancy | 16.67% | 16.67% |
| L1 Hit Rate | 47.04% | 46.91% |
| L2 Hit Rate | 35.54% | 35.16% |
| Block Limit (Shared Mem) | 17 | 7 |

**Analysis:** Both kernels achieve only 16.67% occupancy because the grid is too small
(16 blocks, "only 0.10 full waves"). This is a workload sizing issue, not a kernel
efficiency issue. Neither kernel can saturate the GPU.

### Run 2: Large Grid (num_seqs=32, grid=(32,16)=512 blocks)

| Metric | V2 | V3 |
|--------|----|----|
| **Duration** | **6.49 ms** | **7.49 ms** |
| DRAM Throughput | **94.19%** | 74.78% |
| Memory Throughput | 331 GB/s | 287 GB/s |
| Compute Throughput | 34.24% | 40.89% |
| **Achieved Occupancy** | **93.82%** | **91.50%** |
| L1 Hit Rate | 5.47% | 20.57% |
| L2 Hit Rate | 49.62% | 46.33% |
| Block Limit (Shared Mem) | 17 | 7 |
| Block Limit (Registers) | 6 | 6 |
| Block Limit (Warps) | 6 | 6 |

**Analysis:**

1. **V2 is memory-bound and nearly saturates DRAM bandwidth (94.19%).** This is close to
   the hardware ceiling (~352 GB/s theoretical → 331 GB/s achieved). There is little room
   for further optimization without reducing total memory traffic.

2. **V3's shared memory K loading does not reduce DRAM traffic.** In V2, each warp reads K
   tokens independently from global memory — each K element is read exactly once per warp.
   V3 loads K into shared memory first, but since there is no cross-warp reuse of the same K
   data within a block (each warp processes different tokens), this is just an extra copy step
   with no traffic reduction.

3. **V3's higher L1 hit rate (20.57% vs 5.47%)** is a side effect of shared memory load
   traffic hitting L1, not a genuine cache efficiency gain. The total DRAM bytes transferred
   are similar or higher due to the shared memory write overhead.

4. **Occupancy is similar (93.8% vs 91.5%).** Although V3's shared memory usage limits it to
   7 blocks/SM (vs V2's 17), the actual binding constraint for both kernels is registers and
   warps (6 blocks/SM limit). The shared memory pressure in V3 does not materially reduce
   occupancy in this configuration.

5. **V3 has higher compute throughput (40.89% vs 34.24%)** because the shared memory reads
   in the score computation are faster than global memory reads. However, this gain is offset
   by the cooperative load overhead (`__syncthreads()` barriers + shared memory writes), and
   the kernel is memory-bound overall, so faster compute does not translate to faster
   execution.

---

## 3. Key Takeaways

1. **Fusing gather + attention (V2) is the single biggest win** — eliminates the intermediate
   contiguous buffer and its associated DRAM traffic. V2 achieves 2-3.7x over PyTorch.

2. **Layout A vs B has negligible impact** when each CUDA block handles one (seq, head) pair.
   The memory access pattern is already well-structured regardless of layout.

3. **Shared memory caching of K (V3) is counterproductive** in this kernel design because
   there is no cross-warp data reuse — it adds synchronization cost without reducing DRAM
   traffic. Shared memory optimization would be beneficial in a design where multiple warps
   or thread blocks share the same K data (e.g., multi-query attention / grouped-query
   attention where K/V are shared across heads).

4. **V2 at 94% DRAM throughput is near hardware limit.** Further speedups require either:
   - Reducing total memory traffic (e.g., quantized KV cache, FP16)
   - Algorithmic improvements (e.g., block-sparse attention)
   - Multi-query attention (which would make V3-style shared memory actually useful)

---

## 4. Next Experiments (TODO)

- [ ] **Fragmentation impact:** Contiguous vs 50% fragmented vs fully random block placement,
  measuring latency degradation and L2 hit rate changes. Directly links to KV eviction
  research — "how does eviction-induced layout fragmentation affect GPU decode performance?"
- [ ] **FP16 / BF16 kernels:** Halve memory traffic, potentially 2x speedup on memory-bound V2.
- [ ] **Larger context lengths:** 32k, 64k — test scaling behavior.
- [ ] **Multi-sequence batching:** Realistic serving scenario with mixed context lengths.
