"""Quick test to verify CUDA compilation toolchain works."""

import os
import sys

# Add MSVC to PATH for torch cpp_extension
msvc_bin = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"
if msvc_bin not in os.environ["PATH"]:
    os.environ["PATH"] = msvc_bin + ";" + os.environ["PATH"]

# Also need MSVC include and lib paths
msvc_root = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207"
os.environ.setdefault("INCLUDE", os.path.join(msvc_root, "include"))
os.environ.setdefault("LIB", os.path.join(msvc_root, r"lib\x64"))

import torch
from torch.utils.cpp_extension import load_inline

cuda_src = """
__global__ void add_one_kernel(float* x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] += 1.0f;
    }
}

torch::Tensor add_one(torch::Tensor x) {
    int n = x.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_one_kernel<<<blocks, threads>>>(x.data_ptr<float>(), n);
    return x;
}
"""

cpp_src = "torch::Tensor add_one(torch::Tensor x);"

module = load_inline(
    name="test_cuda",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["add_one"],
    extra_cuda_cflags=["-Xcompiler", "/Zc:preprocessor"],
    verbose=True,
)

x = torch.zeros(10, device="cuda")
result = module.add_one(x)
print(f"Input was zeros, after add_one: {result}")
assert torch.allclose(result, torch.ones(10, device="cuda"))
print("CUDA compilation toolchain works!")
