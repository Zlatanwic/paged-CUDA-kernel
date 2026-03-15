"""
Build helper for CUDA kernels using torch.utils.cpp_extension.load_inline.
"""

import os

# Ensure MSVC is in PATH
msvc_bin = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"
if os.path.exists(msvc_bin) and msvc_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = msvc_bin + ";" + os.environ["PATH"]

from torch.utils.cpp_extension import load


def load_paged_gather():
    """Load the paged gather CUDA kernel."""
    kernel_dir = os.path.dirname(os.path.abspath(__file__))
    return load(
        name="paged_gather",
        sources=[
            os.path.join(kernel_dir, "binding.cpp"),
            os.path.join(kernel_dir, "paged_gather.cu"),
        ],
        extra_cuda_cflags=["-Xcompiler", "/Zc:preprocessor"],
        verbose=True,
    )
