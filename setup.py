from setuptools import setup, find_packages

import os

from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_ext_modules():
    # Build the extension as cuda_iris_matcher._C
    sources = [
        "iris_binding.cu",
        "iris.cu",
        "dot_product.cu",
    ]

    extra_cuda_cflags = ["-O3", "-std=c++17"]
    extra_cflags = ["-O3", "-std=c++17"]

    # Support FORCE_FALLBACK=1 to build with scalar fallback instead of tensor cores
    if os.environ.get("FORCE_FALLBACK", "0") == "1":
        extra_cuda_cflags.append("-DFORCE_FALLBACK")
        print("Building with FORCE_FALLBACK: using scalar __popc + warp shuffles")

    return [
        CUDAExtension(
            name="cuda_iris_matcher._C",
            sources=sources,
            extra_compile_args={"cxx": extra_cflags, "nvcc": extra_cuda_cflags},
        )
    ]


setup(
    name="cuda-iris-matcher",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    ext_modules=get_ext_modules(),
    cmdclass={"build_ext": BuildExtension},
)


