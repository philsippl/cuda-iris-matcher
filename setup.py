from setuptools import setup, find_packages

import os

from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_ext_modules():
    # Build the extension as cuda_iris_matcher._C
    sources = [
        "iris_binding.cu",
        "iris.cu",
    ]

    extra_cuda_cflags = ["-O3", "-std=c++17"]
    extra_cflags = ["-O3", "-std=c++17"]

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


