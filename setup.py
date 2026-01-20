"""
Setup script for cuda-iris-matcher.

Note: CUDA extension is compiled at runtime via JIT compilation.
No CUDAExtension needed at install time - sources are included as package data.
"""
from setuptools import setup

# No CUDAExtension - sources are compiled at runtime via JIT
setup()
