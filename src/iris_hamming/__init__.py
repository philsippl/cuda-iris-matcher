"""Backward-compatible wrapper for the renamed package.

The project was renamed from the Python module `iris_hamming` to `cuda_iris_matcher`.
New code should `import cuda_iris_matcher as ih`.
"""

from cuda_iris_matcher import masked_hamming_cuda, pack_theta_major

__all__ = ["masked_hamming_cuda", "pack_theta_major"]
