"""
JIT compilation of CUDA kernels.

This module compiles CUDA extensions at runtime using torch.utils.cpp_extension.load().
The compiled extension is cached in ~/.cache/torch_extensions/ and reused on subsequent imports.

This approach ensures ABI compatibility - the extension is always compiled against the
runtime PyTorch version, avoiding version mismatch issues.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# Global cache for the compiled module
_compiled_module: Any = None


def get_source_dir() -> Path:
    """Get the directory containing CUDA source files."""
    return Path(__file__).parent / "csrc"


def get_source_files() -> list[str]:
    """Get list of CUDA source files to compile."""
    source_dir = get_source_dir()
    sources = [
        source_dir / "iris_binding.cu",
        source_dir / "iris.cu",
        source_dir / "dot_product.cu",
    ]
    # Verify all sources exist
    for src in sources:
        if not src.exists():
            raise FileNotFoundError(
                f"CUDA source file not found: {src}\n"
                f"This may indicate an incomplete installation. "
                f"Try reinstalling cuda-iris-matcher."
            )
    return [str(s) for s in sources]


def get_extra_cuda_cflags() -> list[str]:
    """Get extra CUDA compiler flags."""
    flags = ["-O3", "-std=c++17"]

    # Support FORCE_FALLBACK=1 to use scalar fallback instead of tensor cores
    if os.environ.get("FORCE_FALLBACK", "0") == "1":
        flags.append("-DFORCE_FALLBACK")
        print(
            "cuda-iris-matcher: Building with FORCE_FALLBACK (scalar __popc + warp shuffles)"
        )

    return flags


def _is_cached() -> bool:
    """Check if the extension is already cached (heuristic)."""
    try:
        import torch

        cache_dir = Path.home() / ".cache" / "torch_extensions"
        if not cache_dir.exists():
            return False

        # Look for our extension in the cache
        # PyTorch organizes cache by python version / torch version
        for item in cache_dir.rglob("cuda_iris_matcher_ops*"):
            if item.is_dir() or item.suffix == ".so":
                return True
        return False
    except Exception:
        return False


def compile_extension(verbose: bool = False) -> Any:
    """
    Compile the CUDA extension using JIT compilation.

    This function is called on first import. The compiled module is cached
    by PyTorch in ~/.cache/torch_extensions/ and reused on subsequent imports.

    The cache key includes:
    - Python version
    - PyTorch version
    - CUDA version
    - Source file hashes

    If any of these change, PyTorch automatically recompiles.

    Args:
        verbose: If True, print compilation progress.

    Returns:
        The compiled extension module.
    """
    global _compiled_module

    if _compiled_module is not None:
        return _compiled_module

    # Import torch here to avoid import at module load time
    from torch.utils.cpp_extension import load

    sources = get_source_files()
    extra_cuda_cflags = get_extra_cuda_cflags()

    # Check verbosity from environment
    verbose = verbose or os.environ.get("CUDA_IRIS_MATCHER_VERBOSE", "0") == "1"

    if verbose or not _is_cached():
        print(
            "cuda-iris-matcher: Compiling CUDA extension (this may take 1-2 minutes on first run)..."
        )
        print("cuda-iris-matcher: Subsequent imports will be instant (cached).")

    try:
        _compiled_module = load(
            name="cuda_iris_matcher_ops",
            sources=sources,
            extra_cuda_cflags=extra_cuda_cflags,
            extra_cflags=["-O3", "-std=c++17"],
            verbose=verbose,
        )
    except Exception as e:
        print(
            f"cuda-iris-matcher: Failed to compile CUDA extension: {e}", file=sys.stderr
        )
        print(
            "cuda-iris-matcher: Make sure you have CUDA toolkit and a C++ compiler installed.",
            file=sys.stderr,
        )
        print(
            "cuda-iris-matcher: Required: nvcc (CUDA compiler), g++ (C++ compiler)",
            file=sys.stderr,
        )
        raise

    return _compiled_module


def get_ops() -> Any:
    """
    Get the compiled CUDA operations module.

    This is the main entry point. Call this to get access to the CUDA kernels.

    Returns:
        The compiled extension module with CUDA operations.

    Example:
        >>> ops = get_ops()
        >>> result = ops.masked_hamming_cuda(data, mask, ...)
    """
    return compile_extension()


# Alias for backward compatibility
_C = property(lambda self: get_ops())
