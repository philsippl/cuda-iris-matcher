from __future__ import annotations

from typing import Tuple

import torch

from . import _C


def masked_hamming_cuda(
    data: torch.Tensor,
    mask: torch.Tensor,
    write_output: bool = False,
    collect_pairs: bool = False,
    threshold: float = 1.0,
    max_pairs: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute minimum fractional hamming distance for all pairs within a single set.
    Only the lower triangle (i > j) is computed.

    data/mask: CUDA int32 contiguous tensors of shape [M, 400].

    Returns:
      - D: [M, M] float32 (or an empty tensor if write_output=False)
      - pairs: [max_pairs, 2] int32 (or an empty tensor if collect_pairs=False)
      - match_count: [1] int32 (or an empty tensor if collect_pairs=False)
    """
    return _C.masked_hamming_cuda(data, mask, write_output, collect_pairs, threshold, max_pairs)


def masked_hamming_ab_cuda(
    data_a: torch.Tensor,
    mask_a: torch.Tensor,
    data_b: torch.Tensor,
    mask_b: torch.Tensor,
    write_output: bool = False,
    collect_pairs: bool = False,
    threshold: float = 1.0,
    max_pairs: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute minimum fractional hamming distance between two different sets A and B.
    Computes the full M_A x M_B matrix (not just lower triangle).

    data_a/mask_a: CUDA int32 contiguous tensors of shape [M_A, 400].
    data_b/mask_b: CUDA int32 contiguous tensors of shape [M_B, 400].

    Returns:
      - D: [M_A, M_B] float32 (or an empty tensor if write_output=False)
      - pairs: [max_pairs, 2] int32 (or an empty tensor if collect_pairs=False)
      - match_count: [1] int32 (or an empty tensor if collect_pairs=False)
    """
    return _C.masked_hamming_ab_cuda(
        data_a, mask_a, data_b, mask_b, write_output, collect_pairs, threshold, max_pairs
    )


def pack_theta_major_cuda(bits: torch.Tensor) -> torch.Tensor:
    """
    Pack iris code bits into theta-major int32 words using CUDA.

    Args:
        bits: CUDA uint8 tensor of shape (M, 16, 200, 2, 2) with values in {0, 1}.
              Modified in-place; do not use after this call.

    Returns:
        Packed int32 tensor of shape (M, 400). Shares storage with input
        (no additional memory allocation).
    """
    return _C.pack_theta_major_cuda(bits)


def repack_to_theta_major_cuda(input: torch.Tensor) -> torch.Tensor:
    """
    Repack int32 words from r-major to theta-major order using CUDA.

    Args:
        input: CUDA int32 tensor of shape (M, 400) packed in r-major order.
               Original layout: bit[r,theta,d0,d1] at linear_bit = r*800 + theta*4 + d0*2 + d1

    Returns:
        New CUDA int32 tensor of shape (M, 400) packed in theta-major order.
        Output layout: bit[r,theta,d0,d1] at linear_bit = theta*64 + r*4 + d0*2 + d1
    """
    return _C.repack_to_theta_major_cuda(input)

