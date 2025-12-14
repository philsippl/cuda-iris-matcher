from __future__ import annotations

from typing import Optional, Tuple

import torch

from . import _C

# Export classification constants from C++ extension
CATEGORY_TRUE_MATCH = _C.CATEGORY_TRUE_MATCH
CATEGORY_FALSE_MATCH = _C.CATEGORY_FALSE_MATCH
CATEGORY_FALSE_NON_MATCH = _C.CATEGORY_FALSE_NON_MATCH
CATEGORY_TRUE_NON_MATCH = _C.CATEGORY_TRUE_NON_MATCH

INCLUDE_TM = _C.INCLUDE_TM
INCLUDE_FM = _C.INCLUDE_FM
INCLUDE_FNM = _C.INCLUDE_FNM
INCLUDE_TNM = _C.INCLUDE_TNM
INCLUDE_ALL = _C.INCLUDE_ALL


def masked_hamming_cuda(
    data: torch.Tensor,
    mask: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    match_threshold: float = 0.35,
    non_match_threshold: float = 0.35,
    is_similarity: bool = False,
    include_flags: int = INCLUDE_ALL,
    max_pairs: int = 1_000_000,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute minimum fractional hamming distance for all pairs within a single set.
    Only the lower triangle (i > j) is computed.

    Args:
        data: CUDA int32 contiguous tensor of shape [M, 400] (packed iris codes)
        mask: CUDA int32 contiguous tensor of shape [M, 400] (packed masks)
        labels: Optional CUDA int32 tensor of shape [M] with identity labels.
                If None, no classification is performed and all pairs are returned.
        match_threshold: Threshold for match classification (default: 0.35)
        non_match_threshold: Threshold for non-match classification (default: 0.35)
        is_similarity: If True, higher values = more similar (use >= for match).
                       If False (default), lower values = more similar (use <= for match).
        include_flags: Bitmask of categories to include (default: INCLUDE_ALL)
                       Use INCLUDE_TM | INCLUDE_FM | ... to combine flags.
        max_pairs: Maximum number of pairs to return (default: 1,000,000)

    Returns:
        Tuple of (pair_indices, categories, distances, count):
        - pair_indices: [N, 2] int32 - (row, col) indices of pairs
        - categories: [N] uint8 - category codes (0=TM, 1=FM, 2=FNM, 3=TNM, 255=unclassified)
        - distances: [N] float32 - distance values
        - count: [1] int32 - actual number of pairs (N == len(pair_indices))

    Note:
        The returned tensors are pre-sliced to contain only the valid entries.
        Synchronization is handled internally.
    """
    return _C.masked_hamming_cuda(
        data, mask, labels,
        match_threshold, non_match_threshold,
        is_similarity, include_flags, max_pairs
    )


def masked_hamming_ab_cuda(
    data_a: torch.Tensor,
    mask_a: torch.Tensor,
    data_b: torch.Tensor,
    mask_b: torch.Tensor,
    labels_a: Optional[torch.Tensor] = None,
    labels_b: Optional[torch.Tensor] = None,
    match_threshold: float = 0.35,
    non_match_threshold: float = 0.35,
    is_similarity: bool = False,
    include_flags: int = INCLUDE_ALL,
    max_pairs: int = 1_000_000,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute minimum fractional hamming distance between two different sets A and B.
    Computes the full M_A x M_B matrix (not just lower triangle).

    Args:
        data_a: CUDA int32 contiguous tensor of shape [M_A, 400] (packed iris codes)
        mask_a: CUDA int32 contiguous tensor of shape [M_A, 400] (packed masks)
        data_b: CUDA int32 contiguous tensor of shape [M_B, 400] (packed iris codes)
        mask_b: CUDA int32 contiguous tensor of shape [M_B, 400] (packed masks)
        labels_a: Optional CUDA int32 tensor of shape [M_A] with identity labels.
        labels_b: Optional CUDA int32 tensor of shape [M_B] with identity labels.
                  Both labels_a and labels_b must be provided, or neither.
                  If None, no classification is performed and all pairs are returned.
        match_threshold: Threshold for match classification (default: 0.35)
        non_match_threshold: Threshold for non-match classification (default: 0.35)
        is_similarity: If True, higher values = more similar (use >= for match).
                       If False (default), lower values = more similar (use <= for match).
        include_flags: Bitmask of categories to include (default: INCLUDE_ALL)
                       Use INCLUDE_TM | INCLUDE_FM | ... to combine flags.
        max_pairs: Maximum number of pairs to return (default: 1,000,000)

    Returns:
        Tuple of (pair_indices, categories, distances, count):
        - pair_indices: [N, 2] int32 - (row, col) indices of pairs
        - categories: [N] uint8 - category codes (0=TM, 1=FM, 2=FNM, 3=TNM, 255=unclassified)
        - distances: [N] float32 - distance values
        - count: [1] int32 - actual number of pairs (N == len(pair_indices))

    Note:
        The returned tensors are pre-sliced to contain only the valid entries.
        Synchronization is handled internally.
    """
    return _C.masked_hamming_ab_cuda(
        data_a, mask_a, data_b, mask_b,
        labels_a, labels_b,
        match_threshold, non_match_threshold,
        is_similarity, include_flags, max_pairs
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
