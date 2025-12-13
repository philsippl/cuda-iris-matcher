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
    data/mask: CUDA int32 contiguous tensors of shape [M, 400].

    Returns:
      - D: [M, M] float32 (or an empty tensor if write_output=False)
      - pairs: [max_pairs, 2] int32 (or an empty tensor if collect_pairs=False)
      - match_count: [1] int32 (or an empty tensor if collect_pairs=False)
    """
    return _C.masked_hamming_cuda(data, mask, write_output, collect_pairs, threshold, max_pairs)


def pack_theta_major_cuda(bits: torch.Tensor) -> torch.Tensor:
    """
    Pack iris code bits into theta-major int32 words using CUDA.

    Args:
        bits: CUDA uint8 tensor of shape (M, 16, 200, 2, 2) with values in {0, 1}.
              This tensor is modified IN-PLACE and should not be used after this call.

    Returns:
        Packed int32 tensor of shape (M, 400). Shares storage with the input tensor
        (no additional memory allocation).
    """
    return _C.pack_theta_major_cuda(bits)


