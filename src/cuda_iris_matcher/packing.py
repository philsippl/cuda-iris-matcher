from __future__ import annotations

import numpy as np


def pack_theta_major(bits_u8: np.ndarray) -> np.ndarray:
    """
    Pack IrisCode-like bits into int32 word vectors in *theta-major* order.

    Expected input shape: (M, 16, 200, 2, 2) with values in {0,1}.

    Output: int32 array of shape (M, 400) representing 12800 bits.

    Theta-major means theta (axis=2 in the input) is the fastest-varying major axis
    in the packed bitstream, so that np.roll(..., axis=2) by 1 corresponds to a
    circular rotation by 64 bits (= 2 uint32 words) in the packed representation.
    """
    if bits_u8.ndim != 5 or bits_u8.shape[1:] != (16, 200, 2, 2):
        raise ValueError(f"expected shape (M,16,200,2,2); got {bits_u8.shape}")

    m = bits_u8.shape[0]
    # (M, 16, 200, 2, 2) -> (M, 200, 16, 2, 2) -> (M, 12800)
    b = np.transpose(bits_u8, (0, 2, 1, 3, 4)).reshape(m, -1).astype(np.uint8)

    # Pack bits (little-endian bit order within each byte)
    packed_bytes = np.packbits(b, axis=1, bitorder="little")  # (M, 1600)
    # View as little-endian uint32 words: (M, 400)
    words_u32 = packed_bytes.view("<u4")
    # Reinterpret as signed int32 (PyTorch int32 compatible)
    return words_u32.view("<i4")



