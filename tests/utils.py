"""Utility functions for testing CUDA iris matcher."""

from typing import Optional, Tuple
import numpy as np


def normalized_HD(
    codebitcount: int, maskbitcount: int, norm_mean: float, norm_gradient: float
) -> float:
    """Compute the normalized Hamming distance using the project's linear approximation.

    This mirrors the open-iris normalization: max(0, norm_mean - (norm_mean - raw) * (norm_gradient * mask + 0.5)),
    where raw = codebitcount / maskbitcount and mask = maskbitcount.

    Args:
        codebitcount: Number of non-matching bits within the valid mask region.
        maskbitcount: Number of valid (unmasked) bits used for comparison. Must be > 0.
        norm_mean: Peak of the non-match distribution used as normalization anchor.
        norm_gradient: Linear coefficient controlling the normalization strength.

    Returns:
        Normalized Hamming distance in [0, 1].
    """
    norm_HD = max(
        0,
        norm_mean
        - (norm_mean - codebitcount / maskbitcount)
        * (norm_gradient * maskbitcount + 0.5),
    )
    return norm_HD


def _iter_shift_order(rotation_shift: int) -> list:
    """Return the rotation evaluation order used by the open-iris implementation.

    Order pattern: 0, -1, 1, -2, 2, ..., -rotation_shift, rotation_shift.

    Args:
        rotation_shift: Maximum absolute number of column shifts to evaluate.

    Returns:
        List of integer shifts in the exact order the reference scans.
    """
    return [0] + [y for x in range(1, rotation_shift + 1) for y in (-x, x)]


def _count_nonmatch_bits_for_shift(
    probe_code: np.ndarray,
    probe_mask: np.ndarray,
    gallery_code: np.ndarray,
    gallery_mask: np.ndarray,
    shift: int,
) -> Tuple[int, int]:
    """Count non-matching iris bits and common mask bits for a specific shift.

    Inputs are boolean arrays of identical shape (H, W, F, 2). For each filter f in [0, F):
      - probe and mask are circularly shifted along the width axis (axis=1)
      - non-match bits are computed as XOR ("!=") of probe and gallery iris bits
      - only bits where both masks are true contribute to the counts

    Args:
        probe_code: Probe code bits, shape (H, W, F, 2), dtype bool/0-1.
        probe_mask: Probe mask bits, shape (H, W, F, 2), dtype bool/0-1.
        gallery_code: Gallery code bits, shape (H, W, F, 2), dtype bool/0-1.
        gallery_mask: Gallery mask bits, shape (H, W, F, 2), dtype bool/0-1.
        shift: Circular column shift applied to probe arrays (negative shifts roll left).

    Returns:
        Tuple (nonmatch_code_bit_count, common_mask_bit_count) as integers aggregated over all dims.
    """
    total_code_nonmatch_bits = 0
    total_common_mask_bits = 0

    num_filters = probe_code.shape[2]
    for filter_index in range(num_filters):
        probe_code_f = probe_code[:, :, filter_index, :]
        probe_mask_f = probe_mask[:, :, filter_index, :]
        gallery_code_f = gallery_code[:, :, filter_index, :]
        gallery_mask_f = gallery_mask[:, :, filter_index, :]

        probe_code_rolled = np.roll(probe_code_f, shift, axis=1)
        probe_mask_rolled = np.roll(probe_mask_f, shift, axis=1)

        code_nonmatch_bits = probe_code_rolled != gallery_code_f
        common_mask_bits = probe_mask_rolled & gallery_mask_f

        total_code_nonmatch_bits += int(np.sum(code_nonmatch_bits & common_mask_bits))
        total_common_mask_bits += int(np.sum(common_mask_bits))

    return total_code_nonmatch_bits, total_common_mask_bits


def rotation_aware_hamming_distance(
    code_probe: np.ndarray,
    mask_probe: np.ndarray,
    code_gallery: np.ndarray,
    mask_gallery: np.ndarray,
    rotation_shift: int = 15,
    normalize: bool = False,
    norm_mean: float = 0.45,
    norm_gradient: float = 0.00005,
) -> Tuple[float, int]:
    """Compute the minimum Hamming distance between probe and gallery vector codes.

    Behavior is identical to open-iris:
      - evaluates rotations in the order 0, -1, 1, -2, 2, ... up to ±rotation_shift
      - per rotation, counts non-matching bits under the intersection of masks
      - returns the minimum distance and its corresponding shift

    Args:
        code_probe: Probe code bits, shape (H, W, F, 2), bool/0-1. Expected (16, 200, 2, 2).
        mask_probe: Probe mask bits, shape (H, W, F, 2), bool/0-1. Expected (16, 200, 2, 2).
        code_gallery: Gallery code bits, shape (H, W, F, 2), bool/0-1. Expected (16, 200, 2, 2).
        mask_gallery: Gallery mask bits, shape (H, W, F, 2), bool/0-1. Expected (16, 200, 2, 2).
        rotation_shift: Maximum absolute column shift to evaluate.
        normalize: If True, apply the normalized HD; otherwise use raw ratio.
        norm_mean: Normalization mean parameter (see normalized_HD).
        norm_gradient: Normalization gradient parameter (see normalized_HD).

    Returns:
        Tuple (best_distance, best_shift):
          - best_distance: float in [0, 1]
          - best_shift: integer in [-rotation_shift, rotation_shift], first minimum by scan order
    """
    if code_probe.shape != code_gallery.shape:
        raise ValueError("probe and gallery vectors are of different sizes")

    # Ensure boolean computation
    code_probe = code_probe.astype(bool, copy=False)
    mask_probe = mask_probe.astype(bool, copy=False)
    code_gallery = code_gallery.astype(bool, copy=False)
    mask_gallery = mask_gallery.astype(bool, copy=False)

    best_dist = 1.0
    rot_shift = 0

    for current_shift in _iter_shift_order(rotation_shift):
        totalcodebitcount, totalmaskbitcount = _count_nonmatch_bits_for_shift(
            code_probe, mask_probe, code_gallery, mask_gallery, current_shift
        )

        if totalmaskbitcount == 0:
            continue

        current_dist = (
            normalized_HD(
                totalcodebitcount, totalmaskbitcount, norm_mean, norm_gradient
            )
            if normalize
            else totalcodebitcount / totalmaskbitcount
        )

        if current_dist < best_dist:
            best_dist = current_dist
            rot_shift = current_shift

    return best_dist, rot_shift


def generate_similar_iris_code(
    base_code: np.ndarray,
    base_mask: np.ndarray,
    noise_ratio: float = 0.05,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate an iris code similar to a base code (for same subject/eye simulations).

    The mask maintains duplicate d1 bits (mask[:,:,:,0] == mask[:,:,:,1]) to match
    the real-world iris code structure.

    Args:
        base_code: Base iris code, shape (16, 200, 2, 2), dtype uint8.
        base_mask: Base mask, shape (16, 200, 2, 2), dtype uint8.
        noise_ratio: Fraction of bits to flip (default 5% for ~0.05 FHD).
        seed: Random seed for reproducibility.

    Returns:
        Tuple (new_code, new_mask) with similar patterns. Mask has duplicate d1 bits.
    """
    rng = np.random.RandomState(seed)

    # Copy base code and add noise
    new_code = base_code.copy()

    # Flip a small fraction of code bits (code has independent d1 values)
    n_bits = base_code.size
    n_flip = int(n_bits * noise_ratio)
    flip_indices = rng.choice(n_bits, size=n_flip, replace=False)
    flat_code = new_code.ravel()
    flat_code[flip_indices] = 1 - flat_code[flip_indices]

    # Slightly modify mask - but maintain duplicate d1 structure
    # Work with the half mask (d1=0 slice) and then duplicate
    half_mask = base_mask[:, :, :, 0].copy()  # Shape (16, 200, 2)
    n_half_bits = half_mask.size
    n_mask_flip = int(n_half_bits * noise_ratio * 0.1)
    mask_flip_indices = rng.choice(n_half_bits, size=n_mask_flip, replace=False)
    flat_half_mask = half_mask.ravel()
    flat_half_mask[mask_flip_indices] = 1 - flat_half_mask[mask_flip_indices]

    # Reconstruct full mask with duplicate d1 bits
    new_mask = np.stack([half_mask, half_mask], axis=-1)

    return new_code, new_mask


def generate_random_iris_code(seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a random iris code and mask.

    The mask has duplicate d1 bits (mask[:,:,:,0] == mask[:,:,:,1]) to match
    the real-world iris code structure where the mask is the same for both
    real and imaginary components of the filter response.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        Tuple (code, mask) with shape (16, 200, 2, 2), dtype uint8.
        Code has independent d1 bits, mask has duplicate d1 bits.
    """
    rng = np.random.RandomState(seed)
    # Code has independent values for d1=0 and d1=1 (real/imaginary parts)
    code = rng.randint(0, 2, (16, 200, 2, 2), dtype=np.uint8)
    # Mask with ~80% valid bits - same for both d1 values (real-world structure)
    # Generate half mask for d1=0, then duplicate to d1=1
    half_mask = (rng.random((16, 200, 2)) < 0.8).astype(np.uint8)
    mask = np.stack([half_mask, half_mask], axis=-1)
    return code, mask
