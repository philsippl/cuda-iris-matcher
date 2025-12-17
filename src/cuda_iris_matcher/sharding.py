"""Multi-GPU sharding for iris matching operations.

This module provides sharded versions of the iris matching functions that:
1. Automatically distribute work across all available CUDA devices
2. Handle cases where data doesn't fit on a single device via tiling
3. Aggregate results from all shards with proper index offsets
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from . import _C
from .ops import (
    DEFAULT_D0_DIM,
    DEFAULT_D1_DIM,
    DEFAULT_R_DIM,
    DEFAULT_THETA_DIM,
    INCLUDE_ALL,
    _resolve_dims,
    pack_theta_major_cuda,
)


@dataclass
class ShardConfig:
    """Configuration for a single shard of computation."""

    device_id: int  # CUDA device ID
    a_start: int  # Start index in A
    a_end: int  # End index in A (exclusive)
    b_start: int  # Start index in B
    b_end: int  # End index in B (exclusive)


def _estimate_memory_bytes(m_a: int, m_b: int, k_words: int, max_pairs: int) -> int:
    """Estimate GPU memory needed for a single tile computation.

    Args:
        m_a: Number of rows in A tile
        m_b: Number of rows in B tile
        k_words: Number of int32 words per row
        max_pairs: Maximum pairs to collect

    Returns:
        Estimated bytes needed (with safety margin)
    """
    # Data + mask + premasked for both A and B
    data_bytes = 2 * (m_a + m_b) * k_words * 4 * 3  # data, mask, premasked
    # Output buffers
    output_bytes = max_pairs * (2 * 4 + 1 + 4 + 4)  # indices, category, distance, count
    # Shared memory and overhead (rough estimate)
    overhead = 256 * 1024 * 1024  # 256 MB overhead

    return int((data_bytes + output_bytes) * 1.5 + overhead)


def _get_available_memory(device_id: int) -> int:
    """Get available GPU memory in bytes."""
    with torch.cuda.device(device_id):
        total = torch.cuda.get_device_properties(device_id).total_memory
        reserved = torch.cuda.memory_reserved(device_id)
        allocated = torch.cuda.memory_allocated(device_id)
        # Use 80% of free memory to leave headroom
        free = total - max(reserved, allocated)
        return int(free * 0.8)


def _is_packed(tensor: torch.Tensor, k_words: int) -> bool:
    """Check if tensor is already packed (int32, shape [M, k_words])."""
    return (
        tensor.dim() == 2
        and tensor.size(1) == k_words
        and tensor.dtype == torch.int32
    )


def _is_unpacked(
    tensor: torch.Tensor, r_dim: int, theta_dim: int, d0_dim: int, d1_dim: int
) -> bool:
    """Check if tensor is unpacked (uint8, shape [M, r, theta, d0, d1])."""
    return (
        tensor.dim() == 5
        and tensor.size(1) == r_dim
        and tensor.size(2) == theta_dim
        and tensor.size(3) == d0_dim
        and tensor.size(4) == d1_dim
        and tensor.dtype == torch.uint8
    )


def _pack_on_device(
    data: torch.Tensor,
    device_id: int,
    r_dim: int,
    theta_dim: int,
    d0_dim: int,
    d1_dim: int,
    batch_size: Optional[int] = None,
) -> torch.Tensor:
    """Pack data on specified GPU device, handling batching if needed.

    Returns packed tensor on the specified GPU device.
    """
    k_words = r_dim * theta_dim * d0_dim * d1_dim // 32
    k_bits = r_dim * theta_dim * d0_dim * d1_dim
    m = data.size(0)

    # If already packed, just move to device
    if _is_packed(data, k_words):
        if data.is_cuda and data.device.index == device_id:
            return data
        return data.cuda(device_id)

    # Need to pack - determine batch size based on memory
    with torch.cuda.device(device_id):
        if batch_size is None:
            available_mem = _get_available_memory(device_id)
            bytes_per_sample = k_bits + k_words * 4  # unpacked + packed
            batch_size = max(1, int(available_mem * 0.4 / bytes_per_sample))
            batch_size = min(batch_size, m)

        # If entire data fits, pack all at once
        if batch_size >= m:
            if data.is_cuda:
                data_gpu = data.cuda(device_id) if data.device.index != device_id else data
            else:
                data_gpu = data.cuda(device_id)
            return pack_theta_major_cuda(
                data_gpu.clone(), None, r_dim, theta_dim, d0_dim, d1_dim
            )

        # Pack in batches, accumulate on GPU
        packed_gpu = torch.zeros((m, k_words), dtype=torch.int32, device=f"cuda:{device_id}")

        for start in range(0, m, batch_size):
            end = min(start + batch_size, m)
            if data.is_cuda:
                batch = data[start:end].cuda(device_id)
            else:
                batch = data[start:end].cuda(device_id)

            packed_batch = pack_theta_major_cuda(
                batch.clone(), None, r_dim, theta_dim, d0_dim, d1_dim
            )
            packed_gpu[start:end] = packed_batch

        return packed_gpu


def _compute_shard_configs(
    m_a: int,
    m_b: int,
    k_words: int,
    max_pairs_per_shard: int,
    num_devices: int,
    min_shards: int = 1,
    max_tile_size: Optional[int] = None,
) -> List[ShardConfig]:
    """Compute shard configurations for distributing A×B computation.

    Args:
        m_a: Total rows in A
        m_b: Total rows in B
        k_words: Number of int32 words per row
        max_pairs_per_shard: Max pairs per shard
        num_devices: Number of CUDA devices available
        min_shards: Minimum number of shards (for testing on single GPU)
        max_tile_size: Maximum rows per tile (None = auto based on memory)

    Returns:
        List of ShardConfig objects
    """
    shards = []

    # Determine tile sizes based on memory constraints
    if max_tile_size is None:
        # Try to fit on device, start with reasonable defaults
        if num_devices > 0:
            available_mem = _get_available_memory(0)
            # Binary search for max tile size
            max_tile_size = max(m_a, m_b)
            while max_tile_size > 64:
                mem_needed = _estimate_memory_bytes(
                    min(max_tile_size, m_a),
                    min(max_tile_size, m_b),
                    k_words,
                    max_pairs_per_shard,
                )
                if mem_needed <= available_mem:
                    break
                max_tile_size //= 2
            max_tile_size = max(64, max_tile_size)
        else:
            max_tile_size = 4096  # Default

    # Compute number of tiles needed
    n_tiles_a = max(1, math.ceil(m_a / max_tile_size))
    n_tiles_b = max(1, math.ceil(m_b / max_tile_size))

    # Ensure minimum shard count
    total_tiles = n_tiles_a * n_tiles_b
    if total_tiles < min_shards:
        # Increase tiling to meet minimum shards
        factor = math.ceil(math.sqrt(min_shards / total_tiles))
        n_tiles_a = max(n_tiles_a, min(m_a, factor))
        n_tiles_b = max(n_tiles_b, min(m_b, factor))

    # Recompute tile sizes
    tile_size_a = math.ceil(m_a / n_tiles_a)
    tile_size_b = math.ceil(m_b / n_tiles_b)

    # Generate shard configs, distributing across devices
    device_idx = 0
    for i in range(n_tiles_a):
        a_start = i * tile_size_a
        a_end = min((i + 1) * tile_size_a, m_a)
        if a_start >= m_a:
            break

        for j in range(n_tiles_b):
            b_start = j * tile_size_b
            b_end = min((j + 1) * tile_size_b, m_b)
            if b_start >= m_b:
                break

            shards.append(
                ShardConfig(
                    device_id=device_idx % num_devices if num_devices > 0 else 0,
                    a_start=a_start,
                    a_end=a_end,
                    b_start=b_start,
                    b_end=b_end,
                )
            )
            device_idx += 1

    return shards


def _compute_self_shard_configs(
    m: int,
    k_words: int,
    max_pairs_per_shard: int,
    num_devices: int,
    min_shards: int = 1,
    max_tile_size: Optional[int] = None,
) -> List[ShardConfig]:
    """Compute shard configurations for self-comparison (lower triangle only).

    For self-comparison, we need tiles that cover the lower triangle (i > j).
    We use the A vs B kernel on tiles where (a_start, a_end) x (b_start, b_end)
    potentially contains lower triangle elements.

    Args:
        m: Total rows
        k_words: Number of int32 words per row
        max_pairs_per_shard: Max pairs per shard
        num_devices: Number of CUDA devices available
        min_shards: Minimum number of shards (for testing on single GPU)
        max_tile_size: Maximum rows per tile (None = auto based on memory)

    Returns:
        List of ShardConfig objects covering the lower triangle
    """
    shards = []

    # Determine tile sizes based on memory constraints
    if max_tile_size is None:
        if num_devices > 0:
            available_mem = _get_available_memory(0)
            max_tile_size = m
            while max_tile_size > 64:
                mem_needed = _estimate_memory_bytes(
                    max_tile_size, max_tile_size, k_words, max_pairs_per_shard
                )
                if mem_needed <= available_mem:
                    break
                max_tile_size //= 2
            max_tile_size = max(64, max_tile_size)
        else:
            max_tile_size = 4096

    # Compute number of tiles
    n_tiles = max(1, math.ceil(m / max_tile_size))

    # Ensure minimum shard count for lower triangle
    # Lower triangle has n*(n+1)/2 tiles
    lower_tri_tiles = n_tiles * (n_tiles + 1) // 2
    if lower_tri_tiles < min_shards:
        # Increase tiling - solve for n in n*(n+1)/2 >= min_shards
        n_tiles = max(n_tiles, math.ceil((-1 + math.sqrt(1 + 8 * min_shards)) / 2))
        n_tiles = min(n_tiles, m)

    tile_size = math.ceil(m / n_tiles)

    # Generate shard configs for lower triangle (including diagonal tiles)
    # For the lower triangle, we need tiles where rows can be > cols
    # This means tiles (i, j) where i >= j
    device_idx = 0
    for i in range(n_tiles):
        a_start = i * tile_size
        a_end = min((i + 1) * tile_size, m)
        if a_start >= m:
            break

        for j in range(i + 1):  # j <= i covers tiles that may have lower triangle elements
            b_start = j * tile_size
            b_end = min((j + 1) * tile_size, m)
            if b_start >= m:
                break

            # Only include if this tile can contain lower triangle elements
            # Tile contains (row, col) pairs where row in [a_start, a_end) and col in [b_start, b_end)
            # Lower triangle needs row > col
            # This is possible if a_end-1 > b_start (max row > min col)
            if a_end > b_start:
                shards.append(
                    ShardConfig(
                        device_id=device_idx % num_devices if num_devices > 0 else 0,
                        a_start=a_start,
                        a_end=a_end,
                        b_start=b_start,
                        b_end=b_end,
                    )
                )
                device_idx += 1

    return shards


def masked_hamming_ab_sharded(
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
    dims: Optional[Tuple[int, int, int, int]] = None,
    r_dim: int = DEFAULT_R_DIM,
    theta_dim: int = DEFAULT_THETA_DIM,
    d0_dim: int = DEFAULT_D0_DIM,
    d1_dim: int = DEFAULT_D1_DIM,
    min_shards: int = 1,
    max_tile_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sharded version of masked_hamming_ab_cuda for multi-GPU and large datasets.

    Automatically distributes computation across all available CUDA devices and
    handles cases where A or B don't fit on a single device. Accepts either
    packed (int32) or unpacked (uint8) data - packing is done on GPU automatically.

    Args:
        data_a: Tensor of shape [M_A, k_words] int32 (packed) OR
                [M_A, r_dim, theta_dim, d0_dim, d1_dim] uint8 (unpacked)
        mask_a: Same shape/dtype as data_a
        data_b: Tensor of shape [M_B, k_words] int32 (packed) OR
                [M_B, r_dim, theta_dim, d0_dim, d1_dim] uint8 (unpacked)
        mask_b: Same shape/dtype as data_b
        labels_a: Optional int32 tensor of shape [M_A] with identity labels.
        labels_b: Optional int32 tensor of shape [M_B] with identity labels.
        match_threshold: Threshold for match classification (default: 0.35)
        non_match_threshold: Threshold for non-match classification (default: 0.35)
        is_similarity: If True, higher values = more similar
        include_flags: Bitmask of categories to include (default: INCLUDE_ALL)
        max_pairs: Maximum total pairs to return (default: 1,000,000)
        dims: Optional tuple (r_dim, theta_dim, d0_dim, d1_dim)
        r_dim, theta_dim, d0_dim, d1_dim: Iris code dimensions
        min_shards: Minimum number of shards (useful for testing on single GPU)
        max_tile_size: Maximum rows per tile (None = auto based on memory)

    Returns:
        Tuple of (pair_indices, categories, distances, count):
        - pair_indices: [N, 2] int32 - (row_in_A, row_in_B) indices of pairs
        - categories: [N] uint8 - category codes
        - distances: [N] float32 - distance values
        - count: [1] int32 - actual number of pairs
    """
    r_dim, theta_dim, d0_dim, d1_dim = _resolve_dims(
        dims, r_dim, theta_dim, d0_dim, d1_dim
    )
    k_words = r_dim * theta_dim * d0_dim * d1_dim // 32

    m_a = data_a.size(0)
    m_b = data_b.size(0)

    # Determine number of devices
    num_devices = torch.cuda.device_count()
    if num_devices == 0:
        raise RuntimeError("No CUDA devices available")

    # Check if data is packed or unpacked
    data_a_is_packed = _is_packed(data_a, k_words)
    data_b_is_packed = _is_packed(data_b, k_words)
    mask_a_is_packed = _is_packed(mask_a, k_words)
    mask_b_is_packed = _is_packed(mask_b, k_words)

    # For single-device case or when data fits, pack everything on GPU first
    # This avoids repeated CPU<->GPU transfers
    primary_device = 0

    # Pack data on GPU (handles batching internally if needed)
    with torch.cuda.device(primary_device):
        if data_a_is_packed:
            data_a_gpu = data_a.cuda(primary_device) if not data_a.is_cuda else data_a
        else:
            data_a_gpu = _pack_on_device(data_a, primary_device, r_dim, theta_dim, d0_dim, d1_dim)

        if mask_a_is_packed:
            mask_a_gpu = mask_a.cuda(primary_device) if not mask_a.is_cuda else mask_a
        else:
            mask_a_gpu = _pack_on_device(mask_a, primary_device, r_dim, theta_dim, d0_dim, d1_dim)

        if data_b_is_packed:
            data_b_gpu = data_b.cuda(primary_device) if not data_b.is_cuda else data_b
        else:
            data_b_gpu = _pack_on_device(data_b, primary_device, r_dim, theta_dim, d0_dim, d1_dim)

        if mask_b_is_packed:
            mask_b_gpu = mask_b.cuda(primary_device) if not mask_b.is_cuda else mask_b
        else:
            mask_b_gpu = _pack_on_device(mask_b, primary_device, r_dim, theta_dim, d0_dim, d1_dim)

    # Compute shard configurations
    shards = _compute_shard_configs(
        m_a, m_b, k_words, max_pairs, num_devices, min_shards, max_tile_size
    )
    max_pairs_per_shard = max(max_pairs // max(1, len(shards)), 1000)

    # For labels, keep on CPU for now (small)
    labels_a_cpu = labels_a.cpu() if labels_a is not None and labels_a.is_cuda else labels_a
    labels_b_cpu = labels_b.cpu() if labels_b is not None and labels_b.is_cuda else labels_b

    # Collect results from all shards
    all_indices = []
    all_categories = []
    all_distances = []
    total_count = 0

    for shard in shards:
        with torch.cuda.device(shard.device_id):
            # For multi-GPU, copy tile to target device; for single GPU, slice in place
            if shard.device_id == primary_device:
                data_a_tile = data_a_gpu[shard.a_start : shard.a_end]
                mask_a_tile = mask_a_gpu[shard.a_start : shard.a_end]
                data_b_tile = data_b_gpu[shard.b_start : shard.b_end]
                mask_b_tile = mask_b_gpu[shard.b_start : shard.b_end]
            else:
                # Multi-GPU: copy tile to this device
                data_a_tile = data_a_gpu[shard.a_start : shard.a_end].cuda(shard.device_id)
                mask_a_tile = mask_a_gpu[shard.a_start : shard.a_end].cuda(shard.device_id)
                data_b_tile = data_b_gpu[shard.b_start : shard.b_end].cuda(shard.device_id)
                mask_b_tile = mask_b_gpu[shard.b_start : shard.b_end].cuda(shard.device_id)

            labels_a_tile = None
            labels_b_tile = None
            if labels_a_cpu is not None and labels_b_cpu is not None:
                labels_a_tile = labels_a_cpu[shard.a_start : shard.a_end].cuda(
                    shard.device_id
                )
                labels_b_tile = labels_b_cpu[shard.b_start : shard.b_end].cuda(
                    shard.device_id
                )

            # Compute remaining pairs budget
            remaining_pairs = max_pairs - total_count
            if remaining_pairs <= 0:
                break

            tile_max_pairs = min(max_pairs_per_shard, remaining_pairs)

            # Run kernel on this tile
            indices, categories, distances, count = _C.masked_hamming_ab_cuda(
                data_a_tile,
                mask_a_tile,
                data_b_tile,
                mask_b_tile,
                labels_a_tile,
                labels_b_tile,
                match_threshold,
                non_match_threshold,
                is_similarity,
                include_flags,
                tile_max_pairs,
                r_dim,
                theta_dim,
                d0_dim,
                d1_dim,
            )

            n = count.item()
            if n > 0:
                # Adjust indices to global coordinates
                indices_cpu = indices[:n].cpu()
                indices_cpu[:, 0] += shard.a_start
                indices_cpu[:, 1] += shard.b_start

                all_indices.append(indices_cpu)
                all_categories.append(categories[:n].cpu())
                all_distances.append(distances[:n].cpu())
                total_count += n

    # Aggregate results
    if total_count == 0:
        # Return empty tensors
        return (
            torch.zeros((0, 2), dtype=torch.int32),
            torch.zeros((0,), dtype=torch.uint8),
            torch.zeros((0,), dtype=torch.float32),
            torch.tensor([0], dtype=torch.int32),
        )

    pair_indices = torch.cat(all_indices, dim=0)
    categories = torch.cat(all_categories, dim=0)
    distances = torch.cat(all_distances, dim=0)
    count = torch.tensor([total_count], dtype=torch.int32)

    return pair_indices, categories, distances, count


def masked_hamming_sharded(
    data: torch.Tensor,
    mask: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    match_threshold: float = 0.35,
    non_match_threshold: float = 0.35,
    is_similarity: bool = False,
    include_flags: int = INCLUDE_ALL,
    max_pairs: int = 1_000_000,
    dims: Optional[Tuple[int, int, int, int]] = None,
    r_dim: int = DEFAULT_R_DIM,
    theta_dim: int = DEFAULT_THETA_DIM,
    d0_dim: int = DEFAULT_D0_DIM,
    d1_dim: int = DEFAULT_D1_DIM,
    min_shards: int = 1,
    max_tile_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sharded version of masked_hamming_cuda for multi-GPU and large datasets.

    Computes the lower triangle (i > j) by tiling and distributing across devices.
    Accepts either packed (int32) or unpacked (uint8) data - packing is done on GPU.

    Args:
        data: Tensor of shape [M, k_words] int32 (packed) OR
              [M, r_dim, theta_dim, d0_dim, d1_dim] uint8 (unpacked)
        mask: Same shape/dtype as data
        labels: Optional int32 tensor of shape [M] with identity labels.
        match_threshold: Threshold for match classification (default: 0.35)
        non_match_threshold: Threshold for non-match classification (default: 0.35)
        is_similarity: If True, higher values = more similar
        include_flags: Bitmask of categories to include (default: INCLUDE_ALL)
        max_pairs: Maximum total pairs to return (default: 1,000,000)
        dims: Optional tuple (r_dim, theta_dim, d0_dim, d1_dim)
        r_dim, theta_dim, d0_dim, d1_dim: Iris code dimensions
        min_shards: Minimum number of shards (useful for testing on single GPU)
        max_tile_size: Maximum rows per tile (None = auto based on memory)

    Returns:
        Tuple of (pair_indices, categories, distances, count):
        - pair_indices: [N, 2] int32 - (row, col) indices of pairs (row > col)
        - categories: [N] uint8 - category codes
        - distances: [N] float32 - distance values
        - count: [1] int32 - actual number of pairs
    """
    r_dim, theta_dim, d0_dim, d1_dim = _resolve_dims(
        dims, r_dim, theta_dim, d0_dim, d1_dim
    )
    k_words = r_dim * theta_dim * d0_dim * d1_dim // 32

    m = data.size(0)

    # Determine number of devices
    num_devices = torch.cuda.device_count()
    if num_devices == 0:
        raise RuntimeError("No CUDA devices available")

    # Check if data is packed or unpacked
    data_is_packed = _is_packed(data, k_words)
    mask_is_packed = _is_packed(mask, k_words)

    # Pack data on primary GPU (handles batching internally if needed)
    primary_device = 0
    with torch.cuda.device(primary_device):
        if data_is_packed:
            data_gpu = data.cuda(primary_device) if not data.is_cuda else data
        else:
            data_gpu = _pack_on_device(data, primary_device, r_dim, theta_dim, d0_dim, d1_dim)

        if mask_is_packed:
            mask_gpu = mask.cuda(primary_device) if not mask.is_cuda else mask
        else:
            mask_gpu = _pack_on_device(mask, primary_device, r_dim, theta_dim, d0_dim, d1_dim)

    # Compute shard configurations for lower triangle
    shards = _compute_self_shard_configs(
        m, k_words, max_pairs, num_devices, min_shards, max_tile_size
    )
    max_pairs_per_shard = max(max_pairs // max(1, len(shards)), 1000)

    # Labels stay on CPU (small)
    labels_cpu = labels.cpu() if labels is not None and labels.is_cuda else labels

    # Collect results from all shards
    all_indices = []
    all_categories = []
    all_distances = []
    total_count = 0

    for shard in shards:
        with torch.cuda.device(shard.device_id):
            # For single GPU, slice in place; for multi-GPU, copy tile to device
            if shard.device_id == primary_device:
                data_a_tile = data_gpu[shard.a_start : shard.a_end]
                mask_a_tile = mask_gpu[shard.a_start : shard.a_end]
                data_b_tile = data_gpu[shard.b_start : shard.b_end]
                mask_b_tile = mask_gpu[shard.b_start : shard.b_end]
            else:
                data_a_tile = data_gpu[shard.a_start : shard.a_end].cuda(shard.device_id)
                mask_a_tile = mask_gpu[shard.a_start : shard.a_end].cuda(shard.device_id)
                data_b_tile = data_gpu[shard.b_start : shard.b_end].cuda(shard.device_id)
                mask_b_tile = mask_gpu[shard.b_start : shard.b_end].cuda(shard.device_id)

            labels_a_tile = None
            labels_b_tile = None
            if labels_cpu is not None:
                labels_a_tile = labels_cpu[shard.a_start : shard.a_end].cuda(
                    shard.device_id
                )
                labels_b_tile = labels_cpu[shard.b_start : shard.b_end].cuda(
                    shard.device_id
                )

            # Compute remaining pairs budget
            remaining_pairs = max_pairs - total_count
            if remaining_pairs <= 0:
                break

            # For diagonal tiles (a_start == b_start), we'll filter to lower triangle later
            # so we need to request more pairs from the kernel (roughly 2x for diagonal)
            is_diagonal = shard.a_start == shard.b_start
            # For diagonal tiles, request full tile size since about half will be filtered
            tile_size_a = shard.a_end - shard.a_start
            tile_size_b = shard.b_end - shard.b_start
            if is_diagonal:
                # Diagonal tile: request all pairs, we'll filter to lower triangle
                tile_max_pairs = tile_size_a * tile_size_b
            else:
                # Off-diagonal lower triangle tile: all pairs are valid
                tile_max_pairs = min(max_pairs_per_shard, remaining_pairs)

            # Run A vs B kernel
            indices, categories, distances, count = _C.masked_hamming_ab_cuda(
                data_a_tile,
                mask_a_tile,
                data_b_tile,
                mask_b_tile,
                labels_a_tile,
                labels_b_tile,
                match_threshold,
                non_match_threshold,
                is_similarity,
                include_flags,
                tile_max_pairs,
                r_dim,
                theta_dim,
                d0_dim,
                d1_dim,
            )

            n = count.item()
            if n > 0:
                # Adjust indices to global coordinates
                indices_cpu = indices[:n].cpu().clone()
                indices_cpu[:, 0] += shard.a_start  # Row from A
                indices_cpu[:, 1] += shard.b_start  # Col from B

                categories_cpu = categories[:n].cpu()
                distances_cpu = distances[:n].cpu()

                # Filter to lower triangle: global_row > global_col
                # This is needed for diagonal tiles where a_start == b_start
                # and off-diagonal tiles that are strictly lower are already all valid
                if shard.a_start == shard.b_start:
                    # Diagonal tile - need to filter
                    valid_mask = indices_cpu[:, 0] > indices_cpu[:, 1]
                    indices_cpu = indices_cpu[valid_mask]
                    categories_cpu = categories_cpu[valid_mask]
                    distances_cpu = distances_cpu[valid_mask]

                if indices_cpu.size(0) > 0:
                    all_indices.append(indices_cpu)
                    all_categories.append(categories_cpu)
                    all_distances.append(distances_cpu)
                    total_count += indices_cpu.size(0)

    # Aggregate results
    if total_count == 0:
        return (
            torch.zeros((0, 2), dtype=torch.int32),
            torch.zeros((0,), dtype=torch.uint8),
            torch.zeros((0,), dtype=torch.float32),
            torch.tensor([0], dtype=torch.int32),
        )

    pair_indices = torch.cat(all_indices, dim=0)
    categories = torch.cat(all_categories, dim=0)
    distances = torch.cat(all_distances, dim=0)
    count = torch.tensor([total_count], dtype=torch.int32)

    return pair_indices, categories, distances, count


def pack_theta_major_batched(
    bits: torch.Tensor,
    dims: Optional[Tuple[int, int, int, int]] = None,
    r_dim: int = DEFAULT_R_DIM,
    theta_dim: int = DEFAULT_THETA_DIM,
    d0_dim: int = DEFAULT_D0_DIM,
    d1_dim: int = DEFAULT_D1_DIM,
    batch_size: Optional[int] = None,
    device_id: int = 0,
) -> torch.Tensor:
    """Pack iris code bits in batches when data doesn't fit on GPU.

    This function handles large datasets that exceed GPU memory by packing
    in batches and accumulating results on CPU.

    Args:
        bits: CPU uint8 tensor of shape (M, r_dim, theta_dim, d0_dim, d1_dim)
              with values in {0, 1}.
        dims: Optional tuple (r_dim, theta_dim, d0_dim, d1_dim)
        r_dim, theta_dim, d0_dim, d1_dim: Iris code dimensions
        batch_size: Number of samples to pack per batch (None = auto based on memory)
        device_id: CUDA device to use for packing

    Returns:
        CPU int32 tensor of shape (M, k_words) with packed bits in theta-major order.

    Example:
        >>> # For 1 million samples that don't fit on GPU
        >>> raw_codes = torch.from_numpy(codes)  # (1M, 16, 200, 2, 2) uint8 on CPU
        >>> packed = pack_theta_major_batched(raw_codes, batch_size=50000)
        >>> # packed is (1M, 400) int32 on CPU, ready for sharded matching
    """
    r_dim, theta_dim, d0_dim, d1_dim = _resolve_dims(
        dims, r_dim, theta_dim, d0_dim, d1_dim
    )
    k_words = r_dim * theta_dim * d0_dim * d1_dim // 32

    if bits.is_cuda:
        # Already on GPU - clone first since pack_theta_major is in-place
        bits_clone = bits.clone()
        return pack_theta_major_cuda(bits_clone, dims, r_dim, theta_dim, d0_dim, d1_dim).cpu()

    m = bits.size(0)

    # Auto-determine batch size based on available memory
    if batch_size is None:
        with torch.cuda.device(device_id):
            available_mem = _get_available_memory(device_id)
            # Each sample: unpacked = r*theta*d0*d1 bytes, packed = k_words*4 bytes
            bytes_per_sample_unpacked = r_dim * theta_dim * d0_dim * d1_dim
            bytes_per_sample_packed = k_words * 4
            # Need space for both unpacked input and packed output
            bytes_per_sample = bytes_per_sample_unpacked + bytes_per_sample_packed
            # Use 50% of available memory for safety
            batch_size = max(1, int(available_mem * 0.5 / bytes_per_sample))
            batch_size = min(batch_size, m)

    # Allocate output on CPU
    packed_cpu = torch.zeros((m, k_words), dtype=torch.int32)

    # Pack in batches
    with torch.cuda.device(device_id):
        for start in range(0, m, batch_size):
            end = min(start + batch_size, m)

            # Copy batch to GPU
            batch_gpu = bits[start:end].cuda(device_id)

            # Pack on GPU
            packed_batch = pack_theta_major_cuda(
                batch_gpu, dims, r_dim, theta_dim, d0_dim, d1_dim
            )

            # Copy back to CPU
            packed_cpu[start:end] = packed_batch.cpu()

            # Free GPU memory
            del batch_gpu, packed_batch
            torch.cuda.empty_cache()

    return packed_cpu


def get_device_count() -> int:
    """Return the number of available CUDA devices."""
    return torch.cuda.device_count()


def get_shard_info(
    m_a: int,
    m_b: int,
    k_words: int = 400,
    max_pairs_per_shard: int = 100000,
    min_shards: int = 1,
    max_tile_size: Optional[int] = None,
) -> List[ShardConfig]:
    """Get shard configuration info for planning (debugging/inspection).

    Args:
        m_a: Number of rows in A
        m_b: Number of rows in B
        k_words: Number of int32 words per row (default 400 for standard iris)
        max_pairs_per_shard: Max pairs per shard
        min_shards: Minimum number of shards
        max_tile_size: Maximum tile size

    Returns:
        List of ShardConfig objects
    """
    num_devices = torch.cuda.device_count()
    return _compute_shard_configs(
        m_a, m_b, k_words, max_pairs_per_shard, num_devices, min_shards, max_tile_size
    )

