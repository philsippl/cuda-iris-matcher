"""Multi-GPU sharding for iris matching operations.

This module provides sharded versions of the iris matching functions that:
1. Automatically distribute work across all available CUDA devices
2. Handle cases where data doesn't fit on a single device via tiling
3. Aggregate results from all shards with proper index offsets
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch

from . import _C
from .ops import (
    DEFAULT_D0_DIM,
    DEFAULT_D1_DIM,
    DEFAULT_R_DIM,
    DEFAULT_THETA_DIM,
    DEFAULT_DOT_VEC_DIM,
    INCLUDE_ALL,
    _resolve_dims,
    pack_theta_major_cuda,
)
from .sampling import SampleBinsType, apply_stratified_sampling


@dataclass
class ShardConfig:
    """Configuration for a single shard of computation."""

    device_id: int  # CUDA device ID (local to this host)
    a_start: int  # Start index in A
    a_end: int  # End index in A (exclusive)
    b_start: int  # Start index in B
    b_end: int  # End index in B (exclusive)
    global_shard_id: int = 0  # Global shard ID across all hosts (for deterministic mapping)


@dataclass
class ShardResult:
    """Result from a single shard computation."""

    indices: torch.Tensor  # [N, 2] int32 with global coordinates
    categories: torch.Tensor  # [N] uint8
    distances: torch.Tensor  # [N] float32
    count: int


@dataclass
class ShardKernelResult:
    """Raw kernel output before CPU transfer (kept on GPU for async execution)."""

    shard: ShardConfig
    indices: torch.Tensor  # On GPU
    categories: torch.Tensor  # On GPU
    distances: torch.Tensor  # On GPU
    count: torch.Tensor  # On GPU


def _filter_shards_for_host(
    shards: List[ShardConfig],
    host_index: int,
    num_hosts: int,
    num_devices_per_host: int,
) -> List[ShardConfig]:
    """Filter shards to only those assigned to this host and remap device IDs.

    Shards are assigned to hosts in a round-robin fashion based on global_shard_id.
    Device IDs are remapped to be local to this host (0 to num_devices_per_host-1).

    Args:
        shards: List of all shard configurations (with global_shard_id set)
        host_index: Index of this host (0 to num_hosts-1)
        num_hosts: Total number of hosts
        num_devices_per_host: Number of CUDA devices on this host

    Returns:
        Filtered list of shards for this host with local device IDs
    """
    if num_hosts <= 1:
        return shards

    filtered = []
    for shard in shards:
        if shard.global_shard_id % num_hosts == host_index:
            # Remap device_id to be local to this host
            local_device_id = (shard.global_shard_id // num_hosts) % num_devices_per_host
            filtered.append(
                ShardConfig(
                    device_id=local_device_id,
                    a_start=shard.a_start,
                    a_end=shard.a_end,
                    b_start=shard.b_start,
                    b_end=shard.b_end,
                    global_shard_id=shard.global_shard_id,
                )
            )
    return filtered


def _run_shards_async(
    shards: List[ShardConfig],
    kernel_fn: Callable[[ShardConfig], ShardKernelResult],
) -> List[ShardResult]:
    """Run shard computations with true async kernel execution.

    Launches all kernels without synchronization, allowing kernels on different
    GPUs to execute concurrently. Only synchronizes at the end before collecting
    results.

    Args:
        shards: List of shard configurations
        kernel_fn: Function that launches kernel and returns raw GPU tensors

    Returns:
        List of ShardResults with data moved to CPU
    """
    if len(shards) == 0:
        return []

    # Group shards by device
    device_ids = sorted(set(s.device_id for s in shards))
    shards_by_device: dict[int, List[ShardConfig]] = {d: [] for d in device_ids}
    for shard in shards:
        shards_by_device[shard.device_id].append(shard)

    # Create a stream per device for async execution
    streams: dict[int, torch.cuda.Stream] = {}
    for device_id in device_ids:
        streams[device_id] = torch.cuda.Stream(device=device_id)

    # Function to launch all shards on a single device
    def launch_on_device(device_id: int) -> List[ShardKernelResult]:
        stream = streams[device_id]
        results = []
        with torch.cuda.stream(stream):
            for shard in shards_by_device[device_id]:
                result = kernel_fn(shard)
                results.append(result)
        return results

    # Launch kernels on all devices in parallel using ThreadPoolExecutor
    # This parallelizes the Python overhead of kernel launches
    all_kernel_results: List[ShardKernelResult] = []
    if len(device_ids) == 1:
        # Single GPU - no need for threads
        all_kernel_results = launch_on_device(device_ids[0])
    else:
        # Multi-GPU - use one thread per device to parallelize launches
        with ThreadPoolExecutor(max_workers=len(device_ids)) as executor:
            futures = {executor.submit(launch_on_device, d): d for d in device_ids}
            # Collect results in device order for determinism
            device_results = {}
            for future in as_completed(futures):
                device_id = futures[future]
                device_results[device_id] = future.result()
            # Flatten in device order
            for device_id in device_ids:
                all_kernel_results.extend(device_results[device_id])

    # Now synchronize all devices - this is where we wait for all GPUs
    for device_id in device_ids:
        streams[device_id].synchronize()
    # Also do a full device sync to ensure all CUDA operations are complete
    for device_id in device_ids:
        torch.cuda.synchronize(device_id)

    # Process results (all kernels have completed)
    final_results: List[ShardResult] = []
    for kr in all_kernel_results:
        n = kr.count.item()
        if n > 0:
            # Transfer to CPU and adjust indices
            indices_cpu = kr.indices[:n].cpu()
            indices_cpu[:, 0] += kr.shard.a_start
            indices_cpu[:, 1] += kr.shard.b_start

            final_results.append(
                ShardResult(
                    indices=indices_cpu,
                    categories=kr.categories[:n].cpu(),
                    distances=kr.distances[:n].cpu(),
                    count=n,
                )
            )

    return final_results


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
    # Note: The CUDA kernel now handles cooperative launch limits internally by batching
    with torch.cuda.device(device_id):
        if batch_size is None:
            available_mem = _get_available_memory(device_id)
            bytes_per_sample = k_bits + k_words * 4  # unpacked + packed
            batch_size = max(1, int(available_mem * 0.4 / bytes_per_sample))
        batch_size = min(batch_size, m)

        # If entire data fits in one batch, pack all at once
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
    global_shard_idx = 0
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
                    device_id=global_shard_idx % num_devices if num_devices > 0 else 0,
                    a_start=a_start,
                    a_end=a_end,
                    b_start=b_start,
                    b_end=b_end,
                    global_shard_id=global_shard_idx,
                )
            )
            global_shard_idx += 1

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
    Uses greedy load balancing to distribute work evenly across GPUs.

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

    # For good load balancing, we need enough tiles for granular distribution
    # Target at least 4*num_devices tiles for reasonable balance (since diagonal tiles have higher weight)
    if num_devices > 1:
        min_tiles_for_balance = 4 * num_devices
        while n_tiles * (n_tiles + 1) // 2 < min_tiles_for_balance and n_tiles < m:
            n_tiles += 1

    tile_size = math.ceil(m / n_tiles)

    # First pass: collect all tiles with their estimated work (pair counts)
    # Profiling shows diagonal and off-diagonal tiles have similar cost per pair
    tiles_with_work: List[Tuple[int, int, int, int, int]] = []  # (a_start, a_end, b_start, b_end, work)
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
            if a_end > b_start:
                a_size = a_end - a_start
                b_size = b_end - b_start
                if a_start == b_start:
                    # Diagonal tiles: lower triangle only
                    work = a_size * (a_size - 1) // 2
                else:
                    # Off-diagonal tiles: full rectangle
                    work = a_size * b_size
                tiles_with_work.append((a_start, a_end, b_start, b_end, work))

    if num_devices <= 0 or num_devices == 1:
        # Single device - no balancing needed
        return [
            ShardConfig(device_id=0, a_start=t[0], a_end=t[1], b_start=t[2], b_end=t[3], global_shard_id=i)
            for i, t in enumerate(tiles_with_work)
        ]

    # Greedy load balancing: assign largest tiles first to least loaded GPU
    # Sort tiles by work (descending)
    tiles_with_work.sort(key=lambda x: x[4], reverse=True)

    # Track work per device
    device_work = [0] * num_devices
    tile_assignments: List[Tuple[int, int, int, int, int, int, int]] = []  # + device_id, global_id

    for idx, (a_start, a_end, b_start, b_end, work) in enumerate(tiles_with_work):
        # Assign to GPU with least work
        min_device = min(range(num_devices), key=lambda d: device_work[d])
        device_work[min_device] += work
        tile_assignments.append((a_start, a_end, b_start, b_end, work, min_device, idx))

    # Create shard configs (sorted by device for better locality)
    tile_assignments.sort(key=lambda x: (x[5], x[0], x[2]))  # Sort by device, then position
    shards = [
        ShardConfig(device_id=t[5], a_start=t[0], a_end=t[1], b_start=t[2], b_end=t[3], global_shard_id=t[6])
        for t in tile_assignments
    ]

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
    host_index: int = 0,
    num_hosts: int = 1,
    sample_bins: SampleBinsType = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sharded version of masked_hamming_ab_cuda for multi-GPU and multi-host datasets.

    Automatically distributes computation across all available CUDA devices and
    handles cases where A or B don't fit on a single device. Accepts either
    packed (int32) or unpacked (uint8) data - packing is done on GPU automatically.

    For multi-host operation:
    - Each host should have the FULL data tensors (or be able to access all rows)
    - Set host_index to this host's index (0 to num_hosts-1)
    - Set num_hosts to total number of hosts
    - Each host will process only its assigned tiles
    - Results from all hosts should be aggregated by the caller

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
        host_index: Index of this host for multi-host operation (default: 0)
        num_hosts: Total number of hosts for multi-host operation (default: 1)

    Returns:
        Tuple of (pair_indices, categories, distances, count):
        - pair_indices: [N, 2] int32 - (row_in_A, row_in_B) indices of pairs
        - categories: [N] uint8 - category codes
        - distances: [N] float32 - distance values
        - count: [1] int32 - actual number of pairs

    Example:
        Large-scale gallery vs probe matching across multiple GPUs:

        >>> import torch
        >>> import cuda_iris_matcher as ih
        >>> # Large gallery (100K enrolled users) and probe set (1K queries)
        >>> gallery = torch.randint(0, 2**31, (100000, 400), dtype=torch.int32)
        >>> gallery_mask = torch.full_like(gallery, 0x7FFFFFFF)
        >>> probe = torch.randint(0, 2**31, (1000, 400), dtype=torch.int32)
        >>> probe_mask = torch.full_like(probe, 0x7FFFFFFF)
        >>> # Distributes 100M comparisons across all GPUs
        >>> pairs, _, dists, count = ih.masked_hamming_ab_sharded(
        ...     gallery, gallery_mask, probe, probe_mask
        ... )
        >>> print(f"Completed {count.item()} comparisons")
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

    # For single GPU without forced sharding, use non-sharded kernel directly
    if num_devices == 1 and min_shards <= 1:
        primary_device = 0
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
        labels_a_gpu = None
        labels_b_gpu = None
        if labels_a is not None:
            labels_a_gpu = labels_a.cuda(primary_device) if not labels_a.is_cuda else labels_a
        if labels_b is not None:
            labels_b_gpu = labels_b.cuda(primary_device) if not labels_b.is_cuda else labels_b
        pair_indices, categories, distances, count = _C.masked_hamming_ab_cuda(
            data_a_gpu, mask_a_gpu, data_b_gpu, mask_b_gpu,
            labels_a_gpu, labels_b_gpu,
            match_threshold, non_match_threshold, is_similarity,
            include_flags, max_pairs, r_dim, theta_dim, d0_dim, d1_dim
        )
        # Apply stratified sampling if configured
        if sample_bins is not None:
            pair_indices, categories, distances, count = apply_stratified_sampling(
                pair_indices, categories, distances, sample_bins
            )
        return pair_indices, categories, distances, count

    # Compute shard configurations
    # For multi-host, compute shards as if all hosts' GPUs were available
    total_devices = num_devices * num_hosts
    all_shards = _compute_shard_configs(
        m_a, m_b, k_words, max_pairs, total_devices, min_shards * num_hosts, max_tile_size
    )

    # Filter to only shards assigned to this host
    shards = _filter_shards_for_host(all_shards, host_index, num_hosts, num_devices)

    # If no shards assigned to this host, return empty results
    if len(shards) == 0:
        return (
            torch.zeros((0, 2), dtype=torch.int32),
            torch.zeros((0,), dtype=torch.uint8),
            torch.zeros((0,), dtype=torch.float32),
            torch.tensor([0], dtype=torch.int32),
        )

    max_pairs_per_shard = max(max_pairs // max(1, len(all_shards)), 1000)

    # For labels, keep on CPU for now (small)
    labels_a_cpu = labels_a.cpu() if labels_a is not None and labels_a.is_cuda else labels_a
    labels_b_cpu = labels_b.cpu() if labels_b is not None and labels_b.is_cuda else labels_b

    # Ensure data is on CPU for direct transfers to each GPU
    data_a_cpu = data_a.cpu() if data_a.is_cuda else data_a
    mask_a_cpu = mask_a.cpu() if mask_a.is_cuda else mask_a
    data_b_cpu = data_b.cpu() if data_b.is_cuda else data_b
    mask_b_cpu = mask_b.cpu() if mask_b.is_cuda else mask_b

    # Determine which row ranges each GPU needs for A and B
    device_ids = sorted(set(s.device_id for s in shards))
    device_a_ranges: dict[int, set[Tuple[int, int]]] = {d: set() for d in device_ids}
    device_b_ranges: dict[int, set[Tuple[int, int]]] = {d: set() for d in device_ids}
    for shard in shards:
        device_a_ranges[shard.device_id].add((shard.a_start, shard.a_end))
        device_b_ranges[shard.device_id].add((shard.b_start, shard.b_end))

    def merge_ranges(ranges: set[Tuple[int, int]]) -> Tuple[int, int]:
        if not ranges:
            return (0, 0)
        return (min(r[0] for r in ranges), max(r[1] for r in ranges))

    device_a_row_ranges = {d: merge_ranges(ranges) for d, ranges in device_a_ranges.items()}
    device_b_row_ranges = {d: merge_ranges(ranges) for d, ranges in device_b_ranges.items()}

    # Transfer data directly from CPU to each GPU in parallel
    def transfer_to_device(device_id: int) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        a_start, a_end = device_a_row_ranges[device_id]
        b_start, b_end = device_b_row_ranges[device_id]
        with torch.cuda.device(device_id):
            # Transfer A slices
            data_a_slice = data_a_cpu[a_start:a_end]
            mask_a_slice = mask_a_cpu[a_start:a_end]
            if data_a_is_packed:
                data_a_gpu = data_a_slice.cuda(device_id)
            else:
                data_a_gpu = _pack_on_device(data_a_slice, device_id, r_dim, theta_dim, d0_dim, d1_dim)
            if mask_a_is_packed:
                mask_a_gpu = mask_a_slice.cuda(device_id)
            else:
                mask_a_gpu = _pack_on_device(mask_a_slice, device_id, r_dim, theta_dim, d0_dim, d1_dim)

            # Transfer B slices
            data_b_slice = data_b_cpu[b_start:b_end]
            mask_b_slice = mask_b_cpu[b_start:b_end]
            if data_b_is_packed:
                data_b_gpu = data_b_slice.cuda(device_id)
            else:
                data_b_gpu = _pack_on_device(data_b_slice, device_id, r_dim, theta_dim, d0_dim, d1_dim)
            if mask_b_is_packed:
                mask_b_gpu = mask_b_slice.cuda(device_id)
            else:
                mask_b_gpu = _pack_on_device(mask_b_slice, device_id, r_dim, theta_dim, d0_dim, d1_dim)

            torch.cuda.synchronize(device_id)
        return device_id, data_a_gpu, mask_a_gpu, data_b_gpu, mask_b_gpu, a_start, b_start

    # Transfer to all GPUs in parallel
    device_data: dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]] = {}
    with ThreadPoolExecutor(max_workers=len(device_ids)) as executor:
        futures = [executor.submit(transfer_to_device, d) for d in device_ids]
        for future in as_completed(futures):
            device_id, da, ma, db, mb, a_off, b_off = future.result()
            device_data[device_id] = (da, ma, db, mb, a_off, b_off)

    # Define kernel launch function
    def launch_kernel(shard: ShardConfig) -> ShardKernelResult:
        data_a_gpu, mask_a_gpu, data_b_gpu, mask_b_gpu, a_offset, b_offset = device_data[shard.device_id]

        # Adjust indices relative to the data slices on this device
        local_a_start = shard.a_start - a_offset
        local_a_end = shard.a_end - a_offset
        local_b_start = shard.b_start - b_offset
        local_b_end = shard.b_end - b_offset

        data_a_tile = data_a_gpu[local_a_start:local_a_end]
        mask_a_tile = mask_a_gpu[local_a_start:local_a_end]
        data_b_tile = data_b_gpu[local_b_start:local_b_end]
        mask_b_tile = mask_b_gpu[local_b_start:local_b_end]

        labels_a_tile = None
        labels_b_tile = None
        if labels_a_cpu is not None and labels_b_cpu is not None:
            labels_a_tile = labels_a_cpu[shard.a_start:shard.a_end].cuda(
                shard.device_id, non_blocking=True
            )
            labels_b_tile = labels_b_cpu[shard.b_start:shard.b_end].cuda(
                shard.device_id, non_blocking=True
            )

        indices, categories, distances, count = _C.masked_hamming_ab_cuda_async(
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
            max_pairs_per_shard,
            r_dim,
            theta_dim,
            d0_dim,
            d1_dim,
        )

        return ShardKernelResult(
            shard=shard,
            indices=indices,
            categories=categories,
            distances=distances,
            count=count,
        )

    # Run shards with async kernel execution across devices
    shard_results = _run_shards_async(shards, launch_kernel)

    # Aggregate results
    if len(shard_results) == 0:
        # Return empty tensors
        return (
            torch.zeros((0, 2), dtype=torch.int32),
            torch.zeros((0,), dtype=torch.uint8),
            torch.zeros((0,), dtype=torch.float32),
            torch.tensor([0], dtype=torch.int32),
        )

    # Collect and concatenate results, respecting max_pairs limit
    all_indices = []
    all_categories = []
    all_distances = []
    total_count = 0

    for result in shard_results:
        remaining = max_pairs - total_count
        if remaining <= 0:
            break
        take_count = min(result.count, remaining)
        all_indices.append(result.indices[:take_count])
        all_categories.append(result.categories[:take_count])
        all_distances.append(result.distances[:take_count])
        total_count += take_count

    pair_indices = torch.cat(all_indices, dim=0)
    categories = torch.cat(all_categories, dim=0)
    distances = torch.cat(all_distances, dim=0)
    count = torch.tensor([total_count], dtype=torch.int32)

    # Apply stratified sampling if configured
    if sample_bins is not None:
        pair_indices, categories, distances, count = apply_stratified_sampling(
            pair_indices, categories, distances, sample_bins
        )

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
    host_index: int = 0,
    num_hosts: int = 1,
    sample_bins: SampleBinsType = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sharded version of masked_hamming_cuda for multi-GPU and multi-host datasets.

    Computes the lower triangle (i > j) by tiling and distributing across devices.
    Accepts either packed (int32) or unpacked (uint8) data - packing is done on GPU.

    For multi-host operation:
    - Each host should have the FULL data tensor (or be able to access all rows)
    - Set host_index to this host's index (0 to num_hosts-1)
    - Set num_hosts to total number of hosts
    - Each host will process only its assigned tiles
    - Results from all hosts should be aggregated by the caller

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
        host_index: Index of this host for multi-host operation (default: 0)
        num_hosts: Total number of hosts for multi-host operation (default: 1)

    Returns:
        Tuple of (pair_indices, categories, distances, count):
        - pair_indices: [N, 2] int32 - (row, col) indices of pairs (row > col)
        - categories: [N] uint8 - category codes
        - distances: [N] float32 - distance values
        - count: [1] int32 - actual number of pairs

    Example:
        Multi-GPU matching (automatically uses all available GPUs):

        >>> import torch
        >>> import cuda_iris_matcher as ih
        >>> # Large dataset on CPU
        >>> data = torch.randint(0, 2**31, (50000, 400), dtype=torch.int32)
        >>> mask = torch.full_like(data, 0x7FFFFFFF)
        >>> # Automatically distributes across GPUs
        >>> pairs, cats, dists, count = ih.masked_hamming_sharded(data, mask)
        >>> print(f"Using {ih.get_device_count()} GPUs, found {count.item()} pairs")

        Force tiling on single GPU (useful for memory-limited scenarios):

        >>> pairs, cats, dists, count = ih.masked_hamming_sharded(
        ...     data, mask, min_shards=4  # Split into at least 4 tiles
        ... )

        Multi-host distributed computation:

        >>> # On host 0 of 4:
        >>> result_0 = ih.masked_hamming_sharded(data, mask, host_index=0, num_hosts=4)
        >>> # On host 1 of 4:
        >>> result_1 = ih.masked_hamming_sharded(data, mask, host_index=1, num_hosts=4)
        >>> # ... aggregate results from all hosts
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

    # For single GPU without forced sharding, use non-sharded kernel directly
    if num_devices == 1 and min_shards <= 1:
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
        labels_gpu = None
        if labels is not None:
            labels_gpu = labels.cuda(primary_device) if not labels.is_cuda else labels
        pair_indices, categories, distances, count = _C.masked_hamming_cuda(
            data_gpu, mask_gpu, labels_gpu,
            match_threshold, non_match_threshold, is_similarity,
            include_flags, max_pairs, r_dim, theta_dim, d0_dim, d1_dim
        )
        # Apply stratified sampling if configured
        if sample_bins is not None:
            pair_indices, categories, distances, count = apply_stratified_sampling(
                pair_indices, categories, distances, sample_bins
            )
        return pair_indices, categories, distances, count

    # Compute shard configurations for lower triangle
    # For multi-host, compute shards as if all hosts' GPUs were available
    total_devices = num_devices * num_hosts
    all_shards = _compute_self_shard_configs(
        m, k_words, max_pairs, total_devices, min_shards * num_hosts, max_tile_size
    )

    # Filter to only shards assigned to this host
    shards = _filter_shards_for_host(all_shards, host_index, num_hosts, num_devices)

    # If no shards assigned to this host, return empty results
    if len(shards) == 0:
        return (
            torch.zeros((0, 2), dtype=torch.int32),
            torch.zeros((0,), dtype=torch.uint8),
            torch.zeros((0,), dtype=torch.float32),
            torch.tensor([0], dtype=torch.int32),
        )

    max_pairs_per_shard = max(max_pairs // max(1, len(all_shards)), 1000)

    # Labels stay on CPU (small)
    labels_cpu = labels.cpu() if labels is not None and labels.is_cuda else labels

    # Ensure data is on CPU for direct transfers to each GPU
    data_cpu = data.cpu() if data.is_cuda else data
    mask_cpu = mask.cpu() if mask.is_cuda else mask

    # Determine which row ranges each GPU needs
    device_ids = sorted(set(s.device_id for s in shards))
    device_ranges: dict[int, set[Tuple[int, int]]] = {d: set() for d in device_ids}
    for shard in shards:
        device_ranges[shard.device_id].add((shard.a_start, shard.a_end))
        device_ranges[shard.device_id].add((shard.b_start, shard.b_end))

    # Merge overlapping ranges and compute unique rows needed per device
    def merge_ranges(ranges: set[Tuple[int, int]]) -> Tuple[int, int]:
        """Merge ranges into a single contiguous range (min_start, max_end)."""
        if not ranges:
            return (0, 0)
        min_start = min(r[0] for r in ranges)
        max_end = max(r[1] for r in ranges)
        return (min_start, max_end)

    device_row_ranges = {d: merge_ranges(ranges) for d, ranges in device_ranges.items()}

    # Transfer data directly from CPU to each GPU (only needed rows) in parallel
    def transfer_to_device(device_id: int) -> Tuple[int, torch.Tensor, torch.Tensor, int]:
        """Transfer needed rows to a GPU, pack if needed, return (device_id, data, mask, offset)."""
        row_start, row_end = device_row_ranges[device_id]
        with torch.cuda.device(device_id):
            data_slice = data_cpu[row_start:row_end]
            mask_slice = mask_cpu[row_start:row_end]

            if data_is_packed:
                data_gpu = data_slice.cuda(device_id)
            else:
                data_gpu = _pack_on_device(data_slice, device_id, r_dim, theta_dim, d0_dim, d1_dim)

            if mask_is_packed:
                mask_gpu = mask_slice.cuda(device_id)
            else:
                mask_gpu = _pack_on_device(mask_slice, device_id, r_dim, theta_dim, d0_dim, d1_dim)

            torch.cuda.synchronize(device_id)
        return device_id, data_gpu, mask_gpu, row_start

    # Transfer to all GPUs in parallel
    device_data: dict[int, Tuple[torch.Tensor, torch.Tensor, int]] = {}
    with ThreadPoolExecutor(max_workers=len(device_ids)) as executor:
        futures = [executor.submit(transfer_to_device, d) for d in device_ids]
        for future in as_completed(futures):
            device_id, data_gpu, mask_gpu, offset = future.result()
            device_data[device_id] = (data_gpu, mask_gpu, offset)

    # Define kernel launch function (returns GPU tensors without sync)
    def launch_kernel(shard: ShardConfig) -> ShardKernelResult:
        data_gpu, mask_gpu, row_offset = device_data[shard.device_id]

        # Adjust indices relative to the data slice on this device
        local_a_start = shard.a_start - row_offset
        local_a_end = shard.a_end - row_offset
        local_b_start = shard.b_start - row_offset
        local_b_end = shard.b_end - row_offset

        data_a_tile = data_gpu[local_a_start:local_a_end]
        mask_a_tile = mask_gpu[local_a_start:local_a_end]
        data_b_tile = data_gpu[local_b_start:local_b_end]
        mask_b_tile = mask_gpu[local_b_start:local_b_end]

        labels_a_tile = None
        labels_b_tile = None
        if labels_cpu is not None:
            labels_a_tile = labels_cpu[shard.a_start:shard.a_end].cuda(
                shard.device_id, non_blocking=True
            )
            labels_b_tile = labels_cpu[shard.b_start:shard.b_end].cuda(
                shard.device_id, non_blocking=True
            )

        is_diagonal = shard.a_start == shard.b_start

        if is_diagonal:
            labels_tile = None
            if labels_cpu is not None:
                labels_tile = labels_cpu[shard.a_start:shard.a_end].cuda(
                    shard.device_id, non_blocking=True
                )

            indices, categories, distances, count = _C.masked_hamming_cuda_async(
                data_a_tile,
                mask_a_tile,
                labels_tile,
                match_threshold,
                non_match_threshold,
                is_similarity,
                include_flags,
                max_pairs_per_shard,
                r_dim,
                theta_dim,
                d0_dim,
                d1_dim,
            )
        else:
            indices, categories, distances, count = _C.masked_hamming_ab_cuda_async(
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
                max_pairs_per_shard,
                r_dim,
                theta_dim,
                d0_dim,
                d1_dim,
            )

        return ShardKernelResult(
            shard=shard,
            indices=indices,
            categories=categories,
            distances=distances,
            count=count,
        )

    # Run shards with async kernel execution across devices
    shard_results = _run_shards_async(shards, launch_kernel)

    # Aggregate results
    if len(shard_results) == 0:
        return (
            torch.zeros((0, 2), dtype=torch.int32),
            torch.zeros((0,), dtype=torch.uint8),
            torch.zeros((0,), dtype=torch.float32),
            torch.tensor([0], dtype=torch.int32),
        )

    # Collect and concatenate results, respecting max_pairs limit
    all_indices = []
    all_categories = []
    all_distances = []
    total_count = 0

    for result in shard_results:
        remaining = max_pairs - total_count
        if remaining <= 0:
            break
        take_count = min(result.count, remaining)
        all_indices.append(result.indices[:take_count])
        all_categories.append(result.categories[:take_count])
        all_distances.append(result.distances[:take_count])
        total_count += take_count

    pair_indices = torch.cat(all_indices, dim=0)
    categories = torch.cat(all_categories, dim=0)
    distances = torch.cat(all_distances, dim=0)
    count = torch.tensor([total_count], dtype=torch.int32)

    # Apply stratified sampling if configured
    if sample_bins is not None:
        pair_indices, categories, distances, count = apply_stratified_sampling(
            pair_indices, categories, distances, sample_bins
        )

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
        Pack a large dataset that doesn't fit on GPU:

        >>> import torch
        >>> import cuda_iris_matcher as ih
        >>> # 1 million iris codes on CPU (too large for GPU)
        >>> raw_codes = torch.randint(0, 2, (1000000, 16, 200, 2, 2), dtype=torch.uint8)
        >>> raw_masks = torch.ones_like(raw_codes)
        >>> # Pack in batches (auto-determines batch size based on GPU memory)
        >>> packed_codes = ih.pack_theta_major_batched(raw_codes)
        >>> packed_masks = ih.pack_theta_major_batched(raw_masks)
        >>> print(packed_codes.shape)  # torch.Size([1000000, 400])
        >>> # Now use with sharded matching
        >>> pairs, _, dists, count = ih.masked_hamming_sharded(packed_codes, packed_masks)
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
    host_index: int = 0,
    num_hosts: int = 1,
) -> List[ShardConfig]:
    """Get shard configuration info for planning (debugging/inspection).

    Args:
        m_a: Number of rows in A
        m_b: Number of rows in B
        k_words: Number of int32 words per row (default 400 for standard iris)
        max_pairs_per_shard: Max pairs per shard
        min_shards: Minimum number of shards
        max_tile_size: Maximum tile size
        host_index: Index of this host for multi-host operation (default: 0)
        num_hosts: Total number of hosts for multi-host operation (default: 1)

    Returns:
        List of ShardConfig objects for this host
    """
    num_devices = torch.cuda.device_count()
    total_devices = num_devices * num_hosts
    all_shards = _compute_shard_configs(
        m_a, m_b, k_words, max_pairs_per_shard, total_devices, min_shards * num_hosts, max_tile_size
    )
    return _filter_shards_for_host(all_shards, host_index, num_hosts, num_devices)


def get_self_shard_info(
    m: int,
    k_words: int = 400,
    max_pairs_per_shard: int = 100000,
    min_shards: int = 1,
    max_tile_size: Optional[int] = None,
    host_index: int = 0,
    num_hosts: int = 1,
) -> List[ShardConfig]:
    """Get shard configuration info for self-comparison (lower triangle).

    Args:
        m: Number of rows
        k_words: Number of int32 words per row (default 400 for standard iris)
        max_pairs_per_shard: Max pairs per shard
        min_shards: Minimum number of shards
        max_tile_size: Maximum tile size
        host_index: Index of this host for multi-host operation (default: 0)
        num_hosts: Total number of hosts for multi-host operation (default: 1)

    Returns:
        List of ShardConfig objects for this host
    """
    num_devices = torch.cuda.device_count()
    total_devices = num_devices * num_hosts
    all_shards = _compute_self_shard_configs(
        m, k_words, max_pairs_per_shard, total_devices, min_shards * num_hosts, max_tile_size
    )
    return _filter_shards_for_host(all_shards, host_index, num_hosts, num_devices)


def get_total_shards(
    m_a: int,
    m_b: Optional[int] = None,
    k_words: int = 400,
    max_pairs_per_shard: int = 100000,
    min_shards: int = 1,
    max_tile_size: Optional[int] = None,
    num_hosts: int = 1,
) -> int:
    """Get total number of shards across all hosts.

    Useful for determining how work will be distributed before launching.

    Args:
        m_a: Number of rows in A (or total rows for self-comparison)
        m_b: Number of rows in B (None for self-comparison)
        k_words: Number of int32 words per row (default 400 for standard iris)
        max_pairs_per_shard: Max pairs per shard
        min_shards: Minimum number of shards per host
        max_tile_size: Maximum tile size
        num_hosts: Total number of hosts

    Returns:
        Total number of shards
    """
    num_devices = torch.cuda.device_count()
    total_devices = num_devices * num_hosts

    if m_b is None:
        # Self-comparison
        shards = _compute_self_shard_configs(
            m_a, k_words, max_pairs_per_shard, total_devices, min_shards * num_hosts, max_tile_size
        )
    else:
        # A vs B comparison
        shards = _compute_shard_configs(
            m_a, m_b, k_words, max_pairs_per_shard, total_devices, min_shards * num_hosts, max_tile_size
        )
    return len(shards)


# =============================================================================
# Dot Product Sharding
# =============================================================================


def _compute_dot_shard_configs(
    m_a: int,
    m_b: int,
    vec_dim: int,
    max_pairs_per_shard: int,
    num_devices: int,
    min_shards: int = 1,
    max_tile_size: Optional[int] = None,
) -> List[ShardConfig]:
    """Compute shard configurations for A vs B dot product comparison.

    Similar to _compute_shard_configs but adapted for f16 dot product memory layout.
    """
    # Estimate memory: data is f16 (2 bytes per element)
    if max_tile_size is None:
        if num_devices > 0:
            available_mem = _get_available_memory(0)
            max_tile_size = min(m_a, m_b)
            while max_tile_size > 64:
                # Memory: tile_a (f16) + tile_b (f16) + output buffers
                mem_a = max_tile_size * vec_dim * 2  # f16
                mem_b = max_tile_size * vec_dim * 2  # f16
                mem_out = max_pairs_per_shard * 13  # indices + cat + score
                mem_needed = mem_a + mem_b + mem_out
                if mem_needed <= available_mem * 0.7:
                    break
                max_tile_size //= 2
            max_tile_size = max(64, max_tile_size)
        else:
            max_tile_size = 4096

    n_tiles_a = max(1, math.ceil(m_a / max_tile_size))
    n_tiles_b = max(1, math.ceil(m_b / max_tile_size))
    total_tiles = n_tiles_a * n_tiles_b

    if total_tiles < min_shards:
        factor = math.ceil(math.sqrt(min_shards / total_tiles))
        n_tiles_a = max(n_tiles_a, min(n_tiles_a * factor, m_a))
        n_tiles_b = max(n_tiles_b, min(n_tiles_b * factor, m_b))

    tile_size_a = math.ceil(m_a / n_tiles_a)
    tile_size_b = math.ceil(m_b / n_tiles_b)

    tiles_with_work: List[Tuple[int, int, int, int, int]] = []
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
            work = (a_end - a_start) * (b_end - b_start)
            tiles_with_work.append((a_start, a_end, b_start, b_end, work))

    if num_devices <= 1:
        return [
            ShardConfig(device_id=0, a_start=t[0], a_end=t[1], b_start=t[2], b_end=t[3], global_shard_id=i)
            for i, t in enumerate(tiles_with_work)
        ]

    # Greedy load balancing
    tiles_with_work.sort(key=lambda x: x[4], reverse=True)
    device_work = [0] * num_devices
    tile_assignments: List[Tuple[int, int, int, int, int, int, int]] = []

    for idx, (a_start, a_end, b_start, b_end, work) in enumerate(tiles_with_work):
        min_device = min(range(num_devices), key=lambda d: device_work[d])
        device_work[min_device] += work
        tile_assignments.append((a_start, a_end, b_start, b_end, work, min_device, idx))

    tile_assignments.sort(key=lambda x: (x[5], x[0], x[2]))
    return [
        ShardConfig(device_id=t[5], a_start=t[0], a_end=t[1], b_start=t[2], b_end=t[3], global_shard_id=t[6])
        for t in tile_assignments
    ]


def _compute_dot_self_shard_configs(
    m: int,
    vec_dim: int,
    max_pairs_per_shard: int,
    num_devices: int,
    min_shards: int = 1,
    max_tile_size: Optional[int] = None,
) -> List[ShardConfig]:
    """Compute shard configurations for self-comparison dot product (lower triangle only)."""
    if max_tile_size is None:
        if num_devices > 0:
            available_mem = _get_available_memory(0)
            max_tile_size = m
            while max_tile_size > 64:
                mem_data = max_tile_size * vec_dim * 2  # f16 data
                mem_out = max_pairs_per_shard * 13
                mem_needed = mem_data * 2 + mem_out  # A tile + B tile + output
                if mem_needed <= available_mem * 0.7:
                    break
                max_tile_size //= 2
            max_tile_size = max(64, max_tile_size)
        else:
            max_tile_size = 4096

    n_tiles = max(1, math.ceil(m / max_tile_size))
    lower_tri_tiles = n_tiles * (n_tiles + 1) // 2

    if lower_tri_tiles < min_shards:
        n_tiles = max(n_tiles, math.ceil((-1 + math.sqrt(1 + 8 * min_shards)) / 2))
        n_tiles = min(n_tiles, m)

    if num_devices > 1:
        min_tiles_for_balance = 4 * num_devices
        while n_tiles * (n_tiles + 1) // 2 < min_tiles_for_balance and n_tiles < m:
            n_tiles += 1

    tile_size = math.ceil(m / n_tiles)

    tiles_with_work: List[Tuple[int, int, int, int, int]] = []
    for i in range(n_tiles):
        a_start = i * tile_size
        a_end = min((i + 1) * tile_size, m)
        if a_start >= m:
            break
        for j in range(i + 1):
            b_start = j * tile_size
            b_end = min((j + 1) * tile_size, m)
            if b_start >= m:
                break
            if a_end > b_start:
                a_size = a_end - a_start
                b_size = b_end - b_start
                if a_start == b_start:
                    work = a_size * (a_size - 1) // 2
                else:
                    work = a_size * b_size
                tiles_with_work.append((a_start, a_end, b_start, b_end, work))

    if num_devices <= 1:
        return [
            ShardConfig(device_id=0, a_start=t[0], a_end=t[1], b_start=t[2], b_end=t[3], global_shard_id=i)
            for i, t in enumerate(tiles_with_work)
        ]

    tiles_with_work.sort(key=lambda x: x[4], reverse=True)
    device_work = [0] * num_devices
    tile_assignments: List[Tuple[int, int, int, int, int, int, int]] = []

    for idx, (a_start, a_end, b_start, b_end, work) in enumerate(tiles_with_work):
        min_device = min(range(num_devices), key=lambda d: device_work[d])
        device_work[min_device] += work
        tile_assignments.append((a_start, a_end, b_start, b_end, work, min_device, idx))

    tile_assignments.sort(key=lambda x: (x[5], x[0], x[2]))
    return [
        ShardConfig(device_id=t[5], a_start=t[0], a_end=t[1], b_start=t[2], b_end=t[3], global_shard_id=t[6])
        for t in tile_assignments
    ]


def dot_product_ab_sharded(
    data_a: torch.Tensor,
    data_b: torch.Tensor,
    labels_a: Optional[torch.Tensor] = None,
    labels_b: Optional[torch.Tensor] = None,
    match_threshold: float = 0.5,
    non_match_threshold: float = 0.5,
    is_similarity: bool = True,
    include_flags: int = INCLUDE_ALL,
    max_pairs: int = 1_000_000,
    vec_dim: int = DEFAULT_DOT_VEC_DIM,
    min_shards: int = 1,
    max_tile_size: Optional[int] = None,
    host_index: int = 0,
    num_hosts: int = 1,
    sample_bins: SampleBinsType = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sharded version of dot_product_ab_cuda for multi-GPU and multi-host datasets.

    Automatically distributes computation across all available CUDA devices.

    Args:
        data_a: CUDA or CPU float16 tensor of shape [M_A, vec_dim]
        data_b: CUDA or CPU float16 tensor of shape [M_B, vec_dim]
        labels_a: Optional int32 tensor of shape [M_A] with identity labels
        labels_b: Optional int32 tensor of shape [M_B] with identity labels
        match_threshold: Threshold for match classification (default: 0.5)
        non_match_threshold: Threshold for non-match classification (default: 0.5)
        is_similarity: If True (default), higher values = more similar
        include_flags: Bitmask of categories to include (default: INCLUDE_ALL)
        max_pairs: Maximum total pairs to return (default: 1,000,000)
        vec_dim: Vector dimension (default: 512)
        min_shards: Minimum number of shards (useful for testing)
        max_tile_size: Maximum rows per tile (None = auto)
        host_index: Index of this host for multi-host operation
        num_hosts: Total number of hosts

    Returns:
        Tuple of (pair_indices, categories, scores, count)
    """
    m_a = data_a.size(0)
    m_b = data_b.size(0)
    actual_vec_dim = data_a.size(1)
    if vec_dim == DEFAULT_DOT_VEC_DIM and actual_vec_dim != vec_dim:
        vec_dim = actual_vec_dim

    num_devices = torch.cuda.device_count()
    if num_devices == 0:
        raise RuntimeError("No CUDA devices available")

    # For single GPU without forced sharding, use non-sharded kernel
    if num_devices == 1 and min_shards <= 1 and num_hosts == 1:
        data_a_gpu = data_a.cuda(0) if not data_a.is_cuda else data_a
        data_b_gpu = data_b.cuda(0) if not data_b.is_cuda else data_b
        labels_a_gpu = labels_a.cuda(0) if labels_a is not None and not labels_a.is_cuda else labels_a
        labels_b_gpu = labels_b.cuda(0) if labels_b is not None and not labels_b.is_cuda else labels_b
        pair_indices, categories, scores, count = _C.dot_product_ab_cuda(
            data_a_gpu, data_b_gpu, labels_a_gpu, labels_b_gpu,
            match_threshold, non_match_threshold, is_similarity,
            include_flags, max_pairs, vec_dim
        )
        # Apply stratified sampling if configured
        if sample_bins is not None:
            pair_indices, categories, scores, count = apply_stratified_sampling(
                pair_indices, categories, scores, sample_bins
            )
        return pair_indices, categories, scores, count

    # Compute shard configurations
    total_devices = num_devices * num_hosts
    all_shards = _compute_dot_shard_configs(
        m_a, m_b, vec_dim, max_pairs, total_devices, min_shards * num_hosts, max_tile_size
    )

    shards = _filter_shards_for_host(all_shards, host_index, num_hosts, num_devices)

    if len(shards) == 0:
        return (
            torch.zeros((0, 2), dtype=torch.int32),
            torch.zeros((0,), dtype=torch.uint8),
            torch.zeros((0,), dtype=torch.float32),
            torch.tensor([0], dtype=torch.int32),
        )

    # For sparse output (max_pairs << total pairs), limit per-shard output
    total_pairs = m_a * m_b
    if max_pairs < total_pairs // 2:
        # Sparse output mode: distribute max_pairs across shards with some headroom
        max_pairs_per_shard = max(max_pairs * 2 // max(1, len(all_shards)), 10000)
    else:
        # Dense output mode: size for full tile output
        max_tile_pairs = 0
        for shard in all_shards:
            a_size = shard.a_end - shard.a_start
            b_size = shard.b_end - shard.b_start
            tile_pairs = a_size * b_size
            max_tile_pairs = max(max_tile_pairs, tile_pairs)
        max_pairs_per_shard = max(max_tile_pairs, max_pairs // max(1, len(all_shards)), 1000)

    # Ensure data is on CPU for transfers
    data_a_cpu = data_a.cpu() if data_a.is_cuda else data_a
    data_b_cpu = data_b.cpu() if data_b.is_cuda else data_b
    labels_a_cpu = labels_a.cpu() if labels_a is not None and labels_a.is_cuda else labels_a
    labels_b_cpu = labels_b.cpu() if labels_b is not None and labels_b.is_cuda else labels_b

    # Determine which row ranges each GPU needs
    device_ids = sorted(set(s.device_id for s in shards))
    device_a_ranges: dict[int, set[Tuple[int, int]]] = {d: set() for d in device_ids}
    device_b_ranges: dict[int, set[Tuple[int, int]]] = {d: set() for d in device_ids}
    for shard in shards:
        device_a_ranges[shard.device_id].add((shard.a_start, shard.a_end))
        device_b_ranges[shard.device_id].add((shard.b_start, shard.b_end))

    def merge_ranges(ranges: set[Tuple[int, int]]) -> Tuple[int, int]:
        if not ranges:
            return (0, 0)
        return (min(r[0] for r in ranges), max(r[1] for r in ranges))

    device_a_row_ranges = {d: merge_ranges(ranges) for d, ranges in device_a_ranges.items()}
    device_b_row_ranges = {d: merge_ranges(ranges) for d, ranges in device_b_ranges.items()}

    # Pre-transfer data to each GPU
    def transfer_to_device(device_id: int) -> Tuple[int, torch.Tensor, torch.Tensor, int, int]:
        a_start, a_end = device_a_row_ranges[device_id]
        b_start, b_end = device_b_row_ranges[device_id]
        with torch.cuda.device(device_id):
            data_a_gpu = data_a_cpu[a_start:a_end].cuda(device_id)
            data_b_gpu = data_b_cpu[b_start:b_end].cuda(device_id)
            torch.cuda.synchronize(device_id)
        return device_id, data_a_gpu, data_b_gpu, a_start, b_start

    device_data: dict[int, Tuple[torch.Tensor, torch.Tensor, int, int]] = {}
    with ThreadPoolExecutor(max_workers=len(device_ids)) as executor:
        futures = [executor.submit(transfer_to_device, d) for d in device_ids]
        for future in as_completed(futures):
            device_id, data_a_gpu, data_b_gpu, a_offset, b_offset = future.result()
            device_data[device_id] = (data_a_gpu, data_b_gpu, a_offset, b_offset)

    def launch_kernel(shard: ShardConfig) -> ShardKernelResult:
        data_a_gpu, data_b_gpu, a_offset, b_offset = device_data[shard.device_id]

        # Get local slices from pre-loaded data
        local_a_start = shard.a_start - a_offset
        local_a_end = shard.a_end - a_offset
        local_b_start = shard.b_start - b_offset
        local_b_end = shard.b_end - b_offset

        data_a_tile = data_a_gpu[local_a_start:local_a_end]
        data_b_tile = data_b_gpu[local_b_start:local_b_end]

        labels_a_tile = None
        labels_b_tile = None
        if labels_a_cpu is not None and labels_b_cpu is not None:
            labels_a_tile = labels_a_cpu[shard.a_start:shard.a_end].cuda(shard.device_id, non_blocking=True)
            labels_b_tile = labels_b_cpu[shard.b_start:shard.b_end].cuda(shard.device_id, non_blocking=True)

        indices, categories, scores, count = _C.dot_product_ab_cuda_async(
            data_a_tile, data_b_tile,
            labels_a_tile, labels_b_tile,
            match_threshold, non_match_threshold,
            is_similarity, include_flags,
            max_pairs_per_shard, vec_dim
        )

        return ShardKernelResult(
            shard=shard,
            indices=indices,
            categories=categories,
            distances=scores,
            count=count,
        )

    shard_results = _run_shards_async(shards, launch_kernel)

    if len(shard_results) == 0:
        return (
            torch.zeros((0, 2), dtype=torch.int32),
            torch.zeros((0,), dtype=torch.uint8),
            torch.zeros((0,), dtype=torch.float32),
            torch.tensor([0], dtype=torch.int32),
        )

    all_indices = []
    all_categories = []
    all_scores = []
    total_count = 0

    for result in shard_results:
        remaining = max_pairs - total_count
        if remaining <= 0:
            break
        take_count = min(result.count, remaining)
        all_indices.append(result.indices[:take_count])
        all_categories.append(result.categories[:take_count])
        all_scores.append(result.distances[:take_count])
        total_count += take_count

    pair_indices = torch.cat(all_indices, dim=0)
    categories = torch.cat(all_categories, dim=0)
    scores = torch.cat(all_scores, dim=0)
    count = torch.tensor([total_count], dtype=torch.int32)

    # Apply stratified sampling if configured
    if sample_bins is not None:
        pair_indices, categories, scores, count = apply_stratified_sampling(
            pair_indices, categories, scores, sample_bins
        )

    return pair_indices, categories, scores, count


def dot_product_sharded(
    data: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    match_threshold: float = 0.5,
    non_match_threshold: float = 0.5,
    is_similarity: bool = True,
    include_flags: int = INCLUDE_ALL,
    max_pairs: int = 1_000_000,
    vec_dim: int = DEFAULT_DOT_VEC_DIM,
    min_shards: int = 1,
    max_tile_size: Optional[int] = None,
    host_index: int = 0,
    num_hosts: int = 1,
    sample_bins: SampleBinsType = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sharded version of dot_product_cuda for multi-GPU and multi-host datasets.

    Computes the lower triangle (i > j) by tiling and distributing across devices.

    Args:
        data: CUDA or CPU float16 tensor of shape [M, vec_dim]
        labels: Optional int32 tensor of shape [M] with identity labels
        match_threshold: Threshold for match classification (default: 0.5)
        non_match_threshold: Threshold for non-match classification (default: 0.5)
        is_similarity: If True (default), higher values = more similar
        include_flags: Bitmask of categories to include (default: INCLUDE_ALL)
        max_pairs: Maximum total pairs to return (default: 1,000,000)
        vec_dim: Vector dimension (default: 512)
        min_shards: Minimum number of shards (useful for testing)
        max_tile_size: Maximum rows per tile (None = auto)
        host_index: Index of this host for multi-host operation
        num_hosts: Total number of hosts

    Returns:
        Tuple of (pair_indices, categories, scores, count)
    """
    m = data.size(0)
    actual_vec_dim = data.size(1)
    if vec_dim == DEFAULT_DOT_VEC_DIM and actual_vec_dim != vec_dim:
        vec_dim = actual_vec_dim

    num_devices = torch.cuda.device_count()
    if num_devices == 0:
        raise RuntimeError("No CUDA devices available")

    # For single GPU without forced sharding, use non-sharded kernel
    if num_devices == 1 and min_shards <= 1 and num_hosts == 1:
        data_gpu = data.cuda(0) if not data.is_cuda else data
        labels_gpu = labels.cuda(0) if labels is not None and not labels.is_cuda else labels
        pair_indices, categories, scores, count = _C.dot_product_cuda(
            data_gpu, labels_gpu,
            match_threshold, non_match_threshold, is_similarity,
            include_flags, max_pairs, vec_dim
        )
        # Apply stratified sampling if configured
        if sample_bins is not None:
            pair_indices, categories, scores, count = apply_stratified_sampling(
                pair_indices, categories, scores, sample_bins
            )
        return pair_indices, categories, scores, count

    # Compute shard configurations for lower triangle
    total_devices = num_devices * num_hosts
    all_shards = _compute_dot_self_shard_configs(
        m, vec_dim, max_pairs, total_devices, min_shards * num_hosts, max_tile_size
    )

    shards = _filter_shards_for_host(all_shards, host_index, num_hosts, num_devices)

    if len(shards) == 0:
        return (
            torch.zeros((0, 2), dtype=torch.int32),
            torch.zeros((0,), dtype=torch.uint8),
            torch.zeros((0,), dtype=torch.float32),
            torch.tensor([0], dtype=torch.int32),
        )

    # For sparse output (max_pairs << total pairs), limit per-shard output
    # to avoid computing and transferring huge amounts of data we'll discard
    total_pairs = m * (m - 1) // 2
    if max_pairs < total_pairs // 2:
        # Sparse output mode: distribute max_pairs across shards with some headroom
        max_pairs_per_shard = max(max_pairs * 2 // max(1, len(all_shards)), 10000)
    else:
        # Dense output mode: size for full tile output
        max_tile_pairs = 0
        for shard in all_shards:
            a_size = shard.a_end - shard.a_start
            b_size = shard.b_end - shard.b_start
            if shard.a_start == shard.b_start:
                tile_pairs = a_size * (a_size - 1) // 2
            else:
                tile_pairs = a_size * b_size
            max_tile_pairs = max(max_tile_pairs, tile_pairs)
        max_pairs_per_shard = max(max_tile_pairs, max_pairs // max(1, len(all_shards)), 1000)

    data_cpu = data.cpu() if data.is_cuda else data
    labels_cpu = labels.cpu() if labels is not None and labels.is_cuda else labels

    # Determine which row ranges each GPU needs
    device_ids = sorted(set(s.device_id for s in shards))
    device_ranges: dict[int, set[Tuple[int, int]]] = {d: set() for d in device_ids}
    for shard in shards:
        device_ranges[shard.device_id].add((shard.a_start, shard.a_end))
        device_ranges[shard.device_id].add((shard.b_start, shard.b_end))

    def merge_ranges(ranges: set[Tuple[int, int]]) -> Tuple[int, int]:
        if not ranges:
            return (0, 0)
        return (min(r[0] for r in ranges), max(r[1] for r in ranges))

    device_row_ranges = {d: merge_ranges(ranges) for d, ranges in device_ranges.items()}

    # Pre-transfer data to each GPU
    def transfer_to_device(device_id: int) -> Tuple[int, torch.Tensor, int]:
        row_start, row_end = device_row_ranges[device_id]
        with torch.cuda.device(device_id):
            data_gpu = data_cpu[row_start:row_end].cuda(device_id)
            torch.cuda.synchronize(device_id)
        return device_id, data_gpu, row_start

    device_data: dict[int, Tuple[torch.Tensor, int]] = {}
    with ThreadPoolExecutor(max_workers=len(device_ids)) as executor:
        futures = [executor.submit(transfer_to_device, d) for d in device_ids]
        for future in as_completed(futures):
            device_id, data_gpu, row_offset = future.result()
            device_data[device_id] = (data_gpu, row_offset)

    def launch_kernel(shard: ShardConfig) -> ShardKernelResult:
        data_gpu, row_offset = device_data[shard.device_id]

        # Get local slices from pre-loaded data
        local_a_start = shard.a_start - row_offset
        local_a_end = shard.a_end - row_offset
        local_b_start = shard.b_start - row_offset
        local_b_end = shard.b_end - row_offset

        data_a_tile = data_gpu[local_a_start:local_a_end]
        data_b_tile = data_gpu[local_b_start:local_b_end]

        labels_a_tile = None
        labels_b_tile = None
        if labels_cpu is not None:
            labels_a_tile = labels_cpu[shard.a_start:shard.a_end].cuda(shard.device_id, non_blocking=True)
            labels_b_tile = labels_cpu[shard.b_start:shard.b_end].cuda(shard.device_id, non_blocking=True)

        indices, categories, scores, count = _C.dot_product_ab_cuda_async(
            data_a_tile, data_b_tile,
            labels_a_tile, labels_b_tile,
            match_threshold, non_match_threshold,
            is_similarity, include_flags,
            max_pairs_per_shard, vec_dim
        )

        # For diagonal tiles, filter to lower triangle using LOCAL indices
        if shard.a_start == shard.b_start:
            valid_mask = indices[:, 0] > indices[:, 1]
            indices = indices[valid_mask]
            categories = categories[valid_mask]
            scores = scores[valid_mask]
            count = torch.tensor([indices.size(0)], dtype=torch.int32, device=indices.device)

        return ShardKernelResult(
            shard=shard,
            indices=indices,
            categories=categories,
            distances=scores,
            count=count,
        )

    shard_results = _run_shards_async(shards, launch_kernel)

    if len(shard_results) == 0:
        return (
            torch.zeros((0, 2), dtype=torch.int32),
            torch.zeros((0,), dtype=torch.uint8),
            torch.zeros((0,), dtype=torch.float32),
            torch.tensor([0], dtype=torch.int32),
        )

    all_indices = []
    all_categories = []
    all_scores = []
    total_count = 0

    for result in shard_results:
        remaining = max_pairs - total_count
        if remaining <= 0:
            break
        take_count = min(result.count, remaining)
        all_indices.append(result.indices[:take_count])
        all_categories.append(result.categories[:take_count])
        all_scores.append(result.distances[:take_count])
        total_count += take_count

    pair_indices = torch.cat(all_indices, dim=0)
    categories = torch.cat(all_categories, dim=0)
    scores = torch.cat(all_scores, dim=0)
    count = torch.tensor([total_count], dtype=torch.int32)

    # Apply stratified sampling if configured
    if sample_bins is not None:
        pair_indices, categories, scores, count = apply_stratified_sampling(
            pair_indices, categories, scores, sample_bins
        )

    return pair_indices, categories, scores, count

