# cuda_iris_matcher API Reference

Auto-generated API documentation.

## Functions

### `masked_hamming_cuda`

```python
masked_hamming_cuda(data: 'torch.Tensor', mask: 'torch.Tensor', labels: 'Optional[torch.Tensor]' = None, match_threshold: 'float' = 0.35, non_match_threshold: 'float' = 0.35, is_similarity: 'bool' = False, include_flags: 'int' = 15, max_pairs: 'int' = 1000000, dims: 'Optional[Tuple[int, int, int, int]]' = None, r_dim: 'int' = 16, theta_dim: 'int' = 200, d0_dim: 'int' = 2, d1_dim: 'int' = 2) -> 'Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]'
```

Compute minimum fractional hamming distance for all pairs within a single set.
Only the lower triangle (i > j) is computed.

Accepts either packed or unpacked data - packing is done on GPU automatically.

Args:
    data: Tensor of shape [M, k_words] int32 (packed) OR
          [M, r_dim, theta_dim, d0_dim, d1_dim] uint8 (unpacked).
          If unpacked, will be packed on GPU automatically.
    mask: Same shape/dtype as data
    labels: Optional CUDA int32 tensor of shape [M] with identity labels.
            If None, no classification is performed and all pairs are returned.
    match_threshold: Threshold for match classification (default: 0.35)
    non_match_threshold: Threshold for non-match classification (default: 0.35)
    is_similarity: If True, higher values = more similar (use >= for match).
                   If False (default), lower values = more similar (use <= for match).
    include_flags: Bitmask of categories to include (default: INCLUDE_ALL)
                   Use INCLUDE_TM | INCLUDE_FM | ... to combine flags.
    max_pairs: Maximum number of pairs to return (default: 1,000,000)
    dims: Optional tuple (r_dim, theta_dim, d0_dim, d1_dim). If provided, overrides
          individual dimension parameters.
    r_dim: Radial dimension of iris code (default: 16)
    theta_dim: Angular dimension of iris code (default: 200)
    d0_dim: First inner dimension (default: 2)
    d1_dim: Second inner dimension (default: 2)

Returns:
    Tuple of (pair_indices, categories, distances, count):
    - pair_indices: [N, 2] int32 - (row, col) indices of pairs
    - categories: [N] uint8 - category codes (0=TM, 1=FM, 2=FNM, 3=TNM, 255=unclassified)
    - distances: [N] float32 - distance values
    - count: [1] int32 - actual number of pairs (N == len(pair_indices))

Note:
    The returned tensors are pre-sliced to contain only the valid entries.
    Synchronization is handled internally.

Example:
    Basic usage with packed data:

    >>> import torch
    >>> import cuda_iris_matcher as ih
    >>> # Create packed iris codes [M, 400] and masks
    >>> data = torch.randint(0, 2**31, (100, 400), dtype=torch.int32, device="cuda")
    >>> mask = torch.full((100, 400), 0x7FFFFFFF, dtype=torch.int32, device="cuda")
    >>> # Compute all pairwise distances
    >>> pairs, cats, dists, count = ih.masked_hamming_cuda(data, mask)
    >>> print(f"Found {count.item()} pairs")

    With identity labels for classification:

    >>> labels = torch.arange(100, dtype=torch.int32, device="cuda")
    >>> pairs, cats, dists, count = ih.masked_hamming_cuda(
    ...     data, mask, labels=labels,
    ...     match_threshold=0.35,
    ...     include_flags=ih.INCLUDE_TM | ih.INCLUDE_FM  # Only matches
    ... )
    >>> true_matches = pairs[cats == ih.CATEGORY_TRUE_MATCH]

---

### `masked_hamming_ab_cuda`

```python
masked_hamming_ab_cuda(data_a: 'torch.Tensor', mask_a: 'torch.Tensor', data_b: 'torch.Tensor', mask_b: 'torch.Tensor', labels_a: 'Optional[torch.Tensor]' = None, labels_b: 'Optional[torch.Tensor]' = None, match_threshold: 'float' = 0.35, non_match_threshold: 'float' = 0.35, is_similarity: 'bool' = False, include_flags: 'int' = 15, max_pairs: 'int' = 1000000, dims: 'Optional[Tuple[int, int, int, int]]' = None, r_dim: 'int' = 16, theta_dim: 'int' = 200, d0_dim: 'int' = 2, d1_dim: 'int' = 2) -> 'Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]'
```

Compute minimum fractional hamming distance between two different sets A and B.
Computes the full M_A x M_B matrix (not just lower triangle).

Accepts either packed or unpacked data - packing is done on GPU automatically.

Args:
    data_a: Tensor of shape [M_A, k_words] int32 (packed) OR
            [M_A, r_dim, theta_dim, d0_dim, d1_dim] uint8 (unpacked)
    mask_a: Same shape/dtype as data_a
    data_b: Tensor of shape [M_B, k_words] int32 (packed) OR
            [M_B, r_dim, theta_dim, d0_dim, d1_dim] uint8 (unpacked)
    mask_b: Same shape/dtype as data_b
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
    dims: Optional tuple (r_dim, theta_dim, d0_dim, d1_dim). If provided, overrides
          individual dimension parameters.
    r_dim: Radial dimension of iris code (default: 16)
    theta_dim: Angular dimension of iris code (default: 200)
    d0_dim: First inner dimension (default: 2)
    d1_dim: Second inner dimension (default: 2)

Returns:
    Tuple of (pair_indices, categories, distances, count):
    - pair_indices: [N, 2] int32 - (row, col) indices of pairs
    - categories: [N] uint8 - category codes (0=TM, 1=FM, 2=FNM, 3=TNM, 255=unclassified)
    - distances: [N] float32 - distance values
    - count: [1] int32 - actual number of pairs (N == len(pair_indices))

Note:
    The returned tensors are pre-sliced to contain only the valid entries.
    Synchronization is handled internally.

Example:
    Compare a gallery set against probe samples:

    >>> import torch
    >>> import cuda_iris_matcher as ih
    >>> # Gallery: 10000 enrolled iris codes
    >>> gallery = torch.randint(0, 2**31, (10000, 400), dtype=torch.int32, device="cuda")
    >>> gallery_mask = torch.full_like(gallery, 0x7FFFFFFF)
    >>> # Probe: 50 query iris codes
    >>> probe = torch.randint(0, 2**31, (50, 400), dtype=torch.int32, device="cuda")
    >>> probe_mask = torch.full_like(probe, 0x7FFFFFFF)
    >>> # Find all matches
    >>> pairs, _, dists, count = ih.masked_hamming_ab_cuda(
    ...     gallery, gallery_mask, probe, probe_mask,
    ...     match_threshold=0.35
    ... )
    >>> # pairs[:, 0] = gallery index, pairs[:, 1] = probe index
    >>> print(f"Found {count.item()} comparisons")

---

### `pack_theta_major`

```python
pack_theta_major_cuda(bits: 'torch.Tensor', dims: 'Optional[Tuple[int, int, int, int]]' = None, r_dim: 'int' = 16, theta_dim: 'int' = 200, d0_dim: 'int' = 2, d1_dim: 'int' = 2) -> 'torch.Tensor'
```

Pack iris code bits into theta-major int32 words using CUDA.

Args:
    bits: CUDA uint8 tensor of shape (M, r_dim, theta_dim, d0_dim, d1_dim) with values in {0, 1}.
          Modified in-place; do not use after this call.
    dims: Optional tuple (r_dim, theta_dim, d0_dim, d1_dim). If provided, overrides
          individual dimension parameters.
    r_dim: Radial dimension of iris code (default: 16)
    theta_dim: Angular dimension of iris code (default: 200)
    d0_dim: First inner dimension (default: 2)
    d1_dim: Second inner dimension (default: 2)

Returns:
    Packed int32 tensor of shape (M, k_words) where k_words = r_dim * theta_dim * d0_dim * d1_dim / 32.
    Shares storage with input (no additional memory allocation).

Constraints:
    - r_dim * d0_dim * d1_dim must be divisible by 32 (for whole-word theta shifts)
    - r_dim * theta_dim * d0_dim * d1_dim must be divisible by 256 (TensorCore alignment)

Example:
    Pack raw iris codes for efficient matching:

    >>> import torch
    >>> import cuda_iris_matcher as ih
    >>> # Raw binary iris codes from feature extractor
    >>> raw_codes = torch.randint(0, 2, (1000, 16, 200, 2, 2), dtype=torch.uint8, device="cuda")
    >>> raw_masks = torch.ones_like(raw_codes)
    >>> # Pack for efficient GPU matching (clone since it's in-place)
    >>> packed_codes = ih.pack_theta_major(raw_codes.clone())
    >>> packed_masks = ih.pack_theta_major(raw_masks.clone())
    >>> print(packed_codes.shape)  # torch.Size([1000, 400])
    >>> print(packed_codes.dtype)  # torch.int32

---

### `repack_to_theta_major`

```python
repack_to_theta_major_cuda(input: 'torch.Tensor', dims: 'Optional[Tuple[int, int, int, int]]' = None, r_dim: 'int' = 16, theta_dim: 'int' = 200, d0_dim: 'int' = 2, d1_dim: 'int' = 2) -> 'torch.Tensor'
```

Repack int32 words from r-major to theta-major order using CUDA.

Args:
    input: CUDA int32 tensor of shape (M, k_words) packed in r-major order.
    dims: Optional tuple (r_dim, theta_dim, d0_dim, d1_dim). If provided, overrides
          individual dimension parameters.
    r_dim: Radial dimension of iris code (default: 16)
    theta_dim: Angular dimension of iris code (default: 200)
    d0_dim: First inner dimension (default: 2)
    d1_dim: Second inner dimension (default: 2)

Returns:
    New CUDA int32 tensor of shape (M, k_words) packed in theta-major order.

---

### `dot_product_cuda`

```python
dot_product_cuda(data: 'torch.Tensor', labels: 'Optional[torch.Tensor]' = None, match_threshold: 'float' = 0.5, non_match_threshold: 'float' = 0.5, is_similarity: 'bool' = True, include_flags: 'int' = 15, max_pairs: 'int' = 1000000, vec_dim: 'int' = 512) -> 'Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]'
```

Compute dot product similarity for all pairs within a single set.
Only the lower triangle (i > j) is computed.

Uses f16 (half precision) vectors and tensor cores for fast computation.

Args:
    data: CUDA float16 tensor of shape [M, vec_dim] containing feature vectors.
    labels: Optional CUDA int32 tensor of shape [M] with identity labels.
            If None, no classification is performed and all pairs are returned.
    match_threshold: Threshold for match classification (default: 0.5)
                    For similarity metrics, pairs with score >= threshold are matches.
    non_match_threshold: Threshold for non-match classification (default: 0.5)
                        For similarity metrics, pairs with score < threshold are non-matches.
    is_similarity: If True (default), higher values = more similar (use >= for match).
                   If False, lower values = more similar (use <= for match).
    include_flags: Bitmask of categories to include (default: INCLUDE_ALL)
                   Use INCLUDE_TM | INCLUDE_FM | ... to combine flags.
    max_pairs: Maximum number of pairs to return (default: 1,000,000)
    vec_dim: Expected vector dimension (default: 512). Will be inferred from data if different.

Returns:
    Tuple of (pair_indices, categories, scores, count):
    - pair_indices: [N, 2] int32 - (row, col) indices of pairs
    - categories: [N] uint8 - category codes (0=TM, 1=FM, 2=FNM, 3=TNM, 255=unclassified)
    - scores: [N] float32 - dot product similarity scores
    - count: [1] int32 - actual number of pairs (N == len(pair_indices))

Note:
    The returned tensors are pre-sliced to contain only the valid entries.
    Synchronization is handled internally.

Example:
    Basic usage with f16 feature vectors:

    >>> import torch
    >>> import cuda_iris_matcher as ih
    >>> # Create normalized f16 feature vectors [M, 512]
    >>> data = torch.randn(100, 512, dtype=torch.float16, device="cuda")
    >>> data = data / data.norm(dim=1, keepdim=True)
    >>> # Compute all pairwise dot products
    >>> pairs, cats, scores, count = ih.dot_product_cuda(data)
    >>> print(f"Found {count.item()} pairs")

    With identity labels for classification:

    >>> labels = torch.arange(100, dtype=torch.int32, device="cuda")
    >>> pairs, cats, scores, count = ih.dot_product_cuda(
    ...     data, labels=labels,
    ...     match_threshold=0.8,
    ...     include_flags=ih.INCLUDE_TM | ih.INCLUDE_FM
    ... )

---

### `dot_product_ab_cuda`

```python
dot_product_ab_cuda(data_a: 'torch.Tensor', data_b: 'torch.Tensor', labels_a: 'Optional[torch.Tensor]' = None, labels_b: 'Optional[torch.Tensor]' = None, match_threshold: 'float' = 0.5, non_match_threshold: 'float' = 0.5, is_similarity: 'bool' = True, include_flags: 'int' = 15, max_pairs: 'int' = 1000000, vec_dim: 'int' = 512) -> 'Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]'
```

Compute dot product similarity between two different sets A and B.
Computes the full M_A x M_B matrix (not just lower triangle).

Uses f16 (half precision) vectors and tensor cores for fast computation.

Args:
    data_a: CUDA float16 tensor of shape [M_A, vec_dim] containing feature vectors.
    data_b: CUDA float16 tensor of shape [M_B, vec_dim] containing feature vectors.
    labels_a: Optional CUDA int32 tensor of shape [M_A] with identity labels.
    labels_b: Optional CUDA int32 tensor of shape [M_B] with identity labels.
              Both labels_a and labels_b must be provided, or neither.
              If None, no classification is performed and all pairs are returned.
    match_threshold: Threshold for match classification (default: 0.5)
    non_match_threshold: Threshold for non-match classification (default: 0.5)
    is_similarity: If True (default), higher values = more similar (use >= for match).
                   If False, lower values = more similar (use <= for match).
    include_flags: Bitmask of categories to include (default: INCLUDE_ALL)
    max_pairs: Maximum number of pairs to return (default: 1,000,000)
    vec_dim: Expected vector dimension (default: 512). Will be inferred from data if different.

Returns:
    Tuple of (pair_indices, categories, scores, count):
    - pair_indices: [N, 2] int32 - (row_a, col_b) indices of pairs
    - categories: [N] uint8 - category codes (0=TM, 1=FM, 2=FNM, 3=TNM, 255=unclassified)
    - scores: [N] float32 - dot product similarity scores
    - count: [1] int32 - actual number of pairs

Example:
    Compare a gallery set against probe samples:

    >>> import torch
    >>> import cuda_iris_matcher as ih
    >>> # Gallery: 10000 enrolled feature vectors
    >>> gallery = torch.randn(10000, 512, dtype=torch.float16, device="cuda")
    >>> gallery = gallery / gallery.norm(dim=1, keepdim=True)
    >>> # Probe: 50 query feature vectors
    >>> probe = torch.randn(50, 512, dtype=torch.float16, device="cuda")
    >>> probe = probe / probe.norm(dim=1, keepdim=True)
    >>> # Find all similarities
    >>> pairs, _, scores, count = ih.dot_product_ab_cuda(
    ...     gallery, probe,
    ...     match_threshold=0.8
    ... )
    >>> # pairs[:, 0] = gallery index, pairs[:, 1] = probe index

---

### `dot_product_dense_cuda`

```python
dot_product_dense_cuda(data: 'torch.Tensor') -> 'torch.Tensor'
```

Compute dense pairwise dot product similarity matrix using optimal backend.

Uses PyTorch's cuBLAS-backed mm for maximum performance.

Args:
    data: CUDA float16 tensor of shape [M, vec_dim] containing feature vectors.

Returns:
    Float32 tensor of shape [M, M] with pairwise dot products.

Note:
    For filtered output with classification, use dot_product_cuda instead.

---

### `dot_product_ab_dense_cuda`

```python
dot_product_ab_dense_cuda(data_a: 'torch.Tensor', data_b: 'torch.Tensor') -> 'torch.Tensor'
```

Compute dense A vs B dot product similarity matrix using optimal backend.

Uses PyTorch's cuBLAS-backed mm for maximum performance.

Args:
    data_a: CUDA float16 tensor of shape [M_A, vec_dim]
    data_b: CUDA float16 tensor of shape [M_B, vec_dim]

Returns:
    Float32 tensor of shape [M_A, M_B] with pairwise dot products.

---

### `masked_hamming_sharded`

```python
masked_hamming_sharded(data: 'torch.Tensor', mask: 'torch.Tensor', labels: 'Optional[torch.Tensor]' = None, match_threshold: 'float' = 0.35, non_match_threshold: 'float' = 0.35, is_similarity: 'bool' = False, include_flags: 'int' = 15, max_pairs: 'int' = 1000000, dims: 'Optional[Tuple[int, int, int, int]]' = None, r_dim: 'int' = 16, theta_dim: 'int' = 200, d0_dim: 'int' = 2, d1_dim: 'int' = 2, min_shards: 'int' = 1, max_tile_size: 'Optional[int]' = None, host_index: 'int' = 0, num_hosts: 'int' = 1) -> 'Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]'
```

Sharded version of masked_hamming_cuda for multi-GPU and multi-host datasets.

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

---

### `masked_hamming_ab_sharded`

```python
masked_hamming_ab_sharded(data_a: 'torch.Tensor', mask_a: 'torch.Tensor', data_b: 'torch.Tensor', mask_b: 'torch.Tensor', labels_a: 'Optional[torch.Tensor]' = None, labels_b: 'Optional[torch.Tensor]' = None, match_threshold: 'float' = 0.35, non_match_threshold: 'float' = 0.35, is_similarity: 'bool' = False, include_flags: 'int' = 15, max_pairs: 'int' = 1000000, dims: 'Optional[Tuple[int, int, int, int]]' = None, r_dim: 'int' = 16, theta_dim: 'int' = 200, d0_dim: 'int' = 2, d1_dim: 'int' = 2, min_shards: 'int' = 1, max_tile_size: 'Optional[int]' = None, host_index: 'int' = 0, num_hosts: 'int' = 1) -> 'Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]'
```

Sharded version of masked_hamming_ab_cuda for multi-GPU and multi-host datasets.

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

---

### `pack_theta_major_batched`

```python
pack_theta_major_batched(bits: 'torch.Tensor', dims: 'Optional[Tuple[int, int, int, int]]' = None, r_dim: 'int' = 16, theta_dim: 'int' = 200, d0_dim: 'int' = 2, d1_dim: 'int' = 2, batch_size: 'Optional[int]' = None, device_id: 'int' = 0) -> 'torch.Tensor'
```

Pack iris code bits in batches when data doesn't fit on GPU.

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

---

### `get_device_count`

```python
get_device_count() -> 'int'
```

Return the number of available CUDA devices.

---

### `get_shard_info`

```python
get_shard_info(m_a: 'int', m_b: 'int', k_words: 'int' = 400, max_pairs_per_shard: 'int' = 100000, min_shards: 'int' = 1, max_tile_size: 'Optional[int]' = None, host_index: 'int' = 0, num_hosts: 'int' = 1) -> 'List[ShardConfig]'
```

Get shard configuration info for planning (debugging/inspection).

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

---

### `get_self_shard_info`

```python
get_self_shard_info(m: 'int', k_words: 'int' = 400, max_pairs_per_shard: 'int' = 100000, min_shards: 'int' = 1, max_tile_size: 'Optional[int]' = None, host_index: 'int' = 0, num_hosts: 'int' = 1) -> 'List[ShardConfig]'
```

Get shard configuration info for self-comparison (lower triangle).

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

---

### `get_total_shards`

```python
get_total_shards(m_a: 'int', m_b: 'Optional[int]' = None, k_words: 'int' = 400, max_pairs_per_shard: 'int' = 100000, min_shards: 'int' = 1, max_tile_size: 'Optional[int]' = None, num_hosts: 'int' = 1) -> 'int'
```

Get total number of shards across all hosts.

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

---

### `dot_product_sharded`

```python
dot_product_sharded(data: 'torch.Tensor', labels: 'Optional[torch.Tensor]' = None, match_threshold: 'float' = 0.5, non_match_threshold: 'float' = 0.5, is_similarity: 'bool' = True, include_flags: 'int' = 15, max_pairs: 'int' = 1000000, vec_dim: 'int' = 512, min_shards: 'int' = 1, max_tile_size: 'Optional[int]' = None, host_index: 'int' = 0, num_hosts: 'int' = 1) -> 'Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]'
```

Sharded version of dot_product_cuda for multi-GPU and multi-host datasets.

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

---

### `dot_product_ab_sharded`

```python
dot_product_ab_sharded(data_a: 'torch.Tensor', data_b: 'torch.Tensor', labels_a: 'Optional[torch.Tensor]' = None, labels_b: 'Optional[torch.Tensor]' = None, match_threshold: 'float' = 0.5, non_match_threshold: 'float' = 0.5, is_similarity: 'bool' = True, include_flags: 'int' = 15, max_pairs: 'int' = 1000000, vec_dim: 'int' = 512, min_shards: 'int' = 1, max_tile_size: 'Optional[int]' = None, host_index: 'int' = 0, num_hosts: 'int' = 1) -> 'Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]'
```

Sharded version of dot_product_ab_cuda for multi-GPU and multi-host datasets.

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

---

## Classes

### `ShardConfig`

Configuration for a single shard of computation.


**Fields:**

- `device_id`: int
- `a_start`: int
- `a_end`: int
- `b_start`: int
- `b_end`: int
- `global_shard_id`: int

---

## Constants

| Name | Value | Description |

|------|-------|-------------|

| `CATEGORY_TRUE_MATCH` | `0` | Classification category |

| `CATEGORY_FALSE_MATCH` | `1` | Classification category |

| `CATEGORY_FALSE_NON_MATCH` | `2` | Classification category |

| `CATEGORY_TRUE_NON_MATCH` | `3` | Classification category |

| `INCLUDE_TM` | `1` | Include flag for filtering |

| `INCLUDE_FM` | `2` | Include flag for filtering |

| `INCLUDE_FNM` | `4` | Include flag for filtering |

| `INCLUDE_TNM` | `8` | Include flag for filtering |

| `INCLUDE_ALL` | `15` | Include flag for filtering |

| `DEFAULT_R_DIM` | `16` | Default dimension |

| `DEFAULT_THETA_DIM` | `200` | Default dimension |

| `DEFAULT_D0_DIM` | `2` | Default dimension |

| `DEFAULT_D1_DIM` | `2` | Default dimension |

| `DEFAULT_DIMS` | `(16, 200, 2, 2)` | Default dimension |

| `DEFAULT_DOT_VEC_DIM` | `512` | Default dimension |
