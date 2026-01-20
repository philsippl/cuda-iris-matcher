from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch

from ._jit_compile import get_ops
from .sampling import StratifiedSamplingFilter, SampleBinsType


def _get_C():
    """Get the compiled CUDA extension module (JIT compiled on first access)."""
    return get_ops()


def _prepare_sampling_params(
    sample_bins: SampleBinsType,
) -> Tuple[int, Optional[torch.Tensor], Optional[torch.Tensor], int]:
    """
    Convert sample_bins to kernel parameters.
    
    Returns:
        (num_bins, thresholds_tensor, probabilities_tensor, seed)
    """
    if sample_bins is None:
        return 0, None, None, 0
    
    # Convert dict to StratifiedSamplingFilter if needed
    if isinstance(sample_bins, dict):
        filter_obj = StratifiedSamplingFilter(sample_bins)
    else:
        filter_obj = sample_bins
    
    # Access the sorted sample_bins (the dataclass sorts them in __post_init__)
    thresholds_list = list(filter_obj.sample_bins.keys())
    probabilities_list = list(filter_obj.sample_bins.values())
    
    num_bins = len(thresholds_list)
    thresholds = torch.tensor(thresholds_list, dtype=torch.float32)
    probabilities = torch.tensor(probabilities_list, dtype=torch.float32)
    seed = filter_obj.seed if filter_obj.seed is not None else 0
    
    return num_bins, thresholds, probabilities, seed

# Export classification constants from C++ extension
CATEGORY_TRUE_MATCH = _get_C().CATEGORY_TRUE_MATCH
CATEGORY_FALSE_MATCH = _get_C().CATEGORY_FALSE_MATCH
CATEGORY_FALSE_NON_MATCH = _get_C().CATEGORY_FALSE_NON_MATCH
CATEGORY_TRUE_NON_MATCH = _get_C().CATEGORY_TRUE_NON_MATCH

INCLUDE_TM = _get_C().INCLUDE_TM
INCLUDE_FM = _get_C().INCLUDE_FM
INCLUDE_FNM = _get_C().INCLUDE_FNM
INCLUDE_TNM = _get_C().INCLUDE_TNM
INCLUDE_ALL = _get_C().INCLUDE_ALL

# Export default dimensions
DEFAULT_R_DIM = _get_C().DEFAULT_R_DIM
DEFAULT_THETA_DIM = _get_C().DEFAULT_THETA_DIM
DEFAULT_D0_DIM = _get_C().DEFAULT_D0_DIM
DEFAULT_D1_DIM = _get_C().DEFAULT_D1_DIM
DEFAULT_DIMS = (DEFAULT_R_DIM, DEFAULT_THETA_DIM, DEFAULT_D0_DIM, DEFAULT_D1_DIM)

# Default vector dimension for dot product
DEFAULT_DOT_VEC_DIM = _get_C().DEFAULT_DOT_VEC_DIM


def dot_product_dense_cuda(data: torch.Tensor) -> torch.Tensor:
    """
    Compute dense pairwise dot product similarity matrix using optimal backend.
    
    Uses PyTorch's cuBLAS-backed mm for maximum performance.
    
    Args:
        data: CUDA float16 tensor of shape [M, vec_dim] containing feature vectors.
    
    Returns:
        Float32 tensor of shape [M, M] with pairwise dot products.
    
    Note:
        For filtered output with classification, use dot_product_cuda instead.
    """
    # Use PyTorch mm (cuBLAS) for best performance
    return torch.mm(data.float(), data.float().t())


def dot_product_ab_dense_cuda(data_a: torch.Tensor, data_b: torch.Tensor) -> torch.Tensor:
    """
    Compute dense A vs B dot product similarity matrix using optimal backend.
    
    Uses PyTorch's cuBLAS-backed mm for maximum performance.
    
    Args:
        data_a: CUDA float16 tensor of shape [M_A, vec_dim]
        data_b: CUDA float16 tensor of shape [M_B, vec_dim]
    
    Returns:
        Float32 tensor of shape [M_A, M_B] with pairwise dot products.
    """
    return torch.mm(data_a.float(), data_b.float().t())


def _resolve_dims(
    dims: Optional[Tuple[int, int, int, int]],
    r_dim: int,
    theta_dim: int,
    d0_dim: int,
    d1_dim: int,
) -> Tuple[int, int, int, int]:
    """Resolve dimensions from either dims tuple or individual parameters."""
    if dims is not None:
        return dims
    return (r_dim, theta_dim, d0_dim, d1_dim)


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


def _ensure_packed(
    tensor: torch.Tensor,
    r_dim: int,
    theta_dim: int,
    d0_dim: int,
    d1_dim: int,
) -> torch.Tensor:
    """Ensure tensor is packed on GPU. Packs if needed, moves to GPU if needed.
    
    Args:
        tensor: Either packed [M, k_words] int32 or unpacked [M, r, theta, d0, d1] uint8
        
    Returns:
        Packed tensor on CUDA, shape [M, k_words] int32
    """
    k_words = r_dim * theta_dim * d0_dim * d1_dim // 32
    
    if _is_packed(tensor, k_words):
        # Already packed, just ensure on GPU
        if tensor.is_cuda:
            return tensor
        return tensor.cuda()
    
    if _is_unpacked(tensor, r_dim, theta_dim, d0_dim, d1_dim):
        # Need to pack - move to GPU first if needed
        if not tensor.is_cuda:
            tensor = tensor.cuda()
        # Clone because pack_theta_major is in-place
        return _get_C().pack_theta_major_cuda(tensor.clone(), r_dim, theta_dim, d0_dim, d1_dim)
    
    raise ValueError(
        f"Invalid tensor shape. Expected packed [M, {k_words}] int32 or "
        f"unpacked [M, {r_dim}, {theta_dim}, {d0_dim}, {d1_dim}] uint8, "
        f"got shape {list(tensor.shape)} dtype {tensor.dtype}"
    )


def masked_hamming_cuda(
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
    sample_bins: SampleBinsType = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
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
        sample_bins: Optional stratified sampling configuration. Can be:
                     - Dict[float, float]: Mapping of bin upper bounds to sampling probabilities
                     - StratifiedSamplingFilter: Pre-configured filter instance
                     - None: No sampling (return all pairs)
                     
                     Example for keeping more close matches:
                         sample_bins={0.3: 1.0, 0.5: 0.1, 1.0: 0.01}
                         Keeps 100% of [0, 0.3], 10% of (0.3, 0.5], 1% of (0.5, 1.0]

    Returns:
        Tuple of (pair_indices, categories, distances, count):
        - pair_indices: [N, 2] int32 - (row, col) indices of pairs
        - categories: [N] uint8 - category codes (0=TM, 1=FM, 2=FNM, 3=TNM, 255=unclassified)
        - distances: [N] float32 - distance values
        - count: [1] int32 - actual number of pairs (N == len(pair_indices))

    Note:
        The returned tensors are pre-sliced to contain only the valid entries.
        Synchronization is handled internally.
        
        When using sample_bins, the sampling is applied after category filtering
        (include_flags). This allows you to first filter by category, then sample
        the remaining pairs based on distance.

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
        
        With stratified sampling:

        >>> pairs, cats, dists, count = ih.masked_hamming_cuda(
        ...     data, mask,
        ...     sample_bins={0.3: 1.0, 0.5: 0.1, 1.0: 0.01}  # Keep more close matches
        ... )
    """
    r_dim, theta_dim, d0_dim, d1_dim = _resolve_dims(dims, r_dim, theta_dim, d0_dim, d1_dim)
    
    # Auto-pack if unpacked data is provided
    data = _ensure_packed(data, r_dim, theta_dim, d0_dim, d1_dim)
    mask = _ensure_packed(mask, r_dim, theta_dim, d0_dim, d1_dim)
    
    # Prepare kernel-level sampling parameters
    num_bins, thresholds, probabilities, seed = _prepare_sampling_params(sample_bins)
    
    pair_indices, categories, distances, count = _get_C().masked_hamming_cuda(
        data, mask, labels,
        match_threshold, non_match_threshold,
        is_similarity, include_flags, max_pairs,
        r_dim, theta_dim, d0_dim, d1_dim,
        num_bins, thresholds, probabilities, seed
    )
    
    return pair_indices, categories, distances, count


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
    dims: Optional[Tuple[int, int, int, int]] = None,
    r_dim: int = DEFAULT_R_DIM,
    theta_dim: int = DEFAULT_THETA_DIM,
    d0_dim: int = DEFAULT_D0_DIM,
    d1_dim: int = DEFAULT_D1_DIM,
    sample_bins: SampleBinsType = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
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
        sample_bins: Optional stratified sampling configuration. Can be:
                     - Dict[float, float]: Mapping of bin upper bounds to sampling probabilities
                     - StratifiedSamplingFilter: Pre-configured filter instance
                     - None: No sampling (return all pairs)
                     
                     Example for keeping more close matches:
                         sample_bins={0.3: 1.0, 0.5: 0.1, 1.0: 0.01}
                         Keeps 100% of [0, 0.3], 10% of (0.3, 0.5], 1% of (0.5, 1.0]

    Returns:
        Tuple of (pair_indices, categories, distances, count):
        - pair_indices: [N, 2] int32 - (row, col) indices of pairs
        - categories: [N] uint8 - category codes (0=TM, 1=FM, 2=FNM, 3=TNM, 255=unclassified)
        - distances: [N] float32 - distance values
        - count: [1] int32 - actual number of pairs (N == len(pair_indices))

    Note:
        The returned tensors are pre-sliced to contain only the valid entries.
        Synchronization is handled internally.
        
        When using sample_bins, the sampling is applied after category filtering
        (include_flags). This allows you to first filter by category, then sample
        the remaining pairs based on distance.

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
        
        With stratified sampling:

        >>> pairs, cats, dists, count = ih.masked_hamming_ab_cuda(
        ...     gallery, gallery_mask, probe, probe_mask,
        ...     sample_bins={0.3: 1.0, 0.5: 0.1, 1.0: 0.01}  # Keep more close matches
        ... )
    """
    r_dim, theta_dim, d0_dim, d1_dim = _resolve_dims(dims, r_dim, theta_dim, d0_dim, d1_dim)
    
    # Auto-pack if unpacked data is provided
    data_a = _ensure_packed(data_a, r_dim, theta_dim, d0_dim, d1_dim)
    mask_a = _ensure_packed(mask_a, r_dim, theta_dim, d0_dim, d1_dim)
    data_b = _ensure_packed(data_b, r_dim, theta_dim, d0_dim, d1_dim)
    mask_b = _ensure_packed(mask_b, r_dim, theta_dim, d0_dim, d1_dim)
    
    # Prepare kernel-level sampling parameters
    num_bins, thresholds, probabilities, seed = _prepare_sampling_params(sample_bins)
    
    pair_indices, categories, distances, count = _get_C().masked_hamming_ab_cuda(
        data_a, mask_a, data_b, mask_b,
        labels_a, labels_b,
        match_threshold, non_match_threshold,
        is_similarity, include_flags, max_pairs,
        r_dim, theta_dim, d0_dim, d1_dim,
        num_bins, thresholds, probabilities, seed
    )
    
    return pair_indices, categories, distances, count


def pack_theta_major_cuda(
    bits: torch.Tensor,
    dims: Optional[Tuple[int, int, int, int]] = None,
    r_dim: int = DEFAULT_R_DIM,
    theta_dim: int = DEFAULT_THETA_DIM,
    d0_dim: int = DEFAULT_D0_DIM,
    d1_dim: int = DEFAULT_D1_DIM,
) -> torch.Tensor:
    """
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
    """
    r_dim, theta_dim, d0_dim, d1_dim = _resolve_dims(dims, r_dim, theta_dim, d0_dim, d1_dim)
    return _get_C().pack_theta_major_cuda(bits, r_dim, theta_dim, d0_dim, d1_dim)


def repack_to_theta_major_cuda(
    input: torch.Tensor,
    dims: Optional[Tuple[int, int, int, int]] = None,
    r_dim: int = DEFAULT_R_DIM,
    theta_dim: int = DEFAULT_THETA_DIM,
    d0_dim: int = DEFAULT_D0_DIM,
    d1_dim: int = DEFAULT_D1_DIM,
) -> torch.Tensor:
    """
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
    """
    r_dim, theta_dim, d0_dim, d1_dim = _resolve_dims(dims, r_dim, theta_dim, d0_dim, d1_dim)
    return _get_C().repack_to_theta_major_cuda(input, r_dim, theta_dim, d0_dim, d1_dim)


# ----------------- Dot Product Similarity Functions -----------------


def dot_product_cuda(
    data: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    match_threshold: float = 0.5,
    non_match_threshold: float = 0.5,
    is_similarity: bool = True,
    include_flags: int = INCLUDE_ALL,
    max_pairs: int = 1_000_000,
    vec_dim: int = DEFAULT_DOT_VEC_DIM,
    sample_bins: SampleBinsType = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
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
        sample_bins: Optional stratified sampling configuration. Can be:
                     - Dict[float, float]: Mapping of bin upper bounds to sampling probabilities
                     - StratifiedSamplingFilter: Pre-configured filter instance
                     - None: No sampling (return all pairs)
                     
                     Example for similarity scores (keep more high-similarity pairs):
                         sample_bins={0.5: 0.01, 0.8: 0.1, 1.0: 1.0}
                         Keeps 1% of [-inf, 0.5], 10% of (0.5, 0.8], 100% of (0.8, 1.0]

    Returns:
        Tuple of (pair_indices, categories, scores, count):
        - pair_indices: [N, 2] int32 - (row, col) indices of pairs
        - categories: [N] uint8 - category codes (0=TM, 1=FM, 2=FNM, 3=TNM, 255=unclassified)
        - scores: [N] float32 - dot product similarity scores
        - count: [1] int32 - actual number of pairs (N == len(pair_indices))

    Note:
        The returned tensors are pre-sliced to contain only the valid entries.
        Synchronization is handled internally.
        
        When using sample_bins, the sampling is applied after category filtering
        (include_flags). This allows you to first filter by category, then sample
        the remaining pairs based on similarity score.

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
        
        With stratified sampling (keep more high-similarity pairs):

        >>> pairs, cats, scores, count = ih.dot_product_cuda(
        ...     data,
        ...     sample_bins={0.5: 0.01, 0.8: 0.1, 1.0: 1.0}
        ... )
    """
    # Prepare kernel-level sampling parameters
    num_bins, thresholds, probabilities, seed = _prepare_sampling_params(sample_bins)
    
    pair_indices, categories, scores, count = _get_C().dot_product_cuda(
        data, labels,
        match_threshold, non_match_threshold,
        is_similarity, include_flags, max_pairs, vec_dim,
        num_bins, thresholds, probabilities, seed
    )
    
    return pair_indices, categories, scores, count


def dot_product_ab_cuda(
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
    sample_bins: SampleBinsType = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
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
        sample_bins: Optional stratified sampling configuration. Can be:
                     - Dict[float, float]: Mapping of bin upper bounds to sampling probabilities
                     - StratifiedSamplingFilter: Pre-configured filter instance
                     - None: No sampling (return all pairs)
                     
                     Example for similarity scores (keep more high-similarity pairs):
                         sample_bins={0.5: 0.01, 0.8: 0.1, 1.0: 1.0}
                         Keeps 1% of [-inf, 0.5], 10% of (0.5, 0.8], 100% of (0.8, 1.0]

    Returns:
        Tuple of (pair_indices, categories, scores, count):
        - pair_indices: [N, 2] int32 - (row_a, col_b) indices of pairs
        - categories: [N] uint8 - category codes (0=TM, 1=FM, 2=FNM, 3=TNM, 255=unclassified)
        - scores: [N] float32 - dot product similarity scores
        - count: [1] int32 - actual number of pairs
        
    Note:
        When using sample_bins, the sampling is applied after category filtering
        (include_flags). This allows you to first filter by category, then sample
        the remaining pairs based on similarity score.

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
        
        With stratified sampling (keep more high-similarity pairs):

        >>> pairs, cats, scores, count = ih.dot_product_ab_cuda(
        ...     gallery, probe,
        ...     sample_bins={0.5: 0.01, 0.8: 0.1, 1.0: 1.0}
        ... )
    """
    # Prepare kernel-level sampling parameters
    num_bins, thresholds, probabilities, seed = _prepare_sampling_params(sample_bins)
    
    pair_indices, categories, scores, count = _get_C().dot_product_ab_cuda(
        data_a, data_b,
        labels_a, labels_b,
        match_threshold, non_match_threshold,
        is_similarity, include_flags, max_pairs, vec_dim,
        num_bins, thresholds, probabilities, seed
    )
    
    return pair_indices, categories, scores, count
