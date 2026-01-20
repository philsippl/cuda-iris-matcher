from ._jit_compile import get_ops

from .ops import (
    masked_hamming_cuda,
    masked_hamming_ab_cuda,
    pack_theta_major_cuda as pack_theta_major,
    repack_to_theta_major_cuda as repack_to_theta_major,
    # Dot product functions (sparse output with filtering)
    dot_product_cuda,
    dot_product_ab_cuda,
    # Dense dot product functions (high performance, full matrix output)
    dot_product_dense_cuda,
    dot_product_ab_dense_cuda,
    # Classification constants
    CATEGORY_TRUE_MATCH,
    CATEGORY_FALSE_MATCH,
    CATEGORY_FALSE_NON_MATCH,
    CATEGORY_TRUE_NON_MATCH,
    INCLUDE_TM,
    INCLUDE_FM,
    INCLUDE_FNM,
    INCLUDE_TNM,
    INCLUDE_ALL,
    # Dimension constants
    DEFAULT_R_DIM,
    DEFAULT_THETA_DIM,
    DEFAULT_D0_DIM,
    DEFAULT_D1_DIM,
    DEFAULT_DIMS,
    # Dot product dimension constant
    DEFAULT_DOT_VEC_DIM,
)

from .sampling import (
    StratifiedSamplingFilter,
)

from .sharding import (
    masked_hamming_sharded,
    masked_hamming_ab_sharded,
    pack_theta_major_batched,
    get_device_count,
    get_shard_info,
    get_self_shard_info,
    get_total_shards,
    ShardConfig,
    # Dot product sharding
    dot_product_sharded,
    dot_product_ab_sharded,
)

__all__ = [
    # JIT compilation access (for tests/advanced usage)
    "get_ops",
    # Main API
    "masked_hamming_cuda",
    "masked_hamming_ab_cuda",
    "pack_theta_major",
    "repack_to_theta_major",
    # Dot product functions (f16 vectors, sparse output with filtering)
    "dot_product_cuda",
    "dot_product_ab_cuda",
    # Dense dot product (high performance, full matrix output)
    "dot_product_dense_cuda",
    "dot_product_ab_dense_cuda",
    # Sharded versions (multi-GPU / multi-host / large dataset support)
    "masked_hamming_sharded",
    "masked_hamming_ab_sharded",
    "pack_theta_major_batched",
    "get_device_count",
    "get_shard_info",
    "get_self_shard_info",
    "get_total_shards",
    "ShardConfig",
    # Dot product sharding
    "dot_product_sharded",
    "dot_product_ab_sharded",
    # Stratified sampling
    "StratifiedSamplingFilter",
    # Classification constants
    "CATEGORY_TRUE_MATCH",
    "CATEGORY_FALSE_MATCH",
    "CATEGORY_FALSE_NON_MATCH",
    "CATEGORY_TRUE_NON_MATCH",
    "INCLUDE_TM",
    "INCLUDE_FM",
    "INCLUDE_FNM",
    "INCLUDE_TNM",
    "INCLUDE_ALL",
    # Dimension constants
    "DEFAULT_R_DIM",
    "DEFAULT_THETA_DIM",
    "DEFAULT_D0_DIM",
    "DEFAULT_D1_DIM",
    "DEFAULT_DIMS",
    # Dot product dimension constant
    "DEFAULT_DOT_VEC_DIM",
]


