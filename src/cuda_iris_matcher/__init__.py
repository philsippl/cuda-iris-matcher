from .ops import (
    masked_hamming_cuda,
    masked_hamming_ab_cuda,
    pack_theta_major_cuda as pack_theta_major,
    repack_to_theta_major_cuda as repack_to_theta_major,
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
)

__all__ = [
    "masked_hamming_cuda",
    "masked_hamming_ab_cuda",
    "pack_theta_major",
    "repack_to_theta_major",
    # Sharded versions (multi-GPU / multi-host / large dataset support)
    "masked_hamming_sharded",
    "masked_hamming_ab_sharded",
    "pack_theta_major_batched",
    "get_device_count",
    "get_shard_info",
    "get_self_shard_info",
    "get_total_shards",
    "ShardConfig",
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
]


