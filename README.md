# cuda-iris-matcher

High-performance CUDA-accelerated library for computing masked fractional Hamming distances on IrisCode-like bit tensors. Supports multi-GPU and multi-host operation for large-scale biometric matching.

## Features

- **Fast**: CUDA kernels with TensorCore acceleration for maximum throughput
- **Rotation-invariant**: Automatically finds minimum distance across theta rotations
- **Scalable**: Multi-GPU sharding for datasets too large for single GPU
- **Flexible**: Accepts packed (int32) or unpacked (uint8) iris codes
- **Classification**: Built-in true/false match categorization with identity labels

## Installation

```bash
pip install -e .
```

**Requirements**: CUDA-capable GPU, PyTorch with CUDA support

## Quick Start

### Basic Usage

```python
import torch
import cuda_iris_matcher as ih

# Create sample data (packed format: [M, 400] int32)
data = torch.randint(0, 2**31, (1024, 400), dtype=torch.int32, device="cuda")
mask = torch.full((1024, 400), 0x7FFFFFFF, dtype=torch.int32, device="cuda")

# Compute pairwise distances for all pairs (lower triangle)
pair_indices, categories, distances, count = ih.masked_hamming_cuda(
    data, mask,
    match_threshold=0.35,
    max_pairs=100_000,
)

print(f"Found {count.item()} pairs")
print(f"Distance range: {distances.min():.4f} - {distances.max():.4f}")
```

### With Unpacked Data

```python
# Unpacked format: [M, r, theta, d0, d1] uint8 with values {0, 1}
raw_codes = torch.randint(0, 2, (100, 16, 200, 2, 2), dtype=torch.uint8, device="cuda")
raw_masks = torch.ones_like(raw_codes)

# Functions accept either format - packing is automatic
pair_idx, categories, distances, count = ih.masked_hamming_cuda(
    raw_codes, raw_masks,
    match_threshold=0.35,
)
```

### With Identity Labels (Classification)

```python
# Labels enable true/false match classification
labels = torch.arange(100, dtype=torch.int32, device="cuda")

pair_idx, categories, distances, count = ih.masked_hamming_cuda(
    data, mask,
    labels=labels,
    match_threshold=0.35,
    include_flags=ih.INCLUDE_TM | ih.INCLUDE_FM,  # Only matches
)

# Filter by category
true_matches = pair_idx[categories == ih.CATEGORY_TRUE_MATCH]
false_matches = pair_idx[categories == ih.CATEGORY_FALSE_MATCH]
```

### A vs B Comparison

```python
# Compare gallery against probe set
gallery = torch.randint(0, 2**31, (10000, 400), dtype=torch.int32, device="cuda")
gallery_mask = torch.full_like(gallery, 0x7FFFFFFF)
probe = torch.randint(0, 2**31, (100, 400), dtype=torch.int32, device="cuda")
probe_mask = torch.full_like(probe, 0x7FFFFFFF)

pair_idx, _, distances, count = ih.masked_hamming_ab_cuda(
    gallery, gallery_mask,
    probe, probe_mask,
    match_threshold=0.35,
)
```

### Multi-GPU Sharding

```python
# Automatically distributes across all available GPUs
pair_idx, categories, distances, count = ih.masked_hamming_sharded(
    large_data, large_mask,
    min_shards=4,  # Force tiling for testing on single GPU
)

# Check device count
print(f"Using {ih.get_device_count()} GPUs")
```

## API Documentation

Generate API docs automatically from docstrings:

```bash
# Install docs dependencies
pip install pdoc

# Generate HTML documentation
python docs/generate.py

# Or serve with live reload
python docs/generate.py --serve
```

Generated docs will be in `docs/api/`. Open `docs/api/cuda_iris_matcher.html` in your browser.

### Main Functions

| Function | Description |
|----------|-------------|
| `masked_hamming_cuda` | Pairwise distances within a set (lower triangle) |
| `masked_hamming_ab_cuda` | Distances between two sets (full matrix) |
| `masked_hamming_sharded` | Multi-GPU version of `masked_hamming_cuda` |
| `masked_hamming_ab_sharded` | Multi-GPU version of `masked_hamming_ab_cuda` |
| `pack_theta_major` | Pack uint8 bits to int32 theta-major format |
| `pack_theta_major_batched` | Pack large CPU datasets in batches |
| `repack_to_theta_major` | Convert r-major to theta-major ordering |

### Constants

```python
# Classification categories
ih.CATEGORY_TRUE_MATCH      # 0 - Same identity, match
ih.CATEGORY_FALSE_MATCH     # 1 - Different identity, match
ih.CATEGORY_FALSE_NON_MATCH # 2 - Same identity, non-match
ih.CATEGORY_TRUE_NON_MATCH  # 3 - Different identity, non-match

# Include flags (combine with |)
ih.INCLUDE_TM   # True matches
ih.INCLUDE_FM   # False matches
ih.INCLUDE_FNM  # False non-matches
ih.INCLUDE_TNM  # True non-matches
ih.INCLUDE_ALL  # All categories

# Default dimensions
ih.DEFAULT_R_DIM      # 16
ih.DEFAULT_THETA_DIM  # 200
ih.DEFAULT_D0_DIM     # 2
ih.DEFAULT_D1_DIM     # 2
ih.DEFAULT_DIMS       # (16, 200, 2, 2)
```

## Data Format

### Unpacked (uint8)
- Shape: `[M, r_dim, theta_dim, d0_dim, d1_dim]` = `[M, 16, 200, 2, 2]`
- Values: `{0, 1}`
- Total bits: 12,800 per sample

### Packed (int32)
- Shape: `[M, k_words]` = `[M, 400]`
- Encoding: Theta-major for efficient rotation search
- Use `pack_theta_major()` to convert

## Tests

```bash
pytest -q
```

## Benchmarking

```bash
# Basic benchmark
python -m benchmarking.benchmark --sizes 1024,2048,4096

# With sharding benchmark
python -m benchmarking.benchmark --sizes 1024,2048 --sharding

# Full options
python -m benchmarking.benchmark --help
```

## License

MIT
