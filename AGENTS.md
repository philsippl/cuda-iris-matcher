# Iris CUDA Matcher - Development Rules

## After Making Changes

Always run the following verification steps after modifying code:

1. **Run all tests:**
   ```bash
   make test
   ```

2. **Run Python benchmark:**
   ```bash
   make pybench
   ```

3. **Run CUDA benchmark:**
   ```bash
   make bench
   ```

## Quick Verification Command

Run all three in sequence:
```bash
make test-tc && make pybench-quick && make quick
```

## Project Structure

- `iris.cu` - Main CUDA kernels (TensorCore hamming distance)
- `iris_params.h` - Configuration and dimension parameters
- `iris_binding.cu` - Python/PyTorch bindings
- `benchmark.cu` - C++ benchmark
- `benchmarking/benchmark.py` - Python benchmark
- `tests/` - Python test suite

## Iris Code Dimensions

The iris code shape is `(r_dim, theta_dim, d0_dim, d1_dim)` with defaults `(16, 200, 2, 2)`.

Constraints:
- `r_dim * d0_dim * d1_dim` must be divisible by 32
- `r_dim * theta_dim * d0_dim * d1_dim` must be divisible by 256

