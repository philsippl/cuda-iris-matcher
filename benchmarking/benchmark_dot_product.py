#!/usr/bin/env python3
"""
Benchmark: Custom CUDA dot product kernel vs PyTorch native operations.

Compares:
1. Custom CUDA kernel (dot_product_cuda)
2. PyTorch mm (matrix multiplication)
3. PyTorch cosine_similarity (for reference)
"""

import argparse
import torch
import time
import numpy as np
from typing import Dict, List, Tuple

import cuda_iris_matcher as ih


def pytorch_dot_product_mm(data: torch.Tensor) -> torch.Tensor:
    """Compute pairwise dot products using PyTorch mm."""
    # data: [M, D] -> result: [M, M]
    return torch.mm(data, data.t())


def pytorch_dot_product_mm_ab(data_a: torch.Tensor, data_b: torch.Tensor) -> torch.Tensor:
    """Compute A x B^T dot products using PyTorch mm."""
    return torch.mm(data_a, data_b.t())


def extract_lower_triangle(matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract lower triangle indices and values (i > j)."""
    M = matrix.size(0)
    indices = torch.tril_indices(M, M, offset=-1, device=matrix.device)
    values = matrix[indices[0], indices[1]]
    return indices, values


def benchmark_pytorch_mm(data: torch.Tensor, warmup: int, repeats: int) -> float:
    """Benchmark PyTorch mm for pairwise dot products (compute only, GPU)."""
    # Warmup
    for _ in range(warmup):
        result = pytorch_dot_product_mm(data)
        del result
        torch.cuda.synchronize()
    
    # Timed runs
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(repeats):
        result = pytorch_dot_product_mm(data)
        del result
    end.record()
    torch.cuda.synchronize()
    
    return start.elapsed_time(end) / repeats


def benchmark_pytorch_full_workflow(data: torch.Tensor, threshold: float, 
                                    warmup: int, repeats: int) -> float:
    """Benchmark PyTorch full workflow: mm + extract lower triangle + threshold filter + CPU copy."""
    M = data.size(0)
    indices = torch.tril_indices(M, M, offset=-1, device=data.device)
    
    # Pre-allocate pinned CPU tensors (like the CUDA kernel does)
    # Cap to avoid huge allocations for large M
    max_pairs = min(M * (M - 1) // 2, 10_000_000)
    cpu_values = torch.empty(max_pairs, dtype=torch.float32, pin_memory=True)
    cpu_i = torch.empty(max_pairs, dtype=torch.int64, pin_memory=True)
    cpu_j = torch.empty(max_pairs, dtype=torch.int64, pin_memory=True)
    
    for _ in range(warmup):
        result = torch.mm(data.float(), data.float().t())
        lower_tri = result[indices[0], indices[1]]
        del result  # Free the large matrix immediately
        mask = lower_tri >= threshold
        filtered_values = lower_tri[mask]
        filtered_i = indices[0][mask]
        filtered_j = indices[1][mask]
        del lower_tri, mask
        # Copy to CPU pinned memory
        n = filtered_values.size(0)
        cpu_values[:n].copy_(filtered_values.cpu(), non_blocking=True)
        cpu_i[:n].copy_(filtered_i.cpu(), non_blocking=True)
        cpu_j[:n].copy_(filtered_j.cpu(), non_blocking=True)
        del filtered_values, filtered_i, filtered_j
        torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(repeats):
        result = torch.mm(data.float(), data.float().t())
        lower_tri = result[indices[0], indices[1]]
        del result
        mask = lower_tri >= threshold
        filtered_values = lower_tri[mask]
        filtered_i = indices[0][mask]
        filtered_j = indices[1][mask]
        del lower_tri, mask
        n = filtered_values.size(0)
        cpu_values[:n].copy_(filtered_values.cpu(), non_blocking=True)
        cpu_i[:n].copy_(filtered_i.cpu(), non_blocking=True)
        cpu_j[:n].copy_(filtered_j.cpu(), non_blocking=True)
        del filtered_values, filtered_i, filtered_j
    end.record()
    torch.cuda.synchronize()
    
    return start.elapsed_time(end) / repeats


def benchmark_pytorch_mm_ab(data_a: torch.Tensor, data_b: torch.Tensor, 
                            warmup: int, repeats: int) -> float:
    """Benchmark PyTorch mm for A vs B dot products."""
    for _ in range(warmup):
        result = pytorch_dot_product_mm_ab(data_a, data_b)
        torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(repeats):
        result = pytorch_dot_product_mm_ab(data_a, data_b)
    end.record()
    torch.cuda.synchronize()
    
    return start.elapsed_time(end) / repeats


def benchmark_custom_kernel(data: torch.Tensor, warmup: int, repeats: int) -> float:
    """Benchmark custom CUDA kernel for pairwise dot products (sparse output)."""
    M = data.size(0)
    # Cap max_pairs to avoid huge allocations (default is 1M)
    max_pairs = min(M * (M - 1) // 2, 10_000_000)
    
    # Warmup
    for _ in range(warmup):
        ih.dot_product_cuda(
            data, match_threshold=2.0, non_match_threshold=-2.0,
            include_flags=ih.INCLUDE_ALL, max_pairs=max_pairs
        )
        torch.cuda.synchronize()
    
    # Timed runs
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(repeats):
        ih.dot_product_cuda(
            data, match_threshold=2.0, non_match_threshold=-2.0,
            include_flags=ih.INCLUDE_ALL, max_pairs=max_pairs
        )
    end.record()
    torch.cuda.synchronize()
    
    return start.elapsed_time(end) / repeats


def benchmark_custom_dense(data: torch.Tensor, warmup: int, repeats: int) -> float:
    """Benchmark dense output (uses PyTorch mm for best performance)."""
    # Warmup
    for _ in range(warmup):
        result = ih.dot_product_dense_cuda(data)
        torch.cuda.synchronize()
    
    # Timed runs
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(repeats):
        result = ih.dot_product_dense_cuda(data)
    end.record()
    torch.cuda.synchronize()
    
    return start.elapsed_time(end) / repeats


def benchmark_custom_kernel_native(data: torch.Tensor, warmup: int, repeats: int) -> float:
    """Benchmark native custom CUDA kernel dense output."""
    from cuda_iris_matcher import _C
    
    # Warmup
    for _ in range(warmup):
        result = _C.dot_product_dense_cuda(data)
        del result
        torch.cuda.synchronize()
    
    # Timed runs
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(repeats):
        result = _C.dot_product_dense_cuda(data)
        del result
    end.record()
    torch.cuda.synchronize()
    
    return start.elapsed_time(end) / repeats


def benchmark_custom_kernel_ab(data_a: torch.Tensor, data_b: torch.Tensor,
                               warmup: int, repeats: int) -> float:
    """Benchmark custom CUDA kernel for A vs B dot products."""
    M_A, M_B = data_a.size(0), data_b.size(0)
    max_pairs = min(M_A * M_B, 10_000_000)
    
    for _ in range(warmup):
        ih.dot_product_ab_cuda(
            data_a, data_b, match_threshold=2.0, non_match_threshold=-2.0,
            include_flags=ih.INCLUDE_ALL, max_pairs=max_pairs
        )
        torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(repeats):
        ih.dot_product_ab_cuda(
            data_a, data_b, match_threshold=2.0, non_match_threshold=-2.0,
            include_flags=ih.INCLUDE_ALL, max_pairs=max_pairs
        )
    end.record()
    torch.cuda.synchronize()
    
    return start.elapsed_time(end) / repeats


def run_self_benchmark(sizes: List[int], vec_dim: int, warmup: int, repeats: int) -> List[Dict]:
    """Run self-comparison benchmark for multiple sizes."""
    results = []
    
    for M in sizes:
        # Aggressive memory cleanup from previous iteration
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        print(f"\n--- Self-comparison: M={M}, vec_dim={vec_dim} ---")
        
        # Create normalized f16 data
        data = torch.randn(M, vec_dim, dtype=torch.float16, device="cuda")
        data = data / data.norm(dim=1, keepdim=True)
        
        total_pairs = M * M
        lower_tri_pairs = M * (M - 1) // 2
        
        # PyTorch mm benchmark (compute only)
        # Check memory AFTER data allocation to get accurate free memory
        torch.cuda.synchronize()
        free_mem = torch.cuda.mem_get_info()[0]  # Free memory in bytes
        max_matrix_bytes = M * M * 4
        # Need space for: data.float() + transpose + result matrix + some overhead
        required_mem = max_matrix_bytes * 2.5
        skip_pytorch = required_mem > free_mem
        
        if skip_pytorch:
            print(f"  [Skipping PyTorch - need ~{required_mem / 1e9:.1f} GB, only {free_mem / 1e9:.1f} GB free]")
            pytorch_ms = float('nan')
            pytorch_tps = float('nan')
            pytorch_full_ms = float('nan')
            pytorch_full_tps = float('nan')
        else:
            pytorch_ms = benchmark_pytorch_mm(data, warmup, repeats)
            pytorch_tps = total_pairs / (pytorch_ms * 1e-3) / 1e6
            
            # PyTorch full workflow (mm + extract lower tri + filter)
            pytorch_full_ms = benchmark_pytorch_full_workflow(data, threshold=0.5, 
                                                              warmup=warmup, repeats=repeats)
            pytorch_full_tps = lower_tri_pairs / (pytorch_full_ms * 1e-3) / 1e6
        
        # Custom CUDA kernel (native dense) benchmark
        # Skip if output matrix would exceed available memory
        if required_mem > free_mem:
            print(f"  [Skipping dense kernels - output {max_matrix_bytes / 1e9:.1f} GB exceeds free {free_mem / 1e9:.1f} GB]")
            native_ms = float('nan')
            native_tps = float('nan')
            native_ratio = float('nan')
        else:
            native_ms = benchmark_custom_kernel_native(data, warmup, repeats)
            native_tps = total_pairs / (native_ms * 1e-3) / 1e6
            native_ratio = native_ms / pytorch_ms if not skip_pytorch else float('nan')
        
        # Clean up before sparse kernel (native kernel creates large output)
        gc.collect()
        torch.cuda.empty_cache()
        
        # Custom sparse kernel with filtering
        # For very large M, even the sparse kernel may have issues
        # Check available memory for output buffers
        torch.cuda.synchronize()
        free_after_native = torch.cuda.mem_get_info()[0]
        # Sparse kernel needs: max_pairs * (8 + 1 + 4) = 13 bytes per pair
        max_pairs_mem = 1_000_000 * 13  # Default max_pairs
        skip_sparse = free_after_native < max_pairs_mem * 2  # 2x safety margin
        
        if skip_sparse:
            print(f"  [Skipping sparse kernel - insufficient memory]")
            sparse_ms = float('nan')
            sparse_tps = float('nan')
            filter_speedup = float('nan')
        else:
            sparse_ms = benchmark_custom_kernel(data, warmup, repeats)
            sparse_tps = lower_tri_pairs / (sparse_ms * 1e-3) / 1e6
            filter_speedup = pytorch_full_ms / sparse_ms if not skip_pytorch else float('nan')
        
        # Memory comparison: full MxM vs sparse output
        full_matrix_mb = M * M * 4 / 1e6  # float32
        
        if not skip_pytorch:
            print(f"  PyTorch mm only:         {pytorch_ms:8.3f} ms ({pytorch_tps:8.2f} M pairs/s)")
            print(f"  PyTorch full workflow:   {pytorch_full_ms:8.3f} ms ({pytorch_full_tps:8.2f} M pairs/s) [mm+extract+filter]")
            if not np.isnan(native_ms):
                print(f"  Custom WMMA dense:       {native_ms:8.3f} ms ({native_tps:8.2f} M pairs/s) - {native_ratio:.1f}x vs mm")
            if not np.isnan(sparse_ms):
                print(f"  Custom fused filter:     {sparse_ms:8.3f} ms ({sparse_tps:8.2f} M pairs/s) - {filter_speedup:.2f}x vs PyTorch workflow")
        else:
            if not np.isnan(native_ms):
                print(f"  Custom WMMA dense:       {native_ms:8.3f} ms ({native_tps:8.2f} M pairs/s)")
            if not np.isnan(sparse_ms):
                print(f"  Custom fused filter:     {sparse_ms:8.3f} ms ({sparse_tps:8.2f} M pairs/s)")
        print(f"  Full matrix memory:      {full_matrix_mb:.1f} MB (saved with fused filter)")
        
        # Clean up to avoid OOM on next iteration
        del data
        torch.cuda.empty_cache()
        
        results.append({
            "M": M,
            "vec_dim": vec_dim,
            "pytorch_ms": pytorch_ms,
            "pytorch_full_ms": pytorch_full_ms,
            "native_ms": native_ms,
            "sparse_ms": sparse_ms,
            "native_ratio": native_ratio,
            "filter_speedup": filter_speedup,
            "pytorch_tps": pytorch_tps,
        })
    
    return results


def run_ab_benchmark(sizes: List[Tuple[int, int]], vec_dim: int, 
                     warmup: int, repeats: int) -> List[Dict]:
    """Run A vs B benchmark for multiple sizes."""
    results = []
    
    for M_A, M_B in sizes:
        torch.cuda.empty_cache()
        print(f"\n--- A vs B: M_A={M_A}, M_B={M_B}, vec_dim={vec_dim} ---")
        
        data_a = torch.randn(M_A, vec_dim, dtype=torch.float16, device="cuda")
        data_a = data_a / data_a.norm(dim=1, keepdim=True)
        data_b = torch.randn(M_B, vec_dim, dtype=torch.float16, device="cuda")
        data_b = data_b / data_b.norm(dim=1, keepdim=True)
        
        # PyTorch mm benchmark
        pytorch_ms = benchmark_pytorch_mm_ab(data_a, data_b, warmup, repeats)
        pairs = M_A * M_B
        pytorch_pairs_per_s = pairs / (pytorch_ms * 1e-3) / 1e6
        
        # Custom kernel benchmark
        custom_ms = benchmark_custom_kernel_ab(data_a, data_b, warmup, repeats)
        custom_pairs_per_s = pairs / (custom_ms * 1e-3) / 1e6
        
        speedup = pytorch_ms / custom_ms
        
        print(f"  PyTorch mm:     {pytorch_ms:8.3f} ms ({pytorch_pairs_per_s:8.2f} M pairs/s)")
        print(f"  Custom kernel:  {custom_ms:8.3f} ms ({custom_pairs_per_s:8.2f} M pairs/s)")
        print(f"  Speedup:        {speedup:.2f}x")
        
        # Clean up
        del data_a, data_b
        torch.cuda.empty_cache()
        
        results.append({
            "M_A": M_A,
            "M_B": M_B,
            "vec_dim": vec_dim,
            "pytorch_ms": pytorch_ms,
            "custom_ms": custom_ms,
            "speedup": speedup,
            "pairs_per_s": custom_pairs_per_s,
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark dot product: custom vs PyTorch")
    parser.add_argument("--sizes", type=str, default="256,512,1024,2048,4096,8192,16384,32768,65536,131072",
                        help="Comma-separated list of M sizes")
    parser.add_argument("--vec-dim", type=int, default=512, help="Vector dimension")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--repeats", type=int, default=10, help="Timed iterations")
    parser.add_argument("--ab", action="store_true", help="Also run A vs B benchmark")
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Vector dimension: {args.vec_dim}")
    print(f"Warmup: {args.warmup}, Repeats: {args.repeats}")
    
    sizes = [int(s) for s in args.sizes.split(",")]
    
    print("\n" + "=" * 60)
    print("SELF-COMPARISON BENCHMARK (pairwise within single set)")
    print("=" * 60)
    self_results = run_self_benchmark(sizes, args.vec_dim, args.warmup, args.repeats)
    
    if args.ab:
        print("\n" + "=" * 60)
        print("A vs B BENCHMARK (cross-set comparison)")
        print("=" * 60)
        ab_sizes = [(s, s) for s in sizes] + [(s, s * 2) for s in sizes[:3]]
        ab_results = run_ab_benchmark(ab_sizes, args.vec_dim, args.warmup, args.repeats)
    
    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY - Dense Compute Performance (GPU only, no CPU copy)")
    print("=" * 90)
    print(f"{'Size':<10} {'PyTorch mm':<14} {'Custom WMMA':<14} {'Ratio':<10}")
    print("-" * 50)
    for r in self_results:
        print(f"{r['M']:<10} {r['pytorch_ms']:<14.3f} {r['native_ms']:<14.3f} {r['native_ratio']:<10.1f}x")
    
    print("\n" + "=" * 90)
    print("SUMMARY - Filtered Workflow Performance (mm + extract lower tri + threshold)")
    print("=" * 90)
    print(f"{'Size':<10} {'PyTorch workflow':<18} {'Custom fused':<14} {'Speedup':<10} {'Memory saved':<14}")
    print("-" * 70)
    for r in self_results:
        mem_mb = r['M'] * r['M'] * 4 / 1e6
        print(f"{r['M']:<10} {r['pytorch_full_ms']:<18.3f} {r['sparse_ms']:<14.3f} {r['filter_speedup']:<10.2f}x {mem_mb:<14.1f} MB")
    
    print("\n" + "=" * 90)
    print("NOTES")
    print("=" * 90)
    print("- 'PyTorch mm only' = pure matrix multiply on GPU (no CPU copy)")
    print("- 'PyTorch workflow' = mm + extract + filter + copy to CPU pinned memory")  
    print("- 'Custom fused filter' = single kernel + copy to CPU pinned memory (like FHD kernel)")
    print("- Both filtered workflows include async copy to CPU pinned memory")
    print("- Custom kernel advantage: avoids allocating full MxM matrix (memory savings shown)")


if __name__ == "__main__":
    main()

