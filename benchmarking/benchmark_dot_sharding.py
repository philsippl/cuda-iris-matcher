#!/usr/bin/env python3
"""
Benchmark: Dot product sharding scalability across multiple GPUs.

Tests that the sharded dot product kernel scales approximately linearly
with the number of GPUs available.
"""

import argparse
import os
import torch
import time
import numpy as np
from typing import Dict, List, Tuple, Optional

import cuda_iris_matcher as ih


def get_device_count() -> int:
    """Get number of CUDA devices available."""
    return torch.cuda.device_count()


def create_test_data(M: int, vec_dim: int, seed: int = 42) -> torch.Tensor:
    """Create normalized random test data on CPU."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    data = np.random.randn(M, vec_dim).astype(np.float32)
    data = data / np.linalg.norm(data, axis=1, keepdims=True)
    return torch.from_numpy(data.astype(np.float16))


def benchmark_sharded_self(
    data: torch.Tensor,
    num_gpus: int,
    warmup: int,
    repeats: int,
    max_pairs: int = 1_000_000,
    verbose: bool = False,
) -> Tuple[float, int]:
    """
    Benchmark sharded dot product for self-comparison.
    
    Returns:
        (time_ms, pair_count)
    """
    M = data.size(0)
    vec_dim = data.size(1)
    
    # Clear GPU caches
    for i in range(get_device_count()):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
    
    # Warmup
    for _ in range(warmup):
        result = ih.dot_product_sharded(
            data,
            match_threshold=100.0,  # Include all pairs
            non_match_threshold=-100.0,
            include_flags=ih.INCLUDE_ALL,
            max_pairs=max_pairs,
            min_shards=1,  # Let it auto-configure based on num_gpus
        )
        del result
        torch.cuda.synchronize()
    
    # Timed runs
    times = []
    pair_count = 0
    
    for _ in range(repeats):
        # Synchronize all GPUs
        for i in range(get_device_count()):
            torch.cuda.synchronize(i)
        
        start = time.perf_counter()
        
        pairs, cats, scores, count = ih.dot_product_sharded(
            data,
            match_threshold=100.0,
            non_match_threshold=-100.0,
            include_flags=ih.INCLUDE_ALL,
            max_pairs=max_pairs,
            min_shards=1,
        )
        
        # Synchronize all GPUs
        for i in range(get_device_count()):
            torch.cuda.synchronize(i)
        
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
        pair_count = count.item()
        
        del pairs, cats, scores, count
    
    avg_time = np.mean(times)
    return avg_time, pair_count


def diagnose_sharding_overhead(
    M: int,
    vec_dim: int,
) -> None:
    """Diagnose where the sharding overhead comes from."""
    import cuda_iris_matcher._C as _C
    
    num_gpus = get_device_count()
    
    print("=" * 80)
    print("SHARDING OVERHEAD DIAGNOSIS")
    print("=" * 80)
    print(f"M={M}, vec_dim={vec_dim}, GPUs={num_gpus}")
    print()
    
    data = create_test_data(M, vec_dim)
    n_pairs = M * (M - 1) // 2
    max_pairs = min(n_pairs, 10_000_000)
    
    # 1. Time just the kernel on one GPU (no sharding overhead)
    print("1. Single kernel on GPU 0 (data pre-loaded):")
    with torch.cuda.device(0):
        torch.cuda.empty_cache()
        data_gpu = data.cuda(0)
        
        # Warmup
        for _ in range(3):
            result = ih.dot_product_cuda(data_gpu, max_pairs=max_pairs)
            del result
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        pairs, cats, scores, count = ih.dot_product_cuda(data_gpu, max_pairs=max_pairs)
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        kernel_time = (end - start) * 1000
        print(f"   Time: {kernel_time:.2f} ms, Pairs: {count.item():,}")
        del data_gpu, pairs, cats, scores, count
    
    # 2. Time data transfer only
    print("\n2. CPU -> GPU transfer time:")
    torch.cuda.synchronize()
    start = time.perf_counter()
    data_gpu = data.cuda(0)
    torch.cuda.synchronize()
    end = time.perf_counter()
    transfer_time = (end - start) * 1000
    print(f"   Time: {transfer_time:.2f} ms")
    del data_gpu
    
    # 3. Time the full sharded flow
    print("\n3. Full sharded flow (all GPUs):")
    for i in range(num_gpus):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
    
    # Warmup
    for _ in range(2):
        result = ih.dot_product_sharded(data, max_pairs=max_pairs)
        del result
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    pairs, cats, scores, count = ih.dot_product_sharded(data, max_pairs=max_pairs)
    for i in range(num_gpus):
        torch.cuda.synchronize(i)
    end = time.perf_counter()
    sharded_time = (end - start) * 1000
    print(f"   Time: {sharded_time:.2f} ms, Pairs: {count.item():,}")
    
    # 4. Breakdown
    print("\n4. Analysis:")
    print(f"   Kernel compute time: {kernel_time:.2f} ms")
    print(f"   Data transfer time:  {transfer_time:.2f} ms")
    print(f"   Sharded total time:  {sharded_time:.2f} ms")
    overhead = sharded_time - kernel_time
    print(f"   Overhead: {overhead:.2f} ms ({overhead/kernel_time*100:.1f}% of kernel time)")
    
    if overhead > 10 * kernel_time:
        print(f"\n   ⚠ SEVERE OVERHEAD: Sharding adds {overhead/kernel_time:.0f}x the kernel time")
        print("      Possible causes:")
        print("      - Too many small shards")
        print("      - Sequential data transfers blocking parallelism")
        print("      - Result aggregation overhead")
    
    del data


def benchmark_single_gpu(
    data: torch.Tensor,
    device_id: int,
    warmup: int,
    repeats: int,
    max_pairs: int = 1_000_000,
    include_transfer: bool = False,
) -> Tuple[float, int]:
    """
    Benchmark non-sharded dot product on a single GPU.
    
    Args:
        include_transfer: If True, include CPU->GPU transfer in timing
    
    Returns:
        (time_ms, pair_count)
    """
    with torch.cuda.device(device_id):
        torch.cuda.empty_cache()
        
        if not include_transfer:
            # Pre-load data to GPU
            data_gpu = data.cuda(device_id)
        
        # Warmup
        for _ in range(warmup):
            if include_transfer:
                data_gpu = data.cuda(device_id)
            result = ih.dot_product_cuda(
                data_gpu,
                match_threshold=100.0,
                non_match_threshold=-100.0,
                include_flags=ih.INCLUDE_ALL,
                max_pairs=max_pairs,
            )
            del result
            if include_transfer:
                del data_gpu
            torch.cuda.synchronize()
        
        # Timed runs
        times = []
        pair_count = 0
        
        for _ in range(repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            if include_transfer:
                data_gpu = data.cuda(device_id)
            
            pairs, cats, scores, count = ih.dot_product_cuda(
                data_gpu,
                match_threshold=100.0,
                non_match_threshold=-100.0,
                include_flags=ih.INCLUDE_ALL,
                max_pairs=max_pairs,
            )
            
            torch.cuda.synchronize()
            end = time.perf_counter()
            
            times.append((end - start) * 1000)
            pair_count = count.item()
            del pairs, cats, scores, count
            if include_transfer:
                del data_gpu
        
        if not include_transfer:
            del data_gpu
        avg_time = np.mean(times)
        return avg_time, pair_count


def benchmark_forced_sharding(
    data: torch.Tensor,
    min_shards: int,
    warmup: int,
    repeats: int,
    max_pairs: int = 1_000_000,
) -> Tuple[float, int]:
    """
    Benchmark sharded dot product with forced number of shards.
    Tests the sharding overhead on a single GPU.
    
    Returns:
        (time_ms, pair_count)
    """
    # Clear GPU caches
    for i in range(get_device_count()):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
    
    # Warmup
    for _ in range(warmup):
        result = ih.dot_product_sharded(
            data,
            match_threshold=100.0,
            non_match_threshold=-100.0,
            include_flags=ih.INCLUDE_ALL,
            max_pairs=max_pairs,
            min_shards=min_shards,
        )
        del result
        torch.cuda.synchronize()
    
    # Timed runs
    times = []
    pair_count = 0
    
    for _ in range(repeats):
        for i in range(get_device_count()):
            torch.cuda.synchronize(i)
        
        start = time.perf_counter()
        
        pairs, cats, scores, count = ih.dot_product_sharded(
            data,
            match_threshold=100.0,
            non_match_threshold=-100.0,
            include_flags=ih.INCLUDE_ALL,
            max_pairs=max_pairs,
            min_shards=min_shards,
        )
        
        for i in range(get_device_count()):
            torch.cuda.synchronize(i)
        
        end = time.perf_counter()
        times.append((end - start) * 1000)
        pair_count = count.item()
        del pairs, cats, scores, count
    
    avg_time = np.mean(times)
    return avg_time, pair_count


def run_scaling_benchmark(
    sizes: List[int],
    vec_dim: int,
    warmup: int,
    repeats: int,
) -> None:
    """Run scaling benchmark across multiple GPU configurations."""
    num_gpus = get_device_count()
    
    print("=" * 80)
    print("DOT PRODUCT SHARDING SCALABILITY BENCHMARK")
    print("=" * 80)
    print(f"Available GPUs: {num_gpus}")
    print(f"Vector dimension: {vec_dim}")
    print(f"Warmup iterations: {warmup}")
    print(f"Timed iterations: {repeats}")
    print()
    print("NOTE: Sharded version includes CPU->GPU data transfer for each shard.")
    print("      This is by design for handling large datasets that don't fit in GPU memory.")
    print()
    
    for M in sizes:
        n_pairs = M * (M - 1) // 2
        max_pairs = min(n_pairs, 10_000_000)
        
        print("-" * 80)
        print(f"Dataset size: M={M:,} vectors ({n_pairs:,} pairs)")
        print("-" * 80)
        
        # Create test data
        data = create_test_data(M, vec_dim)
        
        # Benchmark 1: Single GPU compute-only (data pre-loaded)
        print("\n  1. Single GPU (compute only, data pre-loaded):")
        try:
            compute_time, baseline_pairs = benchmark_single_gpu(
                data, device_id=0, warmup=warmup, repeats=repeats, 
                max_pairs=max_pairs, include_transfer=False
            )
            compute_tps = baseline_pairs / (compute_time / 1000) if compute_time > 0 else 0
            print(f"     Time: {compute_time:.2f} ms")
            print(f"     Throughput: {compute_tps / 1e9:.3f} billion pairs/sec")
        except Exception as e:
            print(f"     Error: {e}")
            compute_time = None
        
        # Benchmark 2: Single GPU with transfer (fair comparison to sharded)
        print("\n  2. Single GPU (with CPU->GPU transfer):")
        try:
            transfer_time, _ = benchmark_single_gpu(
                data, device_id=0, warmup=warmup, repeats=repeats, 
                max_pairs=max_pairs, include_transfer=True
            )
            transfer_tps = baseline_pairs / (transfer_time / 1000) if transfer_time > 0 else 0
            print(f"     Time: {transfer_time:.2f} ms")
            print(f"     Throughput: {transfer_tps / 1e9:.3f} billion pairs/sec")
            if compute_time is not None:
                transfer_overhead = transfer_time - compute_time
                print(f"     Transfer overhead: {transfer_overhead:.2f} ms")
        except Exception as e:
            print(f"     Error: {e}")
            transfer_time = None
        
        # Benchmark 3: Sharded with all GPUs
        print(f"\n  3. Sharded ({num_gpus} GPUs, CPU data, parallel transfers):")
        try:
            sharded_time, pair_count = benchmark_sharded_self(
                data, num_gpus, warmup, repeats, max_pairs
            )
            sharded_tps = pair_count / (sharded_time / 1000) if sharded_time > 0 else 0
            print(f"     Time: {sharded_time:.2f} ms")
            print(f"     Throughput: {sharded_tps / 1e9:.3f} billion pairs/sec")
            
            # Compare to single GPU with transfer (fair comparison)
            if transfer_time is not None:
                speedup = transfer_time / sharded_time
                efficiency = (speedup / num_gpus) * 100
                print(f"     Speedup vs single GPU (with transfer): {speedup:.2f}x")
                print(f"     Scaling efficiency: {efficiency:.1f}%")
                
                if efficiency >= 70:
                    print(f"     ✓ Good scaling (>70% efficiency)")
                elif efficiency >= 50:
                    print(f"     ~ Moderate scaling (50-70% efficiency)")
                else:
                    print(f"     ✗ Poor scaling (<50% efficiency)")
        except Exception as e:
            print(f"     Error: {e}")
            sharded_time = None
        
        # Benchmark 4: Test scaling with different shard counts
        if num_gpus > 1:
            print(f"\n  4. Shard Count Scaling Test (on {num_gpus} GPUs):")
            shard_configs = [1, 2, 4, 8]
            if num_gpus >= 4:
                shard_configs.append(16)
            
            for n_shards in shard_configs:
                try:
                    time_ms, pair_count = benchmark_forced_sharding(
                        data, n_shards, warmup, repeats, max_pairs
                    )
                    tps = pair_count / (time_ms / 1000) if time_ms > 0 else 0
                    
                    print(f"     {n_shards:2d} shards: {time_ms:7.2f} ms, {tps / 1e9:.3f} B pairs/sec")
                except Exception as e:
                    print(f"     {n_shards:2d} shards: Error - {e}")
        
        del data
        torch.cuda.empty_cache()
        print()


def run_gpu_comparison(
    M: int,
    vec_dim: int,
    warmup: int,
    repeats: int,
) -> None:
    """
    Run detailed comparison between different GPU counts.
    Uses CUDA_VISIBLE_DEVICES approach for controlled testing.
    """
    num_gpus = get_device_count()
    
    if num_gpus <= 1:
        print("Need at least 2 GPUs for GPU comparison benchmark")
        return
    
    print("=" * 80)
    print("GPU SCALING COMPARISON")
    print("=" * 80)
    print(f"Dataset size: M={M:,} vectors")
    print(f"Vector dimension: {vec_dim}")
    print(f"Available GPUs: {num_gpus}")
    print()
    
    n_pairs = M * (M - 1) // 2
    max_pairs = min(n_pairs, 10_000_000)
    
    data = create_test_data(M, vec_dim)
    
    # Test with all GPUs
    print("Testing with all available GPUs...")
    all_gpu_time, pair_count = benchmark_sharded_self(
        data, num_gpus, warmup, repeats, max_pairs
    )
    all_gpu_tps = pair_count / (all_gpu_time / 1000)
    
    print(f"  {num_gpus} GPUs: {all_gpu_time:.2f} ms, {all_gpu_tps / 1e9:.3f} B pairs/sec")
    
    # Test single GPU baseline (with transfer for fair comparison)
    print("\nTesting single GPU baseline (with transfer)...")
    single_gpu_time, _ = benchmark_single_gpu(
        data, device_id=0, warmup=warmup, repeats=repeats, 
        max_pairs=max_pairs, include_transfer=True
    )
    single_gpu_tps = pair_count / (single_gpu_time / 1000)
    
    print(f"  1 GPU: {single_gpu_time:.2f} ms, {single_gpu_tps / 1e9:.3f} B pairs/sec")
    
    # Calculate scaling
    speedup = single_gpu_time / all_gpu_time
    efficiency = (speedup / num_gpus) * 100
    linear_speedup = num_gpus
    
    print("\n" + "=" * 80)
    print("SCALING RESULTS")
    print("=" * 80)
    print(f"Single GPU time:    {single_gpu_time:.2f} ms")
    print(f"{num_gpus} GPU time:        {all_gpu_time:.2f} ms")
    print(f"Actual speedup:     {speedup:.2f}x")
    print(f"Ideal speedup:      {linear_speedup:.2f}x")
    print(f"Scaling efficiency: {efficiency:.1f}%")
    
    if efficiency >= 80:
        print("\n✓ PASS: Scaling efficiency >= 80% (near-linear scaling)")
    elif efficiency >= 60:
        print("\n~ ACCEPTABLE: Scaling efficiency 60-80%")
    else:
        print("\n✗ POOR: Scaling efficiency < 60%")
    
    del data


def run_large_scale_test(
    vec_dim: int,
    warmup: int,
    repeats: int,
) -> None:
    """
    Test with very large datasets where sharding is beneficial.
    Focus on memory efficiency and practical use cases.
    """
    num_gpus = get_device_count()
    
    print("=" * 80)
    print("LARGE SCALE MULTI-GPU TEST")
    print("=" * 80)
    print(f"Available GPUs: {num_gpus}")
    print(f"Vector dimension: {vec_dim}")
    print()
    print("Sharding is designed for:")
    print("  1. Datasets too large to fit on a single GPU")
    print("  2. Returning sparse results (filtered pairs) to save memory")
    print()
    print("For small datasets that fit on one GPU, use dot_product_cuda directly.")
    print()
    
    # Test sizes that demonstrate sharding value
    sizes = [8192, 16384, 32768, 65536, 131072, 262144]
    
    for M in sizes:
        n_pairs = M * (M - 1) // 2
        # Use a realistic max_pairs limit
        max_pairs = min(n_pairs, 1_000_000)  # Return at most 1M pairs
        data_size_mb = M * vec_dim * 2 / 1e6  # f16 = 2 bytes
        full_matrix_mb = M * M * 4 / 1e6  # float32 result matrix
        
        print("-" * 80)
        print(f"M={M:,} vectors ({n_pairs:,} pairs)")
        print(f"Data size: {data_size_mb:.1f} MB")
        print(f"Full result matrix would be: {full_matrix_mb:.1f} MB")
        print(f"Sparse result (max {max_pairs:,} pairs): {max_pairs * 13 / 1e6:.1f} MB")
        print("-" * 80)
        
        try:
            data = create_test_data(M, vec_dim)
        except Exception as e:
            print(f"  Cannot create data of size M={M}: {e}")
            continue
        
        # Single GPU baseline
        print("\n  Single GPU (dot_product_cuda):")
        try:
            single_time, pair_count = benchmark_single_gpu(
                data, device_id=0, warmup=warmup, repeats=repeats,
                max_pairs=max_pairs, include_transfer=True
            )
            print(f"    Time: {single_time:.2f} ms")
            print(f"    Pairs returned: {pair_count:,}")
        except Exception as e:
            print(f"    Error: {e}")
            single_time = None
        
        # Multi-GPU sharded
        print(f"\n  Sharded ({num_gpus} GPUs, dot_product_sharded):")
        try:
            sharded_time, pair_count = benchmark_sharded_self(
                data, num_gpus, warmup, repeats, max_pairs
            )
            print(f"    Time: {sharded_time:.2f} ms")
            print(f"    Pairs returned: {pair_count:,}")
            
            if single_time is not None:
                if sharded_time < single_time:
                    speedup = single_time / sharded_time
                    print(f"    Speedup: {speedup:.2f}x vs single GPU")
                else:
                    slowdown = sharded_time / single_time
                    print(f"    Slowdown: {slowdown:.2f}x vs single GPU (expected for small workloads)")
        except Exception as e:
            print(f"    Error: {e}")
        
        del data
        torch.cuda.empty_cache()
        print()
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("PERFORMANCE CHARACTERISTICS:")
    print("  - Small datasets (M < 16K): Single GPU is faster due to sharding overhead")
    print("  - Large datasets (M >= 32K): Sharding matches or exceeds single GPU")
    print("  - For sparse output (returning few pairs), sharding scales well")
    print()
    print("USE dot_product_sharded WHEN:")
    print("  ✓ Dataset is too large for single GPU memory")
    print("  ✓ Processing very large datasets (M >= 32K)")
    print("  ✓ Need to distribute across multiple hosts")
    print("  ✓ Returning sparse results (filtered pairs)")
    print()
    print("USE dot_product_cuda WHEN:")
    print("  ✓ Dataset fits on a single GPU and M < 16K")
    print("  ✓ Need minimum latency for small workloads")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark dot product sharding scalability"
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="1024,2048,4096,8192",
        help="Comma-separated list of dataset sizes (M)"
    )
    parser.add_argument(
        "--vec-dim",
        type=int,
        default=512,
        help="Vector dimension (default: 512)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Warmup iterations (default: 3)"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Timed iterations (default: 5)"
    )
    parser.add_argument(
        "--comparison",
        type=int,
        default=0,
        help="Run GPU comparison with specified M (0 to skip)"
    )
    parser.add_argument(
        "--large-scale",
        action="store_true",
        help="Run large scale test with M=16384,32768"
    )
    parser.add_argument(
        "--diagnose",
        type=int,
        default=0,
        help="Run overhead diagnosis with specified M"
    )
    
    args = parser.parse_args()
    
    sizes = [int(s.strip()) for s in args.sizes.split(",")]
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        return
    
    # Print GPU info
    num_gpus = get_device_count()
    print(f"CUDA devices available: {num_gpus}")
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    print()
    
    if args.diagnose > 0:
        diagnose_sharding_overhead(
            M=args.diagnose,
            vec_dim=args.vec_dim,
        )
    elif args.large_scale:
        run_large_scale_test(
            vec_dim=args.vec_dim,
            warmup=args.warmup,
            repeats=args.repeats,
        )
    elif args.comparison > 0:
        run_gpu_comparison(
            M=args.comparison,
            vec_dim=args.vec_dim,
            warmup=args.warmup,
            repeats=args.repeats,
        )
    else:
        run_scaling_benchmark(
            sizes=sizes,
            vec_dim=args.vec_dim,
            warmup=args.warmup,
            repeats=args.repeats,
        )


if __name__ == "__main__":
    main()

