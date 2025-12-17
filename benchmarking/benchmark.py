#!/usr/bin/env python3
"""
Comprehensive benchmark for CUDA iris matcher.

Measures:
- Kernel execution time
- Throughput (pairs/second)
- GPU memory usage
- CPU/RAM usage
- Scaling with different batch sizes

Usage:
    python benchmark.py [--sizes 256,512,1024,2048,4096] [--warmup 3] [--repeats 5]
"""

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np
import torch

import cuda_iris_matcher as ih


@dataclass
class BenchmarkResult:
    """Single benchmark run result."""
    batch_size: int
    n_pairs: int

    # Timing (milliseconds)
    pack_time_ms: float
    kernel_time_ms: float
    total_time_ms: float

    # Throughput
    pairs_per_second: float
    pairs_per_second_millions: float

    # Memory (MB)
    gpu_memory_allocated_mb: float
    gpu_memory_reserved_mb: float
    peak_gpu_memory_mb: float

    # Device info
    device_name: str
    device_index: int


@dataclass
class BenchmarkSummary:
    """Summary of multiple benchmark runs."""
    batch_size: int
    n_pairs: int

    # Timing statistics (ms)
    pack_time_ms_mean: float
    pack_time_ms_std: float
    kernel_time_ms_mean: float
    kernel_time_ms_std: float
    total_time_ms_mean: float
    total_time_ms_std: float

    # Throughput statistics
    pairs_per_second_millions_mean: float
    pairs_per_second_millions_std: float

    # Memory
    gpu_memory_allocated_mb: float
    peak_gpu_memory_mb: float

    device_name: str
    n_runs: int


def get_gpu_memory_stats() -> dict:
    """Get current GPU memory statistics."""
    if not torch.cuda.is_available():
        return {"allocated_mb": 0, "reserved_mb": 0}

    allocated = torch.cuda.memory_allocated() / (1024 * 1024)
    reserved = torch.cuda.memory_reserved() / (1024 * 1024)
    return {"allocated_mb": allocated, "reserved_mb": reserved}


def run_single_benchmark(
    M: int,
    device_idx: int = 0,
    include_packing: bool = True,
    max_pairs_fraction: float = 0.01,
) -> BenchmarkResult:
    """
    Run a single benchmark iteration.

    Args:
        M: Batch size (number of iris codes)
        device_idx: CUDA device index
        include_packing: Whether to include packing in timing
        max_pairs_fraction: Fraction of total pairs to allocate for output

    Returns:
        BenchmarkResult with timing and memory stats
    """
    torch.cuda.set_device(device_idx)
    device = torch.device(f"cuda:{device_idx}")

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    gc.collect()

    n_pairs = M * (M - 1) // 2
    max_pairs = max(1, int(n_pairs * max_pairs_fraction))

    # Generate random packed data (skip actual iris code generation for speed)
    data = torch.randint(
        low=0, high=2**31, size=(M, 400), dtype=torch.int32, device=device
    ).contiguous()
    mask = torch.full(
        (M, 400), fill_value=0x7FFFFFFF, dtype=torch.int32, device=device
    ).contiguous()

    torch.cuda.synchronize(device)
    mem_before = get_gpu_memory_stats()

    # Timing events
    start_event = torch.cuda.Event(enable_timing=True)
    pack_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()

    # If testing with real unpacked data, packing would go here
    # For this benchmark, data is already packed
    pack_event.record()

    # Run kernel
    pair_indices, categories, distances, count = ih.masked_hamming_cuda(
        data,
        mask,
        match_threshold=1.0,
        non_match_threshold=1.0,
        include_flags=ih.INCLUDE_ALL,
        max_pairs=max_pairs,
    )

    end_event.record()
    torch.cuda.synchronize(device)

    # Get timing
    pack_time_ms = start_event.elapsed_time(pack_event)
    kernel_time_ms = pack_event.elapsed_time(end_event)
    total_time_ms = start_event.elapsed_time(end_event)

    # Get memory stats
    mem_after = get_gpu_memory_stats()
    peak_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    # Calculate throughput
    pairs_per_second = n_pairs / (kernel_time_ms / 1000.0)
    pairs_per_second_millions = pairs_per_second / 1e6

    return BenchmarkResult(
        batch_size=M,
        n_pairs=n_pairs,
        pack_time_ms=pack_time_ms,
        kernel_time_ms=kernel_time_ms,
        total_time_ms=total_time_ms,
        pairs_per_second=pairs_per_second,
        pairs_per_second_millions=pairs_per_second_millions,
        gpu_memory_allocated_mb=mem_after["allocated_mb"],
        gpu_memory_reserved_mb=mem_after["reserved_mb"],
        peak_gpu_memory_mb=peak_memory,
        device_name=torch.cuda.get_device_name(device_idx),
        device_index=device_idx,
    )


def run_benchmark_suite(
    batch_sizes: List[int],
    warmup_runs: int = 3,
    benchmark_runs: int = 5,
    device_idx: int = 0,
) -> List[BenchmarkSummary]:
    """
    Run benchmark suite across multiple batch sizes.

    Args:
        batch_sizes: List of M values to benchmark
        warmup_runs: Number of warmup iterations per size
        benchmark_runs: Number of timed iterations per size
        device_idx: CUDA device index

    Returns:
        List of BenchmarkSummary for each batch size
    """
    results = []

    for M in batch_sizes:
        print(f"\nBenchmarking M={M:,} ({M*(M-1)//2:,} pairs)...")

        # Warmup
        for _ in range(warmup_runs):
            run_single_benchmark(M, device_idx)
            torch.cuda.empty_cache()

        # Benchmark runs
        run_results = []
        for i in range(benchmark_runs):
            result = run_single_benchmark(M, device_idx)
            run_results.append(result)
            torch.cuda.empty_cache()

        # Calculate statistics
        pack_times = [r.pack_time_ms for r in run_results]
        kernel_times = [r.kernel_time_ms for r in run_results]
        total_times = [r.total_time_ms for r in run_results]
        throughputs = [r.pairs_per_second_millions for r in run_results]

        summary = BenchmarkSummary(
            batch_size=M,
            n_pairs=run_results[0].n_pairs,
            pack_time_ms_mean=np.mean(pack_times),
            pack_time_ms_std=np.std(pack_times),
            kernel_time_ms_mean=np.mean(kernel_times),
            kernel_time_ms_std=np.std(kernel_times),
            total_time_ms_mean=np.mean(total_times),
            total_time_ms_std=np.std(total_times),
            pairs_per_second_millions_mean=np.mean(throughputs),
            pairs_per_second_millions_std=np.std(throughputs),
            gpu_memory_allocated_mb=run_results[-1].gpu_memory_allocated_mb,
            peak_gpu_memory_mb=max(r.peak_gpu_memory_mb for r in run_results),
            device_name=run_results[0].device_name,
            n_runs=benchmark_runs,
        )

        results.append(summary)

        print(f"  Kernel: {summary.kernel_time_ms_mean:.2f} \u00b1 {summary.kernel_time_ms_std:.2f} ms")
        print(f"  Throughput: {summary.pairs_per_second_millions_mean:.2f} \u00b1 {summary.pairs_per_second_millions_std:.2f} M pairs/s")
        print(f"  Peak GPU Memory: {summary.peak_gpu_memory_mb:.1f} MB")

    return results


@dataclass
class ShardingBenchmarkResult:
    """Result from sharding benchmark."""
    m_a: int
    m_b: int
    n_pairs: int
    num_devices: int
    num_shards: int
    
    # Timing (ms)
    total_time_ms_mean: float
    total_time_ms_std: float
    
    # Throughput
    pairs_per_second_millions: float
    
    # Comparison with non-sharded
    speedup_vs_single: Optional[float]  # None if not comparable
    
    n_runs: int


def run_sharding_benchmark(
    sizes: List[int],
    warmup_runs: int = 2,
    benchmark_runs: int = 5,
    min_shards_list: Optional[List[int]] = None,
    skip_baseline: bool = False,
    include_packing: bool = False,
) -> List[ShardingBenchmarkResult]:
    """
    Benchmark sharded operations across all available devices.
    
    Args:
        sizes: List of M values (will test M×M self-comparison and M×M A vs B)
        warmup_runs: Number of warmup iterations
        benchmark_runs: Number of timed iterations
        min_shards_list: List of min_shards values to test (for simulating multi-GPU)
        skip_baseline: Skip single-GPU baseline benchmark
        include_packing: Also benchmark with unpacked (uint8) data to measure packing overhead
    
    Returns:
        List of ShardingBenchmarkResult
    """
    num_devices = torch.cuda.device_count()
    results = []
    
    if min_shards_list is None:
        # Default: just test baseline (1) and one sharded config (4 shards for tiling)
        min_shards_list = [1, 4]
    
    # Data format configurations to test
    data_formats = [("packed", True)]
    if include_packing:
        data_formats.append(("unpacked", False))
    
    for M in sizes:
        n_pairs_self = M * (M - 1) // 2
        
        print(f"\n  M={M:,} ({n_pairs_self:,} pairs)")
        
        # Generate packed data on CPU (sharding will distribute to GPUs)
        data_packed = torch.randint(0, 2**31, (M, 400), dtype=torch.int32)
        mask_packed = torch.full((M, 400), 0x7FFFFFFF, dtype=torch.int32)
        
        # Generate unpacked data if needed (shape: [M, r_dim=16, theta_dim=200, d0_dim=2, d1_dim=2])
        if include_packing:
            # Default iris code dimensions (from iris_params.h)
            R_DIM, THETA_DIM, D0_DIM, D1_DIM = 16, 200, 2, 2
            data_unpacked = torch.randint(0, 2, (M, R_DIM, THETA_DIM, D0_DIM, D1_DIM), dtype=torch.uint8)
            mask_unpacked = torch.ones((M, R_DIM, THETA_DIM, D0_DIM, D1_DIM), dtype=torch.uint8)
        
        # Select data based on format
        data = data_packed
        mask = mask_packed
        
        baseline_mean = None
        baseline_throughput = None
        
        if not skip_baseline:
            # Get baseline with non-sharded on single GPU
            data_gpu = data.cuda(0)
            mask_gpu = mask.cuda(0)
            max_pairs_baseline = 100 #max(1, n_pairs_self // 100)
            
            # Warmup + benchmark baseline
            for _ in range(warmup_runs):
                ih.masked_hamming_cuda(data_gpu, mask_gpu, match_threshold=1.0, max_pairs=max_pairs_baseline)
                torch.cuda.synchronize()
            
            baseline_times = []
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            for _ in range(benchmark_runs):
                start_event.record()
                ih.masked_hamming_cuda(data_gpu, mask_gpu, match_threshold=1.0, max_pairs=max_pairs_baseline)
                end_event.record()
                torch.cuda.synchronize()
                baseline_times.append(start_event.elapsed_time(end_event))
            
            baseline_mean = np.mean(baseline_times)
            baseline_std = np.std(baseline_times)
            baseline_throughput = n_pairs_self / (baseline_mean / 1000) / 1e6
            
            results.append(ShardingBenchmarkResult(
                m_a=M, m_b=M, n_pairs=n_pairs_self,
                num_devices=1, num_shards=1,
                total_time_ms_mean=baseline_mean,
                total_time_ms_std=baseline_std,
                pairs_per_second_millions=baseline_throughput,
                speedup_vs_single=1.0,
                n_runs=benchmark_runs,
            ))
            
            del data_gpu, mask_gpu
            torch.cuda.empty_cache()
        
        # Test sharded version (skip if min_shards=1 since that's baseline)
        sharded_min = max(s for s in min_shards_list if s > 1) if any(s > 1 for s in min_shards_list) else None
        
        if sharded_min:
            shard_configs = ih.get_shard_info(M, M, min_shards=sharded_min)
            actual_shards = len(shard_configs)
            max_pairs_sharded = max(1, n_pairs_self // 100)
            
            # Test each data format (packed, and optionally unpacked)
            for format_name, is_packed in data_formats:
                if is_packed:
                    test_data, test_mask = data_packed, mask_packed
                else:
                    test_data, test_mask = data_unpacked, mask_unpacked
                
                # Warmup + benchmark sharded
                for _ in range(warmup_runs):
                    ih.masked_hamming_sharded(test_data, test_mask, match_threshold=1.0, max_pairs=max_pairs_sharded, min_shards=sharded_min)
                
                sharded_times = []
                for _ in range(benchmark_runs):
                    # Sync all GPUs before timing
                    for d in range(num_devices):
                        torch.cuda.synchronize(d)
                    start = time.perf_counter()
                    ih.masked_hamming_sharded(test_data, test_mask, match_threshold=1.0, max_pairs=max_pairs_sharded, min_shards=sharded_min)
                    # Sync all GPUs after to ensure completion
                    for d in range(num_devices):
                        torch.cuda.synchronize(d)
                    sharded_times.append((time.perf_counter() - start) * 1000)
                
                sharded_mean = np.mean(sharded_times)
                sharded_std = np.std(sharded_times)
                sharded_throughput = n_pairs_self / (sharded_mean / 1000) / 1e6
                speedup = baseline_mean / sharded_mean if baseline_mean else None
                
                # Print compact result line
                format_label = f"[{format_name}] " if include_packing else ""
                if baseline_mean:
                    print(f"    Baseline: {baseline_mean:.2f}ms ({baseline_throughput:.1f}M/s) | "
                          f"Sharded {format_label}({actual_shards} tiles): {sharded_mean:.2f}ms ({sharded_throughput:.1f}M/s) | "
                          f"Speedup: {speedup:.2f}x")
                else:
                    print(f"    Sharded {format_label}({actual_shards} tiles): {sharded_mean:.2f}ms ({sharded_throughput:.1f}M/s)")
                
                results.append(ShardingBenchmarkResult(
                    m_a=M, m_b=M, n_pairs=n_pairs_self,
                    num_devices=num_devices, num_shards=actual_shards,
                    total_time_ms_mean=sharded_mean,
                    total_time_ms_std=sharded_std,
                    pairs_per_second_millions=sharded_throughput,
                    speedup_vs_single=speedup,
                    n_runs=benchmark_runs,
                ))
        elif not skip_baseline:
            print(f"    Baseline: {baseline_mean:.2f}ms ({baseline_throughput:.1f}M/s)")
        
        del data_packed, mask_packed
        if include_packing:
            del data_unpacked, mask_unpacked
        gc.collect()
        torch.cuda.empty_cache()
    
    return results


def print_sharding_summary(results: List[ShardingBenchmarkResult]):
    """Print sharding benchmark summary (already printed inline)."""
    # Summary already printed per-size, just show device count
    if results:
        print(f"\n  GPUs available: {torch.cuda.device_count()}")


def run_ab_benchmark(
    sizes_a: List[int],
    sizes_b: List[int],
    warmup_runs: int = 2,
    benchmark_runs: int = 3,
    device_idx: int = 0,
) -> List[dict]:
    """
    Benchmark A vs B kernel with various size combinations.

    Args:
        sizes_a: List of M_A values
        sizes_b: List of M_B values
        warmup_runs: Warmup iterations
        benchmark_runs: Timed iterations
        device_idx: CUDA device index

    Returns:
        List of result dictionaries
    """
    torch.cuda.set_device(device_idx)
    device = torch.device(f"cuda:{device_idx}")
    results = []

    for M_A in sizes_a:
        for M_B in sizes_b:
            n_pairs = M_A * M_B
            print(f"\nBenchmarking A vs B: M_A={M_A}, M_B={M_B} ({n_pairs:,} pairs)...")

            # Create data
            data_a = torch.randint(0, 2**31, (M_A, 400), dtype=torch.int32, device=device)
            mask_a = torch.full((M_A, 400), 0x7FFFFFFF, dtype=torch.int32, device=device)
            data_b = torch.randint(0, 2**31, (M_B, 400), dtype=torch.int32, device=device)
            mask_b = torch.full((M_B, 400), 0x7FFFFFFF, dtype=torch.int32, device=device)

            max_pairs = max(1, n_pairs // 100)

            # Warmup
            for _ in range(warmup_runs):
                ih.masked_hamming_ab_cuda(
                    data_a, mask_a, data_b, mask_b,
                    match_threshold=1.0, non_match_threshold=1.0,
                    include_flags=ih.INCLUDE_ALL, max_pairs=max_pairs
                )
                torch.cuda.synchronize()

            # Benchmark
            times = []
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            for _ in range(benchmark_runs):
                start_event.record()
                ih.masked_hamming_ab_cuda(
                    data_a, mask_a, data_b, mask_b,
                    match_threshold=1.0, non_match_threshold=1.0,
                    include_flags=ih.INCLUDE_ALL, max_pairs=max_pairs
                )
                end_event.record()
                torch.cuda.synchronize()
                times.append(start_event.elapsed_time(end_event))

            mean_time = np.mean(times)
            std_time = np.std(times)
            throughput = n_pairs / (mean_time / 1000) / 1e6

            result = {
                "M_A": M_A,
                "M_B": M_B,
                "n_pairs": n_pairs,
                "kernel_time_ms_mean": mean_time,
                "kernel_time_ms_std": std_time,
                "pairs_per_second_millions": throughput,
            }
            results.append(result)

            print(f"  Time: {mean_time:.2f} \u00b1 {std_time:.2f} ms")
            print(f"  Throughput: {throughput:.2f} M pairs/s")

            del data_a, mask_a, data_b, mask_b
            torch.cuda.empty_cache()

    return results


def print_summary_table(summaries: List[BenchmarkSummary]):
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 90)
    print("BENCHMARK SUMMARY")
    print("=" * 90)
    print(f"{'Batch Size':>12} {'Pairs':>14} {'Kernel (ms)':>14} {'Throughput (M/s)':>18} {'GPU Mem (MB)':>14}")
    print("-" * 90)

    for s in summaries:
        print(
            f"{s.batch_size:>12,} "
            f"{s.n_pairs:>14,} "
            f"{s.kernel_time_ms_mean:>10.2f} \u00b1 {s.kernel_time_ms_std:>4.2f} "
            f"{s.pairs_per_second_millions_mean:>14.2f} \u00b1 {s.pairs_per_second_millions_std:>4.2f} "
            f"{s.peak_gpu_memory_mb:>14.1f}"
        )

    print("=" * 90)
    print(f"Device: {summaries[0].device_name}")
    print(f"Runs per size: {summaries[0].n_runs}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark CUDA iris matcher")
    parser.add_argument(
        "--sizes",
        type=str,
        default="256,512,1024,2048,4096,8192,16384,32768",
        help="Comma-separated batch sizes to benchmark",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Number of warmup runs per size",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of benchmark runs per size",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--ab-benchmark",
        action="store_true",
        help="Also run A vs B benchmark",
    )
    parser.add_argument(
        "--sharding",
        action="store_true",
        help="Run sharding benchmark (multi-GPU and tiled computation)",
    )
    parser.add_argument(
        "--sharding-only",
        action="store_true",
        help="Run only sharding benchmark (skip regular benchmarks)",
    )
    parser.add_argument(
        "--min-shards",
        type=str,
        default=None,
        help="Comma-separated list of min_shards values to test (default: auto)",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline benchmark in sharding tests",
    )
    parser.add_argument(
        "--include-packing",
        action="store_true",
        help="Include benchmark with unpacked (uint8) data to measure packing overhead",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    batch_sizes = [int(s.strip()) for s in args.sizes.split(",")]

    print("=" * 60)
    print("CUDA IRIS MATCHER BENCHMARK")
    print("=" * 60)
    print(f"Device: {torch.cuda.get_device_name(args.device)}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Warmup runs: {args.warmup}")
    print(f"Benchmark runs: {args.repeats}")

    summaries = []
    ab_results = None
    sharding_results = None

    # Run regular benchmarks unless --sharding-only
    if not args.sharding_only:
        # Run self-comparison benchmark
        summaries = run_benchmark_suite(
            batch_sizes,
            warmup_runs=args.warmup,
            benchmark_runs=args.repeats,
            device_idx=args.device,
        )

        print_summary_table(summaries)

        # Run A vs B benchmark if requested
        if args.ab_benchmark:
            print("\n" + "=" * 60)
            print("A VS B BENCHMARK")
            print("=" * 60)
            ab_results = run_ab_benchmark(
                sizes_a=[256, 512, 1024],
                sizes_b=[1024, 2048, 4096],
                warmup_runs=2,
                benchmark_runs=3,
                device_idx=args.device,
            )

    # Run sharding benchmark if requested
    if args.sharding or args.sharding_only:
        print("\n" + "=" * 60)
        print("SHARDING BENCHMARK (Multi-GPU / Tiled)")
        print("=" * 60)
        
        # Parse min_shards list if provided
        min_shards_list = None
        if args.min_shards:
            min_shards_list = [int(x.strip()) for x in args.min_shards.split(",")]
        
        # Use smaller sizes for sharding benchmark by default
        sharding_sizes = [s for s in batch_sizes if s <= 4096] or batch_sizes[:3]
        
        sharding_results = run_sharding_benchmark(
            sizes=sharding_sizes,
            warmup_runs=args.warmup,
            benchmark_runs=args.repeats,
            min_shards_list=min_shards_list,
            skip_baseline=args.skip_baseline,
            include_packing=args.include_packing,
        )
        
        print_sharding_summary(sharding_results)

    # Save results to JSON if requested
    if args.output:
        output_data = {
            "device": torch.cuda.get_device_name(args.device),
            "num_devices": torch.cuda.device_count(),
        }
        if summaries:
            output_data["self_comparison"] = [asdict(s) for s in summaries]
        if ab_results:
            output_data["ab_comparison"] = ab_results
        if sharding_results:
            output_data["sharding"] = [asdict(r) for r in sharding_results]

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
