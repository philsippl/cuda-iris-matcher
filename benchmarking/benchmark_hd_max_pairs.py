#!/usr/bin/env python3
"""
Benchmark for measuring the impact of max_pairs on CUDA iris matcher performance.

This benchmark tests how the max_pairs parameter affects:
- Kernel execution time
- Memory allocation overhead
- Throughput

Usage:
    python benchmark_max_pairs.py [--m 4096] [--max-pairs 1,100,1000,10000] [--warmup 2] [--repeats 5]
"""

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np
import torch

import cuda_iris_matcher as ih


@dataclass
class MaxPairsBenchmarkResult:
    """Result for a single max_pairs benchmark."""
    batch_size: int
    n_pairs_total: int
    max_pairs: int
    max_pairs_fraction: float
    
    # Timing (milliseconds)
    kernel_time_ms_mean: float
    kernel_time_ms_std: float
    
    # Throughput
    pairs_per_second_millions: float
    
    # Memory
    peak_gpu_memory_mb: float
    
    # Counts
    actual_pairs_returned: int
    
    device_name: str
    n_runs: int


def run_max_pairs_benchmark(
    M: int,
    max_pairs_values: List[int],
    warmup_runs: int = 2,
    benchmark_runs: int = 5,
    device_idx: int = 0,
    match_threshold: float = 1.0,
) -> List[MaxPairsBenchmarkResult]:
    """
    Benchmark kernel performance with different max_pairs values.
    
    Args:
        M: Batch size (number of iris codes)
        max_pairs_values: List of max_pairs values to test
        warmup_runs: Number of warmup iterations
        benchmark_runs: Number of timed iterations
        device_idx: CUDA device index
        match_threshold: Threshold for matching (1.0 = include all)
    
    Returns:
        List of MaxPairsBenchmarkResult for each max_pairs value
    """
    torch.cuda.set_device(device_idx)
    device = torch.device(f"cuda:{device_idx}")
    device_name = torch.cuda.get_device_name(device_idx)
    
    n_pairs_total = M * (M - 1) // 2
    results = []
    
    # Generate random packed data once
    data = torch.randint(
        low=0, high=2**31, size=(M, 400), dtype=torch.int32, device=device
    ).contiguous()
    mask = torch.full(
        (M, 400), fill_value=0x7FFFFFFF, dtype=torch.int32, device=device
    ).contiguous()
    
    print(f"\nBenchmarking M={M:,} ({n_pairs_total:,} total pairs)")
    print(f"Testing max_pairs values: {max_pairs_values}")
    
    for max_pairs in max_pairs_values:
        # Reset memory stats
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
        gc.collect()
        
        # Use max_pairs as-is (no clamping) to measure allocation overhead
        # even when max_pairs exceeds actual pairs produced
        fraction = max_pairs / n_pairs_total
        
        # Warmup
        for _ in range(warmup_runs):
            ih.masked_hamming_cuda(
                data, mask,
                match_threshold=match_threshold,
                non_match_threshold=1.0,
                include_flags=ih.INCLUDE_ALL,
                max_pairs=max_pairs,
            )
            torch.cuda.synchronize(device)
        
        # Benchmark runs
        times = []
        actual_returned = 0
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        for _ in range(benchmark_runs):
            start_event.record()
            pair_indices, categories, distances, count = ih.masked_hamming_cuda(
                data, mask,
                match_threshold=match_threshold,
                non_match_threshold=1.0,
                include_flags=ih.INCLUDE_ALL,
                max_pairs=max_pairs,
            )
            end_event.record()
            torch.cuda.synchronize(device)
            
            times.append(start_event.elapsed_time(end_event))
            actual_returned = count.item()
        
        # Statistics
        mean_time = np.mean(times)
        std_time = np.std(times)
        throughput = n_pairs_total / (mean_time / 1000) / 1e6
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        
        result = MaxPairsBenchmarkResult(
            batch_size=M,
            n_pairs_total=n_pairs_total,
            max_pairs=max_pairs,
            max_pairs_fraction=fraction,
            kernel_time_ms_mean=mean_time,
            kernel_time_ms_std=std_time,
            pairs_per_second_millions=throughput,
            peak_gpu_memory_mb=peak_memory,
            actual_pairs_returned=actual_returned,
            device_name=device_name,
            n_runs=benchmark_runs,
        )
        results.append(result)
        
        result.row_str = (
            f"{M:>8,} "
            f"{max_pairs:>12,} "
            f"{fraction:>9.4%} "
            f"{mean_time:>8.2f} ± {std_time:>4.2f} "
            f"{throughput:>12.2f} "
            f"{peak_memory:>10.1f} "
            f"{actual_returned:>12,}"
        )
    
    # Cleanup
    del data, mask
    torch.cuda.empty_cache()
    
    return results


def run_scaling_benchmark(
    batch_sizes: List[int],
    max_pairs_fractions: List[float],
    warmup_runs: int = 2,
    benchmark_runs: int = 5,
    device_idx: int = 0,
) -> List[MaxPairsBenchmarkResult]:
    """
    Benchmark how max_pairs overhead scales with batch size.
    
    Args:
        batch_sizes: List of M values to test
        max_pairs_fractions: List of fractions of total pairs to test
        warmup_runs: Number of warmup iterations
        benchmark_runs: Number of timed iterations
        device_idx: CUDA device index
    
    Returns:
        List of MaxPairsBenchmarkResult for all combinations
    """
    all_results = []
    
    print("\n" + "=" * 80)
    print("MAX_PAIRS SCALING BENCHMARK")
    print("=" * 80)
    
    for M in batch_sizes:
        n_pairs_total = M * (M - 1) // 2
        max_pairs_values = [max(1, int(n_pairs_total * f)) for f in max_pairs_fractions]
        # Also add some fixed small values for comparison
        max_pairs_values = sorted(set([1, 10, 100] + max_pairs_values))
        
        results = run_max_pairs_benchmark(
            M=M,
            max_pairs_values=max_pairs_values,
            warmup_runs=warmup_runs,
            benchmark_runs=benchmark_runs,
            device_idx=device_idx,
        )
        all_results.extend(results)
    
    return all_results


def print_summary(results: List[MaxPairsBenchmarkResult]):
    """Print summary of benchmark results."""
    if not results:
        return
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # Group by batch size
    by_batch = {}
    for r in results:
        if r.batch_size not in by_batch:
            by_batch[r.batch_size] = []
        by_batch[r.batch_size].append(r)
    
    for M, batch_results in sorted(by_batch.items()):
        # Find min and max time
        min_result = min(batch_results, key=lambda r: r.kernel_time_ms_mean)
        max_result = max(batch_results, key=lambda r: r.kernel_time_ms_mean)
        
        overhead = (max_result.kernel_time_ms_mean - min_result.kernel_time_ms_mean) / min_result.kernel_time_ms_mean * 100
        
        print(f"\nM={M:,}:")
        print(f"  Fastest: max_pairs={min_result.max_pairs:,} → {min_result.kernel_time_ms_mean:.2f} ms")
        print(f"  Slowest: max_pairs={max_result.max_pairs:,} → {max_result.kernel_time_ms_mean:.2f} ms")
        print(f"  Max overhead: {overhead:.1f}%")
        print(f"  Memory range: {min_result.peak_gpu_memory_mb:.1f} - {max_result.peak_gpu_memory_mb:.1f} MB")
    
    print(f"\nDevice: {results[0].device_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark max_pairs impact on CUDA iris matcher"
    )
    parser.add_argument(
        "--m",
        type=str,
        default="4096",
        help="Comma-separated batch sizes to benchmark (default: 4096)",
    )
    parser.add_argument(
        "--max-pairs",
        type=str,
        default=None,
        help="Comma-separated max_pairs values to test (default: auto-scale)",
    )
    parser.add_argument(
        "--fractions",
        type=str,
        default="0.0001,0.001,0.01,0.1,0.5,1.0",
        help="Comma-separated fractions of total pairs to test (used if --max-pairs not set)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup runs",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of benchmark runs",
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
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)
    
    batch_sizes = [int(s.strip()) for s in args.m.split(",")]
    
    print("=" * 80)
    print("MAX_PAIRS IMPACT BENCHMARK")
    print("=" * 80)
    print(f"Device: {torch.cuda.get_device_name(args.device)}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Warmup runs: {args.warmup}")
    print(f"Benchmark runs: {args.repeats}")
    
    all_results = []
    
    if args.max_pairs:
        # Use explicit max_pairs values
        max_pairs_values = [int(x.strip()) for x in args.max_pairs.split(",")]
        for M in batch_sizes:
            results = run_max_pairs_benchmark(
                M=M,
                max_pairs_values=max_pairs_values,
                warmup_runs=args.warmup,
                benchmark_runs=args.repeats,
                device_idx=args.device,
            )
            all_results.extend(results)
    else:
        # Use fractions
        fractions = [float(f.strip()) for f in args.fractions.split(",")]
        all_results = run_scaling_benchmark(
            batch_sizes=batch_sizes,
            max_pairs_fractions=fractions,
            warmup_runs=args.warmup,
            benchmark_runs=args.repeats,
            device_idx=args.device,
        )
    
    # Print consolidated results table
    print("\n" + "=" * 95)
    print("RESULTS")
    print("=" * 95)
    print(f"{'M':>8} {'max_pairs':>12} {'fraction':>10} {'time (ms)':>14} {'M pairs/s':>12} {'mem (MB)':>10} {'returned':>12}")
    print("-" * 95)
    for r in all_results:
        print(r.row_str)
    print("-" * 95)
    
    print_summary(all_results)
    
    # Save to JSON if requested
    if args.output:
        output_data = {
            "device": torch.cuda.get_device_name(args.device),
            "results": [asdict(r) for r in all_results],
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
