/**
 * Standalone CUDA benchmark for iris kernel performance testing.
 * Compile with: make benchmark
 * Run with: ./benchmark [M] [warmup_iters] [bench_iters]
 *
 * Defaults: M=10000, warmup=5, bench=20
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "iris_params.h"

// External launchers from iris.cu
extern "C" void launch_masked_hamming_cuda(
    const uint32_t *dData, const uint32_t *dMask, uint32_t *dPremasked, int M,
    const int32_t *dLabels, float match_threshold, float non_match_threshold,
    bool is_similarity, uint8_t include_flags, int32_t *dPairIndices,
    uint8_t *dCategories, float *dOutDistances, unsigned int *dMatchCount,
    unsigned int max_pairs, cudaStream_t stream);

extern "C" void launch_masked_hamming_ab_cuda(
    const uint32_t *dData_A, const uint32_t *dMask_A, uint32_t *dPremasked_A,
    const uint32_t *dData_B, const uint32_t *dMask_B, uint32_t *dPremasked_B,
    int M_A, int M_B, const int32_t *dLabels_A, const int32_t *dLabels_B,
    float match_threshold, float non_match_threshold, bool is_similarity,
    uint8_t include_flags, int32_t *dPairIndices, uint8_t *dCategories,
    float *dOutDistances, unsigned int *dMatchCount, unsigned int max_pairs,
    cudaStream_t stream);

// Check if using tensor cores or fallback
// b1 MMA is available on SM80-SM90 (Ampere, Ada, Hopper)
// Not available on SM75 and below (Turing, Volta, Pascal)
// Not available on SM100+ (Blackwell uses new tcgen05.mma family)
inline bool uses_tensor_cores() {
#ifdef FORCE_FALLBACK
  return false;
#else
  int device;
  cudaDeviceProp prop;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  int sm_version = prop.major * 10 + prop.minor;
  return (sm_version >= 80 && sm_version < 100);
#endif
}

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t _err = (call);                                                 \
    if (_err != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,         \
              __LINE__, cudaGetErrorString(_err));                             \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

void fill_random(uint32_t *ptr, size_t count) {
  for (size_t i = 0; i < count; i++) {
    ptr[i] = ((uint32_t)rand() << 16) ^ (uint32_t)rand();
  }
}

// Fill half-mask (200 words per row) with random data.
// Half-mask only stores d1=0 bits since d1=1 is identical.
void fill_random_half_mask(uint32_t *ptr, int M) {
  for (int m = 0; m < M; m++) {
    uint32_t *row = ptr + m * K_WORDS_HALF;
    for (int w = 0; w < K_WORDS_HALF; w++) {
      row[w] = ((uint32_t)rand() << 16) ^ (uint32_t)rand();
    }
  }
}

void print_gpu_info() {
  int device;
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDevice(&device));
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

  printf("=== GPU Info ===\n");
  printf("Device: %s\n", prop.name);
  printf("Compute capability: %d.%d\n", prop.major, prop.minor);
  printf("SM count: %d\n", prop.multiProcessorCount);
  printf("Memory: %.1f GB\n", prop.totalGlobalMem / 1e9);
  printf("Memory bandwidth: %.0f GB/s\n",
         2.0 * prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8) / 1e9);

  // Print MMA implementation info
  bool tensor_cores = uses_tensor_cores();
  printf("MMA implementation: %s\n",
         tensor_cores ? "TENSOR CORES (native b1 MMA)"
                      : "FALLBACK (scalar __popc + warp shuffles)");
#ifdef FORCE_FALLBACK
  printf("  (FORCE_FALLBACK defined at compile time)\n");
#else
  if (!tensor_cores) {
    int sm_version = prop.major * 10 + prop.minor;
    if (sm_version < 80)
      printf("  (SM%d < SM80: b1 MMA not available)\n", sm_version);
    else if (sm_version >= 100)
      printf("  (SM%d >= SM100 Blackwell: uses tcgen05.mma, legacy b1 MMA "
             "unavailable)\n",
             sm_version);
  }
#endif
  printf("\n");
}

void print_kernel_config(int M) {
  printf("=== Kernel Config ===\n");
  printf("M (samples): %d\n", M);
  printf("K_BITS: %d, K_WORDS: %d\n", K_BITS, K_WORDS);
  printf("Block size: BLOCK_M=%d, BLOCK_N=%d\n", BLOCK_M, BLOCK_N);
  printf("Warps per block: %d\n", WARPS_PER_BLOCK);
  printf("Tiles per warp: %dx%d\n", TILES_M_PER_WARP, TILES_N_PER_WARP);
  printf("K chunks: %d (chunk words: %d)\n", K_CHUNKS, K_CHUNK_WORDS);
  printf("Shift range: [-%d, +%d] (%d shifts)\n", MAX_SHIFT, MAX_SHIFT,
         NUM_SHIFTS);

  dim3 grid((M + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
  size_t smem =
      2 * (2 * BLOCK_M * K_CHUNK_WORDS + 2 * BLOCK_N * K_CHUNK_WORDS_EXT) *
      sizeof(uint32_t);
  printf("Grid: %d x %d = %d blocks\n", grid.x, grid.y, grid.x * grid.y);
  printf("Shared memory per block: %zu bytes\n", smem);
  printf("\n");
}

int main(int argc, char **argv) {
  // Parse args
  int M = (argc > 1) ? atoi(argv[1]) : 10000;
  int warmup_iters = (argc > 2) ? atoi(argv[2]) : 5;
  int bench_iters = (argc > 3) ? atoi(argv[3]) : 20;

  printf("Iris Kernel Benchmark\n");
  printf("=====================\n\n");

  print_gpu_info();
  print_kernel_config(M);

  printf("=== Benchmark Settings ===\n");
  printf("Warmup iterations: %d\n", warmup_iters);
  printf("Benchmark iterations: %d\n", bench_iters);
  printf("\n");

  srand(42);

  // Allocate host memory
  size_t data_size = (size_t)M * K_WORDS * sizeof(uint32_t);
  size_t mask_size = (size_t)M * K_WORDS_HALF * sizeof(uint32_t); // Half-mask
  uint32_t *h_data = (uint32_t *)malloc(data_size);
  uint32_t *h_mask = (uint32_t *)malloc(mask_size);

  printf("Generating random data... ");
  fflush(stdout);
  fill_random(h_data, M * K_WORDS);
  // Use half-mask (200 words per row) - real-world structure
  fill_random_half_mask(h_mask, M);
  printf("done\n");

  // Allocate device memory
  uint32_t *d_data, *d_mask, *d_premasked;
  int32_t *d_pair_indices;
  uint8_t *d_categories;
  float *d_distances;
  unsigned int *d_match_count;

  unsigned int max_pairs = 1000000;

  CHECK_CUDA(cudaMalloc(&d_data, data_size));
  CHECK_CUDA(cudaMalloc(&d_mask, mask_size)); // Half-mask
  CHECK_CUDA(cudaMalloc(&d_premasked, data_size));
  CHECK_CUDA(cudaMalloc(&d_pair_indices, max_pairs * 2 * sizeof(int32_t)));
  CHECK_CUDA(cudaMalloc(&d_categories, max_pairs * sizeof(uint8_t)));
  CHECK_CUDA(cudaMalloc(&d_distances, max_pairs * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_match_count, sizeof(unsigned int)));

  printf("Copying data to GPU... ");
  fflush(stdout);
  CHECK_CUDA(cudaMemcpy(d_data, h_data, data_size, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_mask, h_mask, mask_size, cudaMemcpyHostToDevice));
  printf("done\n\n");

  // Create events for timing
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  // Warmup
  printf("Warmup (%d iterations)... ", warmup_iters);
  fflush(stdout);
  for (int i = 0; i < warmup_iters; i++) {
    CHECK_CUDA(cudaMemset(d_match_count, 0, sizeof(unsigned int)));
    launch_masked_hamming_cuda(d_data, d_mask, d_premasked, M,
                               nullptr, // labels (none)
                               0.3f,    // match_threshold
                               0.3f,    // non_match_threshold
                               false,   // is_similarity
                               INCLUDE_ALL, d_pair_indices, d_categories,
                               d_distances, d_match_count, max_pairs, stream);
  }
  CHECK_CUDA(cudaStreamSynchronize(stream));
  printf("done\n\n");

  // Benchmark
  printf("=== Benchmarking ===\n");

  double *times = (double *)malloc(bench_iters * sizeof(double));
  double total_time = 0.0;

  for (int i = 0; i < bench_iters; i++) {
    CHECK_CUDA(cudaMemset(d_match_count, 0, sizeof(unsigned int)));

    CHECK_CUDA(cudaEventRecord(start, stream));
    launch_masked_hamming_cuda(d_data, d_mask, d_premasked, M,
                               nullptr, // labels (none)
                               0.3f,    // match_threshold
                               0.3f,    // non_match_threshold
                               false,   // is_similarity
                               INCLUDE_ALL, d_pair_indices, d_categories,
                               d_distances, d_match_count, max_pairs, stream);
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    times[i] = ms;
    total_time += ms;

    printf("Iter %2d: %.3f ms\n", i + 1, ms);
  }

  // Calculate statistics
  double mean = total_time / bench_iters;
  double variance = 0.0;
  double min_time = times[0], max_time = times[0];

  for (int i = 0; i < bench_iters; i++) {
    variance += (times[i] - mean) * (times[i] - mean);
    if (times[i] < min_time)
      min_time = times[i];
    if (times[i] > max_time)
      max_time = times[i];
  }
  variance /= bench_iters;
  double stddev = sqrt(variance);

  // Calculate performance metrics
  // Lower triangle comparisons: M*(M-1)/2
  double comparisons = (double)M * (M - 1) / 2.0;
  // Each comparison does NUM_SHIFTS shifts, K_CHUNKS chunks
  // Each chunk: 3 MMA ops (16x8x256), each MMA is 16*8*256 = 32768 ops
  double mma_ops_per_comparison = NUM_SHIFTS * K_CHUNKS * 3 * 16 * 8 * 256;
  double total_ops = comparisons * mma_ops_per_comparison;
  double tops = total_ops / (mean * 1e-3) / 1e12;

  // Pairs per second (in millions) - matches Python script format
  double pairs_per_s_m = comparisons / (mean * 1e-3) / 1e6;

  printf("\n=== Results ===\n");
  printf("Time (mean ± std): %.3f ± %.3f ms\n", mean, stddev);
  printf("Time (min / max): %.3f / %.3f ms\n", min_time, max_time);
  printf("Pairs/s: %.3f M (31 shifts)\n", pairs_per_s_m);
  printf("Throughput: %.2f TOPS (tensor ops)\n", tops);

  // Get match count for verification
  unsigned int h_match_count;
  CHECK_CUDA(cudaMemcpy(&h_match_count, d_match_count, sizeof(unsigned int),
                        cudaMemcpyDeviceToHost));
  printf("Matches found (last iter, threshold=0.3): %u\n", h_match_count);

  // ----------------- A vs B Benchmark -----------------
  printf("\n=== A vs B Benchmark ===\n");

  // Use M/2 for each set to keep total memory similar
  int M_A = M / 2;
  int M_B = M / 2;
  printf("M_A: %d, M_B: %d\n", M_A, M_B);

  // Allocate separate A and B buffers (reuse h_data/h_mask for A, generate new
  // for B)
  size_t data_size_A = (size_t)M_A * K_WORDS * sizeof(uint32_t);
  size_t data_size_B = (size_t)M_B * K_WORDS * sizeof(uint32_t);
  size_t mask_size_A = (size_t)M_A * K_WORDS_HALF * sizeof(uint32_t);
  size_t mask_size_B = (size_t)M_B * K_WORDS_HALF * sizeof(uint32_t);

  uint32_t *h_data_B = (uint32_t *)malloc(data_size_B);
  uint32_t *h_mask_B = (uint32_t *)malloc(mask_size_B);
  fill_random(h_data_B, M_B * K_WORDS);
  fill_random_half_mask(h_mask_B, M_B);

  uint32_t *d_data_A, *d_mask_A, *d_premasked_A;
  uint32_t *d_data_B, *d_mask_B, *d_premasked_B;

  CHECK_CUDA(cudaMalloc(&d_data_A, data_size_A));
  CHECK_CUDA(cudaMalloc(&d_mask_A, mask_size_A));
  CHECK_CUDA(cudaMalloc(&d_premasked_A, data_size_A));
  CHECK_CUDA(cudaMalloc(&d_data_B, data_size_B));
  CHECK_CUDA(cudaMalloc(&d_mask_B, mask_size_B));
  CHECK_CUDA(cudaMalloc(&d_premasked_B, data_size_B));

  CHECK_CUDA(cudaMemcpy(d_data_A, h_data, data_size_A, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_mask_A, h_mask, mask_size_A, cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(d_data_B, h_data_B, data_size_B, cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(d_mask_B, h_mask_B, mask_size_B, cudaMemcpyHostToDevice));

  // Warmup A vs B
  printf("Warmup (%d iterations)... ", warmup_iters);
  fflush(stdout);
  for (int i = 0; i < warmup_iters; i++) {
    CHECK_CUDA(cudaMemset(d_match_count, 0, sizeof(unsigned int)));
    launch_masked_hamming_ab_cuda(
        d_data_A, d_mask_A, d_premasked_A, d_data_B, d_mask_B, d_premasked_B,
        M_A, M_B, nullptr, nullptr, // labels (none)
        0.3f,                       // match_threshold
        0.3f,                       // non_match_threshold
        false,                      // is_similarity
        INCLUDE_ALL, d_pair_indices, d_categories, d_distances, d_match_count,
        max_pairs, stream);
  }
  CHECK_CUDA(cudaStreamSynchronize(stream));
  printf("done\n");

  // Benchmark A vs B
  double total_time_ab = 0.0;
  double min_time_ab = 1e9, max_time_ab = 0.0;

  for (int i = 0; i < bench_iters; i++) {
    CHECK_CUDA(cudaMemset(d_match_count, 0, sizeof(unsigned int)));

    CHECK_CUDA(cudaEventRecord(start, stream));
    launch_masked_hamming_ab_cuda(
        d_data_A, d_mask_A, d_premasked_A, d_data_B, d_mask_B, d_premasked_B,
        M_A, M_B, nullptr, nullptr, // labels (none)
        0.3f,                       // match_threshold
        0.3f,                       // non_match_threshold
        false,                      // is_similarity
        INCLUDE_ALL, d_pair_indices, d_categories, d_distances, d_match_count,
        max_pairs, stream);
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    times[i] = ms;
    total_time_ab += ms;

    if (ms < min_time_ab)
      min_time_ab = ms;
    if (ms > max_time_ab)
      max_time_ab = ms;

    printf("Iter %2d: %.3f ms\n", i + 1, ms);
  }

  double mean_ab = total_time_ab / bench_iters;
  double variance_ab = 0.0;
  for (int i = 0; i < bench_iters; i++) {
    variance_ab += (times[i] - mean_ab) * (times[i] - mean_ab);
  }
  variance_ab /= bench_iters;
  double stddev_ab = sqrt(variance_ab);

  // A vs B comparisons: M_A * M_B (full rectangle)
  double comparisons_ab = (double)M_A * M_B;
  double pairs_per_s_m_ab = comparisons_ab / (mean_ab * 1e-3) / 1e6;

  printf("\n=== A vs B Results ===\n");
  printf("Time (mean ± std): %.3f ± %.3f ms\n", mean_ab, stddev_ab);
  printf("Time (min / max): %.3f / %.3f ms\n", min_time_ab, max_time_ab);
  printf("Pairs/s: %.3f M (31 shifts)\n", pairs_per_s_m_ab);

  CHECK_CUDA(cudaMemcpy(&h_match_count, d_match_count, sizeof(unsigned int),
                        cudaMemcpyDeviceToHost));
  printf("Matches found (last iter, threshold=0.3): %u\n", h_match_count);

  // Cleanup A vs B
  free(h_data_B);
  free(h_mask_B);
  CHECK_CUDA(cudaFree(d_data_A));
  CHECK_CUDA(cudaFree(d_mask_A));
  CHECK_CUDA(cudaFree(d_premasked_A));
  CHECK_CUDA(cudaFree(d_data_B));
  CHECK_CUDA(cudaFree(d_mask_B));
  CHECK_CUDA(cudaFree(d_premasked_B));

  // Cleanup original
  free(times);
  free(h_data);
  free(h_mask);
  CHECK_CUDA(cudaFree(d_data));
  CHECK_CUDA(cudaFree(d_mask));
  CHECK_CUDA(cudaFree(d_premasked));
  CHECK_CUDA(cudaFree(d_pair_indices));
  CHECK_CUDA(cudaFree(d_categories));
  CHECK_CUDA(cudaFree(d_distances));
  CHECK_CUDA(cudaFree(d_match_count));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaStreamDestroy(stream));

  printf("\nBenchmark complete.\n");
  return 0;
}
