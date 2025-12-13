#include <cuda.h>
#include <cuda_pipeline_primitives.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "iris_params.h"

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t _err = (call);                                                 \
    if (_err != cudaSuccess) {                                                 \
      printf("CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,        \
             cudaGetErrorString(_err));                                        \
      return;                                                                  \
    }                                                                          \
  } while (0)

static_assert(BLOCK_M % (WMMA_M * TILES_M_PER_WARP) == 0, "BLOCK_M constraint");
static_assert(BLOCK_N % (WMMA_N * TILES_N_PER_WARP) == 0, "BLOCK_N constraint");
static_assert(WARPS_PER_BLOCK * 32 <= 1024, "Too many threads");

// ----------------- TensorCore MMA: b1 AND+POPC -----------------

__device__ inline void mma_b1_and_popc_16x8x256(int32_t &c0, int32_t &c1,
                                                int32_t &c2, int32_t &c3,
                                                uint32_t a0, uint32_t a1,
                                                uint32_t a2, uint32_t a3,
                                                uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
               "{%0, %1, %2, %3}, "
               "{%4, %5, %6, %7}, "
               "{%8, %9}, "
               "{%0, %1, %2, %3};\n"
               : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// ----------------- Fragment loaders -----------------

__device__ inline void load_a_frag(const uint32_t *As, uint32_t &a0,
                                   uint32_t &a1, uint32_t &a2, uint32_t &a3) {
  int lane = threadIdx.x & 31;
  int group = lane >> 2;
  int tid = lane & 3;
  int words_per_row = K_CHUNK_WORDS;
  a0 = As[group * words_per_row + tid];
  a1 = As[(group + 8) * words_per_row + tid];
  a2 = As[group * words_per_row + tid + 4];
  a3 = As[(group + 8) * words_per_row + tid + 4];
}

// Load B fragment (no intra-word shift). Bs_ext points to an extended buffer
// with layout: [row][word] where word indices 0..9 map to chunk words -1..8.
__device__ inline void load_b_frag(const uint32_t *Bs_ext, uint32_t &b0,
                                   uint32_t &b1) {
  int lane = threadIdx.x & 31;
  int group = lane >> 2;
  int tid = lane & 3;
  int words_per_col = K_CHUNK_WORDS_EXT;

  // Bs_ext layout: index 0 = word -1, index 1 = word 0, etc.
  int base = group * words_per_col + 1; // +1 for offset

  b0 = Bs_ext[base + tid];
  b1 = Bs_ext[base + tid + 4];
}

// ----------------- Minimum Fractional Hamming Distance kernel
// -----------------
//
// Optimized structure: process all shifts within the K-loop to maximize A data
// reuse. For each K chunk: load A once, load B_ext once, then iterate over all
// shifts. This uses 31× more accumulators but provides much better cache
// behavior.

__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32, 1)
    min_hamming_kernel(const uint32_t *__restrict__ premasked,
                       const uint32_t *__restrict__ mask, int M,
                       float *__restrict__ D, bool write_output,
                       bool collect_pairs, float threshold,
                       uint2 *__restrict__ pairs, unsigned int *match_count,
                       unsigned int max_pairs) {
  int warp_id = threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  int group = lane >> 2;
  int tid = lane & 3;

  int cta_row = blockIdx.y;
  int cta_col = blockIdx.x;
  int block_row_start = cta_row * BLOCK_M;
  int block_col_start = cta_col * BLOCK_N;

  if (block_col_start > block_row_start + BLOCK_M - 1)
    return;
  if (block_row_start >= M || block_col_start >= M)
    return;

  extern __shared__ uint32_t smem[];
  size_t stride_A = BLOCK_M * K_CHUNK_WORDS;
  size_t stride_B_ext = BLOCK_N * K_CHUNK_WORDS_EXT;

  uint32_t *A_pm_smem = smem;
  uint32_t *A_m_smem = A_pm_smem + 2 * stride_A;
  uint32_t *B_pm_smem = A_m_smem + 2 * stride_A;
  uint32_t *B_m_smem = B_pm_smem + 2 * stride_B_ext;

  auto A_pm_stage = [&](int s) { return A_pm_smem + s * stride_A; };
  auto A_m_stage = [&](int s) { return A_m_smem + s * stride_A; };
  auto B_pm_stage = [&](int s) { return B_pm_smem + s * stride_B_ext; };
  auto B_m_stage = [&](int s) { return B_m_smem + s * stride_B_ext; };

  int warp_row = warp_id / WARPS_PER_COL;
  int warp_col = warp_id % WARPS_PER_COL;
  int tile_m_base = warp_row * TILES_M_PER_WARP;
  int tile_n_base = warp_col * TILES_N_PER_WARP;

  // Min FHD tracker per output element
  float min_fhd[TILES_M_PER_WARP][TILES_N_PER_WARP][4];
#pragma unroll
  for (int tm = 0; tm < TILES_M_PER_WARP; ++tm)
#pragma unroll
    for (int tn = 0; tn < TILES_N_PER_WARP; ++tn)
#pragma unroll
      for (int r = 0; r < 4; ++r)
        min_fhd[tm][tn][r] = 2.0f;

  auto load_chunk_A = [&](uint32_t *dst, const uint32_t *src, int block_start,
                          int chunk_word) {
    int num_words = BLOCK_M * K_CHUNK_WORDS;
    for (int idx4 = threadIdx.x; idx4 * 4 < num_words; idx4 += blockDim.x) {
      int idx = idx4 * 4;
      int r = idx / K_CHUNK_WORDS;
      int w = idx % K_CHUNK_WORDS;
      int g_row = block_start + r;
      if (g_row < M) {
        __pipeline_memcpy_async(dst + idx,
                                src + g_row * K_WORDS + chunk_word + w, 16);
      } else {
        dst[idx] = dst[idx + 1] = dst[idx + 2] = dst[idx + 3] = 0;
      }
    }
  };

  auto load_chunk_B_ext = [&](uint32_t *dst, const uint32_t *src,
                              int block_start, int chunk_idx, int shift_words) {
    int chunk_word = chunk_idx * K_CHUNK_WORDS;
    int num_elems = BLOCK_N * K_CHUNK_WORDS_EXT;

    for (int idx = threadIdx.x; idx < num_elems; idx += blockDim.x) {
      int r = idx / K_CHUNK_WORDS_EXT;
      int w = idx % K_CHUNK_WORDS_EXT;
      int g_row = block_start + r;
      int g_word = chunk_word + w - 1 + shift_words;

      if (g_row < M) {
        // Wraparound rotation (circular) over the full K_WORDS.
        int gw = g_word % K_WORDS;
        if (gw < 0)
          gw += K_WORDS;
        const uint32_t *src_ptr = src + g_row * K_WORDS + gw;
        __pipeline_memcpy_async(dst + idx, src_ptr, sizeof(uint32_t));
      } else {
        dst[idx] = 0;
      }
    }
  };

  // Process one theta-roll shift at a time (sequential to minimize register
  // pressure)
  for (int shift = -MAX_SHIFT; shift <= MAX_SHIFT; ++shift) {
    int shift_words = shift * WORDS_PER_THETA_SHIFT;
    int32_t c1_frag[TILES_M_PER_WARP][TILES_N_PER_WARP][4];
    int32_t c2_frag[TILES_M_PER_WARP][TILES_N_PER_WARP][4];
    int32_t c3_frag[TILES_M_PER_WARP][TILES_N_PER_WARP][4];

#pragma unroll
    for (int tm = 0; tm < TILES_M_PER_WARP; ++tm)
#pragma unroll
      for (int tn = 0; tn < TILES_N_PER_WARP; ++tn)
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          c1_frag[tm][tn][r] = 0;
          c2_frag[tm][tn][r] = 0;
          c3_frag[tm][tn][r] = 0;
        }

    int stage = 0;
    {
      load_chunk_A(A_pm_stage(stage), premasked, block_row_start, 0);
      load_chunk_A(A_m_stage(stage), mask, block_row_start, 0);
      load_chunk_B_ext(B_pm_stage(stage), premasked, block_col_start, 0,
                       shift_words);
      load_chunk_B_ext(B_m_stage(stage), mask, block_col_start, 0, shift_words);
      __pipeline_commit();
      __pipeline_wait_prior(0);
      __syncthreads();
    }

    for (int kc = 0; kc < K_CHUNKS; ++kc) {
      uint32_t *A_pm_buf = A_pm_stage(stage);
      uint32_t *A_m_buf = A_m_stage(stage);
      uint32_t *B_pm_buf = B_pm_stage(stage);
      uint32_t *B_m_buf = B_m_stage(stage);

      int next_stage = stage ^ 1;
      if (kc + 1 < K_CHUNKS) {
        load_chunk_A(A_pm_stage(next_stage), premasked, block_row_start,
                     (kc + 1) * K_CHUNK_WORDS);
        load_chunk_A(A_m_stage(next_stage), mask, block_row_start,
                     (kc + 1) * K_CHUNK_WORDS);
        load_chunk_B_ext(B_pm_stage(next_stage), premasked, block_col_start,
                         kc + 1, shift_words);
        load_chunk_B_ext(B_m_stage(next_stage), mask, block_col_start, kc + 1,
                         shift_words);
        __pipeline_commit();
      }

#pragma unroll
      for (int tm = 0; tm < TILES_M_PER_WARP; ++tm) {
        int row_off = (tile_m_base + tm) * WMMA_M * K_CHUNK_WORDS;
        uint32_t *A_pm_tile = A_pm_buf + row_off;
        uint32_t *A_m_tile = A_m_buf + row_off;

        uint32_t a_pm0, a_pm1, a_pm2, a_pm3;
        uint32_t a_m0, a_m1, a_m2, a_m3;
        load_a_frag(A_pm_tile, a_pm0, a_pm1, a_pm2, a_pm3);
        load_a_frag(A_m_tile, a_m0, a_m1, a_m2, a_m3);

        uint32_t a_xp0 = a_m0 ^ a_pm0;
        uint32_t a_xp1 = a_m1 ^ a_pm1;
        uint32_t a_xp2 = a_m2 ^ a_pm2;
        uint32_t a_xp3 = a_m3 ^ a_pm3;

#pragma unroll
        for (int tn = 0; tn < TILES_N_PER_WARP; ++tn) {
          int col_off = (tile_n_base + tn) * WMMA_N * K_CHUNK_WORDS_EXT;
          uint32_t *B_pm_tile = B_pm_buf + col_off;
          uint32_t *B_m_tile = B_m_buf + col_off;

          uint32_t b_pm0, b_pm1;
          uint32_t b_m0, b_m1;
          load_b_frag(B_pm_tile, b_pm0, b_pm1);
          load_b_frag(B_m_tile, b_m0, b_m1);

          uint32_t b_xp0 = b_m0 ^ b_pm0;
          uint32_t b_xp1 = b_m1 ^ b_pm1;

          mma_b1_and_popc_16x8x256(c1_frag[tm][tn][0], c1_frag[tm][tn][1],
                                   c1_frag[tm][tn][2], c1_frag[tm][tn][3],
                                   a_xp0, a_xp1, a_xp2, a_xp3, b_pm0, b_pm1);

          mma_b1_and_popc_16x8x256(c2_frag[tm][tn][0], c2_frag[tm][tn][1],
                                   c2_frag[tm][tn][2], c2_frag[tm][tn][3],
                                   a_pm0, a_pm1, a_pm2, a_pm3, b_xp0, b_xp1);

          mma_b1_and_popc_16x8x256(c3_frag[tm][tn][0], c3_frag[tm][tn][1],
                                   c3_frag[tm][tn][2], c3_frag[tm][tn][3], a_m0,
                                   a_m1, a_m2, a_m3, b_m0, b_m1);
        }
      }

      if (kc + 1 < K_CHUNKS)
        __pipeline_wait_prior(0);
      __syncthreads();
      stage ^= 1;
    }

    // Update minimum FHD for this shift
#pragma unroll
    for (int tm = 0; tm < TILES_M_PER_WARP; ++tm)
#pragma unroll
      for (int tn = 0; tn < TILES_N_PER_WARP; ++tn)
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          int32_t c3 = c3_frag[tm][tn][r];
          if (c3 > 0) {
            float fhd =
                (float)(c1_frag[tm][tn][r] + c2_frag[tm][tn][r]) / (float)c3;
            if (fhd < min_fhd[tm][tn][r])
              min_fhd[tm][tn][r] = fhd;
          }
        }
  }

  // Store minimum results
#pragma unroll
  for (int tm = 0; tm < TILES_M_PER_WARP; ++tm) {
    int row_block = block_row_start + (tile_m_base + tm) * WMMA_M;
    int row0 = row_block + group;
    int row1 = row_block + group + 8;

#pragma unroll
    for (int tn = 0; tn < TILES_N_PER_WARP; ++tn) {
      int col_block = block_col_start + (tile_n_base + tn) * WMMA_N;
      int col0 = col_block + tid * 2;
      int col1 = col_block + tid * 2 + 1;

      auto store = [&](int gi, int gj, float val) {
        if (gi >= M || gj >= M || gi <= gj)
          return;
        if (write_output && D) {
          D[gi * M + gj] = (val <= 1.0f) ? val : 0.0f;
        }
        if (collect_pairs && val < threshold && pairs && match_count) {
          unsigned int idx = atomicAdd(match_count, 1);
          if (idx < max_pairs) {
            pairs[idx] = make_uint2(gi, gj);
          }
        }
      };

      store(row0, col0, min_fhd[tm][tn][0]);
      store(row0, col1, min_fhd[tm][tn][1]);
      store(row1, col0, min_fhd[tm][tn][2]);
      store(row1, col1, min_fhd[tm][tn][3]);
    }
  }
}

// ----------------- Preprocessing kernel -----------------

__global__ void preprocess_kernel(const uint32_t *__restrict__ data,
                                  const uint32_t *__restrict__ mask,
                                  uint32_t *__restrict__ premasked, int M) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < M * K_WORDS) {
    premasked[idx] = data[idx] & mask[idx];
  }
}

// ----------------- Pack theta-major kernel -----------------
// Packs (M, 16, 200, 2, 2) uint8 bits into (M, 400) int32 words in-place.
// Uses shared memory to buffer each row before writing compacted output.
// Input layout strides: r=800, theta=4, d0=2, d1=1
// Output is theta-major: theta is fastest-varying in the packed bitstream.

__global__ void pack_theta_major_kernel(uint8_t *data, int M) {
  int m = blockIdx.x;
  if (m >= M)
    return;

  // Use dynamically allocated shared memory (passed via launch config)
  extern __shared__ uint32_t smem_raw[];
  uint8_t *bits_smem = reinterpret_cast<uint8_t *>(smem_raw);

  // Load row m from global memory into shared memory
  // Input row is at offset m * 12800 bytes
  const uint8_t *in_row = data + m * K_BITS;
  for (int i = threadIdx.x; i < K_BITS; i += blockDim.x) {
    bits_smem[i] = in_row[i];
  }
  __syncthreads();

  // Output location (compacted): row m starts at m * 400 words
  uint32_t *out_row = reinterpret_cast<uint32_t *>(data) + m * K_WORDS;

  // Pack 400 words, each thread handles multiple words
  for (int w = threadIdx.x; w < K_WORDS; w += blockDim.x) {
    uint32_t word = 0;

#pragma unroll
    for (int b = 0; b < 32; b++) {
      int linear_bit = w * 32 + b;
      // Theta-major layout: theta varies fastest in groups of 64 bits
      int theta = linear_bit / 64; // 0..199
      int inner = linear_bit % 64; // position within (16, 2, 2) block
      int r = inner / 4;           // 0..15
      int d0 = (inner / 2) % 2;    // 0..1
      int d1 = inner % 2;          // 0..1

      // Source index in original (16, 200, 2, 2) layout
      // Strides: r=800, theta=4, d0=2, d1=1
      int src_idx = r * 800 + theta * 4 + d0 * 2 + d1;

      if (bits_smem[src_idx]) {
        word |= (1u << b);
      }
    }
    out_row[w] = word;
  }
}

// C++/Python-facing launcher for packing kernel.
// Takes buffer of size M * 12800 bytes (uint8), packs in-place to M * 400
// int32. After this call, only the first M * 1600 bytes contain valid data.
extern "C" void launch_pack_theta_major_cuda(uint8_t *data, int M,
                                             cudaStream_t stream) {
  // One block per row, 256 threads, 12800 bytes shared memory (aligned to 4)
  int threads = 256;
  size_t smem = ((K_BITS + 3) / 4) * 4; // 12800 bytes, aligned
  pack_theta_major_kernel<<<M, threads, smem, stream>>>(data, M);
}

// C++/Python-facing launcher (keeps launch config and smem sizing in one
// place).
extern "C" void launch_masked_hamming_cuda(
    const uint32_t *dData, const uint32_t *dMask, uint32_t *dPremasked, int M,
    float *dD, bool write_output, bool collect_pairs, float threshold,
    uint2 *dPairs, unsigned int *dMatchCount, unsigned int max_pairs,
    cudaStream_t stream) {
  int total = M * K_WORDS;
  preprocess_kernel<<<(total + 255) / 256, 256, 0, stream>>>(dData, dMask,
                                                             dPremasked, M);

  dim3 grid((M + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
  dim3 block(WARPS_PER_BLOCK * 32);

  size_t smem =
      2 * (2 * BLOCK_M * K_CHUNK_WORDS + 2 * BLOCK_N * K_CHUNK_WORDS_EXT) *
      sizeof(uint32_t);

  min_hamming_kernel<<<grid, block, smem, stream>>>(
      dPremasked, dMask, M, dD, write_output, collect_pairs, threshold, dPairs,
      dMatchCount, max_pairs);
}
