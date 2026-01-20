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

// ----------------- MMA: b1 AND+POPC (TensorCore + Fallback) -----------------
//
// Computes C[16x8] += popcount(A[16x256] & B[256x8]) using binary AND +
// popcount. On SM80+ (Ampere): Uses native tensor core instruction. On older
// GPUs: Uses scalar __popc with warp shuffles for data gathering.
//
// Fragment layout (same for both implementations):
//   Thread lane l where group=l/4 (0-7), tid=l%4 (0-3):
//   A fragments: a0=A[group, word tid], a1=A[group+8, word tid],
//                a2=A[group, word tid+4], a3=A[group+8, word tid+4]
//   B fragments: b0=B[col group, word tid], b1=B[col group, word tid+4]
//   C outputs:   c0=C[group, tid*2], c1=C[group, tid*2+1],
//                c2=C[group+8, tid*2], c3=C[group+8, tid*2+1]

// Architecture detection for b1 MMA support:
// - SM80-SM90 (Ampere, Ada, Hopper): Native mma.sync.aligned b1 instruction
// - SM75 and below (Turing, Volta, Pascal): Fallback scalar implementation
// - SM100+ (Blackwell): Uses new tcgen05.mma family, b1 MMA not available
#define HAS_B1_MMA (__CUDA_ARCH__ >= 800 && __CUDA_ARCH__ < 1000)

// Query which implementation is active (for benchmark reporting)
__device__ __host__ inline bool mma_uses_tensor_cores() {
#ifdef FORCE_FALLBACK
  return false;
#elif defined(__CUDA_ARCH__) && HAS_B1_MMA
  return true;
#else
  return false;
#endif
}

__device__ inline void mma_b1_and_popc_16x8x256(int32_t &c0, int32_t &c1,
                                                int32_t &c2, int32_t &c3,
                                                uint32_t a0, uint32_t a1,
                                                uint32_t a2, uint32_t a3,
                                                uint32_t b0, uint32_t b1) {
#if !defined(FORCE_FALLBACK) && HAS_B1_MMA
  // Native tensor core binary MMA (SM80+ / Ampere and newer)
  asm volatile("mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
               "{%0, %1, %2, %3}, "
               "{%4, %5, %6, %7}, "
               "{%8, %9}, "
               "{%0, %1, %2, %3};\n"
               : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
  // Fallback for pre-Ampere GPUs using scalar popcount + warp shuffles
  int lane = threadIdx.x & 31;
  int group = lane >> 2; // 0-7, determines which A rows this thread handles
  int tid = lane & 3;    // 0-3, determines which B columns to compute

  // Gather complete A rows for rows 'group' and 'group+8'
  // Each row needs 8 words (256 bits), distributed across 4 threads in same
  // group
  uint32_t a_row0[8], a_row1[8];
  int group_base = group * 4;

#pragma unroll
  for (int w = 0; w < 4; w++) {
    int src_lane = group_base + w;
    a_row0[w] = __shfl_sync(0xffffffff, a0, src_lane);
    a_row0[w + 4] = __shfl_sync(0xffffffff, a2, src_lane);
    a_row1[w] = __shfl_sync(0xffffffff, a1, src_lane);
    a_row1[w + 4] = __shfl_sync(0xffffffff, a3, src_lane);
  }

  // Gather B columns tid*2 and tid*2+1
  // Each column needs 8 words, distributed across threads where group equals
  // col
  uint32_t b_col0[8], b_col1[8];
  int col0 = tid * 2;
  int col1 = tid * 2 + 1;

#pragma unroll
  for (int w = 0; w < 4; w++) {
    int src_lane0 = col0 * 4 + w;
    int src_lane1 = col1 * 4 + w;
    b_col0[w] = __shfl_sync(0xffffffff, b0, src_lane0);
    b_col0[w + 4] = __shfl_sync(0xffffffff, b1, src_lane0);
    b_col1[w] = __shfl_sync(0xffffffff, b0, src_lane1);
    b_col1[w + 4] = __shfl_sync(0xffffffff, b1, src_lane1);
  }

  // Compute AND + popcount for each output element
  int32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;

#pragma unroll
  for (int w = 0; w < 8; w++) {
    acc0 += __popc(a_row0[w] & b_col0[w]); // C[group, tid*2]
    acc1 += __popc(a_row0[w] & b_col1[w]); // C[group, tid*2+1]
    acc2 += __popc(a_row1[w] & b_col0[w]); // C[group+8, tid*2]
    acc3 += __popc(a_row1[w] & b_col1[w]); // C[group+8, tid*2+1]
  }

  c0 += acc0;
  c1 += acc1;
  c2 += acc2;
  c3 += acc3;
#endif
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

// --w--------------- Minimum Fractional Hamming Distance kernel
// -----------------
//
// Optimized structure: process all shifts within the K-loop to maximize A data
// reuse. For each K chunk: load A once, load B_ext once, then iterate over all
// shifts. This uses 31× more accumulators but provides much better cache
// behavior.

__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32, 1) min_hamming_kernel(
    const uint32_t *__restrict__ premasked, const uint32_t *__restrict__ mask,
    int M,
    // Iris config (runtime)
    int k_words, int k_chunks, int words_per_shift,
    // Classification parameters
    const int32_t *__restrict__ labels, // [M] or nullptr (no classification)
    float match_threshold, float non_match_threshold, bool is_similarity,
    uint8_t include_flags,
    // Stratified sampling parameters
    SamplingConfig sampling,
    // Output arrays (sparse format)
    int32_t *__restrict__ pair_indices, // [max_pairs, 2]
    uint8_t *__restrict__ categories,   // [max_pairs]
    float *__restrict__ out_distances,  // [max_pairs]
    unsigned int *match_count, unsigned int max_pairs) {
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
                                src + g_row * k_words + chunk_word + w, 16);
      } else {
        dst[idx] = dst[idx + 1] = dst[idx + 2] = dst[idx + 3] = 0;
      }
    }
  };

  auto load_chunk_B_ext = [&](uint32_t *dst, const uint32_t *src,
                              int block_start, int chunk_idx,
                              int shift_words_arg) {
    int chunk_word = chunk_idx * K_CHUNK_WORDS;
    int num_elems = BLOCK_N * K_CHUNK_WORDS_EXT;

    for (int idx = threadIdx.x; idx < num_elems; idx += blockDim.x) {
      int r = idx / K_CHUNK_WORDS_EXT;
      int w = idx % K_CHUNK_WORDS_EXT;
      int g_row = block_start + r;
      int g_word = chunk_word + w - 1 + shift_words_arg;

      if (g_row < M) {
        // Optimized wraparound: avoid expensive modulo
        int gw = g_word;
        if (gw < 0)
          gw += k_words;
        else if (gw >= k_words)
          gw -= k_words;
        const uint32_t *src_ptr = src + g_row * k_words + gw;
        __pipeline_memcpy_async(dst + idx, src_ptr, sizeof(uint32_t));
      } else {
        dst[idx] = 0;
      }
    }
  };

  // Process one theta-roll shift at a time (sequential to minimize register
  // pressure)
  for (int shift = -MAX_SHIFT; shift <= MAX_SHIFT; ++shift) {
    int shift_words = shift * words_per_shift;
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

    for (int kc = 0; kc < k_chunks; ++kc) {
      uint32_t *A_pm_buf = A_pm_stage(stage);
      uint32_t *A_m_buf = A_m_stage(stage);
      uint32_t *B_pm_buf = B_pm_stage(stage);
      uint32_t *B_m_buf = B_m_stage(stage);

      int next_stage = stage ^ 1;
      if (kc + 1 < k_chunks) {
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

      if (kc + 1 < k_chunks)
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
            min_fhd[tm][tn][r] = fminf(min_fhd[tm][tn][r], fhd);
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
        if (pair_indices == nullptr || match_count == nullptr)
          return;

        // Early exit if buffer is already full (reduces atomic contention)
        // This is a non-atomic read for performance - may slightly over-count
        if (*match_count >= max_pairs)
          return;

        // Classification logic
        uint8_t category;
        bool emit;

        if (labels != nullptr) {
          // Label-aware classification using cached loads
          int32_t label_i = __ldg(&labels[gi]);
          int32_t label_j = __ldg(&labels[gj]);
          bool same_label = (label_i == label_j);

          bool is_match, is_non_match;
          if (is_similarity) {
            is_match = (val >= match_threshold);
            is_non_match = (val < non_match_threshold);
          } else {
            is_match = (val <= match_threshold);
            is_non_match = (val > non_match_threshold);
          }

          // Compute category and emit flag together
          if (same_label) {
            if (is_match) {
              category = CATEGORY_TRUE_MATCH;
              emit = (include_flags & INCLUDE_TM);
            } else if (is_non_match) {
              category = CATEGORY_FALSE_NON_MATCH;
              emit = (include_flags & INCLUDE_FNM);
            } else {
              emit = false; // In gap
            }
          } else {
            if (is_match) {
              category = CATEGORY_FALSE_MATCH;
              emit = (include_flags & INCLUDE_FM);
            } else if (is_non_match) {
              category = CATEGORY_TRUE_NON_MATCH;
              emit = (include_flags & INCLUDE_TNM);
            } else {
              emit = false; // In gap
            }
          }
        } else {
          // No labels: emit all pairs (no classification)
          category = 0xFF;
          emit = true;
        }

        // Apply stratified sampling if enabled
        if (emit && sampling.enabled()) {
          emit = sampling.should_sample(val, gi, gj);
        }

        if (emit) {
          unsigned int idx = atomicAdd(match_count, 1);
          if (idx < max_pairs) {
            pair_indices[idx * 2] = gi;
            pair_indices[idx * 2 + 1] = gj;
            if (categories)
              categories[idx] = category;
            if (out_distances)
              out_distances[idx] = val;
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
                                  uint32_t *__restrict__ premasked, int M,
                                  int k_words) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < M * k_words) {
    premasked[idx] = data[idx] & mask[idx];
  }
}

// ----------------- Pack theta-major kernel -----------------
// Packs (M, R_DIM, THETA_DIM, 2, 2) uint8 bits into (M, K_WORDS) int32 words
// IN-PLACE. Uses cooperative groups for grid-level sync to ensure all reads
// complete before any writes. Input layout strides: r=THETA_DIM*4, theta=4,
// d0=2, d1=1 Output is theta-major: theta is fastest-varying in the packed
// bitstream.

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void pack_theta_major_kernel(uint8_t *data, uint32_t *output,
                                        int M, int row_offset, int r_dim,
                                        int theta_dim, int d0_dim, int d1_dim,
                                        int k_bits, int k_words,
                                        int bits_per_theta_col,
                                        int inner_size) {
  cg::grid_group grid = cg::this_grid();

  int m = blockIdx.x;
  if (m >= M)
    return;

  // Use shared memory to buffer this row's input before grid sync
  extern __shared__ uint32_t smem_raw[];
  uint8_t *bits_smem = reinterpret_cast<uint8_t *>(smem_raw);

  // Load row m from global memory into shared memory
  const uint8_t *in_row = data + m * k_bits;
  for (int i = threadIdx.x; i < k_bits; i += blockDim.x) {
    bits_smem[i] = in_row[i];
  }
  __syncthreads();

  // Pack into registers (not yet written to global memory)
  // Dynamic array sizing: each thread handles k_words/blockDim.x + 1 words max
  constexpr int MAX_WORDS_PER_THREAD = 16; // Generous upper bound
  uint32_t packed_words[MAX_WORDS_PER_THREAD];
  int word_indices[MAX_WORDS_PER_THREAD];
  int num_words = 0;

  for (int w = threadIdx.x; w < k_words && num_words < MAX_WORDS_PER_THREAD;
       w += blockDim.x) {
    uint32_t word = 0;

    for (int b = 0; b < 32; b++) {
      int linear_bit = w * 32 + b;
      // Theta-major output: linear_bit = theta * bits_per_theta_col +
      //   r * inner_size + d0 * d1_dim + d1
      int theta = linear_bit / bits_per_theta_col;
      int inner = linear_bit % bits_per_theta_col;
      int r = inner / inner_size;
      int d_inner = inner % inner_size;
      int d0 = d_inner / d1_dim;
      int d1 = d_inner % d1_dim;
      // R-major input: src_idx = r * (theta_dim * inner_size) +
      //   theta * inner_size + d0 * d1_dim + d1
      int src_idx =
          r * (theta_dim * inner_size) + theta * inner_size + d0 * d1_dim + d1;

      if (bits_smem[src_idx]) {
        word |= (1u << b);
      }
    }
    packed_words[num_words] = word;
    word_indices[num_words] = w;
    num_words++;
  }

  // Grid-level sync: all blocks wait here until everyone has read their data
  grid.sync();

  // Now safe to write - all reads are complete
  // Use row_offset to compute correct output position for batched calls
  uint32_t *out_row = output + (row_offset + m) * k_words;
  for (int i = 0; i < num_words; i++) {
    out_row[word_indices[i]] = packed_words[i];
  }
}

// C++/Python-facing launcher for packing kernel.
// Packs M * k_bits bytes (uint8) into M * k_words int32 words IN-PLACE.
// Uses cooperative kernel launch for grid-level synchronization.
// Automatically batches if M exceeds device cooperative launch limits.
extern "C" void launch_pack_theta_major_cuda(uint8_t *data, int M, int r_dim,
                                             int theta_dim, int d0_dim,
                                             int d1_dim, cudaStream_t stream) {
  IrisConfig cfg = IrisConfig::from_dims(r_dim, theta_dim, d0_dim, d1_dim);
  int threads = 256;
  size_t smem = ((cfg.k_bits + 3) / 4) * 4; // Aligned to 4 bytes

  // Output is written to the same memory, reinterpreted as uint32
  uint32_t *output = reinterpret_cast<uint32_t *>(data);

  // Query max blocks for cooperative launch on current device
  int device;
  cudaGetDevice(&device);
  int numBlocksPerSm;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm,
                                                 pack_theta_major_kernel,
                                                 threads, smem);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  int maxBlocks = numBlocksPerSm * prop.multiProcessorCount;

  // Process in batches if M exceeds cooperative launch limit
  // Use size_t for offset calculations to avoid integer overflow with large M
  for (int start = 0; start < M; start += maxBlocks) {
    int batch_M = min(M - start, maxBlocks);
    // Use size_t to prevent overflow: start * k_bits can exceed INT32_MAX
    uint8_t *batch_data = data + (size_t)start * cfg.k_bits;

    void *args[] = {&batch_data,    &output,      &batch_M,     &start,
                    &cfg.r_dim,     &cfg.theta_dim, &cfg.d0_dim,  &cfg.d1_dim,
                    &cfg.k_bits,    &cfg.k_words, &cfg.bits_per_theta_col,
                    &cfg.inner_size};
    cudaLaunchCooperativeKernel((void *)pack_theta_major_kernel, dim3(batch_M),
                                dim3(threads), args, smem, stream);
    // Must sync between batches since cooperative kernel uses grid sync
    if (start + maxBlocks < M) {
      cudaStreamSynchronize(stream);
    }
  }
}

// ----------------- Repack u32 from r-major to theta-major -----------------
// Input: (M, k_words) int32 packed in r-major order: bit[r,theta,d0,d1] at
//        linear_bit = r*(theta_dim*inner_size) + theta*inner_size + d0*d1_dim +
//        d1
// Output: (M, k_words) int32 packed in theta-major order: bit[r,theta,d0,d1] at
//        linear_bit = theta*bits_per_theta_col + r*inner_size + d0*d1_dim + d1

__global__ void
repack_to_theta_major_kernel(const uint32_t *__restrict__ input,
                             uint32_t *__restrict__ output, int M, int r_dim,
                             int theta_dim, int d0_dim, int d1_dim, int k_words,
                             int bits_per_theta_col, int inner_size) {
  int m = blockIdx.x;
  if (m >= M)
    return;

  extern __shared__ uint32_t smem[];

  // Load entire row into shared memory
  const uint32_t *in_row = input + m * k_words;
  for (int w = threadIdx.x; w < k_words; w += blockDim.x) {
    smem[w] = in_row[w];
  }
  __syncthreads();

  // Repack to theta-major
  uint32_t *out_row = output + m * k_words;
  for (int w = threadIdx.x; w < k_words; w += blockDim.x) {
    uint32_t word = 0;

    for (int b = 0; b < 32; b++) {
      // Output bit position (theta-major): theta*bits_per_theta_col +
      //   r*inner_size + d0*d1_dim + d1
      int dst_linear_bit = w * 32 + b;
      int theta = dst_linear_bit / bits_per_theta_col;
      int inner = dst_linear_bit % bits_per_theta_col;
      int r = inner / inner_size;
      int d_inner = inner % inner_size;
      int d0 = d_inner / d1_dim;
      int d1 = d_inner % d1_dim;

      // Source bit position (r-major): r*(theta_dim*inner_size) +
      //   theta*inner_size + d0*d1_dim + d1
      int src_linear_bit =
          r * (theta_dim * inner_size) + theta * inner_size + d0 * d1_dim + d1;
      int src_word = src_linear_bit / 32;
      int src_bit = src_linear_bit % 32;

      if ((smem[src_word] >> src_bit) & 1) {
        word |= (1u << b);
      }
    }

    out_row[w] = word;
  }
}

// C++/Python-facing launcher for repacking kernel.
// Repacks M * k_words int32 words from r-major to theta-major order.
extern "C" void launch_repack_to_theta_major_cuda(const uint32_t *input,
                                                  uint32_t *output, int M,
                                                  int r_dim, int theta_dim,
                                                  int d0_dim, int d1_dim,
                                                  cudaStream_t stream) {
  IrisConfig cfg = IrisConfig::from_dims(r_dim, theta_dim, d0_dim, d1_dim);
  int threads = 256;
  size_t smem = cfg.k_words * sizeof(uint32_t);

  repack_to_theta_major_kernel<<<M, threads, smem, stream>>>(
      input, output, M, cfg.r_dim, cfg.theta_dim, cfg.d0_dim, cfg.d1_dim,
      cfg.k_words, cfg.bits_per_theta_col, cfg.inner_size);
}

// C++/Python-facing launcher (keeps launch config and smem sizing in one
// place).
extern "C" void launch_masked_hamming_cuda(
    const uint32_t *dData, const uint32_t *dMask, uint32_t *dPremasked, int M,
    int r_dim, int theta_dim, const int32_t *dLabels, float match_threshold,
    float non_match_threshold, bool is_similarity, uint8_t include_flags,
    SamplingConfig sampling,
    int32_t *dPairIndices, uint8_t *dCategories, float *dOutDistances,
    unsigned int *dMatchCount, unsigned int max_pairs, cudaStream_t stream) {
  IrisConfig cfg = IrisConfig::from_dims(r_dim, theta_dim);

  int total = M * cfg.k_words;
  preprocess_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
      dData, dMask, dPremasked, M, cfg.k_words);

  dim3 grid((M + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
  dim3 block(WARPS_PER_BLOCK * 32);

  size_t smem =
      2 * (2 * BLOCK_M * K_CHUNK_WORDS + 2 * BLOCK_N * K_CHUNK_WORDS_EXT) *
      sizeof(uint32_t);

  min_hamming_kernel<<<grid, block, smem, stream>>>(
      dPremasked, dMask, M, cfg.k_words, cfg.k_chunks, cfg.words_per_shift,
      dLabels, match_threshold, non_match_threshold, is_similarity,
      include_flags, sampling, dPairIndices, dCategories, dOutDistances, dMatchCount,
      max_pairs);
}

// ----------------- A vs B kernel: compares all pairs between two sets
// -----------------
//
// Unlike the self-comparison kernel which only computes lower triangle (i > j),
// this kernel computes the full M_A x M_B rectangle.

__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32, 1)
    min_hamming_ab_kernel(
        const uint32_t *__restrict__ premasked_A,
        const uint32_t *__restrict__ mask_A,
        const uint32_t *__restrict__ premasked_B,
        const uint32_t *__restrict__ mask_B, int M_A, int M_B,
        // Iris config (runtime)
        int k_words, int k_chunks, int words_per_shift,
        // Classification parameters
        const int32_t *__restrict__ labels_A, // [M_A] or nullptr
        const int32_t *__restrict__ labels_B, // [M_B] or nullptr
        float match_threshold, float non_match_threshold, bool is_similarity,
        uint8_t include_flags,
        // Stratified sampling parameters
        SamplingConfig sampling,
        // Output arrays (sparse format)
        int32_t *__restrict__ pair_indices, // [max_pairs, 2]
        uint8_t *__restrict__ categories,   // [max_pairs]
        float *__restrict__ out_distances,  // [max_pairs]
        unsigned int *match_count, unsigned int max_pairs) {
  int warp_id = threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  int group = lane >> 2;
  int tid = lane & 3;

  int cta_row = blockIdx.y;
  int cta_col = blockIdx.x;
  int block_row_start = cta_row * BLOCK_M;
  int block_col_start = cta_col * BLOCK_N;

  // No triangle skip for A vs B - compute full rectangle
  if (block_row_start >= M_A || block_col_start >= M_B)
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

  // Lambda to load A chunks (from set A)
  auto load_chunk_A = [&](uint32_t *dst, const uint32_t *src, int block_start,
                          int chunk_word, int M_limit) {
    int num_words = BLOCK_M * K_CHUNK_WORDS;
    for (int idx4 = threadIdx.x; idx4 * 4 < num_words; idx4 += blockDim.x) {
      int idx = idx4 * 4;
      int r = idx / K_CHUNK_WORDS;
      int w = idx % K_CHUNK_WORDS;
      int g_row = block_start + r;
      if (g_row < M_limit) {
        __pipeline_memcpy_async(dst + idx,
                                src + g_row * k_words + chunk_word + w, 16);
      } else {
        dst[idx] = dst[idx + 1] = dst[idx + 2] = dst[idx + 3] = 0;
      }
    }
  };

  // Lambda to load B chunks with extended halo (from set B)
  auto load_chunk_B_ext = [&](uint32_t *dst, const uint32_t *src,
                              int block_start, int chunk_idx,
                              int shift_words_arg, int M_limit) {
    int chunk_word = chunk_idx * K_CHUNK_WORDS;
    int num_elems = BLOCK_N * K_CHUNK_WORDS_EXT;

    for (int idx = threadIdx.x; idx < num_elems; idx += blockDim.x) {
      int r = idx / K_CHUNK_WORDS_EXT;
      int w = idx % K_CHUNK_WORDS_EXT;
      int g_row = block_start + r;
      int g_word = chunk_word + w - 1 + shift_words_arg;

      if (g_row < M_limit) {
        // Optimized wraparound: avoid expensive modulo
        int gw = g_word;
        if (gw < 0)
          gw += k_words;
        else if (gw >= k_words)
          gw -= k_words;
        const uint32_t *src_ptr = src + g_row * k_words + gw;
        __pipeline_memcpy_async(dst + idx, src_ptr, sizeof(uint32_t));
      } else {
        dst[idx] = 0;
      }
    }
  };

  // Process one theta-roll shift at a time
  for (int shift = -MAX_SHIFT; shift <= MAX_SHIFT; ++shift) {
    int shift_words = shift * words_per_shift;
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
      load_chunk_A(A_pm_stage(stage), premasked_A, block_row_start, 0, M_A);
      load_chunk_A(A_m_stage(stage), mask_A, block_row_start, 0, M_A);
      load_chunk_B_ext(B_pm_stage(stage), premasked_B, block_col_start, 0,
                       shift_words, M_B);
      load_chunk_B_ext(B_m_stage(stage), mask_B, block_col_start, 0,
                       shift_words, M_B);
      __pipeline_commit();
      __pipeline_wait_prior(0);
      __syncthreads();
    }

    for (int kc = 0; kc < k_chunks; ++kc) {
      uint32_t *A_pm_buf = A_pm_stage(stage);
      uint32_t *A_m_buf = A_m_stage(stage);
      uint32_t *B_pm_buf = B_pm_stage(stage);
      uint32_t *B_m_buf = B_m_stage(stage);

      int next_stage = stage ^ 1;
      if (kc + 1 < k_chunks) {
        load_chunk_A(A_pm_stage(next_stage), premasked_A, block_row_start,
                     (kc + 1) * K_CHUNK_WORDS, M_A);
        load_chunk_A(A_m_stage(next_stage), mask_A, block_row_start,
                     (kc + 1) * K_CHUNK_WORDS, M_A);
        load_chunk_B_ext(B_pm_stage(next_stage), premasked_B, block_col_start,
                         kc + 1, shift_words, M_B);
        load_chunk_B_ext(B_m_stage(next_stage), mask_B, block_col_start, kc + 1,
                         shift_words, M_B);
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

      if (kc + 1 < k_chunks)
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
            min_fhd[tm][tn][r] = fminf(min_fhd[tm][tn][r], fhd);
          }
        }
  }

  // Store results - no triangle constraint for A vs B
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
        if (gi >= M_A || gj >= M_B)
          return;
        if (pair_indices == nullptr || match_count == nullptr)
          return;

        // Early exit if buffer is already full (reduces atomic contention)
        if (*match_count >= max_pairs)
          return;

        // Classification logic
        uint8_t category;
        bool emit;

        if (labels_A != nullptr && labels_B != nullptr) {
          // Label-aware classification using cached loads
          int32_t label_i = __ldg(&labels_A[gi]);
          int32_t label_j = __ldg(&labels_B[gj]);
          bool same_label = (label_i == label_j);

          bool is_match, is_non_match;
          if (is_similarity) {
            is_match = (val >= match_threshold);
            is_non_match = (val < non_match_threshold);
          } else {
            is_match = (val <= match_threshold);
            is_non_match = (val > non_match_threshold);
          }

          // Compute category and emit flag together
          if (same_label) {
            if (is_match) {
              category = CATEGORY_TRUE_MATCH;
              emit = (include_flags & INCLUDE_TM);
            } else if (is_non_match) {
              category = CATEGORY_FALSE_NON_MATCH;
              emit = (include_flags & INCLUDE_FNM);
            } else {
              emit = false; // In gap
            }
          } else {
            if (is_match) {
              category = CATEGORY_FALSE_MATCH;
              emit = (include_flags & INCLUDE_FM);
            } else if (is_non_match) {
              category = CATEGORY_TRUE_NON_MATCH;
              emit = (include_flags & INCLUDE_TNM);
            } else {
              emit = false; // In gap
            }
          }
        } else {
          // No labels: emit all pairs (no classification)
          category = 0xFF;
          emit = true;
        }

        // Apply stratified sampling if enabled
        if (emit && sampling.enabled()) {
          emit = sampling.should_sample(val, gi, gj);
        }

        if (emit) {
          unsigned int idx = atomicAdd(match_count, 1);
          if (idx < max_pairs) {
            pair_indices[idx * 2] = gi;
            pair_indices[idx * 2 + 1] = gj;
            if (categories)
              categories[idx] = category;
            if (out_distances)
              out_distances[idx] = val;
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

// C++/Python-facing launcher for A vs B comparison.
extern "C" void launch_masked_hamming_ab_cuda(
    const uint32_t *dData_A, const uint32_t *dMask_A, uint32_t *dPremasked_A,
    const uint32_t *dData_B, const uint32_t *dMask_B, uint32_t *dPremasked_B,
    int M_A, int M_B, int r_dim, int theta_dim, const int32_t *dLabels_A,
    const int32_t *dLabels_B, float match_threshold, float non_match_threshold,
    bool is_similarity, uint8_t include_flags, SamplingConfig sampling,
    int32_t *dPairIndices, uint8_t *dCategories, float *dOutDistances,
    unsigned int *dMatchCount, unsigned int max_pairs, cudaStream_t stream) {
  IrisConfig cfg = IrisConfig::from_dims(r_dim, theta_dim);

  // Preprocess both A and B sets
  int total_A = M_A * cfg.k_words;
  int total_B = M_B * cfg.k_words;
  preprocess_kernel<<<(total_A + 255) / 256, 256, 0, stream>>>(
      dData_A, dMask_A, dPremasked_A, M_A, cfg.k_words);
  preprocess_kernel<<<(total_B + 255) / 256, 256, 0, stream>>>(
      dData_B, dMask_B, dPremasked_B, M_B, cfg.k_words);

  dim3 grid((M_B + BLOCK_N - 1) / BLOCK_N, (M_A + BLOCK_M - 1) / BLOCK_M);
  dim3 block(WARPS_PER_BLOCK * 32);

  size_t smem =
      2 * (2 * BLOCK_M * K_CHUNK_WORDS + 2 * BLOCK_N * K_CHUNK_WORDS_EXT) *
      sizeof(uint32_t);

  min_hamming_ab_kernel<<<grid, block, smem, stream>>>(
      dPremasked_A, dMask_A, dPremasked_B, dMask_B, M_A, M_B, cfg.k_words,
      cfg.k_chunks, cfg.words_per_shift, dLabels_A, dLabels_B, match_threshold,
      non_match_threshold, is_similarity, include_flags, sampling, dPairIndices,
      dCategories, dOutDistances, dMatchCount, max_pairs);
}
