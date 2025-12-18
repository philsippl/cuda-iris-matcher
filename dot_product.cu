#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdint.h>
#include <stdio.h>

#include "iris_params.h"

using namespace nvcuda;

// ----------------- Dot Product Configuration -----------------
constexpr int DEFAULT_DOT_VEC_DIM = 512;

// Optimized configuration for WMMA-based GEMM
constexpr int DOT_BLOCK_M = 128;
constexpr int DOT_BLOCK_N = 128; 
constexpr int DOT_BLOCK_K = 32;

constexpr int WMMA_M_DOT = 16;
constexpr int WMMA_N_DOT = 16;
constexpr int WMMA_K_DOT = 16;

// Each warp handles 2x2 = 4 WMMA tiles (32x32 elements)
constexpr int DOT_TILES_M_PER_WARP = 2;
constexpr int DOT_TILES_N_PER_WARP = 2;

// 4x4 = 16 warps = 512 threads
constexpr int DOT_WARPS_M = 4;
constexpr int DOT_WARPS_N = 4;
constexpr int DOT_WARPS_PER_BLOCK = DOT_WARPS_M * DOT_WARPS_N;

constexpr int DOT_WARP_TILE_M = DOT_TILES_M_PER_WARP * WMMA_M_DOT;  // 32
constexpr int DOT_WARP_TILE_N = DOT_TILES_N_PER_WARP * WMMA_N_DOT;  // 32

constexpr int SMEM_RESULT_SIZE = WMMA_M_DOT * WMMA_N_DOT;

// ----------------- Helper for result emission -----------------
__device__ __forceinline__ void emit_tile_results(
    const wmma::fragment<wmma::accumulator, WMMA_M_DOT, WMMA_N_DOT, WMMA_K_DOT, float>& c_frag,
    float* tile_smem,
    int tile_row_start, int tile_col_start,
    int M, int M_limit,
    bool is_self,
    const int32_t* labels, const int32_t* labels_b,
    float match_threshold, float non_match_threshold,
    bool is_similarity, uint8_t include_flags,
    int32_t* pair_indices, uint8_t* categories, float* out_scores,
    unsigned int* match_count, unsigned int max_pairs,
    int lane) {
  
  wmma::store_matrix_sync(tile_smem, c_frag, WMMA_N_DOT, wmma::mem_row_major);
  __syncwarp();
  
  #pragma unroll
  for (int e = lane; e < SMEM_RESULT_SIZE; e += 32) {
    int local_row = e / WMMA_N_DOT;
    int local_col = e % WMMA_N_DOT;
    int gi = tile_row_start + local_row;
    int gj = tile_col_start + local_col;
    
    if (gi >= M || gj >= M_limit) continue;
    if (is_self && gi <= gj) continue;
    
    float score = tile_smem[e];
    if (*match_count >= max_pairs) continue;
    
    uint8_t category;
    bool emit;
    
    const int32_t* labels_j = is_self ? labels : labels_b;
    if (labels != nullptr && labels_j != nullptr) {
      int32_t label_i = labels[gi];
      int32_t label_j = labels_j[gj];
      bool same_label = (label_i == label_j);
      
      bool is_match = is_similarity ? (score >= match_threshold) : (score <= match_threshold);
      bool is_non_match = is_similarity ? (score < non_match_threshold) : (score > non_match_threshold);
      
      if (same_label) {
        if (is_match) { category = CATEGORY_TRUE_MATCH; emit = (include_flags & INCLUDE_TM); }
        else if (is_non_match) { category = CATEGORY_FALSE_NON_MATCH; emit = (include_flags & INCLUDE_FNM); }
        else emit = false;
      } else {
        if (is_match) { category = CATEGORY_FALSE_MATCH; emit = (include_flags & INCLUDE_FM); }
        else if (is_non_match) { category = CATEGORY_TRUE_NON_MATCH; emit = (include_flags & INCLUDE_TNM); }
        else emit = false;
      }
    } else {
      category = 0xFF;
      emit = true;
    }
    
    if (emit) {
      unsigned int out_idx = atomicAdd(match_count, 1);
      if (out_idx < max_pairs) {
        pair_indices[out_idx * 2] = gi;
        pair_indices[out_idx * 2 + 1] = gj;
        if (categories) categories[out_idx] = category;
        if (out_scores) out_scores[out_idx] = score;
      }
    }
  }
}

// ----------------- Dense kernel with double buffering -----------------
__global__ void __launch_bounds__(DOT_WARPS_PER_BLOCK * 32, 1)
dot_product_dense_kernel(
    const half *__restrict__ data,
    float *__restrict__ output,
    int M, int vec_dim) {
  
  int warp_id = threadIdx.x / 32;
  
  int block_row_start = blockIdx.y * DOT_BLOCK_M;
  int block_col_start = blockIdx.x * DOT_BLOCK_N;
  
  if (block_row_start >= M || block_col_start >= M) return;
  
  // Double buffer shared memory
  extern __shared__ char smem_raw[];
  half *A_smem[2];
  half *B_smem[2];
  A_smem[0] = reinterpret_cast<half*>(smem_raw);
  A_smem[1] = A_smem[0] + DOT_BLOCK_M * DOT_BLOCK_K;
  B_smem[0] = A_smem[1] + DOT_BLOCK_M * DOT_BLOCK_K;
  B_smem[1] = B_smem[0] + DOT_BLOCK_N * DOT_BLOCK_K;
  
  int warp_m = warp_id / DOT_WARPS_N;
  int warp_n = warp_id % DOT_WARPS_N;
  int warp_row_base = warp_m * DOT_WARP_TILE_M;
  int warp_col_base = warp_n * DOT_WARP_TILE_N;
  
  wmma::fragment<wmma::accumulator, WMMA_M_DOT, WMMA_N_DOT, WMMA_K_DOT, float>
      c_frag[DOT_TILES_M_PER_WARP][DOT_TILES_N_PER_WARP];
  
  #pragma unroll
  for (int tm = 0; tm < DOT_TILES_M_PER_WARP; ++tm)
    #pragma unroll
    for (int tn = 0; tn < DOT_TILES_N_PER_WARP; ++tn)
      wmma::fill_fragment(c_frag[tm][tn], 0.0f);
  
  int k_chunks = (vec_dim + DOT_BLOCK_K - 1) / DOT_BLOCK_K;
  int buf = 0;
  
  // Load first chunk
  {
    int k_start = 0;
    int k_len = min(DOT_BLOCK_K, vec_dim);
    
    for (int idx = threadIdx.x; idx < DOT_BLOCK_M * DOT_BLOCK_K; idx += blockDim.x) {
      int row = idx / DOT_BLOCK_K;
      int col = idx % DOT_BLOCK_K;
      int g_row = block_row_start + row;
      half val = (g_row < M && col < k_len) ? data[g_row * vec_dim + col] : __float2half(0.0f);
      A_smem[0][idx] = val;
    }
    
    for (int idx = threadIdx.x; idx < DOT_BLOCK_N * DOT_BLOCK_K; idx += blockDim.x) {
      int n = idx / DOT_BLOCK_K;
      int k = idx % DOT_BLOCK_K;
      int g_row = block_col_start + n;
      half val = (g_row < M && k < k_len) ? data[g_row * vec_dim + k] : __float2half(0.0f);
      B_smem[0][n * DOT_BLOCK_K + k] = val;
    }
  }
  
  __syncthreads();
  
  for (int kc = 0; kc < k_chunks; ++kc) {
    int next_buf = 1 - buf;
    
    // Prefetch next chunk while computing current
    if (kc + 1 < k_chunks) {
      int k_start = (kc + 1) * DOT_BLOCK_K;
      int k_len = min(DOT_BLOCK_K, vec_dim - k_start);
      
      for (int idx = threadIdx.x; idx < DOT_BLOCK_M * DOT_BLOCK_K; idx += blockDim.x) {
        int row = idx / DOT_BLOCK_K;
        int col = idx % DOT_BLOCK_K;
        int g_row = block_row_start + row;
        int g_col = k_start + col;
        half val = (g_row < M && col < k_len) ? data[g_row * vec_dim + g_col] : __float2half(0.0f);
        A_smem[next_buf][idx] = val;
      }
      
      for (int idx = threadIdx.x; idx < DOT_BLOCK_N * DOT_BLOCK_K; idx += blockDim.x) {
        int n = idx / DOT_BLOCK_K;
        int k = idx % DOT_BLOCK_K;
        int g_row = block_col_start + n;
        int g_col = k_start + k;
        half val = (g_row < M && k < k_len) ? data[g_row * vec_dim + g_col] : __float2half(0.0f);
        B_smem[next_buf][n * DOT_BLOCK_K + k] = val;
      }
    }
    
    // Compute with current buffer
    int k_len = min(DOT_BLOCK_K, vec_dim - kc * DOT_BLOCK_K);
    
    #pragma unroll
    for (int k = 0; k < DOT_BLOCK_K && k < k_len; k += WMMA_K_DOT) {
      #pragma unroll
      for (int tm = 0; tm < DOT_TILES_M_PER_WARP; ++tm) {
        int a_row = warp_row_base + tm * WMMA_M_DOT;
        wmma::fragment<wmma::matrix_a, WMMA_M_DOT, WMMA_N_DOT, WMMA_K_DOT, half, wmma::row_major> a_frag;
        wmma::load_matrix_sync(a_frag, A_smem[buf] + a_row * DOT_BLOCK_K + k, DOT_BLOCK_K);
        
        #pragma unroll
        for (int tn = 0; tn < DOT_TILES_N_PER_WARP; ++tn) {
          int b_col = warp_col_base + tn * WMMA_N_DOT;
          wmma::fragment<wmma::matrix_b, WMMA_M_DOT, WMMA_N_DOT, WMMA_K_DOT, half, wmma::col_major> b_frag;
          wmma::load_matrix_sync(b_frag, B_smem[buf] + b_col * DOT_BLOCK_K + k, DOT_BLOCK_K);
          wmma::mma_sync(c_frag[tm][tn], a_frag, b_frag, c_frag[tm][tn]);
        }
      }
    }
    
    buf = next_buf;
    __syncthreads();
  }
  
  // Write results
  #pragma unroll
  for (int tm = 0; tm < DOT_TILES_M_PER_WARP; ++tm) {
    int tile_row = block_row_start + warp_row_base + tm * WMMA_M_DOT;
    #pragma unroll
    for (int tn = 0; tn < DOT_TILES_N_PER_WARP; ++tn) {
      int tile_col = block_col_start + warp_col_base + tn * WMMA_N_DOT;
      if (tile_row < M && tile_col < M) {
        wmma::store_matrix_sync(output + tile_row * M + tile_col, c_frag[tm][tn], M, wmma::mem_row_major);
      }
    }
  }
}

// Dense A vs B kernel with double buffering
__global__ void __launch_bounds__(DOT_WARPS_PER_BLOCK * 32, 1)
dot_product_ab_dense_kernel(
    const half *__restrict__ data_a,
    const half *__restrict__ data_b,
    float *__restrict__ output,
    int M_A, int M_B, int vec_dim) {
  
  int warp_id = threadIdx.x / 32;
  
  int block_row_start = blockIdx.y * DOT_BLOCK_M;
  int block_col_start = blockIdx.x * DOT_BLOCK_N;
  
  if (block_row_start >= M_A || block_col_start >= M_B) return;
  
  extern __shared__ char smem_raw[];
  half *A_smem[2];
  half *B_smem[2];
  A_smem[0] = reinterpret_cast<half*>(smem_raw);
  A_smem[1] = A_smem[0] + DOT_BLOCK_M * DOT_BLOCK_K;
  B_smem[0] = A_smem[1] + DOT_BLOCK_M * DOT_BLOCK_K;
  B_smem[1] = B_smem[0] + DOT_BLOCK_N * DOT_BLOCK_K;
  
  int warp_m = warp_id / DOT_WARPS_N;
  int warp_n = warp_id % DOT_WARPS_N;
  int warp_row_base = warp_m * DOT_WARP_TILE_M;
  int warp_col_base = warp_n * DOT_WARP_TILE_N;
  
  wmma::fragment<wmma::accumulator, WMMA_M_DOT, WMMA_N_DOT, WMMA_K_DOT, float>
      c_frag[DOT_TILES_M_PER_WARP][DOT_TILES_N_PER_WARP];
  
  #pragma unroll
  for (int tm = 0; tm < DOT_TILES_M_PER_WARP; ++tm)
    #pragma unroll
    for (int tn = 0; tn < DOT_TILES_N_PER_WARP; ++tn)
      wmma::fill_fragment(c_frag[tm][tn], 0.0f);
  
  int k_chunks = (vec_dim + DOT_BLOCK_K - 1) / DOT_BLOCK_K;
  int buf = 0;
  
  // Load first chunk
  {
    int k_len = min(DOT_BLOCK_K, vec_dim);
    for (int idx = threadIdx.x; idx < DOT_BLOCK_M * DOT_BLOCK_K; idx += blockDim.x) {
      int row = idx / DOT_BLOCK_K;
      int col = idx % DOT_BLOCK_K;
      int g_row = block_row_start + row;
      half val = (g_row < M_A && col < k_len) ? data_a[g_row * vec_dim + col] : __float2half(0.0f);
      A_smem[0][idx] = val;
    }
    for (int idx = threadIdx.x; idx < DOT_BLOCK_N * DOT_BLOCK_K; idx += blockDim.x) {
      int n = idx / DOT_BLOCK_K;
      int k = idx % DOT_BLOCK_K;
      int g_row = block_col_start + n;
      half val = (g_row < M_B && k < k_len) ? data_b[g_row * vec_dim + k] : __float2half(0.0f);
      B_smem[0][n * DOT_BLOCK_K + k] = val;
    }
  }
  __syncthreads();
  
  for (int kc = 0; kc < k_chunks; ++kc) {
    int next_buf = 1 - buf;
    
    if (kc + 1 < k_chunks) {
      int k_start = (kc + 1) * DOT_BLOCK_K;
      int k_len = min(DOT_BLOCK_K, vec_dim - k_start);
      for (int idx = threadIdx.x; idx < DOT_BLOCK_M * DOT_BLOCK_K; idx += blockDim.x) {
        int row = idx / DOT_BLOCK_K;
        int col = idx % DOT_BLOCK_K;
        int g_row = block_row_start + row;
        int g_col = k_start + col;
        half val = (g_row < M_A && col < k_len) ? data_a[g_row * vec_dim + g_col] : __float2half(0.0f);
        A_smem[next_buf][idx] = val;
      }
      for (int idx = threadIdx.x; idx < DOT_BLOCK_N * DOT_BLOCK_K; idx += blockDim.x) {
        int n = idx / DOT_BLOCK_K;
        int k = idx % DOT_BLOCK_K;
        int g_row = block_col_start + n;
        int g_col = k_start + k;
        half val = (g_row < M_B && k < k_len) ? data_b[g_row * vec_dim + g_col] : __float2half(0.0f);
        B_smem[next_buf][n * DOT_BLOCK_K + k] = val;
      }
    }
    
    int k_len = min(DOT_BLOCK_K, vec_dim - kc * DOT_BLOCK_K);
    #pragma unroll
    for (int k = 0; k < DOT_BLOCK_K && k < k_len; k += WMMA_K_DOT) {
      #pragma unroll
      for (int tm = 0; tm < DOT_TILES_M_PER_WARP; ++tm) {
        int a_row = warp_row_base + tm * WMMA_M_DOT;
        wmma::fragment<wmma::matrix_a, WMMA_M_DOT, WMMA_N_DOT, WMMA_K_DOT, half, wmma::row_major> a_frag;
        wmma::load_matrix_sync(a_frag, A_smem[buf] + a_row * DOT_BLOCK_K + k, DOT_BLOCK_K);
        
        #pragma unroll
        for (int tn = 0; tn < DOT_TILES_N_PER_WARP; ++tn) {
          int b_col = warp_col_base + tn * WMMA_N_DOT;
          wmma::fragment<wmma::matrix_b, WMMA_M_DOT, WMMA_N_DOT, WMMA_K_DOT, half, wmma::col_major> b_frag;
          wmma::load_matrix_sync(b_frag, B_smem[buf] + b_col * DOT_BLOCK_K + k, DOT_BLOCK_K);
          wmma::mma_sync(c_frag[tm][tn], a_frag, b_frag, c_frag[tm][tn]);
        }
      }
    }
    buf = next_buf;
    __syncthreads();
  }
  
  #pragma unroll
  for (int tm = 0; tm < DOT_TILES_M_PER_WARP; ++tm) {
    int tile_row = block_row_start + warp_row_base + tm * WMMA_M_DOT;
    #pragma unroll
    for (int tn = 0; tn < DOT_TILES_N_PER_WARP; ++tn) {
      int tile_col = block_col_start + warp_col_base + tn * WMMA_N_DOT;
      if (tile_row < M_A && tile_col < M_B) {
        wmma::store_matrix_sync(output + tile_row * M_B + tile_col, c_frag[tm][tn], M_B, wmma::mem_row_major);
      }
    }
  }
}

// ----------------- Sparse output kernel (with filtering) -----------------
__global__ void __launch_bounds__(DOT_WARPS_PER_BLOCK * 32, 1)
dot_product_kernel(
    const half *__restrict__ data,
    int M, int vec_dim,
    const int32_t *__restrict__ labels,
    float match_threshold, float non_match_threshold,
    bool is_similarity, uint8_t include_flags,
    int32_t *__restrict__ pair_indices,
    uint8_t *__restrict__ categories,
    float *__restrict__ out_scores,
    unsigned int *match_count, unsigned int max_pairs) {
  
  int warp_id = threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  
  int block_row_start = blockIdx.y * DOT_BLOCK_M;
  int block_col_start = blockIdx.x * DOT_BLOCK_N;
  
  if (block_col_start > block_row_start + DOT_BLOCK_M - 1) return;
  if (block_row_start >= M || block_col_start >= M) return;
  
  extern __shared__ char smem_raw[];
  half *A_smem = reinterpret_cast<half*>(smem_raw);
  half *B_smem = A_smem + DOT_BLOCK_M * DOT_BLOCK_K;
  float *tile_smem_base = reinterpret_cast<float*>(B_smem + DOT_BLOCK_N * DOT_BLOCK_K);
  float *my_tile_smem = tile_smem_base + warp_id * SMEM_RESULT_SIZE;
  
  int warp_m = warp_id / DOT_WARPS_N;
  int warp_n = warp_id % DOT_WARPS_N;
  int warp_row_base = warp_m * DOT_WARP_TILE_M;
  int warp_col_base = warp_n * DOT_WARP_TILE_N;
  
  wmma::fragment<wmma::accumulator, WMMA_M_DOT, WMMA_N_DOT, WMMA_K_DOT, float>
      c_frag[DOT_TILES_M_PER_WARP][DOT_TILES_N_PER_WARP];
  
  #pragma unroll
  for (int tm = 0; tm < DOT_TILES_M_PER_WARP; ++tm)
    #pragma unroll
    for (int tn = 0; tn < DOT_TILES_N_PER_WARP; ++tn)
      wmma::fill_fragment(c_frag[tm][tn], 0.0f);
  
  int k_chunks = (vec_dim + DOT_BLOCK_K - 1) / DOT_BLOCK_K;
  
  for (int kc = 0; kc < k_chunks; ++kc) {
    int k_start = kc * DOT_BLOCK_K;
    int k_end = min(k_start + DOT_BLOCK_K, vec_dim);
    int k_len = k_end - k_start;
    
    for (int idx = threadIdx.x; idx < DOT_BLOCK_M * DOT_BLOCK_K; idx += blockDim.x) {
      int row = idx / DOT_BLOCK_K;
      int col = idx % DOT_BLOCK_K;
      int g_row = block_row_start + row;
      int g_col = k_start + col;
      half val = (g_row < M && col < k_len) ? data[g_row * vec_dim + g_col] : __float2half(0.0f);
      A_smem[idx] = val;
    }
    
    for (int idx = threadIdx.x; idx < DOT_BLOCK_N * DOT_BLOCK_K; idx += blockDim.x) {
      int n = idx / DOT_BLOCK_K;
      int k = idx % DOT_BLOCK_K;
      int g_row = block_col_start + n;
      int g_col = k_start + k;
      half val = (g_row < M && k < k_len) ? data[g_row * vec_dim + g_col] : __float2half(0.0f);
      B_smem[n * DOT_BLOCK_K + k] = val;
    }
    
    __syncthreads();
    
    #pragma unroll
    for (int k = 0; k < DOT_BLOCK_K; k += WMMA_K_DOT) {
      #pragma unroll
      for (int tm = 0; tm < DOT_TILES_M_PER_WARP; ++tm) {
        int a_row = warp_row_base + tm * WMMA_M_DOT;
        wmma::fragment<wmma::matrix_a, WMMA_M_DOT, WMMA_N_DOT, WMMA_K_DOT, half, wmma::row_major> a_frag;
        wmma::load_matrix_sync(a_frag, A_smem + a_row * DOT_BLOCK_K + k, DOT_BLOCK_K);
        
        #pragma unroll
        for (int tn = 0; tn < DOT_TILES_N_PER_WARP; ++tn) {
          int b_col = warp_col_base + tn * WMMA_N_DOT;
          wmma::fragment<wmma::matrix_b, WMMA_M_DOT, WMMA_N_DOT, WMMA_K_DOT, half, wmma::col_major> b_frag;
          wmma::load_matrix_sync(b_frag, B_smem + b_col * DOT_BLOCK_K + k, DOT_BLOCK_K);
          wmma::mma_sync(c_frag[tm][tn], a_frag, b_frag, c_frag[tm][tn]);
        }
      }
    }
    __syncthreads();
  }
  
  // Emit filtered results
  #pragma unroll
  for (int tm = 0; tm < DOT_TILES_M_PER_WARP; ++tm) {
    #pragma unroll
    for (int tn = 0; tn < DOT_TILES_N_PER_WARP; ++tn) {
      int tile_row = block_row_start + warp_row_base + tm * WMMA_M_DOT;
      int tile_col = block_col_start + warp_col_base + tn * WMMA_N_DOT;
      if (tile_col > tile_row + WMMA_M_DOT - 1) continue;
      
      emit_tile_results(c_frag[tm][tn], my_tile_smem, tile_row, tile_col, M, M, true,
                       labels, nullptr, match_threshold, non_match_threshold,
                       is_similarity, include_flags, pair_indices, categories, out_scores,
                       match_count, max_pairs, lane);
    }
  }
}

// Sparse A vs B kernel
__global__ void __launch_bounds__(DOT_WARPS_PER_BLOCK * 32, 1)
dot_product_ab_kernel(
    const half *__restrict__ data_a,
    const half *__restrict__ data_b,
    int M_A, int M_B, int vec_dim,
    const int32_t *__restrict__ labels_a,
    const int32_t *__restrict__ labels_b,
    float match_threshold, float non_match_threshold,
    bool is_similarity, uint8_t include_flags,
    int32_t *__restrict__ pair_indices,
    uint8_t *__restrict__ categories,
    float *__restrict__ out_scores,
    unsigned int *match_count, unsigned int max_pairs) {
  
  int warp_id = threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  
  int block_row_start = blockIdx.y * DOT_BLOCK_M;
  int block_col_start = blockIdx.x * DOT_BLOCK_N;
  
  if (block_row_start >= M_A || block_col_start >= M_B) return;
  
  extern __shared__ char smem_raw[];
  half *A_smem = reinterpret_cast<half*>(smem_raw);
  half *B_smem = A_smem + DOT_BLOCK_M * DOT_BLOCK_K;
  float *tile_smem_base = reinterpret_cast<float*>(B_smem + DOT_BLOCK_N * DOT_BLOCK_K);
  float *my_tile_smem = tile_smem_base + warp_id * SMEM_RESULT_SIZE;
  
  int warp_m = warp_id / DOT_WARPS_N;
  int warp_n = warp_id % DOT_WARPS_N;
  int warp_row_base = warp_m * DOT_WARP_TILE_M;
  int warp_col_base = warp_n * DOT_WARP_TILE_N;
  
  wmma::fragment<wmma::accumulator, WMMA_M_DOT, WMMA_N_DOT, WMMA_K_DOT, float>
      c_frag[DOT_TILES_M_PER_WARP][DOT_TILES_N_PER_WARP];
  
  #pragma unroll
  for (int tm = 0; tm < DOT_TILES_M_PER_WARP; ++tm)
    #pragma unroll
    for (int tn = 0; tn < DOT_TILES_N_PER_WARP; ++tn)
      wmma::fill_fragment(c_frag[tm][tn], 0.0f);
  
  int k_chunks = (vec_dim + DOT_BLOCK_K - 1) / DOT_BLOCK_K;
  
  for (int kc = 0; kc < k_chunks; ++kc) {
    int k_start = kc * DOT_BLOCK_K;
    int k_end = min(k_start + DOT_BLOCK_K, vec_dim);
    int k_len = k_end - k_start;
    
    for (int idx = threadIdx.x; idx < DOT_BLOCK_M * DOT_BLOCK_K; idx += blockDim.x) {
      int row = idx / DOT_BLOCK_K;
      int col = idx % DOT_BLOCK_K;
      int g_row = block_row_start + row;
      int g_col = k_start + col;
      half val = (g_row < M_A && col < k_len) ? data_a[g_row * vec_dim + g_col] : __float2half(0.0f);
      A_smem[idx] = val;
    }
    
    for (int idx = threadIdx.x; idx < DOT_BLOCK_N * DOT_BLOCK_K; idx += blockDim.x) {
      int n = idx / DOT_BLOCK_K;
      int k = idx % DOT_BLOCK_K;
      int g_row = block_col_start + n;
      int g_col = k_start + k;
      half val = (g_row < M_B && k < k_len) ? data_b[g_row * vec_dim + g_col] : __float2half(0.0f);
      B_smem[n * DOT_BLOCK_K + k] = val;
    }
    
    __syncthreads();
    
    #pragma unroll
    for (int k = 0; k < DOT_BLOCK_K; k += WMMA_K_DOT) {
      #pragma unroll
      for (int tm = 0; tm < DOT_TILES_M_PER_WARP; ++tm) {
        int a_row = warp_row_base + tm * WMMA_M_DOT;
        wmma::fragment<wmma::matrix_a, WMMA_M_DOT, WMMA_N_DOT, WMMA_K_DOT, half, wmma::row_major> a_frag;
        wmma::load_matrix_sync(a_frag, A_smem + a_row * DOT_BLOCK_K + k, DOT_BLOCK_K);
        
        #pragma unroll
        for (int tn = 0; tn < DOT_TILES_N_PER_WARP; ++tn) {
          int b_col = warp_col_base + tn * WMMA_N_DOT;
          wmma::fragment<wmma::matrix_b, WMMA_M_DOT, WMMA_N_DOT, WMMA_K_DOT, half, wmma::col_major> b_frag;
          wmma::load_matrix_sync(b_frag, B_smem + b_col * DOT_BLOCK_K + k, DOT_BLOCK_K);
          wmma::mma_sync(c_frag[tm][tn], a_frag, b_frag, c_frag[tm][tn]);
        }
      }
    }
    __syncthreads();
  }
  
  #pragma unroll
  for (int tm = 0; tm < DOT_TILES_M_PER_WARP; ++tm) {
    #pragma unroll
    for (int tn = 0; tn < DOT_TILES_N_PER_WARP; ++tn) {
      int tile_row = block_row_start + warp_row_base + tm * WMMA_M_DOT;
      int tile_col = block_col_start + warp_col_base + tn * WMMA_N_DOT;
      
      emit_tile_results(c_frag[tm][tn], my_tile_smem, tile_row, tile_col, M_A, M_B, false,
                       labels_a, labels_b, match_threshold, non_match_threshold,
                       is_similarity, include_flags, pair_indices, categories, out_scores,
                       match_count, max_pairs, lane);
    }
  }
}

// ----------------- C++ Launcher Functions -----------------

extern "C" void launch_dot_product_dense_cuda(
    const half *dData, float *dOutput, int M, int vec_dim, cudaStream_t stream) {
  dim3 grid((M + DOT_BLOCK_N - 1) / DOT_BLOCK_N, (M + DOT_BLOCK_M - 1) / DOT_BLOCK_M);
  dim3 block(DOT_WARPS_PER_BLOCK * 32);
  // Double buffer: 2 * (A + B) tiles
  size_t smem = 2 * (DOT_BLOCK_M * DOT_BLOCK_K + DOT_BLOCK_N * DOT_BLOCK_K) * sizeof(half);
  dot_product_dense_kernel<<<grid, block, smem, stream>>>(dData, dOutput, M, vec_dim);
}

extern "C" void launch_dot_product_ab_dense_cuda(
    const half *dData_A, const half *dData_B, float *dOutput,
    int M_A, int M_B, int vec_dim, cudaStream_t stream) {
  dim3 grid((M_B + DOT_BLOCK_N - 1) / DOT_BLOCK_N, (M_A + DOT_BLOCK_M - 1) / DOT_BLOCK_M);
  dim3 block(DOT_WARPS_PER_BLOCK * 32);
  size_t smem = 2 * (DOT_BLOCK_M * DOT_BLOCK_K + DOT_BLOCK_N * DOT_BLOCK_K) * sizeof(half);
  dot_product_ab_dense_kernel<<<grid, block, smem, stream>>>(dData_A, dData_B, dOutput, M_A, M_B, vec_dim);
}

extern "C" void launch_dot_product_cuda(
    const half *dData, int M, int vec_dim,
    const int32_t *dLabels,
    float match_threshold, float non_match_threshold,
    bool is_similarity, uint8_t include_flags,
    int32_t *dPairIndices, uint8_t *dCategories,
    float *dOutScores, unsigned int *dMatchCount,
    unsigned int max_pairs, cudaStream_t stream) {
  
  dim3 grid((M + DOT_BLOCK_N - 1) / DOT_BLOCK_N, (M + DOT_BLOCK_M - 1) / DOT_BLOCK_M);
  dim3 block(DOT_WARPS_PER_BLOCK * 32);
  size_t smem = (DOT_BLOCK_M * DOT_BLOCK_K + DOT_BLOCK_N * DOT_BLOCK_K) * sizeof(half)
              + DOT_WARPS_PER_BLOCK * SMEM_RESULT_SIZE * sizeof(float);
  
  dot_product_kernel<<<grid, block, smem, stream>>>(
      dData, M, vec_dim, dLabels, match_threshold, non_match_threshold,
      is_similarity, include_flags, dPairIndices, dCategories, dOutScores,
      dMatchCount, max_pairs);
}

extern "C" void launch_dot_product_ab_cuda(
    const half *dData_A, const half *dData_B,
    int M_A, int M_B, int vec_dim,
    const int32_t *dLabels_A, const int32_t *dLabels_B,
    float match_threshold, float non_match_threshold,
    bool is_similarity, uint8_t include_flags,
    int32_t *dPairIndices, uint8_t *dCategories,
    float *dOutScores, unsigned int *dMatchCount,
    unsigned int max_pairs, cudaStream_t stream) {
  
  dim3 grid((M_B + DOT_BLOCK_N - 1) / DOT_BLOCK_N, (M_A + DOT_BLOCK_M - 1) / DOT_BLOCK_M);
  dim3 block(DOT_WARPS_PER_BLOCK * 32);
  size_t smem = (DOT_BLOCK_M * DOT_BLOCK_K + DOT_BLOCK_N * DOT_BLOCK_K) * sizeof(half)
              + DOT_WARPS_PER_BLOCK * SMEM_RESULT_SIZE * sizeof(float);
  
  dot_product_ab_kernel<<<grid, block, smem, stream>>>(
      dData_A, dData_B, M_A, M_B, vec_dim,
      dLabels_A, dLabels_B, match_threshold, non_match_threshold,
      is_similarity, include_flags, dPairIndices, dCategories, dOutScores,
      dMatchCount, max_pairs);
}
