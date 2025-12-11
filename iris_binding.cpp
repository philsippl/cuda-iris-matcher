#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// Constants (must match iris.cu)
constexpr int K_BITS = 12800;
constexpr int K_WORDS = K_BITS / 32; // 400
constexpr int K_CHUNK_BITS = 256;
constexpr int K_CHUNK_WORDS = K_CHUNK_BITS / 32;     // 8
constexpr int K_CHUNK_WORDS_EXT = K_CHUNK_WORDS + 2; // 10
constexpr int MAX_SHIFT = 15;
constexpr int NUM_SHIFTS = 2 * MAX_SHIFT + 1; // 31

// Kernel declarations from iris.cu
__global__ void preprocess_kernel(const uint32_t *__restrict__ data,
                                  const uint32_t *__restrict__ mask,
                                  uint32_t *__restrict__ premasked, int M);

__global__ void __launch_bounds__(/*WARPS_PER_BLOCK * 32*/ 1024, 1)
    min_hamming_kernel(const uint32_t *__restrict__ premasked,
                       const uint32_t *__restrict__ mask, int M,
                       float *__restrict__ D, bool write_output,
                       bool collect_pairs, float threshold,
                       uint2 *__restrict__ pairs, unsigned int *match_count,
                       unsigned int max_pairs);

std::vector<torch::Tensor>
masked_hamming_cuda(torch::Tensor data, torch::Tensor mask, bool write_output,
                    bool collect_pairs, double threshold, int64_t max_pairs) {
  TORCH_CHECK(data.is_cuda(), "data must be CUDA");
  TORCH_CHECK(mask.is_cuda(), "mask must be CUDA");
  TORCH_CHECK(data.scalar_type() == at::kInt, "data dtype must be int32");
  TORCH_CHECK(mask.scalar_type() == at::kInt, "mask dtype must be int32");
  TORCH_CHECK(data.is_contiguous() && mask.is_contiguous(),
              "data/mask must be contiguous");
  TORCH_CHECK(data.sizes() == mask.sizes(), "data/mask shape mismatch");
  TORCH_CHECK(data.dim() == 2 && data.size(1) == K_WORDS,
              "expected data shape [M, ", K_WORDS, "]");

  int64_t M = data.size(0);
  auto opts_int = data.options();
  auto opts_float = data.options().dtype(torch::kFloat32);

  // Buffers
  auto premasked = torch::zeros_like(data);
  torch::Tensor D;
  if (write_output) {
    D = torch::zeros({M, M}, opts_float);
  }
  torch::Tensor pairs;
  torch::Tensor match_count;
  if (collect_pairs && max_pairs > 0) {
    pairs = torch::zeros({max_pairs, 2}, opts_int);
    match_count = torch::zeros({1}, opts_int);
  }

  // Launch preprocess + kernel on the current stream
  const dim3 block_pre(256);
  const dim3 grid_pre((M * K_WORDS + block_pre.x - 1) / block_pre.x);
  auto stream = at::cuda::getDefaultCUDAStream();

  preprocess_kernel<<<grid_pre, block_pre, 0, stream>>>(
      reinterpret_cast<uint32_t *>(data.data_ptr<int>()),
      reinterpret_cast<uint32_t *>(mask.data_ptr<int>()),
      reinterpret_cast<uint32_t *>(premasked.data_ptr<int>()), (int)M);

  // Grid/block/smem match iris.cu
  constexpr int BLOCK_M = 192;
  constexpr int BLOCK_N = 64;
  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 8;
  constexpr int TILES_M_PER_WARP = 2;
  constexpr int TILES_N_PER_WARP = 4;
  constexpr int TILES_M = BLOCK_M / WMMA_M;
  constexpr int TILES_N = BLOCK_N / WMMA_N;
  constexpr int WARPS_PER_ROW = TILES_M / TILES_M_PER_WARP;
  constexpr int WARPS_PER_COL = TILES_N / TILES_N_PER_WARP;
  constexpr int WARPS_PER_BLOCK = WARPS_PER_ROW * WARPS_PER_COL;

  dim3 grid((M + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
  dim3 block(WARPS_PER_BLOCK * 32);
  size_t smem =
      2 * (2 * BLOCK_M * K_CHUNK_WORDS + 2 * BLOCK_N * K_CHUNK_WORDS_EXT) *
      sizeof(uint32_t);

  min_hamming_kernel<<<grid, block, smem, stream>>>(
      reinterpret_cast<uint32_t *>(premasked.data_ptr<int>()),
      reinterpret_cast<uint32_t *>(mask.data_ptr<int>()), (int)M,
      write_output ? D.data_ptr<float>() : nullptr, write_output, collect_pairs,
      (float)threshold,
      collect_pairs && max_pairs > 0
          ? reinterpret_cast<uint2 *>(pairs.data_ptr<int>())
          : nullptr,
      collect_pairs && max_pairs > 0
          ? reinterpret_cast<unsigned int *>(match_count.data_ptr<int>())
          : nullptr,
      (unsigned int)max_pairs);

  return {D, pairs, match_count};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("masked_hamming_cuda", &masked_hamming_cuda,
        "Masked hamming distance (CUDA)");
}
