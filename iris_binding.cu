#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "iris_params.h"

extern "C" void launch_masked_hamming_cuda(
    const uint32_t *dData, const uint32_t *dMask, uint32_t *dPremasked, int M,
    float *dD, bool write_output, bool collect_pairs, float threshold,
    uint2 *dPairs, unsigned int *dMatchCount, unsigned int max_pairs,
    cudaStream_t stream);

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

  auto stream = at::cuda::getDefaultCUDAStream();
  launch_masked_hamming_cuda(
      reinterpret_cast<uint32_t *>(data.data_ptr<int>()),
      reinterpret_cast<uint32_t *>(mask.data_ptr<int>()),
      reinterpret_cast<uint32_t *>(premasked.data_ptr<int>()), (int)M,
      write_output ? D.data_ptr<float>() : nullptr, write_output, collect_pairs,
      (float)threshold,
      collect_pairs && max_pairs > 0
          ? reinterpret_cast<uint2 *>(pairs.data_ptr<int>())
          : nullptr,
      collect_pairs && max_pairs > 0
          ? reinterpret_cast<unsigned int *>(match_count.data_ptr<int>())
          : nullptr,
      (unsigned int)max_pairs, stream);

  return {D, pairs, match_count};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("masked_hamming_cuda", &masked_hamming_cuda,
        "Masked hamming distance (CUDA)");
}
