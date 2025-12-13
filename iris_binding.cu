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

extern "C" void launch_pack_theta_major_cuda(uint8_t *data, int M,
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

torch::Tensor pack_theta_major_cuda(torch::Tensor bits) {
  // Input: (M, 16, 200, 2, 2) uint8 tensor with values in {0, 1}
  // Output: (M, 400) int32 tensor (packed bits, theta-major order)
  //
  // IN-PLACE packing with grid-level sync: the kernel packs data into the
  // input buffer. Uses cooperative groups to ensure all reads complete before
  // any writes. The returned tensor shares storage with the input.
  TORCH_CHECK(bits.is_cuda(), "bits must be CUDA tensor");
  TORCH_CHECK(bits.scalar_type() == at::kByte, "bits dtype must be uint8");
  TORCH_CHECK(bits.is_contiguous(), "bits must be contiguous");
  TORCH_CHECK(bits.dim() == 5, "bits must have shape (M, 16, 200, 2, 2)");
  TORCH_CHECK(bits.size(1) == 16 && bits.size(2) == 200 && bits.size(3) == 2 &&
                  bits.size(4) == 2,
              "bits must have shape (M, 16, 200, 2, 2)");

  int64_t M = bits.size(0);

  auto stream = at::cuda::getDefaultCUDAStream();
  launch_pack_theta_major_cuda(bits.data_ptr<uint8_t>(), (int)M, stream);

  // Return view of packed data (first M * 400 int32 words)
  auto flat = bits.flatten();
  auto sliced = flat.slice(0, 0, M * K_WORDS * 4);
  auto packed = sliced.view(torch::kInt32);
  return packed.view({M, K_WORDS});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("masked_hamming_cuda", &masked_hamming_cuda,
        "Masked hamming distance (CUDA)");
  m.def("pack_theta_major_cuda", &pack_theta_major_cuda,
        "Pack iris bits to theta-major int32 words (CUDA)");
}
