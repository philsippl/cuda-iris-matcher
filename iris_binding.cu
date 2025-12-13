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

extern "C" void launch_masked_hamming_ab_cuda(
    const uint32_t *dData_A, const uint32_t *dMask_A, uint32_t *dPremasked_A,
    const uint32_t *dData_B, const uint32_t *dMask_B, uint32_t *dPremasked_B,
    int M_A, int M_B, float *dD, bool write_output, bool collect_pairs,
    float threshold, uint2 *dPairs, unsigned int *dMatchCount,
    unsigned int max_pairs, cudaStream_t stream);

extern "C" void launch_pack_theta_major_cuda(uint8_t *data, int M,
                                             cudaStream_t stream);

extern "C" void launch_repack_to_theta_major_cuda(const uint32_t *input,
                                                  uint32_t *output, int M,
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

  // GPU buffers for kernel output
  torch::Tensor pairs_gpu;
  torch::Tensor match_count_gpu;
  // Pinned CPU buffers for async copy destination
  torch::Tensor pairs_cpu;
  torch::Tensor match_count_cpu;

  if (collect_pairs && max_pairs > 0) {
    // GPU buffers for kernel to write to
    pairs_gpu = torch::zeros({max_pairs, 2}, opts_int);
    match_count_gpu = torch::zeros({1}, opts_int);

    // Pinned CPU buffers for async copy
    auto cpu_opts = torch::TensorOptions()
                        .device(torch::kCPU)
                        .dtype(torch::kInt32)
                        .pinned_memory(true);
    pairs_cpu = torch::zeros({max_pairs, 2}, cpu_opts);
    match_count_cpu = torch::zeros({1}, cpu_opts);
  }

  auto stream = at::cuda::getDefaultCUDAStream();
  launch_masked_hamming_cuda(
      reinterpret_cast<uint32_t *>(data.data_ptr<int>()),
      reinterpret_cast<uint32_t *>(mask.data_ptr<int>()),
      reinterpret_cast<uint32_t *>(premasked.data_ptr<int>()), (int)M,
      write_output ? D.data_ptr<float>() : nullptr, write_output, collect_pairs,
      (float)threshold,
      collect_pairs && max_pairs > 0
          ? reinterpret_cast<uint2 *>(pairs_gpu.data_ptr<int>())
          : nullptr,
      collect_pairs && max_pairs > 0
          ? reinterpret_cast<unsigned int *>(match_count_gpu.data_ptr<int>())
          : nullptr,
      (unsigned int)max_pairs, stream);

  // Async copy results to pinned host memory
  if (collect_pairs && max_pairs > 0) {
    cudaMemcpyAsync(match_count_cpu.data_ptr(), match_count_gpu.data_ptr(),
                    sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(pairs_cpu.data_ptr(), pairs_gpu.data_ptr(),
                    max_pairs * 2 * sizeof(int), cudaMemcpyDeviceToHost,
                    stream);
  }

  // Return CPU tensors - caller must sync stream before reading
  return {D, pairs_cpu, match_count_cpu};
}

std::vector<torch::Tensor>
masked_hamming_ab_cuda(torch::Tensor data_a, torch::Tensor mask_a,
                       torch::Tensor data_b, torch::Tensor mask_b,
                       bool write_output, bool collect_pairs, double threshold,
                       int64_t max_pairs) {
  TORCH_CHECK(data_a.is_cuda(), "data_a must be CUDA");
  TORCH_CHECK(mask_a.is_cuda(), "mask_a must be CUDA");
  TORCH_CHECK(data_b.is_cuda(), "data_b must be CUDA");
  TORCH_CHECK(mask_b.is_cuda(), "mask_b must be CUDA");
  TORCH_CHECK(data_a.scalar_type() == at::kInt, "data_a dtype must be int32");
  TORCH_CHECK(mask_a.scalar_type() == at::kInt, "mask_a dtype must be int32");
  TORCH_CHECK(data_b.scalar_type() == at::kInt, "data_b dtype must be int32");
  TORCH_CHECK(mask_b.scalar_type() == at::kInt, "mask_b dtype must be int32");
  TORCH_CHECK(data_a.is_contiguous() && mask_a.is_contiguous(),
              "data_a/mask_a must be contiguous");
  TORCH_CHECK(data_b.is_contiguous() && mask_b.is_contiguous(),
              "data_b/mask_b must be contiguous");
  TORCH_CHECK(data_a.sizes() == mask_a.sizes(), "data_a/mask_a shape mismatch");
  TORCH_CHECK(data_b.sizes() == mask_b.sizes(), "data_b/mask_b shape mismatch");
  TORCH_CHECK(data_a.dim() == 2 && data_a.size(1) == K_WORDS,
              "expected data_a shape [M_A, ", K_WORDS, "]");
  TORCH_CHECK(data_b.dim() == 2 && data_b.size(1) == K_WORDS,
              "expected data_b shape [M_B, ", K_WORDS, "]");

  int64_t M_A = data_a.size(0);
  int64_t M_B = data_b.size(0);
  auto opts_int = data_a.options();
  auto opts_float = data_a.options().dtype(torch::kFloat32);

  // Buffers
  auto premasked_a = torch::zeros_like(data_a);
  auto premasked_b = torch::zeros_like(data_b);
  torch::Tensor D;
  if (write_output) {
    D = torch::zeros({M_A, M_B}, opts_float);
  }

  // GPU buffers for kernel output
  torch::Tensor pairs_gpu;
  torch::Tensor match_count_gpu;
  // Pinned CPU buffers for async copy destination
  torch::Tensor pairs_cpu;
  torch::Tensor match_count_cpu;

  if (collect_pairs && max_pairs > 0) {
    // GPU buffers for kernel to write to
    pairs_gpu = torch::zeros({max_pairs, 2}, opts_int);
    match_count_gpu = torch::zeros({1}, opts_int);

    // Pinned CPU buffers for async copy
    auto cpu_opts = torch::TensorOptions()
                        .device(torch::kCPU)
                        .dtype(torch::kInt32)
                        .pinned_memory(true);
    pairs_cpu = torch::zeros({max_pairs, 2}, cpu_opts);
    match_count_cpu = torch::zeros({1}, cpu_opts);
  }

  auto stream = at::cuda::getDefaultCUDAStream();
  launch_masked_hamming_ab_cuda(
      reinterpret_cast<uint32_t *>(data_a.data_ptr<int>()),
      reinterpret_cast<uint32_t *>(mask_a.data_ptr<int>()),
      reinterpret_cast<uint32_t *>(premasked_a.data_ptr<int>()),
      reinterpret_cast<uint32_t *>(data_b.data_ptr<int>()),
      reinterpret_cast<uint32_t *>(mask_b.data_ptr<int>()),
      reinterpret_cast<uint32_t *>(premasked_b.data_ptr<int>()), (int)M_A,
      (int)M_B, write_output ? D.data_ptr<float>() : nullptr, write_output,
      collect_pairs, (float)threshold,
      collect_pairs && max_pairs > 0
          ? reinterpret_cast<uint2 *>(pairs_gpu.data_ptr<int>())
          : nullptr,
      collect_pairs && max_pairs > 0
          ? reinterpret_cast<unsigned int *>(match_count_gpu.data_ptr<int>())
          : nullptr,
      (unsigned int)max_pairs, stream);

  // Async copy results to pinned host memory
  if (collect_pairs && max_pairs > 0) {
    cudaMemcpyAsync(match_count_cpu.data_ptr(), match_count_gpu.data_ptr(),
                    sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(pairs_cpu.data_ptr(), pairs_gpu.data_ptr(),
                    max_pairs * 2 * sizeof(int), cudaMemcpyDeviceToHost,
                    stream);
  }

  // Return CPU tensors - caller must sync stream before reading
  return {D, pairs_cpu, match_count_cpu};
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

torch::Tensor repack_to_theta_major_cuda(torch::Tensor input) {
  // Input: (M, 400) int32 tensor packed in r-major order
  //        bit[r,theta,d0,d1] at linear_bit = r*800 + theta*4 + d0*2 + d1
  // Output: (M, 400) int32 tensor packed in theta-major order
  //        bit[r,theta,d0,d1] at linear_bit = theta*64 + r*4 + d0*2 + d1
  TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
  TORCH_CHECK(input.scalar_type() == at::kInt, "input dtype must be int32");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.dim() == 2 && input.size(1) == K_WORDS,
              "input must have shape (M, ", K_WORDS, ")");

  int64_t M = input.size(0);
  auto output = torch::empty_like(input);

  auto stream = at::cuda::getDefaultCUDAStream();
  launch_repack_to_theta_major_cuda(
      reinterpret_cast<const uint32_t *>(input.data_ptr<int>()),
      reinterpret_cast<uint32_t *>(output.data_ptr<int>()), (int)M, stream);

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("masked_hamming_cuda", &masked_hamming_cuda,
        "Masked hamming distance (CUDA)");
  m.def("masked_hamming_ab_cuda", &masked_hamming_ab_cuda,
        "Masked hamming distance between two sets A and B (CUDA)");
  m.def("pack_theta_major_cuda", &pack_theta_major_cuda,
        "Pack iris bits to theta-major int32 words (CUDA)");
  m.def("repack_to_theta_major_cuda", &repack_to_theta_major_cuda,
        "Repack int32 words from r-major to theta-major order (CUDA)");
}
