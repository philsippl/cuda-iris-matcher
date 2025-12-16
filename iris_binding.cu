#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "iris_params.h"

extern "C" void launch_masked_hamming_cuda(
    const uint32_t *dData, const uint32_t *dMask, uint32_t *dPremasked, int M,
    const int32_t *dLabels, float match_threshold, float non_match_threshold,
    bool is_similarity, uint8_t include_flags,
    int32_t *dPairIndices, uint8_t *dCategories, float *dOutDistances,
    unsigned int *dMatchCount, unsigned int max_pairs,
    cudaStream_t stream);

extern "C" void launch_masked_hamming_ab_cuda(
    const uint32_t *dData_A, const uint32_t *dMask_A, uint32_t *dPremasked_A,
    const uint32_t *dData_B, const uint32_t *dMask_B, uint32_t *dPremasked_B,
    int M_A, int M_B,
    const int32_t *dLabels_A, const int32_t *dLabels_B,
    float match_threshold, float non_match_threshold,
    bool is_similarity, uint8_t include_flags,
    int32_t *dPairIndices, uint8_t *dCategories, float *dOutDistances,
    unsigned int *dMatchCount, unsigned int max_pairs,
    cudaStream_t stream);

extern "C" void launch_pack_theta_major_cuda(uint8_t *data, int M,
                                             cudaStream_t stream);

extern "C" void launch_repack_to_theta_major_cuda(const uint32_t *input,
                                                  uint32_t *output, int M,
                                                  cudaStream_t stream);

extern "C" void launch_pack_half_mask_cuda(const uint8_t *input,
                                           uint32_t *output, int M,
                                           cudaStream_t stream);

std::vector<torch::Tensor>
masked_hamming_cuda(torch::Tensor data, torch::Tensor mask,
                    std::optional<torch::Tensor> labels,
                    double match_threshold, double non_match_threshold,
                    bool is_similarity, int64_t include_flags,
                    int64_t max_pairs) {
  TORCH_CHECK(data.is_cuda(), "data must be CUDA");
  TORCH_CHECK(mask.is_cuda(), "mask must be CUDA");
  TORCH_CHECK(data.scalar_type() == at::kInt, "data dtype must be int32");
  TORCH_CHECK(mask.scalar_type() == at::kInt, "mask dtype must be int32");
  TORCH_CHECK(data.is_contiguous() && mask.is_contiguous(),
              "data/mask must be contiguous");
  TORCH_CHECK(data.dim() == 2 && data.size(1) == K_WORDS,
              "expected data shape [M, ", K_WORDS, "]");
  TORCH_CHECK(mask.dim() == 2 && mask.size(1) == K_WORDS_HALF,
              "expected mask shape [M, ", K_WORDS_HALF, "] (half-mask)");
  TORCH_CHECK(data.size(0) == mask.size(0), "data/mask row count mismatch");

  int64_t M = data.size(0);
  auto opts_int = data.options();
  auto opts_float = data.options().dtype(torch::kFloat32);
  auto opts_byte = data.options().dtype(torch::kUInt8);

  // Validate labels if provided
  if (labels.has_value()) {
    TORCH_CHECK(labels->is_cuda(), "labels must be CUDA");
    TORCH_CHECK(labels->scalar_type() == at::kInt, "labels dtype must be int32");
    TORCH_CHECK(labels->is_contiguous(), "labels must be contiguous");
    TORCH_CHECK(labels->dim() == 1 && labels->size(0) == M,
                "labels must have shape [M]");
  }

  // Buffers
  auto premasked = torch::zeros_like(data);

  // GPU buffers for kernel output
  auto pair_indices_gpu = torch::zeros({max_pairs, 2}, opts_int);
  auto categories_gpu = torch::zeros({max_pairs}, opts_byte);
  auto distances_gpu = torch::zeros({max_pairs}, opts_float);
  auto match_count_gpu = torch::zeros({1}, opts_int);

  // Pinned CPU buffers for async copy
  auto cpu_opts_int = torch::TensorOptions()
                          .device(torch::kCPU)
                          .dtype(torch::kInt32)
                          .pinned_memory(true);
  auto cpu_opts_byte = torch::TensorOptions()
                           .device(torch::kCPU)
                           .dtype(torch::kUInt8)
                           .pinned_memory(true);
  auto cpu_opts_float = torch::TensorOptions()
                            .device(torch::kCPU)
                            .dtype(torch::kFloat32)
                            .pinned_memory(true);

  auto pair_indices_cpu = torch::zeros({max_pairs, 2}, cpu_opts_int);
  auto categories_cpu = torch::zeros({max_pairs}, cpu_opts_byte);
  auto distances_cpu = torch::zeros({max_pairs}, cpu_opts_float);
  auto match_count_cpu = torch::zeros({1}, cpu_opts_int);

  auto stream = at::cuda::getDefaultCUDAStream();
  launch_masked_hamming_cuda(
      reinterpret_cast<uint32_t *>(data.data_ptr<int>()),
      reinterpret_cast<uint32_t *>(mask.data_ptr<int>()),
      reinterpret_cast<uint32_t *>(premasked.data_ptr<int>()), (int)M,
      labels.has_value() ? labels->data_ptr<int32_t>() : nullptr,
      (float)match_threshold, (float)non_match_threshold,
      is_similarity, (uint8_t)include_flags,
      pair_indices_gpu.data_ptr<int32_t>(),
      categories_gpu.data_ptr<uint8_t>(),
      distances_gpu.data_ptr<float>(),
      reinterpret_cast<unsigned int *>(match_count_gpu.data_ptr<int>()),
      (unsigned int)max_pairs, stream);

  // Copy count first and sync to know how many pairs to copy
  cudaMemcpyAsync(match_count_cpu.data_ptr(), match_count_gpu.data_ptr(),
                  sizeof(int), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  int64_t actual_count = match_count_cpu.data_ptr<int>()[0];
  // Clamp to max_pairs in case kernel wrote more
  if (actual_count > max_pairs) {
    actual_count = max_pairs;
  }

  // Only copy the valid entries
  if (actual_count > 0) {
    cudaMemcpyAsync(pair_indices_cpu.data_ptr(), pair_indices_gpu.data_ptr(),
                    actual_count * 2 * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(categories_cpu.data_ptr(), categories_gpu.data_ptr(),
                    actual_count * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(distances_cpu.data_ptr(), distances_gpu.data_ptr(),
                    actual_count * sizeof(float), cudaMemcpyDeviceToHost, stream);
  }

  // Return sliced tensors with only valid entries
  auto pair_indices_out = pair_indices_cpu.slice(0, 0, actual_count);
  auto categories_out = categories_cpu.slice(0, 0, actual_count);
  auto distances_out = distances_cpu.slice(0, 0, actual_count);

  return {pair_indices_out, categories_out, distances_out, match_count_cpu};
}

std::vector<torch::Tensor>
masked_hamming_ab_cuda(torch::Tensor data_a, torch::Tensor mask_a,
                       torch::Tensor data_b, torch::Tensor mask_b,
                       std::optional<torch::Tensor> labels_a,
                       std::optional<torch::Tensor> labels_b,
                       double match_threshold, double non_match_threshold,
                       bool is_similarity, int64_t include_flags,
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
  TORCH_CHECK(data_a.dim() == 2 && data_a.size(1) == K_WORDS,
              "expected data_a shape [M_A, ", K_WORDS, "]");
  TORCH_CHECK(data_b.dim() == 2 && data_b.size(1) == K_WORDS,
              "expected data_b shape [M_B, ", K_WORDS, "]");
  TORCH_CHECK(mask_a.dim() == 2 && mask_a.size(1) == K_WORDS_HALF,
              "expected mask_a shape [M_A, ", K_WORDS_HALF, "] (half-mask)");
  TORCH_CHECK(mask_b.dim() == 2 && mask_b.size(1) == K_WORDS_HALF,
              "expected mask_b shape [M_B, ", K_WORDS_HALF, "] (half-mask)");
  TORCH_CHECK(data_a.size(0) == mask_a.size(0), "data_a/mask_a row count mismatch");
  TORCH_CHECK(data_b.size(0) == mask_b.size(0), "data_b/mask_b row count mismatch");

  int64_t M_A = data_a.size(0);
  int64_t M_B = data_b.size(0);
  auto opts_int = data_a.options();
  auto opts_float = data_a.options().dtype(torch::kFloat32);
  auto opts_byte = data_a.options().dtype(torch::kUInt8);

  // Validate labels if provided (both must be provided or neither)
  bool has_labels = labels_a.has_value() && labels_b.has_value();
  if (labels_a.has_value() || labels_b.has_value()) {
    TORCH_CHECK(has_labels, "Both labels_a and labels_b must be provided, or neither");
  }
  if (has_labels) {
    TORCH_CHECK(labels_a->is_cuda(), "labels_a must be CUDA");
    TORCH_CHECK(labels_b->is_cuda(), "labels_b must be CUDA");
    TORCH_CHECK(labels_a->scalar_type() == at::kInt, "labels_a dtype must be int32");
    TORCH_CHECK(labels_b->scalar_type() == at::kInt, "labels_b dtype must be int32");
    TORCH_CHECK(labels_a->is_contiguous(), "labels_a must be contiguous");
    TORCH_CHECK(labels_b->is_contiguous(), "labels_b must be contiguous");
    TORCH_CHECK(labels_a->dim() == 1 && labels_a->size(0) == M_A,
                "labels_a must have shape [M_A]");
    TORCH_CHECK(labels_b->dim() == 1 && labels_b->size(0) == M_B,
                "labels_b must have shape [M_B]");
  }

  // Buffers
  auto premasked_a = torch::zeros_like(data_a);
  auto premasked_b = torch::zeros_like(data_b);

  // GPU buffers for kernel output
  auto pair_indices_gpu = torch::zeros({max_pairs, 2}, opts_int);
  auto categories_gpu = torch::zeros({max_pairs}, opts_byte);
  auto distances_gpu = torch::zeros({max_pairs}, opts_float);
  auto match_count_gpu = torch::zeros({1}, opts_int);

  // Pinned CPU buffers for async copy
  auto cpu_opts_int = torch::TensorOptions()
                          .device(torch::kCPU)
                          .dtype(torch::kInt32)
                          .pinned_memory(true);
  auto cpu_opts_byte = torch::TensorOptions()
                           .device(torch::kCPU)
                           .dtype(torch::kUInt8)
                           .pinned_memory(true);
  auto cpu_opts_float = torch::TensorOptions()
                            .device(torch::kCPU)
                            .dtype(torch::kFloat32)
                            .pinned_memory(true);

  auto pair_indices_cpu = torch::zeros({max_pairs, 2}, cpu_opts_int);
  auto categories_cpu = torch::zeros({max_pairs}, cpu_opts_byte);
  auto distances_cpu = torch::zeros({max_pairs}, cpu_opts_float);
  auto match_count_cpu = torch::zeros({1}, cpu_opts_int);

  auto stream = at::cuda::getDefaultCUDAStream();
  launch_masked_hamming_ab_cuda(
      reinterpret_cast<uint32_t *>(data_a.data_ptr<int>()),
      reinterpret_cast<uint32_t *>(mask_a.data_ptr<int>()),
      reinterpret_cast<uint32_t *>(premasked_a.data_ptr<int>()),
      reinterpret_cast<uint32_t *>(data_b.data_ptr<int>()),
      reinterpret_cast<uint32_t *>(mask_b.data_ptr<int>()),
      reinterpret_cast<uint32_t *>(premasked_b.data_ptr<int>()),
      (int)M_A, (int)M_B,
      has_labels ? labels_a->data_ptr<int32_t>() : nullptr,
      has_labels ? labels_b->data_ptr<int32_t>() : nullptr,
      (float)match_threshold, (float)non_match_threshold,
      is_similarity, (uint8_t)include_flags,
      pair_indices_gpu.data_ptr<int32_t>(),
      categories_gpu.data_ptr<uint8_t>(),
      distances_gpu.data_ptr<float>(),
      reinterpret_cast<unsigned int *>(match_count_gpu.data_ptr<int>()),
      (unsigned int)max_pairs, stream);

  // Copy count first and sync to know how many pairs to copy
  cudaMemcpyAsync(match_count_cpu.data_ptr(), match_count_gpu.data_ptr(),
                  sizeof(int), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  int64_t actual_count = match_count_cpu.data_ptr<int>()[0];
  // Clamp to max_pairs in case kernel wrote more
  if (actual_count > max_pairs) {
    actual_count = max_pairs;
  }

  // Only copy the valid entries
  if (actual_count > 0) {
    cudaMemcpyAsync(pair_indices_cpu.data_ptr(), pair_indices_gpu.data_ptr(),
                    actual_count * 2 * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(categories_cpu.data_ptr(), categories_gpu.data_ptr(),
                    actual_count * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(distances_cpu.data_ptr(), distances_gpu.data_ptr(),
                    actual_count * sizeof(float), cudaMemcpyDeviceToHost, stream);
  }

  // Return sliced tensors with only valid entries
  auto pair_indices_out = pair_indices_cpu.slice(0, 0, actual_count);
  auto categories_out = categories_cpu.slice(0, 0, actual_count);
  auto distances_out = distances_cpu.slice(0, 0, actual_count);

  return {pair_indices_out, categories_out, distances_out, match_count_cpu};
}

torch::Tensor pack_theta_major_cuda(torch::Tensor bits) {
  // Input: (M, 16, 200, 2, 2) uint8 tensor with values in {0, 1}
  // Output: (M, 400) int32 tensor (packed bits, d1-major order)
  //         linear_bit = d1*6400 + theta*32 + r*2 + d0
  //         First 200 words: d1=0 bits, Last 200 words: d1=1 bits
  //
  // This layout enables half-mask optimization since mask has duplicate d1 bits.
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
  // Output: (M, 400) int32 tensor packed in d1-major order
  //        bit[r,theta,d0,d1] at linear_bit = d1*6400 + theta*32 + r*2 + d0
  //        First 200 words: d1=0 bits, Last 200 words: d1=1 bits
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

torch::Tensor pack_half_mask_cuda(torch::Tensor bits) {
  // Input: (M, 16, 200, 2, 2) uint8 tensor with values in {0, 1}
  // Output: (M, 200) int32 tensor (packed half-mask, d1=0 bits only)
  //         linear_bit = theta*32 + r*2 + d0
  //
  // Half-mask stores only d1=0 bits since d1=1 is identical in real iris masks.
  TORCH_CHECK(bits.is_cuda(), "bits must be CUDA tensor");
  TORCH_CHECK(bits.scalar_type() == at::kByte, "bits dtype must be uint8");
  TORCH_CHECK(bits.is_contiguous(), "bits must be contiguous");
  TORCH_CHECK(bits.dim() == 5, "bits must have shape (M, 16, 200, 2, 2)");
  TORCH_CHECK(bits.size(1) == 16 && bits.size(2) == 200 && bits.size(3) == 2 &&
                  bits.size(4) == 2,
              "bits must have shape (M, 16, 200, 2, 2)");

  int64_t M = bits.size(0);
  auto opts = torch::TensorOptions()
                  .device(bits.device())
                  .dtype(torch::kInt32);
  auto output = torch::empty({M, K_WORDS_HALF}, opts);

  auto stream = at::cuda::getDefaultCUDAStream();
  launch_pack_half_mask_cuda(bits.data_ptr<uint8_t>(),
                             reinterpret_cast<uint32_t *>(output.data_ptr<int>()),
                             (int)M, stream);

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("masked_hamming_cuda", &masked_hamming_cuda,
        "Masked hamming distance with classification (CUDA)",
        py::arg("data"), py::arg("mask"),
        py::arg("labels") = py::none(),
        py::arg("match_threshold") = 0.35,
        py::arg("non_match_threshold") = 0.35,
        py::arg("is_similarity") = false,
        py::arg("include_flags") = INCLUDE_ALL,
        py::arg("max_pairs") = 1000000);
  m.def("masked_hamming_ab_cuda", &masked_hamming_ab_cuda,
        "Masked hamming distance between two sets A and B with classification (CUDA)",
        py::arg("data_a"), py::arg("mask_a"),
        py::arg("data_b"), py::arg("mask_b"),
        py::arg("labels_a") = py::none(),
        py::arg("labels_b") = py::none(),
        py::arg("match_threshold") = 0.35,
        py::arg("non_match_threshold") = 0.35,
        py::arg("is_similarity") = false,
        py::arg("include_flags") = INCLUDE_ALL,
        py::arg("max_pairs") = 1000000);
  m.def("pack_theta_major_cuda", &pack_theta_major_cuda,
        "Pack iris bits to d1-major int32 words (CUDA)");
  m.def("pack_half_mask_cuda", &pack_half_mask_cuda,
        "Pack mask bits to half-size int32 words - d1=0 only (CUDA)");
  m.def("repack_to_theta_major_cuda", &repack_to_theta_major_cuda,
        "Repack int32 words from r-major to d1-major order (CUDA)");

  // Export classification constants
  m.attr("CATEGORY_TRUE_MATCH") = py::int_(CATEGORY_TRUE_MATCH);
  m.attr("CATEGORY_FALSE_MATCH") = py::int_(CATEGORY_FALSE_MATCH);
  m.attr("CATEGORY_FALSE_NON_MATCH") = py::int_(CATEGORY_FALSE_NON_MATCH);
  m.attr("CATEGORY_TRUE_NON_MATCH") = py::int_(CATEGORY_TRUE_NON_MATCH);
  m.attr("INCLUDE_TM") = py::int_(INCLUDE_TM);
  m.attr("INCLUDE_FM") = py::int_(INCLUDE_FM);
  m.attr("INCLUDE_FNM") = py::int_(INCLUDE_FNM);
  m.attr("INCLUDE_TNM") = py::int_(INCLUDE_TNM);
  m.attr("INCLUDE_ALL") = py::int_(INCLUDE_ALL);
}
