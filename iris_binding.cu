#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "iris_params.h"

// Default vector dimension for dot product
constexpr int DEFAULT_DOT_VEC_DIM = 512;

extern "C" void launch_masked_hamming_cuda(
    const uint32_t *dData, const uint32_t *dMask, uint32_t *dPremasked, int M,
    int r_dim, int theta_dim, const int32_t *dLabels, float match_threshold,
    float non_match_threshold, bool is_similarity, uint8_t include_flags,
    SamplingConfig sampling,
    int32_t *dPairIndices, uint8_t *dCategories, float *dOutDistances,
    unsigned int *dMatchCount, unsigned int max_pairs, cudaStream_t stream);

extern "C" void launch_masked_hamming_ab_cuda(
    const uint32_t *dData_A, const uint32_t *dMask_A, uint32_t *dPremasked_A,
    const uint32_t *dData_B, const uint32_t *dMask_B, uint32_t *dPremasked_B,
    int M_A, int M_B, int r_dim, int theta_dim, const int32_t *dLabels_A,
    const int32_t *dLabels_B, float match_threshold, float non_match_threshold,
    bool is_similarity, uint8_t include_flags, SamplingConfig sampling,
    int32_t *dPairIndices, uint8_t *dCategories, float *dOutDistances,
    unsigned int *dMatchCount, unsigned int max_pairs, cudaStream_t stream);

extern "C" void launch_pack_theta_major_cuda(uint8_t *data, int M, int r_dim,
                                             int theta_dim, int d0_dim,
                                             int d1_dim, cudaStream_t stream);

extern "C" void launch_repack_to_theta_major_cuda(const uint32_t *input,
                                                  uint32_t *output, int M,
                                                  int r_dim, int theta_dim,
                                                  int d0_dim, int d1_dim,
                                                  cudaStream_t stream);

// Dot product kernel declarations
extern "C" void launch_dot_product_cuda(
    const half *dData, int M, int vec_dim,
    const int32_t *dLabels,
    float match_threshold, float non_match_threshold,
    bool is_similarity, uint8_t include_flags,
    SamplingConfig sampling,
    int32_t *dPairIndices, uint8_t *dCategories,
    float *dOutScores, unsigned int *dMatchCount,
    unsigned int max_pairs, cudaStream_t stream);

extern "C" void launch_dot_product_ab_cuda(
    const half *dData_A, const half *dData_B,
    int M_A, int M_B, int vec_dim,
    const int32_t *dLabels_A, const int32_t *dLabels_B,
    float match_threshold, float non_match_threshold,
    bool is_similarity, uint8_t include_flags,
    SamplingConfig sampling,
    int32_t *dPairIndices, uint8_t *dCategories,
    float *dOutScores, unsigned int *dMatchCount,
    unsigned int max_pairs, cudaStream_t stream);

// Dense output variants (high performance, no filtering)
extern "C" void launch_dot_product_dense_cuda(
    const half *dData, float *dOutput, int M, int vec_dim, cudaStream_t stream);

extern "C" void launch_dot_product_ab_dense_cuda(
    const half *dData_A, const half *dData_B, float *dOutput,
    int M_A, int M_B, int vec_dim, cudaStream_t stream);

// Helper to construct SamplingConfig from Python parameters
SamplingConfig make_sampling_config(
    int64_t num_bins,
    std::optional<torch::Tensor> thresholds,
    std::optional<torch::Tensor> probabilities,
    int64_t seed) {
  SamplingConfig cfg;
  cfg.num_bins = (int)num_bins;
  cfg.seed = (uint64_t)seed;
  
  if (num_bins > 0 && thresholds.has_value() && probabilities.has_value()) {
    TORCH_CHECK(thresholds->dim() == 1 && thresholds->size(0) >= num_bins,
                "thresholds must have at least num_bins elements");
    TORCH_CHECK(probabilities->dim() == 1 && probabilities->size(0) >= num_bins,
                "probabilities must have at least num_bins elements");
    
    auto thresh_cpu = thresholds->cpu().to(torch::kFloat32);
    auto prob_cpu = probabilities->cpu().to(torch::kFloat32);
    
    for (int i = 0; i < num_bins && i < MAX_SAMPLE_BINS; i++) {
      cfg.thresholds[i] = thresh_cpu[i].item<float>();
      cfg.probabilities[i] = prob_cpu[i].item<float>();
    }
  } else {
    cfg.num_bins = 0;  // Disable sampling if parameters not provided
  }
  
  return cfg;
}

// Async version: returns GPU tensors without synchronization
// Caller must synchronize before reading results
std::vector<torch::Tensor>
masked_hamming_cuda_async(torch::Tensor data, torch::Tensor mask,
                          std::optional<torch::Tensor> labels,
                          double match_threshold, double non_match_threshold,
                          bool is_similarity, int64_t include_flags,
                          int64_t max_pairs, int64_t r_dim, int64_t theta_dim,
                          int64_t d0_dim, int64_t d1_dim,
                          int64_t sampling_num_bins,
                          std::optional<torch::Tensor> sampling_thresholds,
                          std::optional<torch::Tensor> sampling_probabilities,
                          int64_t sampling_seed) {
  TORCH_CHECK(data.is_cuda(), "data must be CUDA");
  TORCH_CHECK(mask.is_cuda(), "mask must be CUDA");
  TORCH_CHECK(data.scalar_type() == at::kInt, "data dtype must be int32");
  TORCH_CHECK(mask.scalar_type() == at::kInt, "mask dtype must be int32");
  TORCH_CHECK(data.is_contiguous() && mask.is_contiguous(),
              "data/mask must be contiguous");
  TORCH_CHECK(data.sizes() == mask.sizes(), "data/mask shape mismatch");

  // Validate iris dimensions
  const char *err = nullptr;
  TORCH_CHECK(validate_iris_config((int)r_dim, (int)theta_dim, (int)d0_dim,
                                   (int)d1_dim, &err),
              err);
  IrisConfig cfg = IrisConfig::from_dims((int)r_dim, (int)theta_dim,
                                         (int)d0_dim, (int)d1_dim);

  TORCH_CHECK(data.dim() == 2 && data.size(1) == cfg.k_words,
              "expected data shape [M, ", cfg.k_words, "] for dims (", r_dim,
              ", ", theta_dim, ", ", d0_dim, ", ", d1_dim, ")");

  int64_t M = data.size(0);
  auto opts_int = data.options();
  auto opts_float = data.options().dtype(torch::kFloat32);
  auto opts_byte = data.options().dtype(torch::kUInt8);

  // Validate labels if provided
  if (labels.has_value()) {
    TORCH_CHECK(labels->is_cuda(), "labels must be CUDA");
    TORCH_CHECK(labels->scalar_type() == at::kInt,
                "labels dtype must be int32");
    TORCH_CHECK(labels->is_contiguous(), "labels must be contiguous");
    TORCH_CHECK(labels->dim() == 1 && labels->size(0) == M,
                "labels must have shape [M]");
  }

  // Buffers
  auto premasked = torch::zeros_like(data);

  // GPU buffers for kernel output (stay on GPU)
  auto pair_indices_gpu = torch::zeros({max_pairs, 2}, opts_int);
  auto categories_gpu = torch::zeros({max_pairs}, opts_byte);
  auto distances_gpu = torch::zeros({max_pairs}, opts_float);
  auto match_count_gpu = torch::zeros({1}, opts_int);

  // Construct sampling config
  SamplingConfig sampling = make_sampling_config(
      sampling_num_bins, sampling_thresholds, sampling_probabilities, sampling_seed);

  // Use current stream (respects torch.cuda.stream() context)
  auto stream = at::cuda::getCurrentCUDAStream();
  launch_masked_hamming_cuda(
      reinterpret_cast<uint32_t *>(data.data_ptr<int>()),
      reinterpret_cast<uint32_t *>(mask.data_ptr<int>()),
      reinterpret_cast<uint32_t *>(premasked.data_ptr<int>()), (int)M,
      (int)r_dim, (int)theta_dim,
      labels.has_value() ? labels->data_ptr<int32_t>() : nullptr,
      (float)match_threshold, (float)non_match_threshold, is_similarity,
      (uint8_t)include_flags, sampling, pair_indices_gpu.data_ptr<int32_t>(),
      categories_gpu.data_ptr<uint8_t>(), distances_gpu.data_ptr<float>(),
      reinterpret_cast<unsigned int *>(match_count_gpu.data_ptr<int>()),
      (unsigned int)max_pairs, stream);

  // Return GPU tensors - caller must sync current stream before reading
  return {pair_indices_gpu, categories_gpu, distances_gpu, match_count_gpu};
}

// Sync version: returns CPU tensors (original behavior)
std::vector<torch::Tensor>
masked_hamming_cuda(torch::Tensor data, torch::Tensor mask,
                    std::optional<torch::Tensor> labels, double match_threshold,
                    double non_match_threshold, bool is_similarity,
                    int64_t include_flags, int64_t max_pairs, int64_t r_dim,
                    int64_t theta_dim, int64_t d0_dim, int64_t d1_dim,
                    int64_t sampling_num_bins,
                    std::optional<torch::Tensor> sampling_thresholds,
                    std::optional<torch::Tensor> sampling_probabilities,
                    int64_t sampling_seed) {
  TORCH_CHECK(data.is_cuda(), "data must be CUDA");
  TORCH_CHECK(mask.is_cuda(), "mask must be CUDA");
  TORCH_CHECK(data.scalar_type() == at::kInt, "data dtype must be int32");
  TORCH_CHECK(mask.scalar_type() == at::kInt, "mask dtype must be int32");
  TORCH_CHECK(data.is_contiguous() && mask.is_contiguous(),
              "data/mask must be contiguous");
  TORCH_CHECK(data.sizes() == mask.sizes(), "data/mask shape mismatch");

  // Validate iris dimensions
  const char *err = nullptr;
  TORCH_CHECK(validate_iris_config((int)r_dim, (int)theta_dim, (int)d0_dim,
                                   (int)d1_dim, &err),
              err);
  IrisConfig cfg = IrisConfig::from_dims((int)r_dim, (int)theta_dim,
                                         (int)d0_dim, (int)d1_dim);

  TORCH_CHECK(data.dim() == 2 && data.size(1) == cfg.k_words,
              "expected data shape [M, ", cfg.k_words, "] for dims (", r_dim,
              ", ", theta_dim, ", ", d0_dim, ", ", d1_dim, ")");

  int64_t M = data.size(0);
  auto opts_int = data.options();
  auto opts_float = data.options().dtype(torch::kFloat32);
  auto opts_byte = data.options().dtype(torch::kUInt8);

  // Validate labels if provided
  if (labels.has_value()) {
    TORCH_CHECK(labels->is_cuda(), "labels must be CUDA");
    TORCH_CHECK(labels->scalar_type() == at::kInt,
                "labels dtype must be int32");
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

  // Construct sampling config
  SamplingConfig sampling = make_sampling_config(
      sampling_num_bins, sampling_thresholds, sampling_probabilities, sampling_seed);

  auto stream = at::cuda::getDefaultCUDAStream();
  launch_masked_hamming_cuda(
      reinterpret_cast<uint32_t *>(data.data_ptr<int>()),
      reinterpret_cast<uint32_t *>(mask.data_ptr<int>()),
      reinterpret_cast<uint32_t *>(premasked.data_ptr<int>()), (int)M,
      (int)r_dim, (int)theta_dim,
      labels.has_value() ? labels->data_ptr<int32_t>() : nullptr,
      (float)match_threshold, (float)non_match_threshold, is_similarity,
      (uint8_t)include_flags, sampling, pair_indices_gpu.data_ptr<int32_t>(),
      categories_gpu.data_ptr<uint8_t>(), distances_gpu.data_ptr<float>(),
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
                    actual_count * 2 * sizeof(int), cudaMemcpyDeviceToHost,
                    stream);
    cudaMemcpyAsync(categories_cpu.data_ptr(), categories_gpu.data_ptr(),
                    actual_count * sizeof(uint8_t), cudaMemcpyDeviceToHost,
                    stream);
    cudaMemcpyAsync(distances_cpu.data_ptr(), distances_gpu.data_ptr(),
                    actual_count * sizeof(float), cudaMemcpyDeviceToHost,
                    stream);
  }

  // Return sliced tensors with only valid entries
  auto pair_indices_out = pair_indices_cpu.slice(0, 0, actual_count);
  auto categories_out = categories_cpu.slice(0, 0, actual_count);
  auto distances_out = distances_cpu.slice(0, 0, actual_count);

  return {pair_indices_out, categories_out, distances_out, match_count_cpu};
}

// Async version: returns GPU tensors without synchronization
// Caller must synchronize before reading results
std::vector<torch::Tensor> masked_hamming_ab_cuda_async(
    torch::Tensor data_a, torch::Tensor mask_a, torch::Tensor data_b,
    torch::Tensor mask_b, std::optional<torch::Tensor> labels_a,
    std::optional<torch::Tensor> labels_b, double match_threshold,
    double non_match_threshold, bool is_similarity, int64_t include_flags,
    int64_t max_pairs, int64_t r_dim, int64_t theta_dim, int64_t d0_dim,
    int64_t d1_dim,
    int64_t sampling_num_bins,
    std::optional<torch::Tensor> sampling_thresholds,
    std::optional<torch::Tensor> sampling_probabilities,
    int64_t sampling_seed) {
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

  // Validate iris dimensions
  const char *err = nullptr;
  TORCH_CHECK(validate_iris_config((int)r_dim, (int)theta_dim, (int)d0_dim,
                                   (int)d1_dim, &err),
              err);
  IrisConfig cfg = IrisConfig::from_dims((int)r_dim, (int)theta_dim,
                                         (int)d0_dim, (int)d1_dim);

  TORCH_CHECK(data_a.dim() == 2 && data_a.size(1) == cfg.k_words,
              "expected data_a shape [M_A, ", cfg.k_words, "]");
  TORCH_CHECK(data_b.dim() == 2 && data_b.size(1) == cfg.k_words,
              "expected data_b shape [M_B, ", cfg.k_words, "]");

  int64_t M_A = data_a.size(0);
  int64_t M_B = data_b.size(0);
  auto opts_int = data_a.options();
  auto opts_float = data_a.options().dtype(torch::kFloat32);
  auto opts_byte = data_a.options().dtype(torch::kUInt8);

  // Validate labels if provided (both must be provided or neither)
  bool has_labels = labels_a.has_value() && labels_b.has_value();
  if (labels_a.has_value() || labels_b.has_value()) {
    TORCH_CHECK(has_labels,
                "Both labels_a and labels_b must be provided, or neither");
  }
  if (has_labels) {
    TORCH_CHECK(labels_a->is_cuda(), "labels_a must be CUDA");
    TORCH_CHECK(labels_b->is_cuda(), "labels_b must be CUDA");
    TORCH_CHECK(labels_a->scalar_type() == at::kInt,
                "labels_a dtype must be int32");
    TORCH_CHECK(labels_b->scalar_type() == at::kInt,
                "labels_b dtype must be int32");
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

  // GPU buffers for kernel output (stay on GPU)
  auto pair_indices_gpu = torch::zeros({max_pairs, 2}, opts_int);
  auto categories_gpu = torch::zeros({max_pairs}, opts_byte);
  auto distances_gpu = torch::zeros({max_pairs}, opts_float);
  auto match_count_gpu = torch::zeros({1}, opts_int);

  // Construct sampling config
  SamplingConfig sampling = make_sampling_config(
      sampling_num_bins, sampling_thresholds, sampling_probabilities, sampling_seed);

  // Use current stream (respects torch.cuda.stream() context)
  auto stream = at::cuda::getCurrentCUDAStream();
  launch_masked_hamming_ab_cuda(
      reinterpret_cast<uint32_t *>(data_a.data_ptr<int>()),
      reinterpret_cast<uint32_t *>(mask_a.data_ptr<int>()),
      reinterpret_cast<uint32_t *>(premasked_a.data_ptr<int>()),
      reinterpret_cast<uint32_t *>(data_b.data_ptr<int>()),
      reinterpret_cast<uint32_t *>(mask_b.data_ptr<int>()),
      reinterpret_cast<uint32_t *>(premasked_b.data_ptr<int>()), (int)M_A,
      (int)M_B, (int)r_dim, (int)theta_dim,
      has_labels ? labels_a->data_ptr<int32_t>() : nullptr,
      has_labels ? labels_b->data_ptr<int32_t>() : nullptr,
      (float)match_threshold, (float)non_match_threshold, is_similarity,
      (uint8_t)include_flags, sampling, pair_indices_gpu.data_ptr<int32_t>(),
      categories_gpu.data_ptr<uint8_t>(), distances_gpu.data_ptr<float>(),
      reinterpret_cast<unsigned int *>(match_count_gpu.data_ptr<int>()),
      (unsigned int)max_pairs, stream);

  // Return GPU tensors - caller must sync current stream before reading
  return {pair_indices_gpu, categories_gpu, distances_gpu, match_count_gpu};
}

// Sync version: returns CPU tensors (original behavior)
std::vector<torch::Tensor> masked_hamming_ab_cuda(
    torch::Tensor data_a, torch::Tensor mask_a, torch::Tensor data_b,
    torch::Tensor mask_b, std::optional<torch::Tensor> labels_a,
    std::optional<torch::Tensor> labels_b, double match_threshold,
    double non_match_threshold, bool is_similarity, int64_t include_flags,
    int64_t max_pairs, int64_t r_dim, int64_t theta_dim, int64_t d0_dim,
    int64_t d1_dim,
    int64_t sampling_num_bins,
    std::optional<torch::Tensor> sampling_thresholds,
    std::optional<torch::Tensor> sampling_probabilities,
    int64_t sampling_seed) {
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

  // Validate iris dimensions
  const char *err = nullptr;
  TORCH_CHECK(validate_iris_config((int)r_dim, (int)theta_dim, (int)d0_dim,
                                   (int)d1_dim, &err),
              err);
  IrisConfig cfg = IrisConfig::from_dims((int)r_dim, (int)theta_dim,
                                         (int)d0_dim, (int)d1_dim);

  TORCH_CHECK(data_a.dim() == 2 && data_a.size(1) == cfg.k_words,
              "expected data_a shape [M_A, ", cfg.k_words, "]");
  TORCH_CHECK(data_b.dim() == 2 && data_b.size(1) == cfg.k_words,
              "expected data_b shape [M_B, ", cfg.k_words, "]");

  int64_t M_A = data_a.size(0);
  int64_t M_B = data_b.size(0);
  auto opts_int = data_a.options();
  auto opts_float = data_a.options().dtype(torch::kFloat32);
  auto opts_byte = data_a.options().dtype(torch::kUInt8);

  // Validate labels if provided (both must be provided or neither)
  bool has_labels = labels_a.has_value() && labels_b.has_value();
  if (labels_a.has_value() || labels_b.has_value()) {
    TORCH_CHECK(has_labels,
                "Both labels_a and labels_b must be provided, or neither");
  }
  if (has_labels) {
    TORCH_CHECK(labels_a->is_cuda(), "labels_a must be CUDA");
    TORCH_CHECK(labels_b->is_cuda(), "labels_b must be CUDA");
    TORCH_CHECK(labels_a->scalar_type() == at::kInt,
                "labels_a dtype must be int32");
    TORCH_CHECK(labels_b->scalar_type() == at::kInt,
                "labels_b dtype must be int32");
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

  // Construct sampling config
  SamplingConfig sampling = make_sampling_config(
      sampling_num_bins, sampling_thresholds, sampling_probabilities, sampling_seed);

  auto stream = at::cuda::getDefaultCUDAStream();
  launch_masked_hamming_ab_cuda(
      reinterpret_cast<uint32_t *>(data_a.data_ptr<int>()),
      reinterpret_cast<uint32_t *>(mask_a.data_ptr<int>()),
      reinterpret_cast<uint32_t *>(premasked_a.data_ptr<int>()),
      reinterpret_cast<uint32_t *>(data_b.data_ptr<int>()),
      reinterpret_cast<uint32_t *>(mask_b.data_ptr<int>()),
      reinterpret_cast<uint32_t *>(premasked_b.data_ptr<int>()), (int)M_A,
      (int)M_B, (int)r_dim, (int)theta_dim,
      has_labels ? labels_a->data_ptr<int32_t>() : nullptr,
      has_labels ? labels_b->data_ptr<int32_t>() : nullptr,
      (float)match_threshold, (float)non_match_threshold, is_similarity,
      (uint8_t)include_flags, sampling, pair_indices_gpu.data_ptr<int32_t>(),
      categories_gpu.data_ptr<uint8_t>(), distances_gpu.data_ptr<float>(),
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
                    actual_count * 2 * sizeof(int), cudaMemcpyDeviceToHost,
                    stream);
    cudaMemcpyAsync(categories_cpu.data_ptr(), categories_gpu.data_ptr(),
                    actual_count * sizeof(uint8_t), cudaMemcpyDeviceToHost,
                    stream);
    cudaMemcpyAsync(distances_cpu.data_ptr(), distances_gpu.data_ptr(),
                    actual_count * sizeof(float), cudaMemcpyDeviceToHost,
                    stream);
  }

  // Return sliced tensors with only valid entries
  auto pair_indices_out = pair_indices_cpu.slice(0, 0, actual_count);
  auto categories_out = categories_cpu.slice(0, 0, actual_count);
  auto distances_out = distances_cpu.slice(0, 0, actual_count);

  return {pair_indices_out, categories_out, distances_out, match_count_cpu};
}

torch::Tensor pack_theta_major_cuda(torch::Tensor bits, int64_t r_dim,
                                    int64_t theta_dim, int64_t d0_dim,
                                    int64_t d1_dim) {
  // Input: (M, r_dim, theta_dim, d0_dim, d1_dim) uint8 tensor with values in
  // {0, 1} Output: (M, k_words) int32 tensor (packed bits, theta-major order)
  //
  // IN-PLACE packing with grid-level sync: the kernel packs data into the
  // input buffer. Uses cooperative groups to ensure all reads complete before
  // any writes. The returned tensor shares storage with the input.
  TORCH_CHECK(bits.is_cuda(), "bits must be CUDA tensor");
  TORCH_CHECK(bits.scalar_type() == at::kByte, "bits dtype must be uint8");
  TORCH_CHECK(bits.is_contiguous(), "bits must be contiguous");

  // Validate iris dimensions
  const char *err = nullptr;
  TORCH_CHECK(validate_iris_config((int)r_dim, (int)theta_dim, (int)d0_dim,
                                   (int)d1_dim, &err),
              err);
  IrisConfig cfg = IrisConfig::from_dims((int)r_dim, (int)theta_dim,
                                         (int)d0_dim, (int)d1_dim);

  TORCH_CHECK(bits.dim() == 5,
              "bits must have shape (M, r_dim, theta_dim, d0_dim, d1_dim)");
  TORCH_CHECK(bits.size(1) == r_dim && bits.size(2) == theta_dim &&
                  bits.size(3) == d0_dim && bits.size(4) == d1_dim,
              "bits must have shape (M, ", r_dim, ", ", theta_dim, ", ", d0_dim,
              ", ", d1_dim, ")");

  int64_t M = bits.size(0);

  auto stream = at::cuda::getDefaultCUDAStream();
  launch_pack_theta_major_cuda(bits.data_ptr<uint8_t>(), (int)M, (int)r_dim,
                               (int)theta_dim, (int)d0_dim, (int)d1_dim,
                               stream);

  // Return view of packed data (first M * k_words int32 words)
  auto flat = bits.flatten();
  auto sliced = flat.slice(0, 0, M * cfg.k_words * 4);
  auto packed = sliced.view(torch::kInt32);
  return packed.view({M, cfg.k_words});
}

torch::Tensor repack_to_theta_major_cuda(torch::Tensor input, int64_t r_dim,
                                         int64_t theta_dim, int64_t d0_dim,
                                         int64_t d1_dim) {
  // Input: (M, k_words) int32 tensor packed in r-major order
  // Output: (M, k_words) int32 tensor packed in theta-major order
  TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
  TORCH_CHECK(input.scalar_type() == at::kInt, "input dtype must be int32");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

  // Validate iris dimensions
  const char *err = nullptr;
  TORCH_CHECK(validate_iris_config((int)r_dim, (int)theta_dim, (int)d0_dim,
                                   (int)d1_dim, &err),
              err);
  IrisConfig cfg = IrisConfig::from_dims((int)r_dim, (int)theta_dim,
                                         (int)d0_dim, (int)d1_dim);

  TORCH_CHECK(input.dim() == 2 && input.size(1) == cfg.k_words,
              "input must have shape (M, ", cfg.k_words, ")");

  int64_t M = input.size(0);
  auto output = torch::empty_like(input);

  auto stream = at::cuda::getDefaultCUDAStream();
  launch_repack_to_theta_major_cuda(
      reinterpret_cast<const uint32_t *>(input.data_ptr<int>()),
      reinterpret_cast<uint32_t *>(output.data_ptr<int>()), (int)M, (int)r_dim,
      (int)theta_dim, (int)d0_dim, (int)d1_dim, stream);

  return output;
}

// ----------------- Dot Product Functions -----------------

// Async version: returns GPU tensors without synchronization
std::vector<torch::Tensor>
dot_product_cuda_async(torch::Tensor data,
                       std::optional<torch::Tensor> labels,
                       double match_threshold, double non_match_threshold,
                       bool is_similarity, int64_t include_flags,
                       int64_t max_pairs, int64_t vec_dim,
                       int64_t sampling_num_bins,
                       std::optional<torch::Tensor> sampling_thresholds,
                       std::optional<torch::Tensor> sampling_probabilities,
                       int64_t sampling_seed) {
  TORCH_CHECK(data.is_cuda(), "data must be CUDA");
  TORCH_CHECK(data.scalar_type() == at::kHalf, "data dtype must be float16");
  TORCH_CHECK(data.is_contiguous(), "data must be contiguous");
  TORCH_CHECK(data.dim() == 2, "data must have shape [M, vec_dim]");
  
  int64_t M = data.size(0);
  int64_t actual_vec_dim = data.size(1);
  
  // Use actual vec_dim from tensor if vec_dim param is default
  if (vec_dim == DEFAULT_DOT_VEC_DIM && actual_vec_dim != vec_dim) {
    vec_dim = actual_vec_dim;
  }
  TORCH_CHECK(data.size(1) == vec_dim,
              "expected data shape [M, ", vec_dim, "], got [", M, ", ", actual_vec_dim, "]");
  
  auto opts_int = data.options().dtype(torch::kInt32);
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
  
  // Construct sampling config
  SamplingConfig sampling = make_sampling_config(
      sampling_num_bins, sampling_thresholds, sampling_probabilities, sampling_seed);
  
  // GPU buffers for output
  auto pair_indices_gpu = torch::zeros({max_pairs, 2}, opts_int);
  auto categories_gpu = torch::zeros({max_pairs}, opts_byte);
  auto scores_gpu = torch::zeros({max_pairs}, opts_float);
  auto match_count_gpu = torch::zeros({1}, opts_int);
  
  auto stream = at::cuda::getCurrentCUDAStream();
  launch_dot_product_cuda(
      reinterpret_cast<const half *>(data.data_ptr<at::Half>()),
      (int)M, (int)vec_dim,
      labels.has_value() ? labels->data_ptr<int32_t>() : nullptr,
      (float)match_threshold, (float)non_match_threshold,
      is_similarity, (uint8_t)include_flags, sampling,
      pair_indices_gpu.data_ptr<int32_t>(),
      categories_gpu.data_ptr<uint8_t>(),
      scores_gpu.data_ptr<float>(),
      reinterpret_cast<unsigned int *>(match_count_gpu.data_ptr<int>()),
      (unsigned int)max_pairs, stream);
  
  return {pair_indices_gpu, categories_gpu, scores_gpu, match_count_gpu};
}

// Sync version: returns CPU tensors
std::vector<torch::Tensor>
dot_product_cuda(torch::Tensor data,
                 std::optional<torch::Tensor> labels,
                 double match_threshold, double non_match_threshold,
                 bool is_similarity, int64_t include_flags,
                 int64_t max_pairs, int64_t vec_dim,
                 int64_t sampling_num_bins,
                 std::optional<torch::Tensor> sampling_thresholds,
                 std::optional<torch::Tensor> sampling_probabilities,
                 int64_t sampling_seed) {
  TORCH_CHECK(data.is_cuda(), "data must be CUDA");
  TORCH_CHECK(data.scalar_type() == at::kHalf, "data dtype must be float16");
  TORCH_CHECK(data.is_contiguous(), "data must be contiguous");
  TORCH_CHECK(data.dim() == 2, "data must have shape [M, vec_dim]");
  
  int64_t M = data.size(0);
  int64_t actual_vec_dim = data.size(1);
  
  if (vec_dim == DEFAULT_DOT_VEC_DIM && actual_vec_dim != vec_dim) {
    vec_dim = actual_vec_dim;
  }
  TORCH_CHECK(data.size(1) == vec_dim,
              "expected data shape [M, ", vec_dim, "], got [", M, ", ", actual_vec_dim, "]");
  
  auto opts_int = data.options().dtype(torch::kInt32);
  auto opts_float = data.options().dtype(torch::kFloat32);
  auto opts_byte = data.options().dtype(torch::kUInt8);
  
  if (labels.has_value()) {
    TORCH_CHECK(labels->is_cuda(), "labels must be CUDA");
    TORCH_CHECK(labels->scalar_type() == at::kInt, "labels dtype must be int32");
    TORCH_CHECK(labels->is_contiguous(), "labels must be contiguous");
    TORCH_CHECK(labels->dim() == 1 && labels->size(0) == M,
                "labels must have shape [M]");
  }
  
  // Construct sampling config
  SamplingConfig sampling = make_sampling_config(
      sampling_num_bins, sampling_thresholds, sampling_probabilities, sampling_seed);
  
  auto pair_indices_gpu = torch::zeros({max_pairs, 2}, opts_int);
  auto categories_gpu = torch::zeros({max_pairs}, opts_byte);
  auto scores_gpu = torch::zeros({max_pairs}, opts_float);
  auto match_count_gpu = torch::zeros({1}, opts_int);
  
  // Pinned CPU buffers
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
  auto scores_cpu = torch::zeros({max_pairs}, cpu_opts_float);
  auto match_count_cpu = torch::zeros({1}, cpu_opts_int);
  
  auto stream = at::cuda::getDefaultCUDAStream();
  launch_dot_product_cuda(
      reinterpret_cast<const half *>(data.data_ptr<at::Half>()),
      (int)M, (int)vec_dim,
      labels.has_value() ? labels->data_ptr<int32_t>() : nullptr,
      (float)match_threshold, (float)non_match_threshold,
      is_similarity, (uint8_t)include_flags, sampling,
      pair_indices_gpu.data_ptr<int32_t>(),
      categories_gpu.data_ptr<uint8_t>(),
      scores_gpu.data_ptr<float>(),
      reinterpret_cast<unsigned int *>(match_count_gpu.data_ptr<int>()),
      (unsigned int)max_pairs, stream);
  
  // Copy count and sync
  cudaMemcpyAsync(match_count_cpu.data_ptr(), match_count_gpu.data_ptr(),
                  sizeof(int), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  
  int64_t actual_count = match_count_cpu.data_ptr<int>()[0];
  if (actual_count > max_pairs) {
    actual_count = max_pairs;
  }
  
  if (actual_count > 0) {
    cudaMemcpyAsync(pair_indices_cpu.data_ptr(), pair_indices_gpu.data_ptr(),
                    actual_count * 2 * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(categories_cpu.data_ptr(), categories_gpu.data_ptr(),
                    actual_count * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(scores_cpu.data_ptr(), scores_gpu.data_ptr(),
                    actual_count * sizeof(float), cudaMemcpyDeviceToHost, stream);
  }
  
  auto pair_indices_out = pair_indices_cpu.slice(0, 0, actual_count);
  auto categories_out = categories_cpu.slice(0, 0, actual_count);
  auto scores_out = scores_cpu.slice(0, 0, actual_count);
  
  return {pair_indices_out, categories_out, scores_out, match_count_cpu};
}

// Async A vs B version
std::vector<torch::Tensor>
dot_product_ab_cuda_async(torch::Tensor data_a, torch::Tensor data_b,
                          std::optional<torch::Tensor> labels_a,
                          std::optional<torch::Tensor> labels_b,
                          double match_threshold, double non_match_threshold,
                          bool is_similarity, int64_t include_flags,
                          int64_t max_pairs, int64_t vec_dim,
                          int64_t sampling_num_bins,
                          std::optional<torch::Tensor> sampling_thresholds,
                          std::optional<torch::Tensor> sampling_probabilities,
                          int64_t sampling_seed) {
  TORCH_CHECK(data_a.is_cuda(), "data_a must be CUDA");
  TORCH_CHECK(data_b.is_cuda(), "data_b must be CUDA");
  TORCH_CHECK(data_a.scalar_type() == at::kHalf, "data_a dtype must be float16");
  TORCH_CHECK(data_b.scalar_type() == at::kHalf, "data_b dtype must be float16");
  TORCH_CHECK(data_a.is_contiguous(), "data_a must be contiguous");
  TORCH_CHECK(data_b.is_contiguous(), "data_b must be contiguous");
  TORCH_CHECK(data_a.dim() == 2, "data_a must have shape [M_A, vec_dim]");
  TORCH_CHECK(data_b.dim() == 2, "data_b must have shape [M_B, vec_dim]");
  
  int64_t M_A = data_a.size(0);
  int64_t M_B = data_b.size(0);
  int64_t actual_vec_dim_a = data_a.size(1);
  int64_t actual_vec_dim_b = data_b.size(1);
  
  TORCH_CHECK(actual_vec_dim_a == actual_vec_dim_b,
              "data_a and data_b must have same vec_dim");
  
  if (vec_dim == DEFAULT_DOT_VEC_DIM && actual_vec_dim_a != vec_dim) {
    vec_dim = actual_vec_dim_a;
  }
  
  auto opts_int = data_a.options().dtype(torch::kInt32);
  auto opts_float = data_a.options().dtype(torch::kFloat32);
  auto opts_byte = data_a.options().dtype(torch::kUInt8);
  
  bool has_labels = labels_a.has_value() && labels_b.has_value();
  if (labels_a.has_value() || labels_b.has_value()) {
    TORCH_CHECK(has_labels, "Both labels_a and labels_b must be provided, or neither");
  }
  if (has_labels) {
    TORCH_CHECK(labels_a->is_cuda() && labels_b->is_cuda(), "labels must be CUDA");
    TORCH_CHECK(labels_a->scalar_type() == at::kInt && labels_b->scalar_type() == at::kInt,
                "labels dtype must be int32");
    TORCH_CHECK(labels_a->is_contiguous() && labels_b->is_contiguous(),
                "labels must be contiguous");
    TORCH_CHECK(labels_a->dim() == 1 && labels_a->size(0) == M_A,
                "labels_a must have shape [M_A]");
    TORCH_CHECK(labels_b->dim() == 1 && labels_b->size(0) == M_B,
                "labels_b must have shape [M_B]");
  }
  
  // Construct sampling config
  SamplingConfig sampling = make_sampling_config(
      sampling_num_bins, sampling_thresholds, sampling_probabilities, sampling_seed);
  
  auto pair_indices_gpu = torch::zeros({max_pairs, 2}, opts_int);
  auto categories_gpu = torch::zeros({max_pairs}, opts_byte);
  auto scores_gpu = torch::zeros({max_pairs}, opts_float);
  auto match_count_gpu = torch::zeros({1}, opts_int);
  
  auto stream = at::cuda::getCurrentCUDAStream();
  launch_dot_product_ab_cuda(
      reinterpret_cast<const half *>(data_a.data_ptr<at::Half>()),
      reinterpret_cast<const half *>(data_b.data_ptr<at::Half>()),
      (int)M_A, (int)M_B, (int)vec_dim,
      has_labels ? labels_a->data_ptr<int32_t>() : nullptr,
      has_labels ? labels_b->data_ptr<int32_t>() : nullptr,
      (float)match_threshold, (float)non_match_threshold,
      is_similarity, (uint8_t)include_flags, sampling,
      pair_indices_gpu.data_ptr<int32_t>(),
      categories_gpu.data_ptr<uint8_t>(),
      scores_gpu.data_ptr<float>(),
      reinterpret_cast<unsigned int *>(match_count_gpu.data_ptr<int>()),
      (unsigned int)max_pairs, stream);
  
  return {pair_indices_gpu, categories_gpu, scores_gpu, match_count_gpu};
}

// Sync A vs B version
std::vector<torch::Tensor>
dot_product_ab_cuda(torch::Tensor data_a, torch::Tensor data_b,
                    std::optional<torch::Tensor> labels_a,
                    std::optional<torch::Tensor> labels_b,
                    double match_threshold, double non_match_threshold,
                    bool is_similarity, int64_t include_flags,
                    int64_t max_pairs, int64_t vec_dim,
                    int64_t sampling_num_bins,
                    std::optional<torch::Tensor> sampling_thresholds,
                    std::optional<torch::Tensor> sampling_probabilities,
                    int64_t sampling_seed) {
  TORCH_CHECK(data_a.is_cuda(), "data_a must be CUDA");
  TORCH_CHECK(data_b.is_cuda(), "data_b must be CUDA");
  TORCH_CHECK(data_a.scalar_type() == at::kHalf, "data_a dtype must be float16");
  TORCH_CHECK(data_b.scalar_type() == at::kHalf, "data_b dtype must be float16");
  TORCH_CHECK(data_a.is_contiguous(), "data_a must be contiguous");
  TORCH_CHECK(data_b.is_contiguous(), "data_b must be contiguous");
  TORCH_CHECK(data_a.dim() == 2, "data_a must have shape [M_A, vec_dim]");
  TORCH_CHECK(data_b.dim() == 2, "data_b must have shape [M_B, vec_dim]");
  
  int64_t M_A = data_a.size(0);
  int64_t M_B = data_b.size(0);
  int64_t actual_vec_dim_a = data_a.size(1);
  int64_t actual_vec_dim_b = data_b.size(1);
  
  TORCH_CHECK(actual_vec_dim_a == actual_vec_dim_b,
              "data_a and data_b must have same vec_dim");
  
  if (vec_dim == DEFAULT_DOT_VEC_DIM && actual_vec_dim_a != vec_dim) {
    vec_dim = actual_vec_dim_a;
  }
  
  auto opts_int = data_a.options().dtype(torch::kInt32);
  auto opts_float = data_a.options().dtype(torch::kFloat32);
  auto opts_byte = data_a.options().dtype(torch::kUInt8);
  
  bool has_labels = labels_a.has_value() && labels_b.has_value();
  if (labels_a.has_value() || labels_b.has_value()) {
    TORCH_CHECK(has_labels, "Both labels_a and labels_b must be provided, or neither");
  }
  if (has_labels) {
    TORCH_CHECK(labels_a->is_cuda() && labels_b->is_cuda(), "labels must be CUDA");
    TORCH_CHECK(labels_a->scalar_type() == at::kInt && labels_b->scalar_type() == at::kInt,
                "labels dtype must be int32");
    TORCH_CHECK(labels_a->is_contiguous() && labels_b->is_contiguous(),
                "labels must be contiguous");
    TORCH_CHECK(labels_a->dim() == 1 && labels_a->size(0) == M_A,
                "labels_a must have shape [M_A]");
    TORCH_CHECK(labels_b->dim() == 1 && labels_b->size(0) == M_B,
                "labels_b must have shape [M_B]");
  }
  
  auto pair_indices_gpu = torch::zeros({max_pairs, 2}, opts_int);
  auto categories_gpu = torch::zeros({max_pairs}, opts_byte);
  auto scores_gpu = torch::zeros({max_pairs}, opts_float);
  auto match_count_gpu = torch::zeros({1}, opts_int);
  
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
  auto scores_cpu = torch::zeros({max_pairs}, cpu_opts_float);
  auto match_count_cpu = torch::zeros({1}, cpu_opts_int);
  
  // Construct sampling config
  SamplingConfig sampling = make_sampling_config(
      sampling_num_bins, sampling_thresholds, sampling_probabilities, sampling_seed);
  
  auto stream = at::cuda::getDefaultCUDAStream();
  launch_dot_product_ab_cuda(
      reinterpret_cast<const half *>(data_a.data_ptr<at::Half>()),
      reinterpret_cast<const half *>(data_b.data_ptr<at::Half>()),
      (int)M_A, (int)M_B, (int)vec_dim,
      has_labels ? labels_a->data_ptr<int32_t>() : nullptr,
      has_labels ? labels_b->data_ptr<int32_t>() : nullptr,
      (float)match_threshold, (float)non_match_threshold,
      is_similarity, (uint8_t)include_flags, sampling,
      pair_indices_gpu.data_ptr<int32_t>(),
      categories_gpu.data_ptr<uint8_t>(),
      scores_gpu.data_ptr<float>(),
      reinterpret_cast<unsigned int *>(match_count_gpu.data_ptr<int>()),
      (unsigned int)max_pairs, stream);
  
  cudaMemcpyAsync(match_count_cpu.data_ptr(), match_count_gpu.data_ptr(),
                  sizeof(int), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  
  int64_t actual_count = match_count_cpu.data_ptr<int>()[0];
  if (actual_count > max_pairs) {
    actual_count = max_pairs;
  }
  
  if (actual_count > 0) {
    cudaMemcpyAsync(pair_indices_cpu.data_ptr(), pair_indices_gpu.data_ptr(),
                    actual_count * 2 * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(categories_cpu.data_ptr(), categories_gpu.data_ptr(),
                    actual_count * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(scores_cpu.data_ptr(), scores_gpu.data_ptr(),
                    actual_count * sizeof(float), cudaMemcpyDeviceToHost, stream);
  }
  
  auto pair_indices_out = pair_indices_cpu.slice(0, 0, actual_count);
  auto categories_out = categories_cpu.slice(0, 0, actual_count);
  auto scores_out = scores_cpu.slice(0, 0, actual_count);
  
  return {pair_indices_out, categories_out, scores_out, match_count_cpu};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("masked_hamming_cuda", &masked_hamming_cuda,
        "Masked hamming distance with classification (CUDA)", py::arg("data"),
        py::arg("mask"), py::arg("labels") = py::none(),
        py::arg("match_threshold") = 0.35,
        py::arg("non_match_threshold") = 0.35, py::arg("is_similarity") = false,
        py::arg("include_flags") = INCLUDE_ALL, py::arg("max_pairs") = 1000000,
        py::arg("r_dim") = DEFAULT_R_DIM,
        py::arg("theta_dim") = DEFAULT_THETA_DIM,
        py::arg("d0_dim") = DEFAULT_D0_DIM, py::arg("d1_dim") = DEFAULT_D1_DIM,
        py::arg("sampling_num_bins") = 0,
        py::arg("sampling_thresholds") = py::none(),
        py::arg("sampling_probabilities") = py::none(),
        py::arg("sampling_seed") = 0);
  m.def("masked_hamming_cuda_async", &masked_hamming_cuda_async,
        "Async masked hamming distance (returns GPU tensors, caller must sync)",
        py::arg("data"), py::arg("mask"), py::arg("labels") = py::none(),
        py::arg("match_threshold") = 0.35,
        py::arg("non_match_threshold") = 0.35, py::arg("is_similarity") = false,
        py::arg("include_flags") = INCLUDE_ALL, py::arg("max_pairs") = 1000000,
        py::arg("r_dim") = DEFAULT_R_DIM,
        py::arg("theta_dim") = DEFAULT_THETA_DIM,
        py::arg("d0_dim") = DEFAULT_D0_DIM, py::arg("d1_dim") = DEFAULT_D1_DIM,
        py::arg("sampling_num_bins") = 0,
        py::arg("sampling_thresholds") = py::none(),
        py::arg("sampling_probabilities") = py::none(),
        py::arg("sampling_seed") = 0);
  m.def("masked_hamming_ab_cuda", &masked_hamming_ab_cuda,
        "Masked hamming distance between two sets A and B with classification "
        "(CUDA)",
        py::arg("data_a"), py::arg("mask_a"), py::arg("data_b"),
        py::arg("mask_b"), py::arg("labels_a") = py::none(),
        py::arg("labels_b") = py::none(), py::arg("match_threshold") = 0.35,
        py::arg("non_match_threshold") = 0.35, py::arg("is_similarity") = false,
        py::arg("include_flags") = INCLUDE_ALL, py::arg("max_pairs") = 1000000,
        py::arg("r_dim") = DEFAULT_R_DIM,
        py::arg("theta_dim") = DEFAULT_THETA_DIM,
        py::arg("d0_dim") = DEFAULT_D0_DIM, py::arg("d1_dim") = DEFAULT_D1_DIM,
        py::arg("sampling_num_bins") = 0,
        py::arg("sampling_thresholds") = py::none(),
        py::arg("sampling_probabilities") = py::none(),
        py::arg("sampling_seed") = 0);
  m.def("masked_hamming_ab_cuda_async", &masked_hamming_ab_cuda_async,
        "Async masked hamming A vs B (returns GPU tensors, caller must sync)",
        py::arg("data_a"), py::arg("mask_a"), py::arg("data_b"),
        py::arg("mask_b"), py::arg("labels_a") = py::none(),
        py::arg("labels_b") = py::none(), py::arg("match_threshold") = 0.35,
        py::arg("non_match_threshold") = 0.35, py::arg("is_similarity") = false,
        py::arg("include_flags") = INCLUDE_ALL, py::arg("max_pairs") = 1000000,
        py::arg("r_dim") = DEFAULT_R_DIM,
        py::arg("theta_dim") = DEFAULT_THETA_DIM,
        py::arg("d0_dim") = DEFAULT_D0_DIM, py::arg("d1_dim") = DEFAULT_D1_DIM,
        py::arg("sampling_num_bins") = 0,
        py::arg("sampling_thresholds") = py::none(),
        py::arg("sampling_probabilities") = py::none(),
        py::arg("sampling_seed") = 0);
  m.def("pack_theta_major_cuda", &pack_theta_major_cuda,
        "Pack iris bits to theta-major int32 words (CUDA)", py::arg("bits"),
        py::arg("r_dim") = DEFAULT_R_DIM,
        py::arg("theta_dim") = DEFAULT_THETA_DIM,
        py::arg("d0_dim") = DEFAULT_D0_DIM, py::arg("d1_dim") = DEFAULT_D1_DIM);
  m.def("repack_to_theta_major_cuda", &repack_to_theta_major_cuda,
        "Repack int32 words from r-major to theta-major order (CUDA)",
        py::arg("input"), py::arg("r_dim") = DEFAULT_R_DIM,
        py::arg("theta_dim") = DEFAULT_THETA_DIM,
        py::arg("d0_dim") = DEFAULT_D0_DIM, py::arg("d1_dim") = DEFAULT_D1_DIM);

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

  // Export default dimensions
  m.attr("DEFAULT_R_DIM") = py::int_(DEFAULT_R_DIM);
  m.attr("DEFAULT_THETA_DIM") = py::int_(DEFAULT_THETA_DIM);
  m.attr("DEFAULT_D0_DIM") = py::int_(DEFAULT_D0_DIM);
  m.attr("DEFAULT_D1_DIM") = py::int_(DEFAULT_D1_DIM);

  // Dot product functions
  m.def("dot_product_cuda", &dot_product_cuda,
        "Dot product similarity with classification (CUDA, f16)",
        py::arg("data"),
        py::arg("labels") = py::none(),
        py::arg("match_threshold") = 0.5,
        py::arg("non_match_threshold") = 0.5,
        py::arg("is_similarity") = true,
        py::arg("include_flags") = INCLUDE_ALL,
        py::arg("max_pairs") = 1000000,
        py::arg("vec_dim") = DEFAULT_DOT_VEC_DIM,
        py::arg("sampling_num_bins") = 0,
        py::arg("sampling_thresholds") = py::none(),
        py::arg("sampling_probabilities") = py::none(),
        py::arg("sampling_seed") = 0);
  m.def("dot_product_cuda_async", &dot_product_cuda_async,
        "Async dot product similarity (returns GPU tensors, caller must sync)",
        py::arg("data"),
        py::arg("labels") = py::none(),
        py::arg("match_threshold") = 0.5,
        py::arg("non_match_threshold") = 0.5,
        py::arg("is_similarity") = true,
        py::arg("include_flags") = INCLUDE_ALL,
        py::arg("max_pairs") = 1000000,
        py::arg("vec_dim") = DEFAULT_DOT_VEC_DIM,
        py::arg("sampling_num_bins") = 0,
        py::arg("sampling_thresholds") = py::none(),
        py::arg("sampling_probabilities") = py::none(),
        py::arg("sampling_seed") = 0);
  m.def("dot_product_ab_cuda", &dot_product_ab_cuda,
        "Dot product similarity between two sets A and B (CUDA, f16)",
        py::arg("data_a"), py::arg("data_b"),
        py::arg("labels_a") = py::none(),
        py::arg("labels_b") = py::none(),
        py::arg("match_threshold") = 0.5,
        py::arg("non_match_threshold") = 0.5,
        py::arg("is_similarity") = true,
        py::arg("include_flags") = INCLUDE_ALL,
        py::arg("max_pairs") = 1000000,
        py::arg("vec_dim") = DEFAULT_DOT_VEC_DIM,
        py::arg("sampling_num_bins") = 0,
        py::arg("sampling_thresholds") = py::none(),
        py::arg("sampling_probabilities") = py::none(),
        py::arg("sampling_seed") = 0);
  m.def("dot_product_ab_cuda_async", &dot_product_ab_cuda_async,
        "Async dot product A vs B (returns GPU tensors, caller must sync)",
        py::arg("data_a"), py::arg("data_b"),
        py::arg("labels_a") = py::none(),
        py::arg("labels_b") = py::none(),
        py::arg("match_threshold") = 0.5,
        py::arg("non_match_threshold") = 0.5,
        py::arg("is_similarity") = true,
        py::arg("include_flags") = INCLUDE_ALL,
        py::arg("max_pairs") = 1000000,
        py::arg("vec_dim") = DEFAULT_DOT_VEC_DIM,
        py::arg("sampling_num_bins") = 0,
        py::arg("sampling_thresholds") = py::none(),
        py::arg("sampling_probabilities") = py::none(),
        py::arg("sampling_seed") = 0);
  
  // Dense output variants (high performance)
  m.def("dot_product_dense_cuda", [](torch::Tensor data) {
    TORCH_CHECK(data.is_cuda(), "data must be CUDA");
    TORCH_CHECK(data.scalar_type() == at::kHalf, "data dtype must be float16");
    TORCH_CHECK(data.is_contiguous(), "data must be contiguous");
    TORCH_CHECK(data.dim() == 2, "data must have shape [M, vec_dim]");
    
    int64_t M = data.size(0);
    int64_t vec_dim = data.size(1);
    auto output = torch::zeros({M, M}, data.options().dtype(torch::kFloat32));
    
    auto stream = at::cuda::getCurrentCUDAStream();
    launch_dot_product_dense_cuda(
        reinterpret_cast<const half *>(data.data_ptr<at::Half>()),
        output.data_ptr<float>(),
        (int)M, (int)vec_dim, stream);
    
    return output;
  }, "Dense dot product similarity (returns [M, M] matrix, high performance)", py::arg("data"));

  m.def("dot_product_ab_dense_cuda", [](torch::Tensor data_a, torch::Tensor data_b) {
    TORCH_CHECK(data_a.is_cuda() && data_b.is_cuda(), "data must be CUDA");
    TORCH_CHECK(data_a.scalar_type() == at::kHalf && data_b.scalar_type() == at::kHalf,
                "data dtype must be float16");
    TORCH_CHECK(data_a.is_contiguous() && data_b.is_contiguous(), "data must be contiguous");
    TORCH_CHECK(data_a.dim() == 2 && data_b.dim() == 2, "data must be 2D");
    TORCH_CHECK(data_a.size(1) == data_b.size(1), "vec_dim must match");
    
    int64_t M_A = data_a.size(0);
    int64_t M_B = data_b.size(0);
    int64_t vec_dim = data_a.size(1);
    auto output = torch::zeros({M_A, M_B}, data_a.options().dtype(torch::kFloat32));
    
    auto stream = at::cuda::getCurrentCUDAStream();
    launch_dot_product_ab_dense_cuda(
        reinterpret_cast<const half *>(data_a.data_ptr<at::Half>()),
        reinterpret_cast<const half *>(data_b.data_ptr<at::Half>()),
        output.data_ptr<float>(),
        (int)M_A, (int)M_B, (int)vec_dim, stream);
    
    return output;
  }, "Dense A vs B dot product (returns [M_A, M_B] matrix, high performance)",
     py::arg("data_a"), py::arg("data_b"));

  // Export default dot product vec_dim
  m.attr("DEFAULT_DOT_VEC_DIM") = py::int_(DEFAULT_DOT_VEC_DIM);
}
