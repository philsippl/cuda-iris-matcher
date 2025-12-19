#pragma once

#include <stdint.h>

// ----------------- Runtime Iris Configuration -----------------
// Python layout: (r_dim, theta_dim, d0_dim, d1_dim) where:
//   - r_dim: radial dimension (default 16)
//   - theta_dim: angular/theta dimension (default 200)
//   - d0_dim: first inner dimension (default 2)
//   - d1_dim: second inner dimension (default 2)
//
// Constraints:
//   - (r_dim * d0_dim * d1_dim) must be divisible by 32 (for whole-word theta
//   shifts)
//   - (r_dim * theta_dim * d0_dim * d1_dim) must be divisible by 256
//   (TensorCore chunk)

struct IrisConfig {
  int r_dim;              // Radial dimension
  int theta_dim;          // Angular dimension
  int d0_dim;             // First inner dimension
  int d1_dim;             // Second inner dimension
  int inner_size;         // d0_dim * d1_dim
  int k_bits;             // Total bits = r_dim * theta_dim * d0_dim * d1_dim
  int k_words;            // Total words = k_bits / 32
  int k_chunks;           // Number of chunks = k_bits / 256
  int words_per_shift;    // Words per theta shift = r_dim * inner_size / 32
  int bits_per_theta_col; // Bits per theta column = r_dim * inner_size

  // Construct from all 4 dimensions
  __host__ __device__ static IrisConfig from_dims(int r, int theta, int d0,
                                                  int d1) {
    IrisConfig cfg;
    cfg.r_dim = r;
    cfg.theta_dim = theta;
    cfg.d0_dim = d0;
    cfg.d1_dim = d1;
    cfg.inner_size = d0 * d1;
    cfg.k_bits = r * theta * d0 * d1;
    cfg.k_words = cfg.k_bits / 32;
    cfg.k_chunks = cfg.k_bits / 256;
    cfg.bits_per_theta_col = r * d0 * d1;
    cfg.words_per_shift = cfg.bits_per_theta_col / 32;
    return cfg;
  }

  // Construct from r and theta with default inner dims (2, 2)
  __host__ __device__ static IrisConfig from_dims(int r, int theta) {
    return from_dims(r, theta, 2, 2);
  }

  // Default configuration (16, 200, 2, 2)
  __host__ __device__ static IrisConfig default_config() {
    return from_dims(16, 200, 2, 2);
  }
};

// Validate iris configuration (host-side only)
inline bool validate_iris_config(int r_dim, int theta_dim, int d0_dim,
                                 int d1_dim, const char **error) {
  if (r_dim <= 0 || theta_dim <= 0 || d0_dim <= 0 || d1_dim <= 0) {
    if (error)
      *error = "all dimensions must be positive";
    return false;
  }
  int inner_size = d0_dim * d1_dim;
  int bits_per_theta_col = r_dim * inner_size;
  if (bits_per_theta_col % 32 != 0) {
    if (error)
      *error = "r_dim * d0_dim * d1_dim must be divisible by 32 for whole-word "
               "theta shifts";
    return false;
  }
  int k_bits = r_dim * theta_dim * inner_size;
  if (k_bits % 256 != 0) {
    if (error)
      *error = "total bits (r_dim * theta_dim * d0_dim * d1_dim) must be "
               "divisible by 256";
    return false;
  }
  return true;
}

// Overload for backwards compatibility (assumes d0=2, d1=2)
inline bool validate_iris_config(int r_dim, int theta_dim, const char **error) {
  return validate_iris_config(r_dim, theta_dim, 2, 2, error);
}

// ----------------- Default compile-time constants -----------------
// These are used for backward compatibility and the benchmark
constexpr int DEFAULT_R_DIM = 16;
constexpr int DEFAULT_THETA_DIM = 200;
constexpr int DEFAULT_D0_DIM = 2;
constexpr int DEFAULT_D1_DIM = 2;
constexpr int DEFAULT_K_BITS =
    DEFAULT_R_DIM * DEFAULT_THETA_DIM * DEFAULT_D0_DIM * DEFAULT_D1_DIM;
constexpr int DEFAULT_K_WORDS = DEFAULT_K_BITS / 32;

// ----------------- Fixed TensorCore constants -----------------
// Chunking (TensorCore k=256b) - these are hardware-fixed
constexpr int K_CHUNK_BITS = 256;
constexpr int K_CHUNK_WORDS = K_CHUNK_BITS / 32; // 8

// Extended chunk size for B (we keep +2 halo for fragment indexing convenience)
constexpr int K_CHUNK_WORDS_EXT = K_CHUNK_WORDS + 2; // 10

// Shift range for minimum FHD (theta-roll steps, matches np.roll(axis=1))
constexpr int MAX_SHIFT = 15;
constexpr int NUM_SHIFTS = 2 * MAX_SHIFT + 1; // 31

// Warp MMA tile
constexpr int WMMA_M = 16; // tile rows
constexpr int WMMA_N = 8;  // tile cols

// ----------------- CTA tile config (defaults) -----------------
#ifndef BLOCK_M_VAL
#define BLOCK_M_VAL 192
#endif
#ifndef BLOCK_N_VAL
#define BLOCK_N_VAL 64
#endif

constexpr int BLOCK_M = BLOCK_M_VAL;
constexpr int BLOCK_N = BLOCK_N_VAL;

constexpr int TILES_M_PER_WARP = 2;
constexpr int TILES_N_PER_WARP = 4;

constexpr int TILES_M = BLOCK_M / WMMA_M;
constexpr int TILES_N = BLOCK_N / WMMA_N;

constexpr int WARPS_PER_ROW = TILES_M / TILES_M_PER_WARP;
constexpr int WARPS_PER_COL = TILES_N / TILES_N_PER_WARP;
constexpr int WARPS_PER_BLOCK = WARPS_PER_ROW * WARPS_PER_COL;

// ----------------- Classification categories -----------------
// Match classification for biometric verification
// Category encoding: 2 bits (0-3)
constexpr uint8_t CATEGORY_TRUE_MATCH =
    0; // Same label, distance <= match_threshold
constexpr uint8_t CATEGORY_FALSE_MATCH =
    1; // Diff label, distance <= match_threshold
constexpr uint8_t CATEGORY_FALSE_NON_MATCH =
    2; // Same label, distance > non_match_threshold
constexpr uint8_t CATEGORY_TRUE_NON_MATCH =
    3; // Diff label, distance > non_match_threshold

// Include flags bitmask
constexpr uint8_t INCLUDE_TM = 0x1;  // Include True Matches
constexpr uint8_t INCLUDE_FM = 0x2;  // Include False Matches
constexpr uint8_t INCLUDE_FNM = 0x4; // Include False Non-Matches
constexpr uint8_t INCLUDE_TNM = 0x8; // Include True Non-Matches
constexpr uint8_t INCLUDE_ALL = 0xF; // Include all categories

// ----------------- Stratified Sampling Configuration -----------------
// Maximum number of bins for stratified sampling
constexpr int MAX_SAMPLE_BINS = 8;

struct SamplingConfig {
  int num_bins;                         // Number of active bins (0 = no sampling)
  float thresholds[MAX_SAMPLE_BINS];    // Bin upper bounds (sorted ascending)
  float probabilities[MAX_SAMPLE_BINS]; // Sampling probability for each bin [0, 1]
  uint64_t seed;                        // RNG seed for reproducibility

  // Check if sampling is enabled
  __host__ __device__ bool enabled() const { return num_bins > 0; }

  // Get sampling probability for a given score
  __device__ float get_probability(float score) const {
    // Find the bin this score falls into (first threshold >= score)
    for (int i = 0; i < num_bins; i++) {
      if (score <= thresholds[i]) {
        return probabilities[i];
      }
    }
    // Score exceeds all thresholds, use last bin's probability
    return probabilities[num_bins - 1];
  }

  // Fast hash-based RNG for sampling decisions
  // Uses pair indices to generate deterministic but pseudo-random result
  __device__ bool should_sample(float score, int row, int col) const {
    if (!enabled()) return true;
    
    float prob = get_probability(score);
    if (prob >= 1.0f) return true;
    if (prob <= 0.0f) return false;
    
    // Hash-based RNG: combine seed with pair indices
    // Using a variant of xxHash for speed
    uint64_t h = seed;
    h ^= (uint64_t)row * 0x9E3779B97F4A7C15ULL;
    h ^= (uint64_t)col * 0xC6BC279692B5C323ULL;
    h = h * 0x87C37B91114253D5ULL;
    h ^= h >> 33;
    h = h * 0xC2B2AE3D27D4EB4FULL;
    h ^= h >> 29;
    
    // Convert to float in [0, 1)
    float rand_val = (h & 0xFFFFFFFFULL) / 4294967296.0f;
    return rand_val < prob;
  }
  
  // Create disabled sampling config
  __host__ static SamplingConfig disabled() {
    SamplingConfig cfg;
    cfg.num_bins = 0;
    cfg.seed = 0;
    return cfg;
  }
};
