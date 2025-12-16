#pragma once

#include <stdint.h>

// ----------------- Problem params -----------------
constexpr int K_BITS = 12800;
constexpr int K_WORDS = K_BITS / 32; // 400

// Half-mask optimization: mask has duplicate d1 bits (d1=0 == d1=1)
// We use d1-major layout: first half = d1=0 bits, second half = d1=1 bits
// This allows computing c3 (mask AND) with half the work since mask bits are
// identical
constexpr int K_BITS_HALF = K_BITS / 2;   // 6400 unique mask bits
constexpr int K_WORDS_HALF = K_WORDS / 2; // 200 words per half

// Chunking (TensorCore k=256b)
constexpr int K_CHUNK_BITS = 256;
constexpr int K_CHUNK_WORDS = K_CHUNK_BITS / 32; // 8
constexpr int K_CHUNKS = K_BITS / K_CHUNK_BITS;  // 50
constexpr int K_CHUNKS_HALF = K_CHUNKS / 2;      // 25 chunks for mask

// Shift range for minimum FHD (theta-roll steps, matches np.roll(axis=1))
constexpr int MAX_SHIFT = 15;
constexpr int NUM_SHIFTS = 2 * MAX_SHIFT + 1; // 31

// D1-major layout: linear_bit = d1*6400 + theta*32 + r*2 + d0
// Rolling by 1 theta step rotates 16*2 = 32 bits = 1 word per half (2 words
// total)
constexpr int WORDS_PER_THETA_SHIFT = 2;

// Extended chunk size for B (we keep +2 halo for fragment indexing convenience)
constexpr int K_CHUNK_WORDS_EXT = K_CHUNK_WORDS + 2; // 10

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
