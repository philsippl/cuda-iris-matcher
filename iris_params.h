#pragma once

#include <stdint.h>

// ----------------- Problem params -----------------
constexpr int K_BITS = 12800;
constexpr int K_WORDS = K_BITS / 32; // 400

// Chunking (TensorCore k=256b)
constexpr int K_CHUNK_BITS = 256;
constexpr int K_CHUNK_WORDS = K_CHUNK_BITS / 32; // 8
constexpr int K_CHUNKS = K_BITS / K_CHUNK_BITS;  // 50

// Shift range for minimum FHD (theta-roll steps, matches np.roll(axis=1))
constexpr int MAX_SHIFT = 15;
constexpr int NUM_SHIFTS = 2 * MAX_SHIFT + 1; // 31

// Iris layout in Python: (16, 200, 2, 2). Rolling axis=1 by 1 step rotates
// 16*2*2 = 64 bits = 2 uint32 words in our packed representation.
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
