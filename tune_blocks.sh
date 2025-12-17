#!/usr/bin/env bash
set -euo pipefail

# Search over a grid of BLOCK_M and BLOCK_N values to find best pairs/s.
# Requires nvcc and a supported GPU. Adjust ranges below to suit your GPU.

NVCC_FLAGS_BASE="-O3 -arch=sm_89"
# NOTE: benchmark runtime scales ~O(M^2). Keep this moderate unless you really
# want a long tuning run.
M_PERF=${M_PERF:-5000}       # benchmark problem size
WARMUP_ITERS=${WARMUP_ITERS:-1}
BENCH_ITERS=${BENCH_ITERS:-3}
REPEATS=${REPEATS:-1}        # repeats per config to smooth variance
TARGET=${TARGET:-benchmark}  # make target to build/run: benchmark|benchmark_fallback
# Optional: skip configs above a chosen dynamic shared-mem budget.
# Default 0 = don't skip; rely on the benchmark to succeed/fail.
MAX_SMEM_BYTES=${MAX_SMEM_BYTES:-0}

# Grids (multiples of 32 to satisfy tiling assertions)
# Extended to larger sizes - max warps = (BM/32)*(BN/32) <= 32
BLOCK_M_LIST=(${BLOCK_M_LIST:-128 160 192 224 256 288 320 384 512})
BLOCK_N_LIST=(${BLOCK_N_LIST:-32 64 96 128 160 192 224 256 320})

best_cfg=""
best_cps=0

run_cfg() {
  local bm=$1
  local bn=$2

  # Skip invalid shapes (must be multiples of 32 in M and N due to tiling:
  # 16x8 tile, 2x4 tiles per warp => 32x32 per warp).
  if (( bm % 32 != 0 )); then return; fi
  if (( bn % 32 != 0 )); then return; fi
  # Ensure warps per block <= 32 (max 1024 threads).
  # Valid large configs: 256x128(32w), 384x64(24w), 512x64(32w), 320x96(30w)
  local warps=$(( (bm / 32) * (bn / 32) ))
  if (( warps > 32 )); then return; fi
  if (( warps < 1 )); then return; fi

  # Optionally skip configs that exceed a dynamic shared-memory budget.
  # Kernel uses: smem = 2*(2*BM*K_CHUNK_WORDS + 2*BN*K_CHUNK_WORDS_EXT)*4 bytes.
  # With K_CHUNK_WORDS=8 and K_CHUNK_WORDS_EXT=10, this simplifies to:
  # smem_bytes = 128*BM + 160*BN
  local smem_bytes=$(( 128*bm + 160*bn ))
  if (( MAX_SMEM_BYTES > 0 )) && (( smem_bytes > MAX_SMEM_BYTES )); then
      echo "BM=$bm BN=$bn -> SKIP (smem=${smem_bytes}B > MAX_SMEM_BYTES=${MAX_SMEM_BYTES})"
      return
    fi
  local sum_cps=0
  for _ in $(seq 1 $REPEATS); do
    # Build with overrides - check for compile errors
    make clean >/dev/null 2>&1
    if ! make "$TARGET" NVCCFLAGS="$NVCC_FLAGS_BASE -DBLOCK_M_VAL=$bm -DBLOCK_N_VAL=$bn" >/dev/null 2>&1; then
      echo "BM=$bm BN=$bn -> COMPILE FAILED"
      return
    fi

    # Run and capture output
    local output
    if ! output=$(./"$TARGET" "$M_PERF" "$WARMUP_ITERS" "$BENCH_ITERS" 2>&1); then
      echo "BM=$bm BN=$bn -> RUN FAILED"
      return
    fi
    
    # Check for CUDA errors in output
    if echo "$output" | grep -q "CUDA error"; then
      echo "BM=$bm BN=$bn -> CUDA ERROR"
      return
    fi

    # Extract pairs/s from the SELF "=== Results ===" section only.
    # benchmark prints:
    #   === Results ===
    #   ...
    #   Pairs/s: XXX.XXX M (31 shifts)
    local cps
    cps=$(echo "$output" | awk '
      /^=== Results ===/ { in_self=1; next }
      /^=== A vs B / { in_self=0 }
      in_self && /^Pairs\/s:/ { print $2; exit }
      END { if (NR==0) exit 1 }
    ' || true)

    # Validate parse
    if [[ -z "${cps}" ]] || ! [[ "${cps}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
      echo "BM=$bm BN=$bn -> PARSE FAILED"
      return
    fi
  
    
    sum_cps=$(python - <<EOF
print($sum_cps + float("$cps"))
EOF
)
  done
  # average
  avg_cps=$(python - <<EOF
print($sum_cps / $REPEATS)
EOF
)
  echo "BM=$bm BN=$bn -> ${avg_cps} M pairs/s (M=$M_PERF warmup=$WARMUP_ITERS bench=$BENCH_ITERS target=$TARGET)"
  if python - <<EOF
import sys
sys.exit(0 if $avg_cps > $best_cps else 1)
EOF
  then
    best_cps=$avg_cps
    best_cfg="BM=$bm BN=$bn"
  fi
}

for bm in "${BLOCK_M_LIST[@]}"; do
  for bn in "${BLOCK_N_LIST[@]}"; do
    run_cfg "$bm" "$bn"
  done
done

echo "Best: $best_cfg -> ${best_cps} M pairs/s"

