#!/usr/bin/env bash
set -euo pipefail

# Search over a grid of BLOCK_M and BLOCK_N values to find best pairs/s.
# Requires nvcc and a supported GPU. Adjust ranges below to suit your GPU.

NVCC_FLAGS_BASE="-O3 -arch=sm_89"
M_PERF=${M_PERF:-16384}   # performance problem size (large for realistic test)
REPEATS=${REPEATS:-1}      # repeats per config to smooth variance

# Grids (multiples of 32 to satisfy tiling assertions)
# Extended to larger sizes - max warps = (BM/32)*(BN/32) <= 32
BLOCK_M_LIST=(${BLOCK_M_LIST:-64 96 128 160 192 224 256 288 320 384 512})
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
  local sum_cps=0
  for _ in $(seq 1 $REPEATS); do
    # Build with overrides - check for compile errors
    make clean >/dev/null 2>&1
    if ! make NVCCFLAGS="$NVCC_FLAGS_BASE -DBLOCK_M_VAL=$bm -DBLOCK_N_VAL=$bn" >/dev/null 2>&1; then
      echo "BM=$bm BN=$bn -> COMPILE FAILED"
      return
    fi

    # Run and capture output
    local output
    output=$(./iris 2>&1)
    
    # Check for CUDA errors in output
    if echo "$output" | grep -q "CUDA error"; then
      echo "BM=$bm BN=$bn -> CUDA ERROR"
      return
    fi

    # Extract last pairs/s value
    local cps
    cps=$(echo "$output" | awk '
      /^GPU: .*pairs\/s=/ {
        for (i = 1; i <= NF; ++i) {
          if ($i ~ /pairs\/s=/) {
            split($i, a, "=");
            gsub(/[^0-9.]/, "", a[2]);
            val = a[2];
          }
        }
      }
      END {
        if (val == "") exit 1;
        print val;
      }
    ')
  
    
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
  echo "BM=$bm BN=$bn -> ${avg_cps} M pairs/s"
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

