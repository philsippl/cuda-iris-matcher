#!/bin/bash
# Run Python tests for both tensor core and fallback implementations
set -e

cd "$(dirname "$0")"

# Detect GPU compute capability
get_sm_version() {
    python3 -c "import torch; p = torch.cuda.get_device_properties(0); print(p.major * 10 + p.minor)" 2>/dev/null || echo "0"
}

SM_VERSION=$(get_sm_version)
echo "Detected GPU SM version: $SM_VERSION"

# Determine if tensor cores are available (SM80-SM90)
if [ "$SM_VERSION" -ge 80 ] && [ "$SM_VERSION" -lt 100 ]; then
    HAS_TENSOR_CORES=1
    echo "Tensor core b1 MMA: AVAILABLE"
else
    HAS_TENSOR_CORES=0
    echo "Tensor core b1 MMA: NOT AVAILABLE (SM$SM_VERSION)"
fi

FAILED=0

# Test tensor core version (if supported)
if [ "$HAS_TENSOR_CORES" -eq 1 ]; then
    echo ""
    echo "=============================================="
    echo "  TESTING: TENSOR CORE VERSION"
    echo "=============================================="
    echo ""
    
    # Build without FORCE_FALLBACK
    echo "Building tensor core version..."
    FORCE_FALLBACK=0 pip install -e . --no-build-isolation -q
    
    echo "Running tests..."
    if python -m pytest tests/python_tests/ -v; then
        echo "✓ Tensor core tests PASSED"
    else
        echo "✗ Tensor core tests FAILED"
        FAILED=1
    fi
else
    echo ""
    echo "Skipping tensor core tests (not supported on SM$SM_VERSION)"
fi

# Test fallback version (always)
echo ""
echo "=============================================="
echo "  TESTING: FALLBACK VERSION (scalar __popc)"
echo "=============================================="
echo ""

# Build with FORCE_FALLBACK
echo "Building fallback version..."
FORCE_FALLBACK=1 pip install -e . --no-build-isolation -q

echo "Running tests..."
if python -m pytest tests/python_tests/ -v; then
    echo "✓ Fallback tests PASSED"
else
    echo "✗ Fallback tests FAILED"
    FAILED=1
fi

# Summary
echo ""
echo "=============================================="
echo "  SUMMARY"
echo "=============================================="
if [ "$HAS_TENSOR_CORES" -eq 1 ]; then
    echo "Tensor core version: tested"
else
    echo "Tensor core version: skipped (SM$SM_VERSION)"
fi
echo "Fallback version: tested"

if [ "$FAILED" -eq 0 ]; then
    echo ""
    echo "All tests PASSED ✓"
    exit 0
else
    echo ""
    echo "Some tests FAILED ✗"
    exit 1
fi

