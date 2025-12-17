import sys
import os
import numpy as np
import torch
import pytest

# Add parent tests directory to path for utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import cuda_iris_matcher as ih
from utils import rotation_aware_hamming_distance


@pytest.mark.parametrize(
    "dims,M",
    [
        # Default dimensions
        ((16, 200, 2, 2), 16),
        # Non-default but constraint-valid:
        # - r_dim*d0_dim*d1_dim must be divisible by 32
        # - r_dim*theta_dim*d0_dim*d1_dim must be divisible by 256
        ((16, 256, 2, 2), 8),
    ],
)
def test_matches_numpy_roll(dims, M):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    code = np.random.randint(0, 2, (M, *dims), dtype=np.uint8)
    mask = np.random.randint(0, 2, (M, *dims), dtype=np.uint8)

    # Pack on GPU using CUDA kernel
    code_cuda = torch.from_numpy(code).cuda()
    mask_cuda = torch.from_numpy(mask).cuda()
    data_t = ih.pack_theta_major(code_cuda, dims=dims)
    mask_t = ih.pack_theta_major(mask_cuda, dims=dims)

    # New interface: returns (pair_indices, categories, distances, count)
    # Use high threshold and INCLUDE_ALL to get all pairs
    max_pairs = M * (M - 1) // 2
    pair_indices, categories, distances, count = ih.masked_hamming_cuda(
        data_t, mask_t, match_threshold=1.0, non_match_threshold=1.0,
        include_flags=ih.INCLUDE_ALL, max_pairs=max_pairs, dims=dims,
    )
    torch.cuda.synchronize()

    n_pairs = count.item()

    # NumPy reference - compute for lower triangle (i > j)
    ref = {}
    for i in range(M):
        for j in range(i):
            dist, _ = rotation_aware_hamming_distance(code[i], mask[i], code[j], mask[j])
            ref[(i, j)] = dist

    # Verify each pair returned by CUDA matches reference
    for k in range(n_pairs):
        i, j = pair_indices[k].tolist()
        cuda_dist = distances[k].item()
        ref_dist = ref.get((i, j), ref.get((j, i), -1))
        assert abs(cuda_dist - ref_dist) <= 1e-5, (
            f"dims={(r_dim,theta_dim,d0_dim,d1_dim)} pair=({i},{j}): cuda={cuda_dist}, ref={ref_dist}"
        )


def test_repack_to_theta_major():
    """Test that repack_to_theta_major correctly converts r-major to theta-major order."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    M = 16
    code = np.random.randint(0, 2, (M, 16, 200, 2, 2), dtype=np.uint8)

    # Pack to r-major order (NumPy reference)
    # r-major: bit[r,theta,d0,d1] at linear_bit = r*800 + theta*4 + d0*2 + d1
    packed_r_major = np.zeros((M, 400), dtype=np.uint32)
    for m in range(M):
        for r in range(16):
            for theta in range(200):
                for d0 in range(2):
                    for d1 in range(2):
                        if code[m, r, theta, d0, d1]:
                            linear_bit = r * 800 + theta * 4 + d0 * 2 + d1
                            word = linear_bit // 32
                            bit = linear_bit % 32
                            packed_r_major[m, word] |= np.uint32(1 << bit)

    # Pack to theta-major order (NumPy reference)
    # theta-major: bit[r,theta,d0,d1] at linear_bit = theta*64 + r*4 + d0*2 + d1
    packed_theta_major_ref = np.zeros((M, 400), dtype=np.uint32)
    for m in range(M):
        for r in range(16):
            for theta in range(200):
                for d0 in range(2):
                    for d1 in range(2):
                        if code[m, r, theta, d0, d1]:
                            linear_bit = theta * 64 + r * 4 + d0 * 2 + d1
                            word = linear_bit // 32
                            bit = linear_bit % 32
                            packed_theta_major_ref[m, word] |= np.uint32(1 << bit)

    # Use CUDA repack function
    input_cuda = torch.from_numpy(packed_r_major.view(np.int32)).cuda()
    output_cuda = ih.repack_to_theta_major(input_cuda)
    torch.cuda.synchronize()

    # Compare
    output_np = output_cuda.cpu().numpy().view(np.uint32)
    assert np.array_equal(output_np, packed_theta_major_ref)


def test_masked_hamming_ab_matches_numpy():
    """Test that A vs B kernel matches NumPy reference for different sets."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    M_A = 12
    M_B = 10
    code_a = np.random.randint(0, 2, (M_A, 16, 200, 2, 2), dtype=np.uint8)
    mask_a = np.random.randint(0, 2, (M_A, 16, 200, 2, 2), dtype=np.uint8)
    code_b = np.random.randint(0, 2, (M_B, 16, 200, 2, 2), dtype=np.uint8)
    mask_b = np.random.randint(0, 2, (M_B, 16, 200, 2, 2), dtype=np.uint8)

    # Pack on GPU using CUDA kernel
    code_a_cuda = torch.from_numpy(code_a).cuda()
    mask_a_cuda = torch.from_numpy(mask_a).cuda()
    code_b_cuda = torch.from_numpy(code_b).cuda()
    mask_b_cuda = torch.from_numpy(mask_b).cuda()

    data_a_t = ih.pack_theta_major(code_a_cuda)
    mask_a_t = ih.pack_theta_major(mask_a_cuda)
    data_b_t = ih.pack_theta_major(code_b_cuda)
    mask_b_t = ih.pack_theta_major(mask_b_cuda)

    # New interface: returns (pair_indices, categories, distances, count)
    max_pairs = M_A * M_B
    pair_indices, categories, distances, count = ih.masked_hamming_ab_cuda(
        data_a_t, mask_a_t, data_b_t, mask_b_t,
        match_threshold=1.0, non_match_threshold=1.0,
        include_flags=ih.INCLUDE_ALL, max_pairs=max_pairs
    )
    torch.cuda.synchronize()

    n_pairs = count.item()

    # NumPy reference - compute full M_A x M_B matrix
    ref = {}
    for i in range(M_A):
        for j in range(M_B):
            dist, _ = rotation_aware_hamming_distance(code_a[i], mask_a[i], code_b[j], mask_b[j])
            ref[(i, j)] = dist

    # Verify each pair returned by CUDA matches reference
    for k in range(n_pairs):
        i, j = pair_indices[k].tolist()
        cuda_dist = distances[k].item()
        ref_dist = ref.get((i, j), -1)
        assert abs(cuda_dist - ref_dist) <= 1e-5, f"Pair ({i},{j}): cuda={cuda_dist}, ref={ref_dist}"


def test_masked_hamming_ab_pair_collection():
    """Test that A vs B kernel correctly collects matching pairs."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    M_A = 8
    M_B = 6

    code_a = np.random.randint(0, 2, (M_A, 16, 200, 2, 2), dtype=np.uint8)
    mask_a = np.random.randint(0, 2, (M_A, 16, 200, 2, 2), dtype=np.uint8)
    code_b = np.random.randint(0, 2, (M_B, 16, 200, 2, 2), dtype=np.uint8)
    mask_b = np.random.randint(0, 2, (M_B, 16, 200, 2, 2), dtype=np.uint8)

    # Pack on GPU
    code_a_cuda = torch.from_numpy(code_a).cuda()
    mask_a_cuda = torch.from_numpy(mask_a).cuda()
    code_b_cuda = torch.from_numpy(code_b).cuda()
    mask_b_cuda = torch.from_numpy(mask_b).cuda()

    data_a_t = ih.pack_theta_major(code_a_cuda)
    mask_a_t = ih.pack_theta_major(mask_a_cuda)
    data_b_t = ih.pack_theta_major(code_b_cuda)
    mask_b_t = ih.pack_theta_major(mask_b_cuda)

    threshold = 0.4
    max_pairs = 1000

    # New interface: use match_threshold to filter pairs
    pair_indices, categories, distances, count = ih.masked_hamming_ab_cuda(
        data_a_t, mask_a_t, data_b_t, mask_b_t,
        match_threshold=threshold, non_match_threshold=threshold,
        include_flags=ih.INCLUDE_ALL, max_pairs=max_pairs
    )
    torch.cuda.synchronize()

    n_pairs = count.item()

    # Compute reference distances for all pairs
    ref_dists = {}
    for i in range(M_A):
        for j in range(M_B):
            dist, _ = rotation_aware_hamming_distance(code_a[i], mask_a[i], code_b[j], mask_b[j])
            ref_dists[(i, j)] = dist

    # Verify all collected pairs
    for k in range(min(n_pairs, max_pairs)):
        i, j = pair_indices[k].tolist()
        cuda_dist = distances[k].item()
        assert 0 <= i < M_A, f"Invalid row index: {i}"
        assert 0 <= j < M_B, f"Invalid col index: {j}"
        ref_dist = ref_dists[(i, j)]
        assert abs(cuda_dist - ref_dist) <= 1e-5, f"Pair ({i},{j}): cuda={cuda_dist}, ref={ref_dist}"

    # With no labels, all pairs should be returned (unclassified)
    expected_count = M_A * M_B
    assert n_pairs == expected_count, f"Expected {expected_count} pairs (all unclassified), got {n_pairs}"


def test_masked_hamming_ab_asymmetric_sizes():
    """Test A vs B with asymmetric sizes (M_A != M_B)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Test with significantly different sizes
    M_A = 5
    M_B = 20

    code_a = np.random.randint(0, 2, (M_A, 16, 200, 2, 2), dtype=np.uint8)
    mask_a = np.ones((M_A, 16, 200, 2, 2), dtype=np.uint8)  # Full mask
    code_b = np.random.randint(0, 2, (M_B, 16, 200, 2, 2), dtype=np.uint8)
    mask_b = np.ones((M_B, 16, 200, 2, 2), dtype=np.uint8)  # Full mask

    code_a_cuda = torch.from_numpy(code_a).cuda()
    mask_a_cuda = torch.from_numpy(mask_a).cuda()
    code_b_cuda = torch.from_numpy(code_b).cuda()
    mask_b_cuda = torch.from_numpy(mask_b).cuda()

    data_a_t = ih.pack_theta_major(code_a_cuda)
    mask_a_t = ih.pack_theta_major(mask_a_cuda)
    data_b_t = ih.pack_theta_major(code_b_cuda)
    mask_b_t = ih.pack_theta_major(mask_b_cuda)

    # New interface
    max_pairs = M_A * M_B
    pair_indices, categories, distances, count = ih.masked_hamming_ab_cuda(
        data_a_t, mask_a_t, data_b_t, mask_b_t,
        match_threshold=1.0, non_match_threshold=1.0,
        include_flags=ih.INCLUDE_ALL, max_pairs=max_pairs
    )
    torch.cuda.synchronize()

    n_pairs = count.item()

    # Verify we got all pairs (no labels means all unclassified)
    assert n_pairs == M_A * M_B, f"Expected {M_A * M_B} pairs, got {n_pairs}"

    # Verify all distances are valid
    for k in range(n_pairs):
        dist = distances[k].item()
        assert 0 <= dist <= 1, f"Distance {dist} should be in [0, 1]"

