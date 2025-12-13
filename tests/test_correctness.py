import numpy as np
import torch
import pytest

import cuda_iris_matcher as ih


def test_matches_numpy_roll():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    M = 16
    code = np.random.randint(0, 2, (M, 16, 200, 2, 2), dtype=np.uint8)
    mask = np.random.randint(0, 2, (M, 16, 200, 2, 2), dtype=np.uint8)

    # Pack on GPU using CUDA kernel
    code_cuda = torch.from_numpy(code).cuda()
    mask_cuda = torch.from_numpy(mask).cuda()
    data_t = ih.pack_theta_major(code_cuda)
    mask_t = ih.pack_theta_major(mask_cuda)

    D, pairs, match_count = ih.masked_hamming_cuda(
        data_t, mask_t, write_output=True, collect_pairs=False, threshold=0.5, max_pairs=0
    )

    # NumPy reference
    ref = np.zeros((M, M), dtype=np.float32)
    for i in range(M):
        for j in range(i):
            mi = mask[i].astype(bool)
            di = code[i].astype(bool)
            best = 2.0
            for s in range(-15, 16):
                djr = np.roll(code[j], s, axis=1).astype(bool)
                mjr = np.roll(mask[j], s, axis=1).astype(bool)
                inter = mi & mjr
                valid = int(inter.sum())
                if valid == 0:
                    continue
                ham = int(((di ^ djr) & inter).sum())
                fhd = ham / valid
                if fhd < best:
                    best = fhd
            ref[i, j] = best if best <= 1.0 else 0.0

    D_cpu = D.cpu().numpy()
    assert np.max(np.abs(D_cpu - ref)) <= 1e-5


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

    D, pairs, match_count = ih.masked_hamming_ab_cuda(
        data_a_t, mask_a_t, data_b_t, mask_b_t,
        write_output=True, collect_pairs=False, threshold=0.5, max_pairs=0
    )

    # NumPy reference - compute full M_A x M_B matrix
    ref = np.zeros((M_A, M_B), dtype=np.float32)
    for i in range(M_A):
        for j in range(M_B):
            mi = mask_a[i].astype(bool)
            di = code_a[i].astype(bool)
            best = 2.0
            for s in range(-15, 16):
                djr = np.roll(code_b[j], s, axis=1).astype(bool)
                mjr = np.roll(mask_b[j], s, axis=1).astype(bool)
                inter = mi & mjr
                valid = int(inter.sum())
                if valid == 0:
                    continue
                ham = int(((di ^ djr) & inter).sum())
                fhd = ham / valid
                if fhd < best:
                    best = fhd
            ref[i, j] = best if best <= 1.0 else 0.0

    D_cpu = D.cpu().numpy()
    max_err = np.max(np.abs(D_cpu - ref))
    assert max_err <= 1e-5, f"Max error: {max_err}"


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

    D, pairs, match_count = ih.masked_hamming_ab_cuda(
        data_a_t, mask_a_t, data_b_t, mask_b_t,
        write_output=True, collect_pairs=True, threshold=threshold, max_pairs=max_pairs
    )
    torch.cuda.synchronize()

    D_cpu = D.cpu().numpy()
    count = match_count.item()

    # Verify all collected pairs are below threshold
    for k in range(min(count, max_pairs)):
        i, j = pairs[k].tolist()
        assert 0 <= i < M_A, f"Invalid row index: {i}"
        assert 0 <= j < M_B, f"Invalid col index: {j}"
        assert D_cpu[i, j] < threshold, f"Pair ({i},{j}) has distance {D_cpu[i,j]} >= {threshold}"

    # Verify we found all pairs below threshold
    expected_count = np.sum(D_cpu < threshold)
    assert count == expected_count, f"Expected {expected_count} pairs, got {count}"


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

    D, _, _ = ih.masked_hamming_ab_cuda(
        data_a_t, mask_a_t, data_b_t, mask_b_t,
        write_output=True, collect_pairs=False, threshold=1.0, max_pairs=0
    )
    torch.cuda.synchronize()

    # Verify output shape
    assert D.shape == (M_A, M_B), f"Expected shape ({M_A}, {M_B}), got {D.shape}"

    # Verify all values are valid distances
    D_cpu = D.cpu().numpy()
    assert np.all(D_cpu >= 0) and np.all(D_cpu <= 1), "Distances should be in [0, 1]"

