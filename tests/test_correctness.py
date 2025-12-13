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

