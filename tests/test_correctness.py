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

    data_words = ih.pack_theta_major(code)
    mask_words = ih.pack_theta_major(mask)
    data_t = torch.from_numpy(data_words).cuda()
    mask_t = torch.from_numpy(mask_words).cuda()

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


