import numpy as np
import torch
import pytest

import cuda_iris_matcher as ih


def numpy_dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """Compute dot product between two vectors using NumPy."""
    return np.dot(a.astype(np.float32), b.astype(np.float32))


@pytest.mark.parametrize("vec_dim", [512, 256, 1024])
def test_dot_product_matches_numpy(vec_dim):
    """Test that dot product kernel matches NumPy reference."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    M = 16
    # Create random normalized f16 vectors
    data_np = np.random.randn(M, vec_dim).astype(np.float32)
    # Normalize
    data_np = data_np / np.linalg.norm(data_np, axis=1, keepdims=True)
    data_np = data_np.astype(np.float16)

    data_cuda = torch.from_numpy(data_np).cuda()

    # Run CUDA kernel
    max_pairs = M * (M - 1) // 2
    pair_indices, categories, scores, count = ih.dot_product_cuda(
        data_cuda,
        match_threshold=1.0,
        non_match_threshold=-1.0,
        include_flags=ih.INCLUDE_ALL,
        max_pairs=max_pairs,
        vec_dim=vec_dim,
    )
    torch.cuda.synchronize()

    n_pairs = count.item()

    # Compute reference - lower triangle (i > j)
    ref = {}
    for i in range(M):
        for j in range(i):
            score = numpy_dot_product(data_np[i], data_np[j])
            ref[(i, j)] = score

    # Verify each pair
    for k in range(n_pairs):
        i, j = pair_indices[k].tolist()
        cuda_score = scores[k].item()
        ref_score = ref.get((i, j), ref.get((j, i), None))
        assert ref_score is not None, f"Pair ({i}, {j}) not in reference"
        # Allow some tolerance for f16 precision
        assert abs(cuda_score - ref_score) < 1e-2, (
            f"Pair ({i},{j}): cuda={cuda_score:.6f}, ref={ref_score:.6f}, "
            f"diff={abs(cuda_score - ref_score):.6f}"
        )


def test_dot_product_ab_matches_numpy():
    """Test that A vs B dot product kernel matches NumPy reference."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    M_A = 12
    M_B = 10
    vec_dim = 512

    data_a_np = np.random.randn(M_A, vec_dim).astype(np.float32)
    data_a_np = data_a_np / np.linalg.norm(data_a_np, axis=1, keepdims=True)
    data_a_np = data_a_np.astype(np.float16)

    data_b_np = np.random.randn(M_B, vec_dim).astype(np.float32)
    data_b_np = data_b_np / np.linalg.norm(data_b_np, axis=1, keepdims=True)
    data_b_np = data_b_np.astype(np.float16)

    data_a_cuda = torch.from_numpy(data_a_np).cuda()
    data_b_cuda = torch.from_numpy(data_b_np).cuda()

    max_pairs = M_A * M_B
    pair_indices, categories, scores, count = ih.dot_product_ab_cuda(
        data_a_cuda, data_b_cuda,
        match_threshold=1.0,
        non_match_threshold=-1.0,
        include_flags=ih.INCLUDE_ALL,
        max_pairs=max_pairs,
    )
    torch.cuda.synchronize()

    n_pairs = count.item()

    # Compute reference - full M_A x M_B matrix
    ref = {}
    for i in range(M_A):
        for j in range(M_B):
            score = numpy_dot_product(data_a_np[i], data_b_np[j])
            ref[(i, j)] = score

    # Verify each pair
    for k in range(n_pairs):
        i, j = pair_indices[k].tolist()
        cuda_score = scores[k].item()
        ref_score = ref.get((i, j), None)
        assert ref_score is not None, f"Pair ({i}, {j}) not in reference"
        assert abs(cuda_score - ref_score) < 1e-2, (
            f"Pair ({i},{j}): cuda={cuda_score:.6f}, ref={ref_score:.6f}"
        )


def test_dot_product_classification():
    """Test that classification with labels works correctly."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    M = 20
    vec_dim = 512

    # Create vectors - some identical pairs to ensure matches
    data_np = np.random.randn(M, vec_dim).astype(np.float32)
    data_np = data_np / np.linalg.norm(data_np, axis=1, keepdims=True)
    data_np = data_np.astype(np.float16)

    data_cuda = torch.from_numpy(data_np).cuda()

    # Create labels: 5 groups of 4
    labels = torch.tensor([i // 4 for i in range(M)], dtype=torch.int32, device="cuda")

    threshold = 0.5  # Moderate threshold
    max_pairs = M * (M - 1) // 2

    pair_indices, categories, scores, count = ih.dot_product_cuda(
        data_cuda,
        labels=labels,
        match_threshold=threshold,
        non_match_threshold=threshold,
        is_similarity=True,
        include_flags=ih.INCLUDE_ALL,
        max_pairs=max_pairs,
    )
    torch.cuda.synchronize()

    n_pairs = count.item()

    # Verify classifications
    for k in range(n_pairs):
        i, j = pair_indices[k].tolist()
        score = scores[k].item()
        cat = categories[k].item()

        label_i = labels[i].item()
        label_j = labels[j].item()
        same_label = (label_i == label_j)

        is_match = (score >= threshold)
        is_non_match = (score < threshold)

        if same_label:
            if is_match:
                assert cat == ih.CATEGORY_TRUE_MATCH, f"Expected TM for same_label match"
            elif is_non_match:
                assert cat == ih.CATEGORY_FALSE_NON_MATCH, f"Expected FNM for same_label non-match"
        else:
            if is_match:
                assert cat == ih.CATEGORY_FALSE_MATCH, f"Expected FM for diff_label match"
            elif is_non_match:
                assert cat == ih.CATEGORY_TRUE_NON_MATCH, f"Expected TNM for diff_label non-match"


def test_dot_product_include_flags():
    """Test filtering with include_flags."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    M = 16
    vec_dim = 256

    data_np = np.random.randn(M, vec_dim).astype(np.float32)
    data_np = data_np / np.linalg.norm(data_np, axis=1, keepdims=True)
    data_np = data_np.astype(np.float16)

    data_cuda = torch.from_numpy(data_np).cuda()
    labels = torch.tensor([i % 4 for i in range(M)], dtype=torch.int32, device="cuda")

    threshold = 0.5
    max_pairs = M * (M - 1) // 2

    # Request only True Matches
    pair_indices, categories, scores, count = ih.dot_product_cuda(
        data_cuda,
        labels=labels,
        match_threshold=threshold,
        non_match_threshold=threshold,
        is_similarity=True,
        include_flags=ih.INCLUDE_TM,
        max_pairs=max_pairs,
    )
    torch.cuda.synchronize()

    n_pairs = count.item()

    # All returned should be True Matches
    for k in range(n_pairs):
        cat = categories[k].item()
        assert cat == ih.CATEGORY_TRUE_MATCH, f"Expected TM, got {cat}"


def test_dot_product_asymmetric_sizes():
    """Test A vs B with asymmetric sizes."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    M_A = 5
    M_B = 20
    vec_dim = 512

    data_a = torch.randn(M_A, vec_dim, dtype=torch.float16, device="cuda")
    data_a = data_a / data_a.norm(dim=1, keepdim=True)

    data_b = torch.randn(M_B, vec_dim, dtype=torch.float16, device="cuda")
    data_b = data_b / data_b.norm(dim=1, keepdim=True)

    max_pairs = M_A * M_B
    pair_indices, categories, scores, count = ih.dot_product_ab_cuda(
        data_a, data_b,
        match_threshold=1.0,
        non_match_threshold=-1.0,
        include_flags=ih.INCLUDE_ALL,
        max_pairs=max_pairs,
    )
    torch.cuda.synchronize()

    n_pairs = count.item()

    # Should get all M_A * M_B pairs
    assert n_pairs == M_A * M_B, f"Expected {M_A * M_B} pairs, got {n_pairs}"

    # Verify all scores are valid (between -1 and 1 for normalized vectors)
    for k in range(n_pairs):
        score = scores[k].item()
        assert -1.1 <= score <= 1.1, f"Score {score} out of expected range"


def test_dot_product_default_vec_dim():
    """Test that default vec_dim of 512 works."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    M = 10
    data = torch.randn(M, ih.DEFAULT_DOT_VEC_DIM, dtype=torch.float16, device="cuda")
    data = data / data.norm(dim=1, keepdim=True)

    pair_indices, categories, scores, count = ih.dot_product_cuda(data)
    torch.cuda.synchronize()

    n_pairs = count.item()
    expected = M * (M - 1) // 2
    assert n_pairs == expected, f"Expected {expected} pairs, got {n_pairs}"


def test_dot_product_larger_dataset():
    """Test with a larger dataset for performance validation."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    M = 256
    vec_dim = 512

    data = torch.randn(M, vec_dim, dtype=torch.float16, device="cuda")
    data = data / data.norm(dim=1, keepdim=True)

    max_pairs = M * (M - 1) // 2

    # Just verify it runs without error
    pair_indices, categories, scores, count = ih.dot_product_cuda(
        data,
        match_threshold=1.0,
        non_match_threshold=-1.0,
        include_flags=ih.INCLUDE_ALL,
        max_pairs=max_pairs,
    )
    torch.cuda.synchronize()

    n_pairs = count.item()
    assert n_pairs == max_pairs, f"Expected {max_pairs} pairs, got {n_pairs}"


def test_dot_product_dense_output():
    """Test dense output functions match numpy reference."""
    vec_dim = 512
    M = 100
    
    # Generate normalized data
    data = torch.randn(M, vec_dim, dtype=torch.float16, device="cuda")
    data = data / data.norm(dim=1, keepdim=True)
    
    # Test dense output (uses PyTorch mm for best performance)
    dense_result = ih.dot_product_dense_cuda(data)
    
    # Compare with numpy
    data_np = data.cpu().numpy().astype(np.float32)
    expected = data_np @ data_np.T
    
    assert dense_result.shape == (M, M), f"Expected shape ({M}, {M}), got {dense_result.shape}"
    np.testing.assert_allclose(
        dense_result.cpu().numpy(), expected, rtol=1e-2, atol=1e-2
    )


def test_dot_product_dense_ab_output():
    """Test dense A vs B output."""
    vec_dim = 512
    M_A, M_B = 80, 100
    
    data_a = torch.randn(M_A, vec_dim, dtype=torch.float16, device="cuda")
    data_a = data_a / data_a.norm(dim=1, keepdim=True)
    data_b = torch.randn(M_B, vec_dim, dtype=torch.float16, device="cuda")
    data_b = data_b / data_b.norm(dim=1, keepdim=True)
    
    dense_result = ih.dot_product_ab_dense_cuda(data_a, data_b)
    
    data_a_np = data_a.cpu().numpy().astype(np.float32)
    data_b_np = data_b.cpu().numpy().astype(np.float32)
    expected = data_a_np @ data_b_np.T
    
    assert dense_result.shape == (M_A, M_B), f"Expected shape ({M_A}, {M_B}), got {dense_result.shape}"
    np.testing.assert_allclose(
        dense_result.cpu().numpy(), expected, rtol=1e-2, atol=1e-2
    )


def test_dot_product_native_dense_kernel():
    """Test native CUDA dense kernel correctness."""
    from cuda_iris_matcher import _C
    
    vec_dim = 512
    M = 128
    
    data = torch.randn(M, vec_dim, dtype=torch.float16, device="cuda")
    data = data / data.norm(dim=1, keepdim=True)
    
    # Native kernel result
    native_result = _C.dot_product_dense_cuda(data)
    
    # Reference using PyTorch mm
    expected = torch.mm(data.float(), data.float().t())
    
    torch.testing.assert_close(native_result, expected, rtol=1e-2, atol=1e-2)


def test_dot_product_non_normalized():
    """Test with non-normalized vectors (dot product can exceed 1)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    M = 8
    vec_dim = 128

    # Create non-normalized vectors
    data_np = np.random.randn(M, vec_dim).astype(np.float16) * 2

    data_cuda = torch.from_numpy(data_np).cuda()

    max_pairs = M * (M - 1) // 2
    pair_indices, categories, scores, count = ih.dot_product_cuda(
        data_cuda,
        match_threshold=100.0,  # High threshold
        non_match_threshold=-100.0,
        include_flags=ih.INCLUDE_ALL,
        max_pairs=max_pairs,
    )
    torch.cuda.synchronize()

    n_pairs = count.item()
    assert n_pairs == max_pairs

    # Verify against numpy
    for k in range(n_pairs):
        i, j = pair_indices[k].tolist()
        cuda_score = scores[k].item()
        ref_score = numpy_dot_product(data_np[i], data_np[j])
        # Larger tolerance for non-normalized and larger values
        assert abs(cuda_score - ref_score) < 0.5, (
            f"Pair ({i},{j}): cuda={cuda_score:.4f}, ref={ref_score:.4f}"
        )


# =============================================================================
# Sharded Dot Product Tests
# =============================================================================


def test_dot_product_sharded_basic():
    """Test sharded dot product matches non-sharded version."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    M = 64
    vec_dim = 128

    # Create normalized random data
    data_np = np.random.randn(M, vec_dim).astype(np.float32)
    data_np = (data_np / np.linalg.norm(data_np, axis=1, keepdims=True)).astype(np.float16)
    data = torch.from_numpy(data_np)

    max_pairs = M * (M - 1) // 2

    # Run non-sharded version
    pair_indices1, categories1, scores1, count1 = ih.dot_product_cuda(
        data.cuda(),
        match_threshold=100.0,
        non_match_threshold=-100.0,
        include_flags=ih.INCLUDE_ALL,
        max_pairs=max_pairs,
    )
    torch.cuda.synchronize()

    # Run sharded version with forced sharding (min_shards=4)
    pair_indices2, categories2, scores2, count2 = ih.dot_product_sharded(
        data,  # CPU tensor - will be transferred
        match_threshold=100.0,
        non_match_threshold=-100.0,
        include_flags=ih.INCLUDE_ALL,
        max_pairs=max_pairs,
        min_shards=4,
    )

    n_pairs1 = count1.item()
    n_pairs2 = count2.item()
    assert n_pairs1 == n_pairs2, f"Count mismatch: {n_pairs1} vs {n_pairs2}"

    # Build dictionaries for comparison
    scores_dict1 = {}
    for k in range(n_pairs1):
        i, j = pair_indices1[k].tolist()
        scores_dict1[(i, j)] = scores1[k].item()

    scores_dict2 = {}
    for k in range(n_pairs2):
        i, j = pair_indices2[k].tolist()
        scores_dict2[(i, j)] = scores2[k].item()

    assert set(scores_dict1.keys()) == set(scores_dict2.keys()), "Pair sets differ"

    for key in scores_dict1:
        assert abs(scores_dict1[key] - scores_dict2[key]) < 1e-2, (
            f"Pair {key}: {scores_dict1[key]:.4f} vs {scores_dict2[key]:.4f}"
        )


def test_dot_product_ab_sharded_basic():
    """Test sharded dot_product_ab matches non-sharded version."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    M_A = 32
    M_B = 48
    vec_dim = 128

    # Create normalized random data
    data_a_np = np.random.randn(M_A, vec_dim).astype(np.float32)
    data_a_np = (data_a_np / np.linalg.norm(data_a_np, axis=1, keepdims=True)).astype(np.float16)
    data_b_np = np.random.randn(M_B, vec_dim).astype(np.float32)
    data_b_np = (data_b_np / np.linalg.norm(data_b_np, axis=1, keepdims=True)).astype(np.float16)

    data_a = torch.from_numpy(data_a_np)
    data_b = torch.from_numpy(data_b_np)

    max_pairs = M_A * M_B

    # Run non-sharded version
    pair_indices1, categories1, scores1, count1 = ih.dot_product_ab_cuda(
        data_a.cuda(),
        data_b.cuda(),
        match_threshold=100.0,
        non_match_threshold=-100.0,
        include_flags=ih.INCLUDE_ALL,
        max_pairs=max_pairs,
    )
    torch.cuda.synchronize()

    # Run sharded version with forced sharding (min_shards=4)
    pair_indices2, categories2, scores2, count2 = ih.dot_product_ab_sharded(
        data_a,
        data_b,
        match_threshold=100.0,
        non_match_threshold=-100.0,
        include_flags=ih.INCLUDE_ALL,
        max_pairs=max_pairs,
        min_shards=4,
    )

    n_pairs1 = count1.item()
    n_pairs2 = count2.item()
    assert n_pairs1 == n_pairs2, f"Count mismatch: {n_pairs1} vs {n_pairs2}"

    # Build dictionaries for comparison
    scores_dict1 = {}
    for k in range(n_pairs1):
        i, j = pair_indices1[k].tolist()
        scores_dict1[(i, j)] = scores1[k].item()

    scores_dict2 = {}
    for k in range(n_pairs2):
        i, j = pair_indices2[k].tolist()
        scores_dict2[(i, j)] = scores2[k].item()

    assert set(scores_dict1.keys()) == set(scores_dict2.keys()), "Pair sets differ"

    for key in scores_dict1:
        assert abs(scores_dict1[key] - scores_dict2[key]) < 1e-2, (
            f"Pair {key}: {scores_dict1[key]:.4f} vs {scores_dict2[key]:.4f}"
        )


def test_dot_product_sharded_with_labels():
    """Test sharded dot product with labels for classification."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    M = 64
    vec_dim = 128

    data_np = np.random.randn(M, vec_dim).astype(np.float32)
    data_np = (data_np / np.linalg.norm(data_np, axis=1, keepdims=True)).astype(np.float16)
    data = torch.from_numpy(data_np)

    # Create labels: 4 classes of 16 each
    labels = torch.tensor([i // 16 for i in range(M)], dtype=torch.int32)

    max_pairs = M * (M - 1) // 2

    # Run sharded version
    pair_indices, categories, scores, count = ih.dot_product_sharded(
        data,
        labels=labels,
        match_threshold=0.5,
        non_match_threshold=0.5,
        include_flags=ih.INCLUDE_ALL,
        max_pairs=max_pairs,
        min_shards=4,
    )

    # Verify all pairs are returned
    assert count.item() == max_pairs

    # Verify category assignment
    for k in range(count.item()):
        i, j = pair_indices[k].tolist()
        cat = categories[k].item()
        score = scores[k].item()
        same_label = labels[i].item() == labels[j].item()

        if same_label:
            # Same identity
            if score >= 0.5:
                assert cat == ih.CATEGORY_TRUE_MATCH
            else:
                assert cat == ih.CATEGORY_FALSE_NON_MATCH
        else:
            # Different identity
            if score >= 0.5:
                assert cat == ih.CATEGORY_FALSE_MATCH
            else:
                assert cat == ih.CATEGORY_TRUE_NON_MATCH


def test_dot_product_sharded_large():
    """Test sharded dot product with larger dataset and compare to non-sharded."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    M = 256
    vec_dim = 128

    data_np = np.random.randn(M, vec_dim).astype(np.float32)
    data_np = (data_np / np.linalg.norm(data_np, axis=1, keepdims=True)).astype(np.float16)
    data = torch.from_numpy(data_np)

    max_pairs = M * (M - 1) // 2

    # Run non-sharded version for reference
    pair_indices1, categories1, scores1, count1 = ih.dot_product_cuda(
        data.cuda(),
        match_threshold=100.0,  # Include all pairs
        non_match_threshold=-100.0,
        include_flags=ih.INCLUDE_ALL,
        max_pairs=max_pairs,
    )
    torch.cuda.synchronize()

    # Run sharded version with forced sharding (min_shards=8)
    pair_indices2, categories2, scores2, count2 = ih.dot_product_sharded(
        data,
        match_threshold=100.0,
        non_match_threshold=-100.0,
        include_flags=ih.INCLUDE_ALL,
        max_pairs=max_pairs,
        min_shards=8,
    )

    # Verify same count
    n_pairs1 = count1.item()
    n_pairs2 = count2.item()
    assert n_pairs1 == n_pairs2, f"Count mismatch: {n_pairs1} vs {n_pairs2}"

    # Build dictionaries for comparison (sample a subset for speed)
    scores_dict1 = {}
    for k in range(min(n_pairs1, 1000)):
        i, j = pair_indices1[k].tolist()
        scores_dict1[(i, j)] = scores1[k].item()

    scores_dict2 = {}
    for k in range(n_pairs2):
        i, j = pair_indices2[k].tolist()
        if (i, j) in scores_dict1:
            scores_dict2[(i, j)] = scores2[k].item()

    # Verify scores match
    for key in scores_dict1:
        if key in scores_dict2:
            assert abs(scores_dict1[key] - scores_dict2[key]) < 1e-2, (
                f"Pair {key}: {scores_dict1[key]:.4f} vs {scores_dict2[key]:.4f}"
            )

