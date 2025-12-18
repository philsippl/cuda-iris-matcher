"""Tests for sharding functionality.

These tests verify that:
1. Sharded computation produces identical results to non-sharded
2. All expected pairs are computed (none missed due to tiling)
3. Index offsets are correctly applied
4. Works with both single and multiple shards on a single device
"""

import sys
import os
import numpy as np
import torch
import pytest

# Add parent tests directory to path for utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import cuda_iris_matcher as ih
from utils import rotation_aware_hamming_distance


class TestShardingAB:
    """Tests for masked_hamming_ab_sharded."""

    def test_single_shard_matches_original(self):
        """Verify single shard produces same results as non-sharded version."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        M_A = 20
        M_B = 15
        
        code_a = np.random.randint(0, 2, (M_A, 16, 200, 2, 2), dtype=np.uint8)
        mask_a = np.random.randint(0, 2, (M_A, 16, 200, 2, 2), dtype=np.uint8)
        code_b = np.random.randint(0, 2, (M_B, 16, 200, 2, 2), dtype=np.uint8)
        mask_b = np.random.randint(0, 2, (M_B, 16, 200, 2, 2), dtype=np.uint8)

        # Pack on GPU
        data_a = ih.pack_theta_major(torch.from_numpy(code_a).cuda())
        mask_a_t = ih.pack_theta_major(torch.from_numpy(mask_a).cuda())
        data_b = ih.pack_theta_major(torch.from_numpy(code_b).cuda())
        mask_b_t = ih.pack_theta_major(torch.from_numpy(mask_b).cuda())

        max_pairs = M_A * M_B

        # Non-sharded version
        indices_orig, cat_orig, dist_orig, count_orig = ih.masked_hamming_ab_cuda(
            data_a, mask_a_t, data_b, mask_b_t,
            match_threshold=1.0, non_match_threshold=1.0,
            include_flags=ih.INCLUDE_ALL, max_pairs=max_pairs
        )
        torch.cuda.synchronize()

        # Sharded version with min_shards=1 (should be equivalent)
        indices_sharded, cat_sharded, dist_sharded, count_sharded = ih.masked_hamming_ab_sharded(
            data_a, mask_a_t, data_b, mask_b_t,
            match_threshold=1.0, non_match_threshold=1.0,
            include_flags=ih.INCLUDE_ALL, max_pairs=max_pairs,
            min_shards=1
        )

        # Both should return the same count
        assert count_orig.item() == count_sharded.item() == max_pairs

        # Convert to sets of (i, j, dist) for comparison (order may differ)
        orig_set = set()
        for k in range(count_orig.item()):
            i, j = indices_orig[k].tolist()
            d = round(dist_orig[k].item(), 5)
            orig_set.add((i, j, d))

        sharded_set = set()
        for k in range(count_sharded.item()):
            i, j = indices_sharded[k].tolist()
            d = round(dist_sharded[k].item(), 5)
            sharded_set.add((i, j, d))

        assert orig_set == sharded_set, "Sharded results differ from original"

    def test_multiple_shards_complete_coverage(self):
        """Verify that multiple shards cover all pairs correctly."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        M_A = 24
        M_B = 18

        code_a = np.random.randint(0, 2, (M_A, 16, 200, 2, 2), dtype=np.uint8)
        mask_a = np.random.randint(0, 2, (M_A, 16, 200, 2, 2), dtype=np.uint8)
        code_b = np.random.randint(0, 2, (M_B, 16, 200, 2, 2), dtype=np.uint8)
        mask_b = np.random.randint(0, 2, (M_B, 16, 200, 2, 2), dtype=np.uint8)

        data_a = ih.pack_theta_major(torch.from_numpy(code_a).cuda())
        mask_a_t = ih.pack_theta_major(torch.from_numpy(mask_a).cuda())
        data_b = ih.pack_theta_major(torch.from_numpy(code_b).cuda())
        mask_b_t = ih.pack_theta_major(torch.from_numpy(mask_b).cuda())

        max_pairs = M_A * M_B

        # Force multiple shards with small tile size
        indices, categories, distances, count = ih.masked_hamming_ab_sharded(
            data_a, mask_a_t, data_b, mask_b_t,
            match_threshold=1.0, non_match_threshold=1.0,
            include_flags=ih.INCLUDE_ALL, max_pairs=max_pairs,
            min_shards=4,  # Force at least 4 shards
            max_tile_size=10  # Small tiles to force more shards
        )

        # Verify all pairs are present
        assert count.item() == max_pairs, f"Expected {max_pairs} pairs, got {count.item()}"

        # Verify all (i, j) combinations are present with correct indices
        found_pairs = set()
        for k in range(count.item()):
            i, j = indices[k].tolist()
            assert 0 <= i < M_A, f"Invalid row index: {i}"
            assert 0 <= j < M_B, f"Invalid col index: {j}"
            found_pairs.add((i, j))

        expected_pairs = {(i, j) for i in range(M_A) for j in range(M_B)}
        assert found_pairs == expected_pairs, "Not all pairs were computed"

    def test_sharded_matches_numpy_reference(self):
        """Verify sharded results match NumPy reference implementation."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        M_A = 12
        M_B = 10

        np.random.seed(42)
        code_a = np.random.randint(0, 2, (M_A, 16, 200, 2, 2), dtype=np.uint8)
        mask_a = np.random.randint(0, 2, (M_A, 16, 200, 2, 2), dtype=np.uint8)
        code_b = np.random.randint(0, 2, (M_B, 16, 200, 2, 2), dtype=np.uint8)
        mask_b = np.random.randint(0, 2, (M_B, 16, 200, 2, 2), dtype=np.uint8)

        data_a = ih.pack_theta_major(torch.from_numpy(code_a).cuda())
        mask_a_t = ih.pack_theta_major(torch.from_numpy(mask_a).cuda())
        data_b = ih.pack_theta_major(torch.from_numpy(code_b).cuda())
        mask_b_t = ih.pack_theta_major(torch.from_numpy(mask_b).cuda())

        max_pairs = M_A * M_B

        # Run sharded version with multiple shards
        indices, categories, distances, count = ih.masked_hamming_ab_sharded(
            data_a, mask_a_t, data_b, mask_b_t,
            match_threshold=1.0, non_match_threshold=1.0,
            include_flags=ih.INCLUDE_ALL, max_pairs=max_pairs,
            min_shards=6  # Force multiple shards
        )

        # Compute reference with NumPy
        ref = {}
        for i in range(M_A):
            for j in range(M_B):
                dist, _ = rotation_aware_hamming_distance(
                    code_a[i], mask_a[i], code_b[j], mask_b[j]
                )
                ref[(i, j)] = dist

        # Verify each pair
        for k in range(count.item()):
            i, j = indices[k].tolist()
            cuda_dist = distances[k].item()
            ref_dist = ref[(i, j)]
            assert abs(cuda_dist - ref_dist) <= 1e-5, \
                f"Pair ({i},{j}): cuda={cuda_dist}, ref={ref_dist}"

    def test_input_from_cpu(self):
        """Test that sharding works with CPU input tensors."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        M_A = 16
        M_B = 12

        code_a = np.random.randint(0, 2, (M_A, 16, 200, 2, 2), dtype=np.uint8)
        mask_a = np.random.randint(0, 2, (M_A, 16, 200, 2, 2), dtype=np.uint8)
        code_b = np.random.randint(0, 2, (M_B, 16, 200, 2, 2), dtype=np.uint8)
        mask_b = np.random.randint(0, 2, (M_B, 16, 200, 2, 2), dtype=np.uint8)

        # Pack on GPU then move to CPU
        data_a = ih.pack_theta_major(torch.from_numpy(code_a).cuda()).cpu()
        mask_a_t = ih.pack_theta_major(torch.from_numpy(mask_a).cuda()).cpu()
        data_b = ih.pack_theta_major(torch.from_numpy(code_b).cuda()).cpu()
        mask_b_t = ih.pack_theta_major(torch.from_numpy(mask_b).cuda()).cpu()

        max_pairs = M_A * M_B

        # Should work with CPU input
        indices, categories, distances, count = ih.masked_hamming_ab_sharded(
            data_a, mask_a_t, data_b, mask_b_t,
            match_threshold=1.0, non_match_threshold=1.0,
            include_flags=ih.INCLUDE_ALL, max_pairs=max_pairs,
            min_shards=4
        )

        assert count.item() == max_pairs


class TestShardingSelf:
    """Tests for masked_hamming_sharded (self-comparison)."""

    def test_single_shard_matches_original(self):
        """Verify single shard produces same results as non-sharded version."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        M = 25

        code = np.random.randint(0, 2, (M, 16, 200, 2, 2), dtype=np.uint8)
        mask = np.random.randint(0, 2, (M, 16, 200, 2, 2), dtype=np.uint8)

        data = ih.pack_theta_major(torch.from_numpy(code).cuda())
        mask_t = ih.pack_theta_major(torch.from_numpy(mask).cuda())

        max_pairs = M * (M - 1) // 2

        # Non-sharded version
        indices_orig, cat_orig, dist_orig, count_orig = ih.masked_hamming_cuda(
            data, mask_t,
            match_threshold=1.0, non_match_threshold=1.0,
            include_flags=ih.INCLUDE_ALL, max_pairs=max_pairs
        )
        torch.cuda.synchronize()

        # Sharded version with min_shards=1
        indices_sharded, cat_sharded, dist_sharded, count_sharded = ih.masked_hamming_sharded(
            data, mask_t,
            match_threshold=1.0, non_match_threshold=1.0,
            include_flags=ih.INCLUDE_ALL, max_pairs=max_pairs,
            min_shards=1
        )

        # Both should return the same count
        assert count_orig.item() == count_sharded.item() == max_pairs

        # Convert to sets for comparison
        orig_set = set()
        for k in range(count_orig.item()):
            i, j = indices_orig[k].tolist()
            d = round(dist_orig[k].item(), 5)
            orig_set.add((i, j, d))

        sharded_set = set()
        for k in range(count_sharded.item()):
            i, j = indices_sharded[k].tolist()
            d = round(dist_sharded[k].item(), 5)
            sharded_set.add((i, j, d))

        assert orig_set == sharded_set, "Sharded results differ from original"

    def test_multiple_shards_lower_triangle_only(self):
        """Verify that sharded self-comparison only computes lower triangle (i > j)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        M = 30

        code = np.random.randint(0, 2, (M, 16, 200, 2, 2), dtype=np.uint8)
        mask = np.random.randint(0, 2, (M, 16, 200, 2, 2), dtype=np.uint8)

        data = ih.pack_theta_major(torch.from_numpy(code).cuda())
        mask_t = ih.pack_theta_major(torch.from_numpy(mask).cuda())

        expected_pairs = M * (M - 1) // 2

        # Force multiple shards
        indices, categories, distances, count = ih.masked_hamming_sharded(
            data, mask_t,
            match_threshold=1.0, non_match_threshold=1.0,
            include_flags=ih.INCLUDE_ALL, max_pairs=expected_pairs * 2,
            min_shards=6,
            max_tile_size=10
        )

        # Verify count
        assert count.item() == expected_pairs, \
            f"Expected {expected_pairs} pairs (lower triangle), got {count.item()}"

        # Verify all pairs satisfy i > j (lower triangle)
        found_pairs = set()
        for k in range(count.item()):
            i, j = indices[k].tolist()
            assert i > j, f"Pair ({i},{j}) violates lower triangle constraint"
            assert 0 <= i < M, f"Invalid row index: {i}"
            assert 0 <= j < M, f"Invalid col index: {j}"
            found_pairs.add((i, j))

        # Verify all expected pairs are present
        expected_set = {(i, j) for i in range(M) for j in range(i)}
        assert found_pairs == expected_set, "Not all lower triangle pairs were computed"

    def test_sharded_matches_numpy_reference(self):
        """Verify sharded self-comparison matches NumPy reference."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        M = 16

        np.random.seed(123)
        code = np.random.randint(0, 2, (M, 16, 200, 2, 2), dtype=np.uint8)
        mask = np.random.randint(0, 2, (M, 16, 200, 2, 2), dtype=np.uint8)

        data = ih.pack_theta_major(torch.from_numpy(code).cuda())
        mask_t = ih.pack_theta_major(torch.from_numpy(mask).cuda())

        max_pairs = M * (M - 1) // 2

        # Run sharded version
        indices, categories, distances, count = ih.masked_hamming_sharded(
            data, mask_t,
            match_threshold=1.0, non_match_threshold=1.0,
            include_flags=ih.INCLUDE_ALL, max_pairs=max_pairs,
            min_shards=4
        )

        # Compute reference
        ref = {}
        for i in range(M):
            for j in range(i):
                dist, _ = rotation_aware_hamming_distance(code[i], mask[i], code[j], mask[j])
                ref[(i, j)] = dist

        # Verify each pair
        for k in range(count.item()):
            i, j = indices[k].tolist()
            cuda_dist = distances[k].item()
            ref_dist = ref[(i, j)]
            assert abs(cuda_dist - ref_dist) <= 1e-5, \
                f"Pair ({i},{j}): cuda={cuda_dist}, ref={ref_dist}"

    def test_diagonal_tile_filtering(self):
        """Test that diagonal tiles correctly filter to lower triangle."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Use size that creates diagonal tiles
        M = 20

        code = np.random.randint(0, 2, (M, 16, 200, 2, 2), dtype=np.uint8)
        mask = np.ones((M, 16, 200, 2, 2), dtype=np.uint8)  # Full mask

        data = ih.pack_theta_major(torch.from_numpy(code).cuda())
        mask_t = ih.pack_theta_major(torch.from_numpy(mask).cuda())

        # Force small tiles to ensure diagonal tiles
        indices, categories, distances, count = ih.masked_hamming_sharded(
            data, mask_t,
            match_threshold=1.0, non_match_threshold=1.0,
            include_flags=ih.INCLUDE_ALL, max_pairs=M * M,
            min_shards=9,  # 3x3 grid of tiles
            max_tile_size=7
        )

        expected_pairs = M * (M - 1) // 2
        assert count.item() == expected_pairs

        # No duplicates
        pairs_list = [(indices[k, 0].item(), indices[k, 1].item()) for k in range(count.item())]
        assert len(pairs_list) == len(set(pairs_list)), "Duplicate pairs found"


class TestShardingWithLabels:
    """Tests for sharding with label-based classification."""

    def test_ab_sharded_with_labels(self):
        """Test A vs B sharding with labels."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        M_A = 16
        M_B = 12

        code_a = np.random.randint(0, 2, (M_A, 16, 200, 2, 2), dtype=np.uint8)
        mask_a = np.ones((M_A, 16, 200, 2, 2), dtype=np.uint8)
        code_b = np.random.randint(0, 2, (M_B, 16, 200, 2, 2), dtype=np.uint8)
        mask_b = np.ones((M_B, 16, 200, 2, 2), dtype=np.uint8)

        # Labels: first half same label, second half different
        labels_a = torch.zeros(M_A, dtype=torch.int32)
        labels_a[M_A // 2:] = 1
        labels_b = torch.zeros(M_B, dtype=torch.int32)
        labels_b[M_B // 2:] = 1

        data_a = ih.pack_theta_major(torch.from_numpy(code_a).cuda())
        mask_a_t = ih.pack_theta_major(torch.from_numpy(mask_a).cuda())
        data_b = ih.pack_theta_major(torch.from_numpy(code_b).cuda())
        mask_b_t = ih.pack_theta_major(torch.from_numpy(mask_b).cuda())

        # Run with labels - filter to only true matches/non-matches
        indices, categories, distances, count = ih.masked_hamming_ab_sharded(
            data_a, mask_a_t, data_b, mask_b_t,
            labels_a=labels_a.cuda(), labels_b=labels_b.cuda(),
            match_threshold=0.35, non_match_threshold=0.35,
            include_flags=ih.INCLUDE_TM | ih.INCLUDE_TNM,
            max_pairs=M_A * M_B,
            min_shards=4
        )

        # Verify categories are correctly assigned
        for k in range(count.item()):
            i, j = indices[k].tolist()
            cat = categories[k].item()
            same_label = (labels_a[i].item() == labels_b[j].item())
            dist = distances[k].item()

            if same_label:
                assert cat in [ih.CATEGORY_TRUE_MATCH, ih.CATEGORY_FALSE_NON_MATCH], \
                    f"Same label pair ({i},{j}) has wrong category {cat}"
            else:
                assert cat in [ih.CATEGORY_FALSE_MATCH, ih.CATEGORY_TRUE_NON_MATCH], \
                    f"Different label pair ({i},{j}) has wrong category {cat}"

    def test_self_sharded_with_labels(self):
        """Test self-comparison sharding with labels."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        M = 20

        code = np.random.randint(0, 2, (M, 16, 200, 2, 2), dtype=np.uint8)
        mask = np.ones((M, 16, 200, 2, 2), dtype=np.uint8)

        # Labels: alternating
        labels = torch.arange(M, dtype=torch.int32) % 4

        data = ih.pack_theta_major(torch.from_numpy(code).cuda())
        mask_t = ih.pack_theta_major(torch.from_numpy(mask).cuda())

        indices, categories, distances, count = ih.masked_hamming_sharded(
            data, mask_t,
            labels=labels.cuda(),
            match_threshold=0.35, non_match_threshold=0.35,
            include_flags=ih.INCLUDE_ALL,
            max_pairs=M * M,
            min_shards=4
        )

        # Verify label consistency
        for k in range(count.item()):
            i, j = indices[k].tolist()
            cat = categories[k].item()
            same_label = (labels[i].item() == labels[j].item())

            if cat == ih.CATEGORY_TRUE_MATCH or cat == ih.CATEGORY_FALSE_NON_MATCH:
                assert same_label, f"Category {cat} should have same labels"
            elif cat == ih.CATEGORY_FALSE_MATCH or cat == ih.CATEGORY_TRUE_NON_MATCH:
                assert not same_label, f"Category {cat} should have different labels"


class TestShardInfo:
    """Tests for shard configuration inspection."""

    def test_get_shard_info(self):
        """Test shard info helper function."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        shards = ih.get_shard_info(
            m_a=100, m_b=80,
            min_shards=4,
            max_tile_size=50
        )

        # Should have at least 4 shards
        assert len(shards) >= 4

        # Verify shards cover the full range
        a_covered = set()
        b_covered = set()
        for shard in shards:
            for i in range(shard.a_start, shard.a_end):
                a_covered.add(i)
            for j in range(shard.b_start, shard.b_end):
                b_covered.add(j)

        assert a_covered == set(range(100)), "Not all A indices covered"
        assert b_covered == set(range(80)), "Not all B indices covered"

    def test_device_count(self):
        """Test device count function."""
        count = ih.get_device_count()
        assert count >= 0
        if torch.cuda.is_available():
            assert count >= 1


class TestBatchPacking:
    """Tests for batch packing when data doesn't fit on GPU."""

    def test_batch_packing_matches_regular(self):
        """Verify batch packing produces same results as regular packing."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        M = 100
        code = np.random.randint(0, 2, (M, 16, 200, 2, 2), dtype=np.uint8)

        # Regular packing (all on GPU at once)
        code_gpu = torch.from_numpy(code).cuda()
        packed_regular = ih.pack_theta_major(code_gpu).cpu()

        # Batch packing (from CPU, in batches)
        code_cpu = torch.from_numpy(code)
        packed_batched = ih.pack_theta_major_batched(code_cpu, batch_size=30)

        assert packed_regular.shape == packed_batched.shape
        assert torch.equal(packed_regular, packed_batched)

    def test_batch_packing_small_batches(self):
        """Test with very small batch sizes."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        M = 50
        code = np.random.randint(0, 2, (M, 16, 200, 2, 2), dtype=np.uint8)

        # Regular packing
        code_gpu = torch.from_numpy(code).cuda()
        packed_regular = ih.pack_theta_major(code_gpu).cpu()

        # Batch packing with tiny batches
        code_cpu = torch.from_numpy(code)
        packed_batched = ih.pack_theta_major_batched(code_cpu, batch_size=7)

        assert torch.equal(packed_regular, packed_batched)

    def test_batch_packing_auto_batch_size(self):
        """Test that auto batch size works."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        M = 64
        code = np.random.randint(0, 2, (M, 16, 200, 2, 2), dtype=np.uint8)

        # Batch packing with auto batch size
        code_cpu = torch.from_numpy(code)
        packed = ih.pack_theta_major_batched(code_cpu)  # No batch_size specified

        assert packed.shape == (M, 400)
        assert packed.dtype == torch.int32

    def test_batch_packing_gpu_input(self):
        """Test that GPU input is handled correctly (no batching needed)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        M = 32
        code = np.random.randint(0, 2, (M, 16, 200, 2, 2), dtype=np.uint8)

        # Regular packing (pack_theta_major is in-place, so use fresh copy)
        code_gpu1 = torch.from_numpy(code.copy()).cuda()
        packed_regular = ih.pack_theta_major(code_gpu1).cpu()

        # Batch packing with GPU input (should clone internally since in-place)
        code_gpu2 = torch.from_numpy(code.copy()).cuda()
        packed_batched = ih.pack_theta_major_batched(code_gpu2)

        assert torch.equal(packed_regular, packed_batched)


class TestEdgeCases:
    """Edge case tests for sharding."""

    def test_single_element_matrices(self):
        """Test with very small matrices."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        code_a = np.random.randint(0, 2, (1, 16, 200, 2, 2), dtype=np.uint8)
        mask_a = np.ones((1, 16, 200, 2, 2), dtype=np.uint8)
        code_b = np.random.randint(0, 2, (1, 16, 200, 2, 2), dtype=np.uint8)
        mask_b = np.ones((1, 16, 200, 2, 2), dtype=np.uint8)

        data_a = ih.pack_theta_major(torch.from_numpy(code_a).cuda())
        mask_a_t = ih.pack_theta_major(torch.from_numpy(mask_a).cuda())
        data_b = ih.pack_theta_major(torch.from_numpy(code_b).cuda())
        mask_b_t = ih.pack_theta_major(torch.from_numpy(mask_b).cuda())

        indices, categories, distances, count = ih.masked_hamming_ab_sharded(
            data_a, mask_a_t, data_b, mask_b_t,
            match_threshold=1.0, non_match_threshold=1.0,
            include_flags=ih.INCLUDE_ALL, max_pairs=10,
            min_shards=1
        )

        assert count.item() == 1  # 1x1 matrix
        assert indices[0].tolist() == [0, 0]

    def test_asymmetric_sizes(self):
        """Test with very different A and B sizes."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        M_A = 5
        M_B = 50

        code_a = np.random.randint(0, 2, (M_A, 16, 200, 2, 2), dtype=np.uint8)
        mask_a = np.ones((M_A, 16, 200, 2, 2), dtype=np.uint8)
        code_b = np.random.randint(0, 2, (M_B, 16, 200, 2, 2), dtype=np.uint8)
        mask_b = np.ones((M_B, 16, 200, 2, 2), dtype=np.uint8)

        data_a = ih.pack_theta_major(torch.from_numpy(code_a).cuda())
        mask_a_t = ih.pack_theta_major(torch.from_numpy(mask_a).cuda())
        data_b = ih.pack_theta_major(torch.from_numpy(code_b).cuda())
        mask_b_t = ih.pack_theta_major(torch.from_numpy(mask_b).cuda())

        indices, categories, distances, count = ih.masked_hamming_ab_sharded(
            data_a, mask_a_t, data_b, mask_b_t,
            match_threshold=1.0, non_match_threshold=1.0,
            include_flags=ih.INCLUDE_ALL, max_pairs=M_A * M_B,
            min_shards=5
        )

        assert count.item() == M_A * M_B

    def test_max_pairs_limit(self):
        """Test that max_pairs is approximately respected across shards.
        
        Note: Since shards are processed independently and we can't partially
        take results from a shard, the actual count may slightly exceed max_pairs
        in some cases. This test verifies we're reasonably close to the limit.
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        M_A = 20
        M_B = 20
        max_pairs = 200  # Half of M_A * M_B = 400

        code_a = np.random.randint(0, 2, (M_A, 16, 200, 2, 2), dtype=np.uint8)
        mask_a = np.ones((M_A, 16, 200, 2, 2), dtype=np.uint8)
        code_b = np.random.randint(0, 2, (M_B, 16, 200, 2, 2), dtype=np.uint8)
        mask_b = np.ones((M_B, 16, 200, 2, 2), dtype=np.uint8)

        data_a = ih.pack_theta_major(torch.from_numpy(code_a).cuda())
        mask_a_t = ih.pack_theta_major(torch.from_numpy(mask_a).cuda())
        data_b = ih.pack_theta_major(torch.from_numpy(code_b).cuda())
        mask_b_t = ih.pack_theta_major(torch.from_numpy(mask_b).cuda())

        indices, categories, distances, count = ih.masked_hamming_ab_sharded(
            data_a, mask_a_t, data_b, mask_b_t,
            match_threshold=1.0, non_match_threshold=1.0,
            include_flags=ih.INCLUDE_ALL, max_pairs=max_pairs,
            min_shards=4
        )

        # Should be less than total possible pairs (since we limited it)
        total_possible = M_A * M_B
        assert count.item() < total_possible, \
            f"Count {count.item()} should be less than total {total_possible}"
        # Should not be way over the limit (allow some slack for shard boundaries)
        assert count.item() <= max_pairs * 2, \
            f"Count {count.item()} significantly exceeds max_pairs {max_pairs}"


class TestAutoPackingNonSharded:
    """Tests for automatic packing on non-sharded functions."""

    def test_masked_hamming_cuda_with_unpacked_cpu(self):
        """Test masked_hamming_cuda can accept unpacked CPU data."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        M = 25
        code = np.random.randint(0, 2, (M, 16, 200, 2, 2), dtype=np.uint8)
        mask = np.ones((M, 16, 200, 2, 2), dtype=np.uint8)

        # Create unpacked tensors on CPU
        code_cpu = torch.from_numpy(code)
        mask_cpu = torch.from_numpy(mask)

        # Run with unpacked data - should auto-pack and move to GPU
        indices, categories, distances, count = ih.masked_hamming_cuda(
            code_cpu, mask_cpu,
            match_threshold=1.0, non_match_threshold=1.0,
            include_flags=ih.INCLUDE_ALL, max_pairs=1000
        )

        # Verify results are correct (lower triangle)
        n_pairs = count.item()
        assert n_pairs > 0, "Should have found some pairs"
        assert n_pairs <= M * (M - 1) // 2, "Should be lower triangle only"
        # All pairs should have row > col
        for i in range(n_pairs):
            assert indices[i, 0] > indices[i, 1], f"Pair {i} not in lower triangle"

    def test_masked_hamming_cuda_with_unpacked_gpu(self):
        """Test masked_hamming_cuda can accept unpacked GPU data."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        M = 25
        code = np.random.randint(0, 2, (M, 16, 200, 2, 2), dtype=np.uint8)
        mask = np.ones((M, 16, 200, 2, 2), dtype=np.uint8)

        # Create unpacked tensors on GPU
        code_gpu = torch.from_numpy(code).cuda()
        mask_gpu = torch.from_numpy(mask).cuda()

        # Run with unpacked GPU data - should auto-pack in place
        indices, categories, distances, count = ih.masked_hamming_cuda(
            code_gpu, mask_gpu,
            match_threshold=1.0, non_match_threshold=1.0,
            include_flags=ih.INCLUDE_ALL, max_pairs=1000
        )

        # Verify results are correct
        n_pairs = count.item()
        assert n_pairs > 0, "Should have found some pairs"
        assert n_pairs <= M * (M - 1) // 2, "Should be lower triangle only"

    def test_masked_hamming_cuda_matches_packed(self):
        """Test unpacked input produces same results as pre-packed input."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        M = 25
        code = np.random.randint(0, 2, (M, 16, 200, 2, 2), dtype=np.uint8)
        mask = np.ones((M, 16, 200, 2, 2), dtype=np.uint8)

        # Pre-packed path
        code_gpu = torch.from_numpy(code).cuda()
        mask_gpu = torch.from_numpy(mask).cuda()
        packed_code = ih.pack_theta_major(code_gpu.clone())
        packed_mask = ih.pack_theta_major(mask_gpu.clone())
        
        indices1, cat1, dist1, count1 = ih.masked_hamming_cuda(
            packed_code, packed_mask,
            match_threshold=1.0, non_match_threshold=1.0,
            include_flags=ih.INCLUDE_ALL, max_pairs=10000
        )

        # Unpacked path (fresh copy)
        code_cpu = torch.from_numpy(code)
        mask_cpu = torch.from_numpy(mask)
        
        indices2, cat2, dist2, count2 = ih.masked_hamming_cuda(
            code_cpu, mask_cpu,
            match_threshold=1.0, non_match_threshold=1.0,
            include_flags=ih.INCLUDE_ALL, max_pairs=10000
        )

        # Should get same results
        assert count1.item() == count2.item(), f"Counts differ: {count1.item()} vs {count2.item()}"
        n = count1.item()
        
        # Convert to numpy for sorting
        idx1 = indices1[:n].cpu().numpy()
        idx2 = indices2[:n].cpu().numpy()
        dist1_np = dist1[:n].cpu().numpy()
        dist2_np = dist2[:n].cpu().numpy()
        
        # Sort by (row, col) for comparison
        order1 = np.lexsort((idx1[:, 1], idx1[:, 0]))
        order2 = np.lexsort((idx2[:, 1], idx2[:, 0]))
        
        np.testing.assert_array_equal(idx1[order1], idx2[order2], "Indices differ")
        np.testing.assert_allclose(dist1_np[order1], dist2_np[order2], err_msg="Distances differ")

    def test_masked_hamming_ab_cuda_with_unpacked(self):
        """Test masked_hamming_ab_cuda can accept unpacked data."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        M_A, M_B = 15, 20
        code_a = np.random.randint(0, 2, (M_A, 16, 200, 2, 2), dtype=np.uint8)
        mask_a = np.ones((M_A, 16, 200, 2, 2), dtype=np.uint8)
        code_b = np.random.randint(0, 2, (M_B, 16, 200, 2, 2), dtype=np.uint8)
        mask_b = np.ones((M_B, 16, 200, 2, 2), dtype=np.uint8)

        # Create unpacked tensors on CPU
        code_a_cpu = torch.from_numpy(code_a)
        mask_a_cpu = torch.from_numpy(mask_a)
        code_b_cpu = torch.from_numpy(code_b)
        mask_b_cpu = torch.from_numpy(mask_b)

        # Run with unpacked data - should auto-pack and move to GPU
        indices, categories, distances, count = ih.masked_hamming_ab_cuda(
            code_a_cpu, mask_a_cpu, code_b_cpu, mask_b_cpu,
            match_threshold=1.0, non_match_threshold=1.0,
            include_flags=ih.INCLUDE_ALL, max_pairs=1000
        )

        # Verify results
        n_pairs = count.item()
        assert n_pairs > 0, "Should have found some pairs"
        assert n_pairs <= M_A * M_B, "Should not exceed total pairs"

    def test_masked_hamming_ab_cuda_matches_packed(self):
        """Test AB unpacked input produces same results as pre-packed input."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        M_A, M_B = 15, 20
        code_a = np.random.randint(0, 2, (M_A, 16, 200, 2, 2), dtype=np.uint8)
        mask_a = np.ones((M_A, 16, 200, 2, 2), dtype=np.uint8)
        code_b = np.random.randint(0, 2, (M_B, 16, 200, 2, 2), dtype=np.uint8)
        mask_b = np.ones((M_B, 16, 200, 2, 2), dtype=np.uint8)

        # Pre-packed path
        packed_a = ih.pack_theta_major(torch.from_numpy(code_a).cuda().clone())
        packed_mask_a = ih.pack_theta_major(torch.from_numpy(mask_a).cuda().clone())
        packed_b = ih.pack_theta_major(torch.from_numpy(code_b).cuda().clone())
        packed_mask_b = ih.pack_theta_major(torch.from_numpy(mask_b).cuda().clone())
        
        indices1, cat1, dist1, count1 = ih.masked_hamming_ab_cuda(
            packed_a, packed_mask_a, packed_b, packed_mask_b,
            match_threshold=1.0, non_match_threshold=1.0,
            include_flags=ih.INCLUDE_ALL, max_pairs=10000
        )

        # Unpacked path (fresh copy from numpy)
        code_a_cpu = torch.from_numpy(code_a)
        mask_a_cpu = torch.from_numpy(mask_a)
        code_b_cpu = torch.from_numpy(code_b)
        mask_b_cpu = torch.from_numpy(mask_b)
        
        indices2, cat2, dist2, count2 = ih.masked_hamming_ab_cuda(
            code_a_cpu, mask_a_cpu, code_b_cpu, mask_b_cpu,
            match_threshold=1.0, non_match_threshold=1.0,
            include_flags=ih.INCLUDE_ALL, max_pairs=10000
        )

        # Should get same results
        assert count1.item() == count2.item(), f"Counts differ: {count1.item()} vs {count2.item()}"
        n = count1.item()
        
        # Convert to numpy for sorting
        idx1 = indices1[:n].cpu().numpy()
        idx2 = indices2[:n].cpu().numpy()
        dist1_np = dist1[:n].cpu().numpy()
        dist2_np = dist2[:n].cpu().numpy()
        
        # Sort by (row, col) for comparison
        order1 = np.lexsort((idx1[:, 1], idx1[:, 0]))
        order2 = np.lexsort((idx2[:, 1], idx2[:, 0]))
        
        np.testing.assert_array_equal(idx1[order1], idx2[order2], "Indices differ")
        np.testing.assert_allclose(dist1_np[order1], dist2_np[order2], err_msg="Distances differ")


class TestMultiHostSharding:
    """Tests for multi-host sharding support."""

    def test_shard_distribution_across_hosts(self):
        """Verify shards are evenly distributed across hosts."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        M = 1000
        num_hosts = 3

        # Get shards for each host
        all_global_ids = set()
        for host_idx in range(num_hosts):
            shards = ih.get_self_shard_info(
                m=M, min_shards=6, host_index=host_idx, num_hosts=num_hosts
            )
            # Check no duplicate global IDs
            for s in shards:
                assert s.global_shard_id not in all_global_ids, \
                    f"Duplicate global_shard_id {s.global_shard_id}"
                all_global_ids.add(s.global_shard_id)
                # Check device ID is valid for local host
                assert 0 <= s.device_id < torch.cuda.device_count(), \
                    f"Invalid device_id {s.device_id}"

        # Verify total shards matches expected
        total = ih.get_total_shards(m_a=M, num_hosts=num_hosts, min_shards=6)
        assert len(all_global_ids) == total, \
            f"Collected {len(all_global_ids)} shards but expected {total}"

    def test_multi_host_self_results_combine(self):
        """Verify multi-host self-comparison results combine correctly."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        M = 200  # Smaller for faster test
        num_hosts = 2
        # Use very high max_pairs to avoid per-shard limits affecting results
        max_pairs = 1000000

        # Generate test data
        data = torch.randint(0, 2**31, (M, 400), dtype=torch.int32)
        mask = torch.full((M, 400), 0x7FFFFFFF, dtype=torch.int32)

        # Run single-host version (baseline) - use same min_shards*num_hosts
        # to get same tiling as multi-host
        idx_single, cat_single, dist_single, cnt_single = ih.masked_hamming_sharded(
            data, mask, max_pairs=max_pairs, min_shards=4 * num_hosts
        )
        n_single = cnt_single.item()

        # Run multi-host version and combine
        all_indices = []
        all_distances = []
        total_count = 0
        for host_idx in range(num_hosts):
            idx, cat, dist, cnt = ih.masked_hamming_sharded(
                data, mask,
                max_pairs=max_pairs,
                min_shards=4,
                host_index=host_idx,
                num_hosts=num_hosts
            )
            n = cnt.item()
            if n > 0:
                all_indices.append(idx[:n])
                all_distances.append(dist[:n])
                total_count += n

        # Combined count should match single-host count
        assert total_count == n_single, \
            f"Multi-host total {total_count} != single-host {n_single}"

        # Combined results should have same pairs
        if n_single > 0:
            combined_indices = torch.cat(all_indices, dim=0)
            combined_distances = torch.cat(all_distances, dim=0)

            # Sort both by (row, col) for comparison
            single_order = torch.argsort(
                idx_single[:n_single, 0] * M + idx_single[:n_single, 1]
            )
            combined_order = torch.argsort(
                combined_indices[:, 0] * M + combined_indices[:, 1]
            )

            torch.testing.assert_close(
                idx_single[:n_single][single_order],
                combined_indices[combined_order],
                msg="Indices differ between single-host and multi-host"
            )
            torch.testing.assert_close(
                dist_single[:n_single][single_order],
                combined_distances[combined_order],
                msg="Distances differ between single-host and multi-host"
            )

    def test_multi_host_ab_results_combine(self):
        """Verify multi-host AB comparison results combine correctly."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        M_A, M_B = 300, 200
        num_hosts = 2

        # Generate test data
        data_a = torch.randint(0, 2**31, (M_A, 400), dtype=torch.int32)
        mask_a = torch.full((M_A, 400), 0x7FFFFFFF, dtype=torch.int32)
        data_b = torch.randint(0, 2**31, (M_B, 400), dtype=torch.int32)
        mask_b = torch.full((M_B, 400), 0x7FFFFFFF, dtype=torch.int32)

        # Run single-host version (baseline)
        idx_single, cat_single, dist_single, cnt_single = ih.masked_hamming_ab_sharded(
            data_a, mask_a, data_b, mask_b, max_pairs=100000, min_shards=4
        )
        n_single = cnt_single.item()

        # Run multi-host version and combine
        all_indices = []
        all_distances = []
        total_count = 0
        for host_idx in range(num_hosts):
            idx, cat, dist, cnt = ih.masked_hamming_ab_sharded(
                data_a, mask_a, data_b, mask_b,
                max_pairs=100000,
                min_shards=4,
                host_index=host_idx,
                num_hosts=num_hosts
            )
            n = cnt.item()
            if n > 0:
                all_indices.append(idx[:n])
                all_distances.append(dist[:n])
                total_count += n

        # Combined count should match single-host count
        assert total_count == n_single, \
            f"Multi-host total {total_count} != single-host {n_single}"

        # Combined results should have same pairs
        if n_single > 0:
            combined_indices = torch.cat(all_indices, dim=0)
            combined_distances = torch.cat(all_distances, dim=0)

            # Sort both by (row_a, row_b) for comparison
            single_order = torch.argsort(
                idx_single[:n_single, 0] * M_B + idx_single[:n_single, 1]
            )
            combined_order = torch.argsort(
                combined_indices[:, 0] * M_B + combined_indices[:, 1]
            )

            torch.testing.assert_close(
                idx_single[:n_single][single_order],
                combined_indices[combined_order],
                msg="Indices differ between single-host and multi-host"
            )
            torch.testing.assert_close(
                dist_single[:n_single][single_order],
                combined_distances[combined_order],
                msg="Distances differ between single-host and multi-host"
            )

    def test_empty_host_returns_empty(self):
        """Verify host with no assigned shards returns empty results."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        M = 100
        # Use very few shards so some hosts get nothing when num_hosts is large
        data = torch.randint(0, 2**31, (M, 400), dtype=torch.int32)
        mask = torch.full((M, 400), 0x7FFFFFFF, dtype=torch.int32)

        # With min_shards=1 and num_hosts=10, some hosts may get 0 shards
        idx, cat, dist, cnt = ih.masked_hamming_sharded(
            data, mask,
            max_pairs=100,
            min_shards=1,
            host_index=9,  # High index
            num_hosts=10   # Many hosts
        )

        # Should return valid (possibly empty) tensors
        assert idx.shape[1] == 2
        assert cnt.shape == (1,)
        # Count should be >= 0
        assert cnt.item() >= 0

    def test_deterministic_shard_assignment(self):
        """Verify shard assignment is deterministic across calls."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        M = 2000
        num_hosts = 4

        # Get shards multiple times
        for _ in range(3):
            shards_by_host = {}
            for host_idx in range(num_hosts):
                shards = ih.get_self_shard_info(
                    m=M, min_shards=8, host_index=host_idx, num_hosts=num_hosts
                )
                # Store as tuple of (a_start, a_end, b_start, b_end)
                shards_by_host[host_idx] = [
                    (s.a_start, s.a_end, s.b_start, s.b_end, s.global_shard_id)
                    for s in shards
                ]

            # First iteration, save for comparison
            if _ == 0:
                reference = shards_by_host
            else:
                # Compare to reference
                for host_idx in range(num_hosts):
                    assert shards_by_host[host_idx] == reference[host_idx], \
                        f"Shard assignment not deterministic for host {host_idx}"
