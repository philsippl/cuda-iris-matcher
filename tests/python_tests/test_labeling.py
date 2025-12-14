"""
Tests for the labeling/classification system in CUDA iris matcher.

Creates synthetic iris codes for multiple subjects and eye sides, then verifies:
1. Same subject + same eye => low hamming distance (match)
2. Different subjects => high hamming distance (non-match)
3. Classification flags (TM, FM, FNM, TNM) work correctly
"""

from typing import List
import numpy as np
import torch
import pytest
import sys
import os

# Add parent tests directory to path for utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import cuda_iris_matcher as ih
from utils import (
    rotation_aware_hamming_distance,
    generate_similar_iris_code,
    generate_random_iris_code,
)


def create_subject_iris_codes(
    n_subjects: int,
    signups_per_subject: List[int],
    intra_subject_noise: float = 0.08,
    seed: int = 42,
) -> tuple:
    """
    Create synthetic iris codes for multiple subjects with realistic properties.

    Args:
        n_subjects: Number of distinct subjects
        signups_per_subject: List of signup counts per subject
        intra_subject_noise: Noise ratio for same-subject codes (affects FHD)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (codes, masks, labels, label_strings):
        - codes: np.ndarray [N, 16, 200, 2, 2] uint8
        - masks: np.ndarray [N, 16, 200, 2, 2] uint8
        - labels: np.ndarray [N] int32 (encoded labels)
        - label_strings: list of str (original label strings)
    """
    all_codes = []
    all_masks = []
    all_label_strings = []

    for subject_id in range(n_subjects):
        # Two eye sides: Left (L) and Right (R)
        for eye_side in ["L", "R"]:
            label_str = f"subject_{subject_id}_{eye_side}"

            # Generate base code for this subject+eye combination
            base_code, base_mask = generate_random_iris_code(
                seed=seed + subject_id * 100 + (0 if eye_side == "L" else 50)
            )

            n_signups = signups_per_subject[subject_id]
            for signup_idx in range(n_signups):
                if signup_idx == 0:
                    # First signup: use base code directly
                    code, mask = base_code.copy(), base_mask.copy()
                else:
                    # Additional signups: add small noise to base code
                    code, mask = generate_similar_iris_code(
                        base_code,
                        base_mask,
                        noise_ratio=intra_subject_noise,
                        seed=seed + subject_id * 1000 + signup_idx,
                    )

                all_codes.append(code)
                all_masks.append(mask)
                all_label_strings.append(label_str)

    # Stack arrays
    codes = np.stack(all_codes, axis=0)
    masks = np.stack(all_masks, axis=0)

    # Encode labels to integers
    unique_labels = sorted(set(all_label_strings))
    label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
    labels = np.array([label_encoder[s] for s in all_label_strings], dtype=np.int32)

    return codes, masks, labels, all_label_strings


class TestLabelingSystem:
    """Test suite for the labeling/classification system."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Create test data:
        # Subject 0: 2 signups (4 iris codes: 2 eyes x 2 signups)
        # Subject 1: 2 signups (4 iris codes: 2 eyes x 2 signups)
        # Subject 2: 1 signup (2 iris codes: 2 eyes x 1 signup)
        # Total: 10 iris codes
        self.n_subjects = 3
        self.signups_per_subject = [2, 2, 1]

        self.codes, self.masks, self.labels, self.label_strings = create_subject_iris_codes(
            n_subjects=self.n_subjects,
            signups_per_subject=self.signups_per_subject,
            intra_subject_noise=0.08,  # ~8% bit flip for same subject
            seed=42,
        )

        self.M = len(self.codes)
        assert self.M == 10, f"Expected 10 iris codes, got {self.M}"

        # Pack for CUDA
        code_cuda = torch.from_numpy(self.codes).cuda()
        mask_cuda = torch.from_numpy(self.masks).cuda()
        self.data_t = ih.pack_theta_major(code_cuda)
        self.mask_t = ih.pack_theta_major(mask_cuda)
        self.labels_t = torch.from_numpy(self.labels).cuda()

    def test_same_subject_low_distance_reference(self):
        """Verify that same subject + same eye have low hamming distance (NumPy reference)."""
        # Calculate all pairwise distances using NumPy reference
        same_subject_distances = []
        diff_subject_distances = []

        for i in range(self.M):
            for j in range(i):
                dist, _ = rotation_aware_hamming_distance(
                    self.codes[i], self.masks[i], self.codes[j], self.masks[j]
                )

                if self.labels[i] == self.labels[j]:
                    same_subject_distances.append((i, j, dist))
                else:
                    diff_subject_distances.append((i, j, dist))

        # Same subject should have low distance (< 0.25 for our synthetic data)
        for i, j, dist in same_subject_distances:
            assert dist < 0.25, (
                f"Same subject pair ({i},{j}) has high distance {dist:.4f}. "
                f"Labels: {self.label_strings[i]}, {self.label_strings[j]}"
            )

        # Different subjects should have higher distance (> 0.30 typically)
        # Note: random codes have ~0.5 FHD on average
        for i, j, dist in diff_subject_distances:
            assert dist > 0.30, (
                f"Different subject pair ({i},{j}) has low distance {dist:.4f}. "
                f"Labels: {self.label_strings[i]}, {self.label_strings[j]}"
            )

    def test_same_subject_low_distance_cuda(self):
        """Verify that same subject + same eye have low hamming distance (CUDA kernel)."""
        max_pairs = self.M * (self.M - 1) // 2

        # Get all pairs from CUDA kernel
        pair_indices, categories, distances, count = ih.masked_hamming_cuda(
            self.data_t,
            self.mask_t,
            labels=self.labels_t,
            match_threshold=1.0,  # High threshold to get all pairs
            non_match_threshold=1.0,
            include_flags=ih.INCLUDE_ALL,
            max_pairs=max_pairs,
        )
        torch.cuda.synchronize()

        n_pairs = count.item()

        # Build distance lookup from CUDA results
        cuda_distances = {}
        for k in range(n_pairs):
            i, j = pair_indices[k].tolist()
            cuda_distances[(i, j)] = distances[k].item()

        # Categorize pairs by label
        same_subject_distances = []
        diff_subject_distances = []

        for i in range(self.M):
            for j in range(i):
                dist = cuda_distances.get((i, j), cuda_distances.get((j, i)))
                assert dist is not None, f"Pair ({i},{j}) not found in CUDA results"

                if self.labels[i] == self.labels[j]:
                    same_subject_distances.append((i, j, dist))
                else:
                    diff_subject_distances.append((i, j, dist))

        # Same subject should have low distance (< 0.25 for our synthetic data)
        for i, j, dist in same_subject_distances:
            assert dist < 0.25, (
                f"[CUDA] Same subject pair ({i},{j}) has high distance {dist:.4f}. "
                f"Labels: {self.label_strings[i]}, {self.label_strings[j]}"
            )

        # Different subjects should have higher distance (> 0.30 typically)
        for i, j, dist in diff_subject_distances:
            assert dist > 0.30, (
                f"[CUDA] Different subject pair ({i},{j}) has low distance {dist:.4f}. "
                f"Labels: {self.label_strings[i]}, {self.label_strings[j]}"
            )

    def test_cuda_distances_match_reference(self):
        """Verify CUDA kernel distances match NumPy reference."""
        max_pairs = self.M * (self.M - 1) // 2

        pair_indices, categories, distances, count = ih.masked_hamming_cuda(
            self.data_t,
            self.mask_t,
            labels=self.labels_t,
            match_threshold=0.35,
            non_match_threshold=0.35,
            include_flags=ih.INCLUDE_ALL,
            max_pairs=max_pairs,
        )
        torch.cuda.synchronize()

        n_pairs = count.item()

        # Calculate reference distances
        for k in range(n_pairs):
            i, j = pair_indices[k].tolist()
            cuda_dist = distances[k].item()
            ref_dist, _ = rotation_aware_hamming_distance(
                self.codes[i], self.masks[i], self.codes[j], self.masks[j]
            )
            assert abs(cuda_dist - ref_dist) <= 1e-5, (
                f"Pair ({i},{j}): CUDA={cuda_dist:.6f}, ref={ref_dist:.6f}"
            )

    def test_include_true_matches_only(self):
        """Test INCLUDE_TM flag returns only true matches."""
        max_pairs = self.M * (self.M - 1) // 2
        threshold = 0.35

        pair_indices, categories, distances, count = ih.masked_hamming_cuda(
            self.data_t,
            self.mask_t,
            labels=self.labels_t,
            match_threshold=threshold,
            non_match_threshold=threshold,
            include_flags=ih.INCLUDE_TM,
            max_pairs=max_pairs,
        )
        torch.cuda.synchronize()

        n_pairs = count.item()

        # All returned pairs should be true matches (same label, distance <= threshold)
        for k in range(n_pairs):
            i, j = pair_indices[k].tolist()
            cat = categories[k].item()
            dist = distances[k].item()

            assert cat == ih.CATEGORY_TRUE_MATCH, f"Expected TM, got category {cat}"
            assert self.labels[i] == self.labels[j], (
                f"TM pair ({i},{j}) has different labels: "
                f"{self.label_strings[i]} vs {self.label_strings[j]}"
            )
            assert dist <= threshold, f"TM pair ({i},{j}) has distance {dist} > {threshold}"

    def test_include_true_non_matches_only(self):
        """Test INCLUDE_TNM flag returns only true non-matches."""
        max_pairs = self.M * (self.M - 1) // 2
        threshold = 0.35

        pair_indices, categories, distances, count = ih.masked_hamming_cuda(
            self.data_t,
            self.mask_t,
            labels=self.labels_t,
            match_threshold=threshold,
            non_match_threshold=threshold,
            include_flags=ih.INCLUDE_TNM,
            max_pairs=max_pairs,
        )
        torch.cuda.synchronize()

        n_pairs = count.item()

        # All returned pairs should be true non-matches (diff label, distance > threshold)
        for k in range(n_pairs):
            i, j = pair_indices[k].tolist()
            cat = categories[k].item()
            dist = distances[k].item()

            assert cat == ih.CATEGORY_TRUE_NON_MATCH, f"Expected TNM, got category {cat}"
            assert self.labels[i] != self.labels[j], (
                f"TNM pair ({i},{j}) has same labels: "
                f"{self.label_strings[i]} vs {self.label_strings[j]}"
            )
            assert dist > threshold, f"TNM pair ({i},{j}) has distance {dist} <= {threshold}"

    def test_include_false_matches_only(self):
        """Test INCLUDE_FM flag returns only false matches (impostors)."""
        max_pairs = self.M * (self.M - 1) // 2
        # Use a high threshold to potentially catch some false matches
        threshold = 0.6

        pair_indices, categories, distances, count = ih.masked_hamming_cuda(
            self.data_t,
            self.mask_t,
            labels=self.labels_t,
            match_threshold=threshold,
            non_match_threshold=threshold,
            include_flags=ih.INCLUDE_FM,
            max_pairs=max_pairs,
        )
        torch.cuda.synchronize()

        n_pairs = count.item()

        # All returned pairs should be false matches (diff label, distance <= threshold)
        for k in range(n_pairs):
            i, j = pair_indices[k].tolist()
            cat = categories[k].item()
            dist = distances[k].item()

            assert cat == ih.CATEGORY_FALSE_MATCH, f"Expected FM, got category {cat}"
            assert self.labels[i] != self.labels[j], (
                f"FM pair ({i},{j}) has same labels: "
                f"{self.label_strings[i]} vs {self.label_strings[j]}"
            )
            assert dist <= threshold, f"FM pair ({i},{j}) has distance {dist} > {threshold}"

    def test_include_false_non_matches_only(self):
        """Test INCLUDE_FNM flag returns only false non-matches."""
        max_pairs = self.M * (self.M - 1) // 2
        # Use a low threshold to potentially catch some false non-matches
        threshold = 0.05

        pair_indices, categories, distances, count = ih.masked_hamming_cuda(
            self.data_t,
            self.mask_t,
            labels=self.labels_t,
            match_threshold=threshold,
            non_match_threshold=threshold,
            include_flags=ih.INCLUDE_FNM,
            max_pairs=max_pairs,
        )
        torch.cuda.synchronize()

        n_pairs = count.item()

        # All returned pairs should be false non-matches (same label, distance > threshold)
        for k in range(n_pairs):
            i, j = pair_indices[k].tolist()
            cat = categories[k].item()
            dist = distances[k].item()

            assert cat == ih.CATEGORY_FALSE_NON_MATCH, f"Expected FNM, got category {cat}"
            assert self.labels[i] == self.labels[j], (
                f"FNM pair ({i},{j}) has different labels: "
                f"{self.label_strings[i]} vs {self.label_strings[j]}"
            )
            assert dist > threshold, f"FNM pair ({i},{j}) has distance {dist} <= {threshold}"

    def test_include_all_categories(self):
        """Test INCLUDE_ALL returns all pairs with correct categories."""
        max_pairs = self.M * (self.M - 1) // 2
        threshold = 0.35

        pair_indices, categories, distances, count = ih.masked_hamming_cuda(
            self.data_t,
            self.mask_t,
            labels=self.labels_t,
            match_threshold=threshold,
            non_match_threshold=threshold,
            include_flags=ih.INCLUDE_ALL,
            max_pairs=max_pairs,
        )
        torch.cuda.synchronize()

        n_pairs = count.item()

        # Should get all lower-triangle pairs
        assert n_pairs == max_pairs, f"Expected {max_pairs} pairs, got {n_pairs}"

        # Count each category
        cat_counts = {
            ih.CATEGORY_TRUE_MATCH: 0,
            ih.CATEGORY_FALSE_MATCH: 0,
            ih.CATEGORY_FALSE_NON_MATCH: 0,
            ih.CATEGORY_TRUE_NON_MATCH: 0,
        }

        for k in range(n_pairs):
            i, j = pair_indices[k].tolist()
            cat = categories[k].item()
            dist = distances[k].item()
            same_label = self.labels[i] == self.labels[j]

            # Verify category is correct
            is_match = dist <= threshold
            is_non_match = dist > threshold

            if same_label and is_match:
                expected_cat = ih.CATEGORY_TRUE_MATCH
            elif not same_label and is_match:
                expected_cat = ih.CATEGORY_FALSE_MATCH
            elif same_label and is_non_match:
                expected_cat = ih.CATEGORY_FALSE_NON_MATCH
            else:
                expected_cat = ih.CATEGORY_TRUE_NON_MATCH

            assert cat == expected_cat, (
                f"Pair ({i},{j}): expected category {expected_cat}, got {cat}. "
                f"same_label={same_label}, dist={dist:.4f}"
            )

            cat_counts[cat] += 1

        # Verify we have some pairs in expected categories
        print(f"\nCategory distribution:")
        print(f"  True Matches (TM):       {cat_counts[ih.CATEGORY_TRUE_MATCH]}")
        print(f"  False Matches (FM):      {cat_counts[ih.CATEGORY_FALSE_MATCH]}")
        print(f"  False Non-Matches (FNM): {cat_counts[ih.CATEGORY_FALSE_NON_MATCH]}")
        print(f"  True Non-Matches (TNM):  {cat_counts[ih.CATEGORY_TRUE_NON_MATCH]}")

        # With our test data, we should have some TM and TNM at minimum
        assert cat_counts[ih.CATEGORY_TRUE_MATCH] > 0, "Expected some True Matches"
        assert cat_counts[ih.CATEGORY_TRUE_NON_MATCH] > 0, "Expected some True Non-Matches"

    def test_returned_count_matches_expected(self):
        """Test that returned count matches the number of valid entries in output arrays."""
        max_pairs = self.M * (self.M - 1) // 2
        threshold = 0.35

        pair_indices, categories, distances, count = ih.masked_hamming_cuda(
            self.data_t,
            self.mask_t,
            labels=self.labels_t,
            match_threshold=threshold,
            non_match_threshold=threshold,
            include_flags=ih.INCLUDE_ALL,
            max_pairs=max_pairs,
        )
        torch.cuda.synchronize()

        n_pairs = count.item()

        # Verify count is within valid range
        assert 0 <= n_pairs <= max_pairs, f"Count {n_pairs} out of range [0, {max_pairs}]"

        # Verify that we can access all n_pairs entries without error
        # and that indices are valid
        seen_pairs = set()
        for k in range(n_pairs):
            i, j = pair_indices[k].tolist()
            cat = categories[k].item()
            dist = distances[k].item()

            # Indices should be valid
            assert 0 <= i < self.M, f"Invalid row index {i} at position {k}"
            assert 0 <= j < self.M, f"Invalid col index {j} at position {k}"
            assert i > j, f"Expected lower triangle (i > j), got ({i}, {j})"

            # Category should be valid
            assert cat in [
                ih.CATEGORY_TRUE_MATCH,
                ih.CATEGORY_FALSE_MATCH,
                ih.CATEGORY_FALSE_NON_MATCH,
                ih.CATEGORY_TRUE_NON_MATCH,
            ], f"Invalid category {cat} at position {k}"

            # Distance should be valid
            assert 0.0 <= dist <= 1.0, f"Invalid distance {dist} at position {k}"

            # No duplicate pairs
            pair_key = (i, j)
            assert pair_key not in seen_pairs, f"Duplicate pair {pair_key} at position {k}"
            seen_pairs.add(pair_key)

        # Verify total count matches expected (all lower triangle pairs with INCLUDE_ALL)
        expected_total = self.M * (self.M - 1) // 2
        assert n_pairs == expected_total, (
            f"Expected {expected_total} pairs (all lower triangle), got {n_pairs}"
        )

    def test_category_counts_match_reference(self):
        """Test that returned tensor lengths match count and expected reference counts."""
        max_pairs = self.M * (self.M - 1) // 2
        threshold = 0.35

        # Get CUDA results
        pair_indices, categories, distances, count = ih.masked_hamming_cuda(
            self.data_t,
            self.mask_t,
            labels=self.labels_t,
            match_threshold=threshold,
            non_match_threshold=threshold,
            include_flags=ih.INCLUDE_ALL,
            max_pairs=max_pairs,
        )

        n_pairs = count.item()

        # Verify tensor lengths match count
        assert len(pair_indices) == n_pairs, (
            f"pair_indices length {len(pair_indices)} != count {n_pairs}"
        )
        assert len(categories) == n_pairs, (
            f"categories length {len(categories)} != count {n_pairs}"
        )
        assert len(distances) == n_pairs, (
            f"distances length {len(distances)} != count {n_pairs}"
        )

        # Compute expected count from reference
        expected_count = 0
        for i in range(self.M):
            for j in range(i):
                dist, _ = rotation_aware_hamming_distance(
                    self.codes[i], self.masks[i], self.codes[j], self.masks[j]
                )
                same_label = self.labels[i] == self.labels[j]
                is_match = dist <= threshold
                is_non_match = dist > threshold

                # With INCLUDE_ALL and equal thresholds, all pairs should be emitted
                if (same_label and is_match) or (not same_label and is_match) or \
                   (same_label and is_non_match) or (not same_label and is_non_match):
                    expected_count += 1

        # Verify count matches expected
        assert n_pairs == expected_count, (
            f"count {n_pairs} != expected {expected_count}"
        )

    def test_combined_flags(self):
        """Test combining multiple include flags."""
        max_pairs = self.M * (self.M - 1) // 2
        threshold = 0.35

        # Include only TM and FM (all matches, regardless of label correctness)
        pair_indices, categories, distances, count = ih.masked_hamming_cuda(
            self.data_t,
            self.mask_t,
            labels=self.labels_t,
            match_threshold=threshold,
            non_match_threshold=threshold,
            include_flags=ih.INCLUDE_TM | ih.INCLUDE_FM,
            max_pairs=max_pairs,
        )
        torch.cuda.synchronize()

        n_pairs = count.item()

        # All returned pairs should be matches (distance <= threshold)
        for k in range(n_pairs):
            i, j = pair_indices[k].tolist()
            cat = categories[k].item()
            dist = distances[k].item()

            assert cat in [ih.CATEGORY_TRUE_MATCH, ih.CATEGORY_FALSE_MATCH], (
                f"Expected TM or FM, got category {cat}"
            )
            assert dist <= threshold, f"Match pair ({i},{j}) has distance {dist} > {threshold}"

    def test_no_labels_returns_unclassified(self):
        """Test that omitting labels returns all pairs as unclassified."""
        max_pairs = self.M * (self.M - 1) // 2

        pair_indices, categories, distances, count = ih.masked_hamming_cuda(
            self.data_t,
            self.mask_t,
            labels=None,  # No labels
            match_threshold=0.35,
            non_match_threshold=0.35,
            include_flags=ih.INCLUDE_ALL,
            max_pairs=max_pairs,
        )
        torch.cuda.synchronize()

        n_pairs = count.item()
        assert n_pairs == max_pairs, f"Expected {max_pairs} pairs, got {n_pairs}"

        # All categories should be 0xFF (unclassified)
        for k in range(n_pairs):
            cat = categories[k].item()
            assert cat == 0xFF, f"Expected unclassified (255), got {cat}"

    def test_gap_between_thresholds(self):
        """Test that pairs in the gap between thresholds are not emitted."""
        max_pairs = self.M * (self.M - 1) // 2

        # Use thresholds that create a gap: match < 0.3, non-match > 0.4
        match_threshold = 0.3
        non_match_threshold = 0.4

        pair_indices, categories, distances, count = ih.masked_hamming_cuda(
            self.data_t,
            self.mask_t,
            labels=self.labels_t,
            match_threshold=match_threshold,
            non_match_threshold=non_match_threshold,
            include_flags=ih.INCLUDE_ALL,
            max_pairs=max_pairs,
        )
        torch.cuda.synchronize()

        n_pairs = count.item()

        # Count pairs in the gap
        gap_count = 0
        for i in range(self.M):
            for j in range(i):
                dist, _ = rotation_aware_hamming_distance(
                    self.codes[i], self.masks[i], self.codes[j], self.masks[j]
                )
                if match_threshold < dist <= non_match_threshold:
                    gap_count += 1

        # Pairs in gap should not be emitted
        expected_pairs = max_pairs - gap_count
        assert n_pairs == expected_pairs, (
            f"Expected {expected_pairs} pairs (excluding {gap_count} in gap), got {n_pairs}"
        )

        # Verify all returned distances are outside the gap
        for k in range(n_pairs):
            dist = distances[k].item()
            in_gap = match_threshold < dist <= non_match_threshold
            assert not in_gap, f"Pair in gap returned with distance {dist}"


class TestLabelingAB:
    """Test labeling with A vs B comparisons."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures for A vs B testing."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Gallery: Subject 0 and 1 (2 signups each)
        # Probe: Subject 1 and 2 (1 signup each)
        self.n_subjects = 3
        self.signups_gallery = [2, 2, 0]  # Subject 0 and 1 in gallery
        self.signups_probe = [0, 1, 1]  # Subject 1 and 2 in probe

        # Create gallery data
        (
            self.codes_gallery,
            self.masks_gallery,
            self.labels_gallery,
            self.label_strings_gallery,
        ) = create_subject_iris_codes(
            n_subjects=self.n_subjects,
            signups_per_subject=self.signups_gallery,
            intra_subject_noise=0.08,
            seed=100,
        )

        # Create probe data with different seed for variation
        (
            self.codes_probe,
            self.masks_probe,
            self.labels_probe,
            self.label_strings_probe,
        ) = create_subject_iris_codes(
            n_subjects=self.n_subjects,
            signups_per_subject=self.signups_probe,
            intra_subject_noise=0.08,
            seed=200,
        )

        self.M_A = len(self.codes_gallery)
        self.M_B = len(self.codes_probe)

        # Pack for CUDA
        code_a_cuda = torch.from_numpy(self.codes_gallery).cuda()
        mask_a_cuda = torch.from_numpy(self.masks_gallery).cuda()
        code_b_cuda = torch.from_numpy(self.codes_probe).cuda()
        mask_b_cuda = torch.from_numpy(self.masks_probe).cuda()

        self.data_a = ih.pack_theta_major(code_a_cuda)
        self.mask_a = ih.pack_theta_major(mask_a_cuda)
        self.data_b = ih.pack_theta_major(code_b_cuda)
        self.mask_b = ih.pack_theta_major(mask_b_cuda)

        self.labels_a = torch.from_numpy(self.labels_gallery).cuda()
        self.labels_b = torch.from_numpy(self.labels_probe).cuda()

    def test_ab_classification(self):
        """Test A vs B classification returns correct categories."""
        max_pairs = self.M_A * self.M_B
        threshold = 0.35

        pair_indices, categories, distances, count = ih.masked_hamming_ab_cuda(
            self.data_a,
            self.mask_a,
            self.data_b,
            self.mask_b,
            labels_a=self.labels_a,
            labels_b=self.labels_b,
            match_threshold=threshold,
            non_match_threshold=threshold,
            include_flags=ih.INCLUDE_ALL,
            max_pairs=max_pairs,
        )
        torch.cuda.synchronize()

        n_pairs = count.item()

        # Verify categories
        for k in range(n_pairs):
            i, j = pair_indices[k].tolist()
            cat = categories[k].item()
            dist = distances[k].item()
            same_label = self.labels_gallery[i] == self.labels_probe[j]

            is_match = dist <= threshold
            is_non_match = dist > threshold

            if same_label and is_match:
                expected_cat = ih.CATEGORY_TRUE_MATCH
            elif not same_label and is_match:
                expected_cat = ih.CATEGORY_FALSE_MATCH
            elif same_label and is_non_match:
                expected_cat = ih.CATEGORY_FALSE_NON_MATCH
            else:
                expected_cat = ih.CATEGORY_TRUE_NON_MATCH

            assert cat == expected_cat, (
                f"Pair ({i},{j}): expected category {expected_cat}, got {cat}. "
                f"same_label={same_label}, dist={dist:.4f}"
            )
