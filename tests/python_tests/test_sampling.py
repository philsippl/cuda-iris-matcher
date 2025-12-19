"""Tests for stratified sampling functionality."""

import pytest
import torch

from cuda_iris_matcher import (
    StratifiedSamplingFilter,
    masked_hamming_cuda,
    dot_product_cuda,
)
# Import apply_stratified_sampling from internal module for testing
from cuda_iris_matcher.sampling import apply_stratified_sampling


class TestStratifiedSamplingFilter:
    """Tests for StratifiedSamplingFilter class."""
    
    def test_basic_creation(self):
        """Test basic filter creation."""
        filter = StratifiedSamplingFilter(
            sample_bins={0.3: 1.0, 0.5: 0.1, 1.0: 0.01},
            seed=42
        )
        assert filter.sample_bins == {0.3: 1.0, 0.5: 0.1, 1.0: 0.01}
        assert filter.seed == 42
    
    def test_bins_sorted(self):
        """Test that bins are sorted by threshold."""
        filter = StratifiedSamplingFilter(
            sample_bins={1.0: 0.01, 0.3: 1.0, 0.5: 0.1}
        )
        # Should be sorted
        keys = list(filter.sample_bins.keys())
        assert keys == [0.3, 0.5, 1.0]
    
    def test_empty_bins_raises(self):
        """Test that empty sample_bins raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            StratifiedSamplingFilter(sample_bins={})
    
    def test_invalid_probability_raises(self):
        """Test that invalid probabilities raise error."""
        with pytest.raises(ValueError, match="must be in"):
            StratifiedSamplingFilter(sample_bins={0.5: 1.5})
        
        with pytest.raises(ValueError, match="must be in"):
            StratifiedSamplingFilter(sample_bins={0.5: -0.1})
    
    def test_apply_empty_input(self):
        """Test applying filter to empty results."""
        filter = StratifiedSamplingFilter(sample_bins={0.5: 0.5})
        
        pairs = torch.zeros((0, 2), dtype=torch.int32)
        cats = torch.zeros((0,), dtype=torch.uint8)
        scores = torch.zeros((0,), dtype=torch.float32)
        
        out_pairs, out_cats, out_scores, out_count = filter.apply(pairs, cats, scores)
        
        assert out_count.item() == 0
        assert out_pairs.shape == (0, 2)
    
    def test_apply_keep_all(self):
        """Test filter with 100% sampling keeps all."""
        filter = StratifiedSamplingFilter(
            sample_bins={1.0: 1.0},  # Keep 100%
            seed=42
        )
        
        n = 100
        pairs = torch.arange(n * 2, dtype=torch.int32).reshape(n, 2)
        cats = torch.zeros(n, dtype=torch.uint8)
        scores = torch.rand(n, dtype=torch.float32)
        
        out_pairs, out_cats, out_scores, out_count = filter.apply(pairs, cats, scores)
        
        assert out_count.item() == n
        assert out_pairs.shape == (n, 2)
    
    def test_apply_keep_none(self):
        """Test filter with 0% sampling keeps none."""
        filter = StratifiedSamplingFilter(
            sample_bins={1.0: 0.0},  # Keep 0%
            seed=42
        )
        
        n = 100
        pairs = torch.arange(n * 2, dtype=torch.int32).reshape(n, 2)
        cats = torch.zeros(n, dtype=torch.uint8)
        scores = torch.rand(n, dtype=torch.float32)
        
        out_pairs, out_cats, out_scores, out_count = filter.apply(pairs, cats, scores)
        
        assert out_count.item() == 0
    
    def test_apply_stratified(self):
        """Test stratified sampling with different bins."""
        filter = StratifiedSamplingFilter(
            sample_bins={0.3: 1.0, 0.6: 0.5, 1.0: 0.0},
            seed=42
        )
        
        n = 1000
        pairs = torch.arange(n * 2, dtype=torch.int32).reshape(n, 2)
        cats = torch.zeros(n, dtype=torch.uint8)
        
        # Create scores in each bin
        scores = torch.zeros(n, dtype=torch.float32)
        scores[:333] = 0.15  # Should all be kept (100%)
        scores[333:666] = 0.45  # ~50% should be kept
        scores[666:] = 0.8  # None should be kept (0%)
        
        out_pairs, out_cats, out_scores, out_count = filter.apply(pairs, cats, scores)
        
        # All low scores should be kept
        low_score_kept = (out_scores <= 0.3).sum().item()
        assert low_score_kept == 333
        
        # No high scores should be kept
        high_score_kept = (out_scores > 0.6).sum().item()
        assert high_score_kept == 0
        
        # Middle scores should be roughly 50%
        mid_score_kept = ((out_scores > 0.3) & (out_scores <= 0.6)).sum().item()
        assert 100 < mid_score_kept < 250  # Approximate 50% of 333
    
    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results."""
        n = 100
        pairs = torch.arange(n * 2, dtype=torch.int32).reshape(n, 2)
        cats = torch.zeros(n, dtype=torch.uint8)
        scores = torch.rand(n, dtype=torch.float32)
        
        filter1 = StratifiedSamplingFilter(sample_bins={1.0: 0.5}, seed=42)
        filter2 = StratifiedSamplingFilter(sample_bins={1.0: 0.5}, seed=42)
        
        _, _, _, count1 = filter1.apply(pairs, cats, scores)
        _, _, _, count2 = filter2.apply(pairs, cats, scores)
        
        assert count1.item() == count2.item()
    
    def test_reset_seed(self):
        """Test resetting seed produces same results."""
        filter = StratifiedSamplingFilter(sample_bins={1.0: 0.5}, seed=42)
        
        n = 100
        pairs = torch.arange(n * 2, dtype=torch.int32).reshape(n, 2)
        cats = torch.zeros(n, dtype=torch.uint8)
        scores = torch.rand(n, dtype=torch.float32)
        
        _, _, _, count1 = filter.apply(pairs, cats, scores)
        filter.reset_seed()
        _, _, _, count2 = filter.apply(pairs, cats, scores)
        
        assert count1.item() == count2.item()
    
    def test_cuda_tensors(self):
        """Test filter works with CUDA tensors."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        filter = StratifiedSamplingFilter(sample_bins={1.0: 0.5}, seed=42)
        
        n = 100
        pairs = torch.arange(n * 2, dtype=torch.int32, device="cuda").reshape(n, 2)
        cats = torch.zeros(n, dtype=torch.uint8, device="cuda")
        scores = torch.rand(n, dtype=torch.float32, device="cuda")
        
        out_pairs, out_cats, out_scores, out_count = filter.apply(pairs, cats, scores)
        
        assert out_pairs.is_cuda
        assert out_cats.is_cuda
        assert out_scores.is_cuda
        assert out_count.is_cuda


class TestApplyStratifiedSampling:
    """Tests for the apply_stratified_sampling function."""
    
    def test_with_dict(self):
        """Test using a dict instead of filter instance."""
        n = 100
        pairs = torch.arange(n * 2, dtype=torch.int32).reshape(n, 2)
        cats = torch.zeros(n, dtype=torch.uint8)
        scores = torch.rand(n, dtype=torch.float32) * 0.5  # All <= 0.5
        
        out_pairs, out_cats, out_scores, out_count = apply_stratified_sampling(
            pairs, cats, scores,
            sample_bins={0.5: 1.0, 1.0: 0.0},
            seed=42
        )
        
        # All should be kept since all scores <= 0.5
        assert out_count.item() == n
    
    def test_with_filter_instance(self):
        """Test using a pre-created filter instance."""
        filter = StratifiedSamplingFilter(sample_bins={1.0: 0.5}, seed=42)
        
        n = 100
        pairs = torch.arange(n * 2, dtype=torch.int32).reshape(n, 2)
        cats = torch.zeros(n, dtype=torch.uint8)
        scores = torch.rand(n, dtype=torch.float32)
        
        out_pairs, out_cats, out_scores, out_count = apply_stratified_sampling(
            pairs, cats, scores, sample_bins=filter
        )
        
        # Approximately 50% should be kept
        assert 30 < out_count.item() < 70


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestIntegrationWithKernels:
    """Integration tests with actual CUDA kernels."""
    
    def test_hamming_with_sample_bins_dict(self):
        """Test masked_hamming_cuda with sample_bins as dict."""
        m = 50
        data = torch.randint(0, 2**31, (m, 400), dtype=torch.int32, device="cuda")
        mask = torch.full((m, 400), 0x7FFFFFFF, dtype=torch.int32, device="cuda")
        
        # Without sampling
        pairs1, cats1, dists1, count1 = masked_hamming_cuda(data, mask)
        
        # With sampling - keep 50%
        pairs2, cats2, dists2, count2 = masked_hamming_cuda(
            data, mask,
            sample_bins={1.0: 0.5}
        )
        
        # Sampled should have roughly half the pairs
        assert count2.item() < count1.item()
        assert count2.item() > 0
    
    def test_hamming_with_sample_bins_filter(self):
        """Test masked_hamming_cuda with StratifiedSamplingFilter."""
        m = 50
        data = torch.randint(0, 2**31, (m, 400), dtype=torch.int32, device="cuda")
        mask = torch.full((m, 400), 0x7FFFFFFF, dtype=torch.int32, device="cuda")
        
        filter = StratifiedSamplingFilter(
            sample_bins={0.3: 1.0, 0.5: 0.1, 1.0: 0.01},
            seed=42
        )
        
        pairs, cats, dists, count = masked_hamming_cuda(
            data, mask,
            sample_bins=filter
        )
        
        assert pairs.shape[0] == count.item()
        assert cats.shape[0] == count.item()
        assert dists.shape[0] == count.item()
    
    def test_dot_product_with_sample_bins(self):
        """Test dot_product_cuda with sample_bins."""
        m = 50
        data = torch.randn(m, 512, dtype=torch.float16, device="cuda")
        data = data / data.norm(dim=1, keepdim=True)
        
        # Without sampling
        pairs1, cats1, scores1, count1 = dot_product_cuda(data)
        
        # With sampling
        filter = StratifiedSamplingFilter(
            sample_bins={0.5: 0.01, 0.8: 0.1, 1.0: 1.0},
            seed=42
        )
        pairs2, cats2, scores2, count2 = dot_product_cuda(data, sample_bins=filter)
        
        # Sampled should have fewer pairs
        assert count2.item() <= count1.item()
    
    def test_sample_bins_with_include_flags(self):
        """Test that sample_bins works correctly with include_flags."""
        from cuda_iris_matcher import INCLUDE_ALL, CATEGORY_TRUE_MATCH
        
        m = 50
        data = torch.randint(0, 2**31, (m, 400), dtype=torch.int32, device="cuda")
        mask = torch.full((m, 400), 0x7FFFFFFF, dtype=torch.int32, device="cuda")
        labels = torch.arange(m, dtype=torch.int32, device="cuda")
        # Make some labels match
        labels[10:20] = 0  # These will be same-identity pairs
        
        # First filter by category (include_flags), then sample
        pairs, cats, dists, count = masked_hamming_cuda(
            data, mask,
            labels=labels,
            include_flags=INCLUDE_ALL,
            sample_bins={0.5: 0.5, 1.0: 0.1}
        )
        
        # Verify results have valid structure
        assert pairs.shape[0] == count.item()
        assert cats.shape[0] == count.item()
        assert dists.shape[0] == count.item()

