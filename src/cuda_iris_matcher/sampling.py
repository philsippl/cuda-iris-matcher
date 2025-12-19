"""Stratified sampling utilities for filtering comparison results by score/distance bins."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

import torch


@dataclass
class StratifiedSamplingFilter:
    """Filter for stratified sampling of pair results based on distance/score bins.
    
    Allows different sampling rates for different score ranges, enabling efficient
    data reduction while preserving the distribution characteristics you care about.
    
    For distance metrics (hamming, is_similarity=False):
        Lower values are "better" (more similar). Bins are defined as ranges from 0 upward.
        Example: {0.3: 1.0, 0.5: 0.1, 1.0: 0.01} means:
            - Keep 100% of pairs with distance in [0, 0.3]
            - Keep 10% of pairs with distance in (0.3, 0.5]
            - Keep 1% of pairs with distance in (0.5, 1.0]
    
    For similarity metrics (dot product, is_similarity=True):
        Higher values are "better" (more similar). Bins are still defined with ascending
        thresholds, but you'd typically use higher sampling rates for higher values.
        Example: {0.5: 0.01, 0.8: 0.1, 1.0: 1.0} means:
            - Keep 1% of pairs with similarity in [-inf, 0.5]
            - Keep 10% of pairs with similarity in (0.5, 0.8]
            - Keep 100% of pairs with similarity in (0.8, 1.0]
    
    Attributes:
        sample_bins: Dict mapping bin upper bounds (thresholds) to sampling probabilities.
                     Keys should be in ascending order. Values are probabilities in [0, 1].
                     The first bin starts at -inf, the last bin ends at +inf if not explicitly bounded.
        seed: Optional random seed for reproducibility.
    
    Example:
        Basic usage with hamming distances:
        
        >>> import cuda_iris_matcher as ih
        >>> # Create filter: keep all close matches, sample distant ones
        >>> filter = ih.StratifiedSamplingFilter(
        ...     sample_bins={
        ...         0.3: 1.0,    # Keep 100% of [0, 0.3]
        ...         0.5: 0.1,    # Keep 10% of (0.3, 0.5]
        ...         1.0: 0.01    # Keep 1% of (0.5, 1.0]
        ...     },
        ...     seed=42
        ... )
        >>> # Compute distances and apply sampling
        >>> pairs, cats, dists, count = ih.masked_hamming_cuda(
        ...     data, mask, sample_bins=filter
        ... )
        
        Or apply to existing results:
        
        >>> pairs, cats, dists, count = ih.masked_hamming_cuda(data, mask)
        >>> pairs, cats, dists, count = filter.apply(pairs, cats, dists)
        
        With dot product similarity:
        
        >>> filter = ih.StratifiedSamplingFilter(
        ...     sample_bins={
        ...         0.5: 0.01,   # Keep 1% of [-inf, 0.5]
        ...         0.8: 0.1,    # Keep 10% of (0.5, 0.8]
        ...         1.0: 1.0     # Keep 100% of (0.8, 1.0]
        ...     },
        ...     seed=42
        ... )
        >>> pairs, cats, scores, count = ih.dot_product_cuda(
        ...     data, sample_bins=filter
        ... )
    """
    
    sample_bins: Dict[float, float]
    seed: Optional[int] = None
    _generator: Optional[torch.Generator] = field(default=None, repr=False, compare=False)
    
    def __post_init__(self):
        """Validate sample_bins configuration."""
        if not self.sample_bins:
            raise ValueError("sample_bins cannot be empty")
        
        # Validate probabilities
        for threshold, prob in self.sample_bins.items():
            if not isinstance(threshold, (int, float)):
                raise TypeError(f"Bin threshold must be numeric, got {type(threshold)}")
            if not isinstance(prob, (int, float)):
                raise TypeError(f"Sampling probability must be numeric, got {type(prob)}")
            if not 0.0 <= prob <= 1.0:
                raise ValueError(f"Sampling probability must be in [0, 1], got {prob}")
        
        # Sort bins by threshold
        self.sample_bins = dict(sorted(self.sample_bins.items()))
        
        # Initialize generator if seed provided
        if self.seed is not None:
            self._generator = torch.Generator()
            self._generator.manual_seed(self.seed)
    
    def apply(
        self,
        pair_indices: torch.Tensor,
        categories: torch.Tensor,
        scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply stratified sampling to pair results.
        
        Args:
            pair_indices: [N, 2] int32 tensor of (row, col) pair indices
            categories: [N] uint8 tensor of category codes
            scores: [N] float32 tensor of distance/similarity scores
        
        Returns:
            Tuple of (pair_indices, categories, scores, count) with sampled results:
            - pair_indices: [M, 2] int32 - sampled pair indices
            - categories: [M] uint8 - sampled categories
            - scores: [M] float32 - sampled scores
            - count: [1] int32 - number of sampled pairs (M)
        """
        if pair_indices.numel() == 0:
            count = torch.tensor([0], dtype=torch.int32, device=pair_indices.device)
            return pair_indices, categories, scores, count
        
        n_pairs = pair_indices.size(0)
        device = scores.device
        
        # Generate random values for sampling decision
        if self._generator is not None:
            rand_vals = torch.rand(n_pairs, generator=self._generator, device='cpu').to(device)
        else:
            rand_vals = torch.rand(n_pairs, device=device)
        
        # Compute sampling probability for each pair based on its score
        sample_probs = self._get_sample_probabilities(scores)
        
        # Keep pairs where random value < sampling probability
        keep_mask = rand_vals < sample_probs
        
        # Apply mask
        sampled_pairs = pair_indices[keep_mask]
        sampled_cats = categories[keep_mask]
        sampled_scores = scores[keep_mask]
        sampled_count = torch.tensor([sampled_pairs.size(0)], dtype=torch.int32, device=device)
        
        return sampled_pairs, sampled_cats, sampled_scores, sampled_count
    
    def _get_sample_probabilities(self, scores: torch.Tensor) -> torch.Tensor:
        """Compute sampling probability for each score based on bin configuration.
        
        Args:
            scores: [N] float32 tensor of scores
        
        Returns:
            [N] float32 tensor of sampling probabilities
        """
        device = scores.device
        n = scores.size(0)
        
        # Start with zeros
        probs = torch.zeros(n, dtype=torch.float32, device=device)
        
        # Get sorted thresholds and probabilities
        thresholds = list(self.sample_bins.keys())
        probabilities = list(self.sample_bins.values())
        
        # Assign probabilities based on bins
        # First bin: score <= first_threshold
        # Middle bins: prev_threshold < score <= current_threshold  
        # Last bin: score > last_threshold (if any scores exceed)
        
        prev_threshold = float('-inf')
        for threshold, prob in zip(thresholds, probabilities):
            bin_mask = (scores > prev_threshold) & (scores <= threshold)
            probs[bin_mask] = prob
            prev_threshold = threshold
        
        # Handle scores above the highest threshold (use last probability)
        if thresholds:
            above_mask = scores > thresholds[-1]
            probs[above_mask] = probabilities[-1]
        
        return probs
    
    def reset_seed(self, seed: Optional[int] = None):
        """Reset the random generator with a new seed.
        
        Args:
            seed: New seed value. If None, uses the original seed.
        """
        if seed is not None:
            self.seed = seed
        if self.seed is not None:
            if self._generator is None:
                self._generator = torch.Generator()
            self._generator.manual_seed(self.seed)


def apply_stratified_sampling(
    pair_indices: torch.Tensor,
    categories: torch.Tensor,
    scores: torch.Tensor,
    sample_bins: Union[Dict[float, float], StratifiedSamplingFilter],
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply stratified sampling to pair comparison results.
    
    Convenience function that creates a StratifiedSamplingFilter and applies it.
    For repeated use with the same configuration, prefer creating a
    StratifiedSamplingFilter instance directly.
    
    Args:
        pair_indices: [N, 2] int32 tensor of (row, col) pair indices
        categories: [N] uint8 tensor of category codes
        scores: [N] float32 tensor of distance/similarity scores
        sample_bins: Either a dict mapping thresholds to probabilities, or a
                     StratifiedSamplingFilter instance.
        seed: Random seed (ignored if sample_bins is a StratifiedSamplingFilter)
    
    Returns:
        Tuple of (pair_indices, categories, scores, count) with sampled results.
    
    Example:
        >>> pairs, cats, dists, count = ih.masked_hamming_cuda(data, mask)
        >>> # Sample: keep all close, few distant
        >>> pairs, cats, dists, count = ih.apply_stratified_sampling(
        ...     pairs, cats, dists,
        ...     sample_bins={0.3: 1.0, 0.5: 0.1, 1.0: 0.01},
        ...     seed=42
        ... )
    """
    if isinstance(sample_bins, StratifiedSamplingFilter):
        return sample_bins.apply(pair_indices, categories, scores)
    
    # Create filter from dict
    filter_obj = StratifiedSamplingFilter(sample_bins=sample_bins, seed=seed)
    return filter_obj.apply(pair_indices, categories, scores)


# Type alias for sample_bins parameter
SampleBinsType = Union[Dict[float, float], StratifiedSamplingFilter, None]

