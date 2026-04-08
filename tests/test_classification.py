"""
Tests for trajectory classification using synthetic trajectories
with known motion types.
"""

import numpy as np
import pytest

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from classification import classify_linear_motion_simple, classify_linear_motion


class TestSimpleLinearClassification:
    """Test simple geometric linear classification."""

    def test_perfectly_linear_unidirectional(self):
        """A straight line should be classified as linear_unidirectional."""
        xy = np.array([[i, 0] for i in range(50)], dtype=float)
        result = classify_linear_motion_simple(xy)
        assert result['classification'] == 'linear_unidirectional'
        assert result['directionality_ratio'] == pytest.approx(1.0, abs=0.01)

    def test_random_walk_non_linear(self):
        """A random walk should typically be classified as non_linear."""
        rng = np.random.default_rng(42)
        xy = np.cumsum(rng.normal(0, 1, (200, 2)), axis=0)
        result = classify_linear_motion_simple(xy)
        assert result['classification'] == 'non_linear'

    def test_too_few_points(self):
        """Fewer than 3 points should be unclassified."""
        xy = np.array([[0, 0], [1, 1]], dtype=float)
        result = classify_linear_motion_simple(xy)
        assert result['classification'] == 'unclassified'

    def test_stationary_unclassified(self):
        """All points at same location should be unclassified."""
        xy = np.array([[5.0, 5.0]] * 10)
        result = classify_linear_motion_simple(xy)
        assert result['classification'] == 'unclassified'

    def test_directionality_ratio_range(self):
        """Directionality ratio should always be in [0, 1]."""
        rng = np.random.default_rng(123)
        xy = np.cumsum(rng.normal(0, 1, (50, 2)), axis=0)
        result = classify_linear_motion_simple(xy)
        if not np.isnan(result['directionality_ratio']):
            assert 0 <= result['directionality_ratio'] <= 1

    def test_nan_handling(self):
        """NaN values should be handled gracefully."""
        xy = np.array([[0, 0], [np.nan, np.nan], [1, 0], [2, 0], [3, 0]], dtype=float)
        result = classify_linear_motion_simple(xy)
        assert result['classification'] in ('linear_unidirectional', 'linear_bidirectional', 'non_linear')


class TestTensorLinearClassification:
    """Test tensor-based linear classification."""

    def test_nan_inputs(self):
        result = classify_linear_motion(np.nan, 0.8, 0.9)
        assert result == 'unclassified'

    def test_non_linear(self):
        result = classify_linear_motion(2.0, 0.3, 0.5,
                                        eig_ratio_cutoff=20.0,
                                        step_align_cutoff=0.7)
        assert result == 'non_linear'

    def test_linear_unidirectional(self):
        result = classify_linear_motion(50.0, 0.9, 0.9,
                                        eig_ratio_cutoff=20.0,
                                        step_align_cutoff=0.7)
        assert result == 'linear_unidirectional'

    def test_linear_bidirectional(self):
        result = classify_linear_motion(50.0, 0.9, 0.3,
                                        eig_ratio_cutoff=20.0,
                                        step_align_cutoff=0.7)
        assert result == 'linear_bidirectional'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
