"""
Tests for trajectory metric calculations using synthetic trajectories
with known properties.
"""

import numpy as np
import pytest

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from metrics import (
    get_radius_of_gyration_simple,
    get_trajectory_metrics_tensor,
    get_mean_step_length,
    get_scaled_rg,
    calculate_step_angle_variation,
)


class TestRadiusOfGyrationSimple:
    """Test simple Rg calculation."""

    def test_stationary_point(self):
        """All points at same location should have Rg = 0."""
        xy = np.array([[5.0, 5.0]] * 10)
        rg = get_radius_of_gyration_simple(xy)
        assert rg == pytest.approx(0.0, abs=1e-10)

    def test_single_point(self):
        """Single point should return NaN."""
        xy = np.array([[1.0, 2.0]])
        rg = get_radius_of_gyration_simple(xy)
        assert np.isnan(rg)

    def test_nan_handling(self):
        """NaN rows should be excluded."""
        xy = np.array([[0, 0], [np.nan, np.nan], [1, 0], [2, 0]])
        rg = get_radius_of_gyration_simple(xy)
        assert not np.isnan(rg)
        assert rg > 0

    def test_linear_trajectory(self):
        """Linear trajectory should have positive Rg."""
        xy = np.array([[i, 0] for i in range(20)], dtype=float)
        rg = get_radius_of_gyration_simple(xy)
        assert rg > 0

    def test_symmetric_trajectory(self):
        """Square trajectory should have known Rg."""
        xy = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        rg = get_radius_of_gyration_simple(xy)
        # Rg = sqrt(sum(var_x + var_y)) = sqrt(0.25 + 0.25) = sqrt(0.5) ~ 0.707
        assert rg == pytest.approx(np.sqrt(0.5), abs=0.01)


class TestTensorMetrics:
    """Test tensor-based metrics."""

    def test_single_point_returns_nan(self):
        xy = np.array([[1.0, 2.0]])
        result = get_trajectory_metrics_tensor(xy)
        assert np.isnan(result['rg'])

    def test_linear_high_eigenvalue_ratio(self):
        """Perfectly linear trajectory should have very high eigenvalue ratio."""
        xy = np.array([[i, 0] for i in range(50)], dtype=float)
        result = get_trajectory_metrics_tensor(xy)
        assert result['eigenvalue_ratio'] > 100

    def test_circular_low_eigenvalue_ratio(self):
        """Circular trajectory should have eigenvalue ratio near 1."""
        t = np.linspace(0, 2 * np.pi, 100)
        xy = np.column_stack([np.cos(t), np.sin(t)])
        result = get_trajectory_metrics_tensor(xy)
        assert result['eigenvalue_ratio'] < 5

    def test_directionality_ratio_range(self):
        """Directionality ratio should be between 0 and 1."""
        xy = np.cumsum(np.random.default_rng(42).normal(0, 1, (50, 2)), axis=0)
        result = get_trajectory_metrics_tensor(xy)
        assert 0 <= result['directionality_ratio'] <= 1


class TestMeanStepLength:
    """Test mean step length calculation."""

    def test_uniform_steps(self):
        """Uniform steps should give exact step length."""
        xy = np.array([[i, 0] for i in range(10)], dtype=float)
        assert get_mean_step_length(xy) == pytest.approx(1.0, abs=1e-10)

    def test_single_point(self):
        xy = np.array([[1.0, 2.0]])
        assert np.isnan(get_mean_step_length(xy))

    def test_diagonal_steps(self):
        xy = np.array([[i, i] for i in range(5)], dtype=float)
        expected = np.sqrt(2)
        assert get_mean_step_length(xy) == pytest.approx(expected, abs=1e-10)


class TestScaledRg:
    """Test scaled Rg calculation."""

    def test_zero_step_length(self):
        assert np.isnan(get_scaled_rg(1.0, 0.0))

    def test_nan_inputs(self):
        assert np.isnan(get_scaled_rg(np.nan, 1.0))
        assert np.isnan(get_scaled_rg(1.0, np.nan))

    def test_positive_result(self):
        result = get_scaled_rg(5.0, 1.0)
        assert result > 0
        expected = np.sqrt(np.pi / 2) * 5.0 / 1.0
        assert result == pytest.approx(expected, abs=1e-10)


class TestStepAngleVariation:
    """Test step angle variation calculation."""

    def test_straight_line_zero_variation(self):
        """Straight line should have zero angle variation."""
        xy = np.array([[i, 0] for i in range(10)], dtype=float)
        result = calculate_step_angle_variation(xy)
        assert result['mean_angle_change'] == pytest.approx(0.0, abs=1e-10)

    def test_too_few_points(self):
        xy = np.array([[0, 0], [1, 1]], dtype=float)
        result = calculate_step_angle_variation(xy)
        assert np.isnan(result['mean_angle_change'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
