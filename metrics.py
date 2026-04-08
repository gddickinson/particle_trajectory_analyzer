"""
metrics.py - Trajectory metric calculations.

Contains functions for calculating radius of gyration (Rg), scaled Rg (sRg),
mean step length, step angle variation, and tensor-based shape metrics.

The simple Rg and sRg method was adapted from R code of J. Alfredo Freites
(jfreites@uci.edu). The simple sRg is computed as in Golan and Sherman
Nat Comm 2017.
"""

import numpy as np
import math
from scipy import stats


def get_radius_of_gyration_simple(xy):
    """Calculate radius of gyration using simple formula.

    Args:
        xy: Nx2 matrix of coordinates (can contain NaN values)

    Returns:
        Radius of gyration value
    """
    xy = xy[~np.isnan(xy).any(axis=1)]

    if len(xy) < 2:
        return np.nan

    avg = np.nanmean(xy, axis=0)
    avg2 = np.nanmean(xy**2, axis=0)
    rg = np.sqrt(np.sum(avg2 - avg**2))

    return rg


def get_trajectory_metrics_tensor(xy):
    """Calculate radius of gyration, shape metrics, and linear motion metrics
    using tensor-based approach.

    Args:
        xy: Nx2 matrix of coordinates (can contain NaN values)

    Returns:
        Dictionary with metrics: rg, asymmetry, skewness, kurtosis,
        eigenvalue_ratio, step_alignment, and directionality_ratio
    """
    xy = xy[~np.isnan(xy).any(axis=1)]

    num_points = len(xy)
    if num_points < 2:
        return {
            'rg': np.nan,
            'asymmetry': np.nan,
            'skewness': np.nan,
            'kurtosis': np.nan,
            'eigenvalue_ratio': np.nan,
            'step_alignment': np.nan,
            'directionality_ratio': np.nan
        }

    center = np.mean(xy, axis=0)
    normed_points = xy - center[None, :]
    gyration_tensor = np.einsum('im,in->mn', normed_points, normed_points) / num_points

    eig_values, eig_vectors = np.linalg.eig(gyration_tensor)
    eig_values = np.real(eig_values)

    idx = eig_values.argsort()[::-1]
    eig_values = eig_values[idx]
    eig_vectors = eig_vectors[:, idx]
    eig_vectors = np.real(eig_vectors)

    rg = np.sqrt(np.sum(eig_values))

    # Calculate asymmetry
    try:
        asymmetry_numerator = pow((eig_values[0] - eig_values[1]), 2)
        asymmetry_denominator = 2 * (pow((eig_values[0] + eig_values[1]), 2))
        asymmetry = -math.log(1 - (asymmetry_numerator / asymmetry_denominator))
    except (ValueError, ZeroDivisionError):
        asymmetry = np.nan

    # Calculate skewness and kurtosis
    try:
        principal_eigenvector = eig_vectors[:, 0]
        points_a = xy[:-1]
        points_b = xy[1:]
        steps = points_b - points_a
        proj_steps = np.dot(steps, principal_eigenvector) / np.linalg.norm(principal_eigenvector)
        skewness = stats.skew(proj_steps)
        kurtosis = stats.kurtosis(proj_steps)
    except (ValueError, ZeroDivisionError):
        skewness = np.nan
        kurtosis = np.nan
        steps = np.diff(xy, axis=0)

    # Calculate eigenvalue ratio
    try:
        eigenvalue_ratio = eig_values[0] / eig_values[1] if eig_values[1] > 0 else np.inf
    except (IndexError, ZeroDivisionError):
        eigenvalue_ratio = np.nan

    # Calculate step alignment with principal axis
    try:
        if 'steps' not in dir():
            steps = np.diff(xy, axis=0)
        step_norms = np.linalg.norm(steps, axis=1)
        valid_steps = step_norms > 0
        normalized_steps = steps[valid_steps] / step_norms[valid_steps, None]
        cos_angles = np.abs(np.dot(normalized_steps, principal_eigenvector))
        step_alignment = np.mean(cos_angles)
    except (ValueError, ZeroDivisionError, IndexError):
        step_alignment = np.nan

    # Calculate directionality ratio
    try:
        if 'steps' not in dir():
            steps = np.diff(xy, axis=0)
        net_displacement = np.linalg.norm(xy[-1] - xy[0])
        path_length = np.sum(np.linalg.norm(steps, axis=1))
        directionality_ratio = net_displacement / path_length if path_length > 0 else np.nan
    except (ValueError, ZeroDivisionError, IndexError):
        directionality_ratio = np.nan

    return {
        'rg': rg,
        'asymmetry': asymmetry,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'eigenvalue_ratio': eigenvalue_ratio,
        'step_alignment': step_alignment,
        'directionality_ratio': directionality_ratio
    }


def get_mean_step_length(xy):
    """Calculate mean step length of trajectory.

    Args:
        xy: Nx2 matrix of coordinates (can contain NaN values)

    Returns:
        Mean step length
    """
    xy = xy[~np.isnan(xy).any(axis=1)]

    if len(xy) < 2:
        return np.nan

    steps = np.diff(xy, axis=0)
    step_lengths = np.sqrt(np.sum(steps**2, axis=1))
    return np.mean(step_lengths)


def get_scaled_rg(rg, mean_step_length):
    """Calculate scaled radius of gyration as in Golan and Sherman Nat Comm 2017.

    Args:
        rg: Radius of gyration
        mean_step_length: Mean step length

    Returns:
        Scaled radius of gyration value
    """
    if np.isnan(rg) or np.isnan(mean_step_length) or mean_step_length == 0:
        return np.nan

    s_rg = np.sqrt(np.pi / 2) * rg / mean_step_length
    return s_rg


def calculate_step_angle_variation(xy):
    """Calculate the variation in angles between consecutive steps.

    Lower variation indicates more linear motion.

    Args:
        xy: Nx2 matrix of coordinates (can contain NaN values)

    Returns:
        Dictionary with angle statistics
    """
    xy = xy[~np.isnan(xy).any(axis=1)]

    if len(xy) < 3:
        return {
            'mean_angle_change': np.nan,
            'std_angle_change': np.nan,
            'max_angle_change': np.nan
        }

    steps = np.diff(xy, axis=0)

    angle_changes = []
    for i in range(len(steps) - 1):
        vec1 = steps[i]
        vec2 = steps[i + 1]

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            continue

        cos_angle = np.dot(vec1, vec2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(np.abs(cos_angle))
        angle_changes.append(np.degrees(angle))

    if len(angle_changes) == 0:
        return {
            'mean_angle_change': np.nan,
            'std_angle_change': np.nan,
            'max_angle_change': np.nan
        }

    return {
        'mean_angle_change': np.mean(angle_changes),
        'std_angle_change': np.std(angle_changes),
        'max_angle_change': np.max(angle_changes)
    }
