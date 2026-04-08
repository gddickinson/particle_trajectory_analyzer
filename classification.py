"""
classification.py - Trajectory classification functions.

Classifies trajectories as linear (unidirectional/bidirectional) or non-linear
using either simple geometric methods or tensor-based eigenvalue analysis.
"""

import numpy as np


def classify_linear_motion_simple(xy, directionality_threshold=0.8, perpendicular_threshold=0.15):
    """Classify trajectory as linear or non-linear using geometric methods.

    This method uses:
    1. Directionality ratio: net displacement / total path length
    2. Mean perpendicular distance: average distance of points from the straight
       line connecting start to end

    Args:
        xy: Nx2 matrix of coordinates (can contain NaN values)
        directionality_threshold: Minimum directionality ratio for linear
            classification (default: 0.8)
        perpendicular_threshold: Maximum normalized perpendicular distance for
            linear classification (default: 0.15)

    Returns:
        Dictionary with:
            - classification: 'linear_unidirectional', 'linear_bidirectional',
              or 'non_linear'
            - directionality_ratio: Net displacement / total path length
            - mean_perpendicular_distance: Average perpendicular distance
            - normalized_perpendicular_distance: Perpendicular distance normalized
              by trajectory length
    """
    xy = xy[~np.isnan(xy).any(axis=1)]

    if len(xy) < 3:
        return {
            'classification': 'unclassified',
            'directionality_ratio': np.nan,
            'mean_perpendicular_distance': np.nan,
            'normalized_perpendicular_distance': np.nan
        }

    start_point = xy[0]
    end_point = xy[-1]
    net_displacement = np.linalg.norm(end_point - start_point)

    steps = np.diff(xy, axis=0)
    step_lengths = np.linalg.norm(steps, axis=1)
    total_path_length = np.sum(step_lengths)

    if total_path_length == 0:
        return {
            'classification': 'unclassified',
            'directionality_ratio': np.nan,
            'mean_perpendicular_distance': np.nan,
            'normalized_perpendicular_distance': np.nan
        }

    directionality_ratio = net_displacement / total_path_length

    if net_displacement > 0:
        direction = (end_point - start_point) / net_displacement

        perpendicular_distances = []
        for point in xy:
            vec_to_point = point - start_point
            projection_length = np.dot(vec_to_point, direction)
            projection = projection_length * direction
            perpendicular = vec_to_point - projection
            perpendicular_dist = np.linalg.norm(perpendicular)
            perpendicular_distances.append(perpendicular_dist)

        mean_perpendicular_distance = np.mean(perpendicular_distances)
        normalized_perpendicular_distance = (
            mean_perpendicular_distance / net_displacement
            if net_displacement > 0 else np.inf
        )
    else:
        mean_perpendicular_distance = np.nan
        normalized_perpendicular_distance = np.nan

    is_directional = directionality_ratio >= directionality_threshold
    is_straight = (
        normalized_perpendicular_distance <= perpendicular_threshold
        if not np.isnan(normalized_perpendicular_distance) else False
    )

    if is_directional and is_straight:
        classification = 'linear_unidirectional'
    elif not is_directional and is_straight:
        classification = 'linear_bidirectional'
    else:
        classification = 'non_linear'

    return {
        'classification': classification,
        'directionality_ratio': directionality_ratio,
        'mean_perpendicular_distance': mean_perpendicular_distance,
        'normalized_perpendicular_distance': normalized_perpendicular_distance
    }


def classify_linear_motion(eigenvalue_ratio, step_alignment, directionality_ratio,
                           eig_ratio_cutoff=20.0, step_align_cutoff=0.7):
    """Classify trajectory as linear or non-linear based on eigenvalue ratio
    and step alignment.

    Classifies trajectories based on three key metrics:
    - 'linear_unidirectional' for trajectories with high linearity and
      directionality
    - 'linear_bidirectional' for trajectories with high linearity but low
      directionality (back-and-forth motion)
    - 'non_linear' for trajectories that don't meet linearity criteria

    Args:
        eigenvalue_ratio: Ratio of largest to smallest eigenvalue
        step_alignment: Average absolute cosine similarity between steps and
            principal eigenvector
        directionality_ratio: Net displacement divided by total path length
        eig_ratio_cutoff: Threshold for eigenvalue ratio (default: 20.0)
        step_align_cutoff: Threshold for step alignment (default: 0.7)

    Returns:
        Classification string: 'linear_unidirectional', 'linear_bidirectional',
        or 'non_linear'
    """
    if np.isnan(eigenvalue_ratio) or np.isnan(step_alignment):
        return 'unclassified'

    is_linear = (
        eigenvalue_ratio >= eig_ratio_cutoff
        and step_alignment >= step_align_cutoff
    )

    if not is_linear:
        return 'non_linear'

    if np.isnan(directionality_ratio):
        return 'linear'

    if directionality_ratio >= 0.7:
        return 'linear_unidirectional'
    else:
        return 'linear_bidirectional'
