"""
trajectory_sim.py - Synthetic trajectory generators (numpy/pandas only).

Factored out of trajectory_generator.py so the generators can be used without
matplotlib/seaborn/scikit-learn (e.g. by the web app's synthetic demo dataset).
trajectory_generator.py imports everything below, so its behaviour is unchanged.

Contents: N_TRAJECTORIES, N_POINTS, RANDOM_SEED, TRAJECTORY_PARAMS,
generate_linear_unidirectional, generate_linear_bidirectional,
generate_random_diffusion, generate_confined_diffusion,
generate_directed_random, generate_trajectories, trajectories_to_dataframe.
"""

import numpy as np
import pandas as pd

# Number of trajectories to generate per motion type
N_TRAJECTORIES = 50

# Number of points per trajectory
N_POINTS = 100

# Random seed for reproducibility
RANDOM_SEED = 42

# Parameters for trajectory generation
TRAJECTORY_PARAMS = {
    'linear_unidirectional': {
        'step_size': 1.0,
        'noise_level': 0.1,  # Perpendicular noise
    },
    'linear_bidirectional': {
        'step_size': 1.0,
        'noise_level': 0.1,
        'oscillation_period': 20,  # Points per oscillation
    },
    'random_diffusion': {
        'step_size': 1.0,
    },
    'confined_diffusion': {
        'step_size': 1.0,
        'confinement_radius': 10.0,
    },
    'directed_random': {
        'step_size': 1.0,
        'drift_magnitude': 0.5,
    }
}


def generate_linear_unidirectional(n_points, step_size=1.0, noise_level=0.1, angle=None):
    """Generate a linear unidirectional trajectory with perpendicular noise.

    Args:
        n_points: Number of points in the trajectory
        step_size: Size of each step
        noise_level: Standard deviation of perpendicular noise
        angle: Direction angle in radians (random if None)

    Returns:
        Nx2 array of trajectory coordinates
    """
    # Random angle if not specified
    if angle is None:
        angle = np.random.uniform(0, 2 * np.pi)

    # Direction vector
    direction = np.array([np.cos(angle), np.sin(angle)])

    # Generate steps with noise perpendicular to direction
    steps = np.zeros((n_points, 2))

    for i in range(1, n_points):
        # Perpendicular vector
        perp_vector = np.array([-direction[1], direction[0]])

        # Add noise perpendicular to direction
        perpendicular_noise = np.random.normal(0, noise_level) * perp_vector

        # Create step
        steps[i] = step_size * direction + perpendicular_noise

    # Accumulate steps to get trajectory
    trajectory = np.cumsum(steps, axis=0)

    return trajectory


def generate_linear_bidirectional(n_points, step_size=1.0, noise_level=0.1, oscillation_period=20, angle=None):
    """Generate a linear bidirectional (oscillating) trajectory.

    Args:
        n_points: Number of points in the trajectory
        step_size: Size of each step
        noise_level: Standard deviation of perpendicular noise
        oscillation_period: Points per complete oscillation
        angle: Direction angle in radians (random if None)

    Returns:
        Nx2 array of trajectory coordinates
    """
    # Random angle if not specified
    if angle is None:
        angle = np.random.uniform(0, 2 * np.pi)

    # Direction vector
    direction = np.array([np.cos(angle), np.sin(angle)])

    # Perpendicular vector
    perp_vector = np.array([-direction[1], direction[0]])

    # Generate oscillating movement along the direction
    steps = np.zeros((n_points, 2))

    for i in range(1, n_points):
        # Oscillating direction - cosine function controls direction reversal
        direction_factor = np.cos(2 * np.pi * i / oscillation_period)

        # Perpendicular noise
        perpendicular_noise = np.random.normal(0, noise_level) * perp_vector

        # Create step
        steps[i] = step_size * direction_factor * direction + perpendicular_noise

    # Accumulate steps to get trajectory
    trajectory = np.cumsum(steps, axis=0)

    return trajectory


def generate_random_diffusion(n_points, step_size=1.0):
    """Generate a random diffusion (Brownian motion) trajectory.

    Args:
        n_points: Number of points in the trajectory
        step_size: Average size of each step

    Returns:
        Nx2 array of trajectory coordinates
    """
    # Generate random steps
    steps = np.random.normal(0, step_size, (n_points, 2))

    # Set first step to zero
    steps[0] = [0, 0]

    # Accumulate steps to get trajectory
    trajectory = np.cumsum(steps, axis=0)

    return trajectory


def generate_confined_diffusion(n_points, step_size=1.0, confinement_radius=10.0):
    """Generate a confined diffusion trajectory (Brownian motion within boundaries).

    Args:
        n_points: Number of points in the trajectory
        step_size: Average size of each step
        confinement_radius: Radius of the confinement region

    Returns:
        Nx2 array of trajectory coordinates
    """
    trajectory = np.zeros((n_points, 2))

    for i in range(1, n_points):
        # Generate random step
        step = np.random.normal(0, step_size, 2)

        # Proposed new position
        proposed_position = trajectory[i-1] + step

        # Calculate distance from origin
        distance = np.sqrt(np.sum(proposed_position**2))

        # If outside confinement, reflect back
        if distance > confinement_radius:
            # Vector from origin to proposed position
            direction = proposed_position / distance

            # Reflection point on the boundary
            reflection_point = confinement_radius * direction

            # Reflect the excess distance back inside
            excess_distance = distance - confinement_radius

            # New position after reflection
            trajectory[i] = reflection_point - excess_distance * direction
        else:
            # Accept the step if inside confinement
            trajectory[i] = proposed_position

    return trajectory


def generate_directed_random(n_points, step_size=1.0, drift_magnitude=0.5, drift_angle=None):
    """Generate a directed random motion trajectory (Brownian motion with drift).

    Args:
        n_points: Number of points in the trajectory
        step_size: Average size of each random step
        drift_magnitude: Magnitude of the drift vector
        drift_angle: Direction angle of drift in radians (random if None)

    Returns:
        Nx2 array of trajectory coordinates
    """
    # Random drift angle if not specified
    if drift_angle is None:
        drift_angle = np.random.uniform(0, 2 * np.pi)

    # Drift vector
    drift_vector = drift_magnitude * np.array([np.cos(drift_angle), np.sin(drift_angle)])

    # Generate steps with drift
    steps = np.random.normal(0, step_size, (n_points, 2))

    # Add drift to each step
    steps += drift_vector

    # Set first step to zero
    steps[0] = [0, 0]

    # Accumulate steps to get trajectory
    trajectory = np.cumsum(steps, axis=0)

    return trajectory


def generate_trajectories(n_trajectories=50, n_points=100):
    """Generate trajectories for all motion types.

    Args:
        n_trajectories: Number of trajectories per motion type
        n_points: Number of points per trajectory

    Returns:
        Dictionary with motion types as keys and lists of trajectories as values
    """
    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)

    trajectories = {}

    # Linear unidirectional
    trajectories['linear_unidirectional'] = []
    for i in range(n_trajectories):
        params = TRAJECTORY_PARAMS['linear_unidirectional']
        traj = generate_linear_unidirectional(
            n_points,
            step_size=params['step_size'],
            noise_level=params['noise_level']
        )
        trajectories['linear_unidirectional'].append(traj)

    # Linear bidirectional
    trajectories['linear_bidirectional'] = []
    for i in range(n_trajectories):
        params = TRAJECTORY_PARAMS['linear_bidirectional']
        traj = generate_linear_bidirectional(
            n_points,
            step_size=params['step_size'],
            noise_level=params['noise_level'],
            oscillation_period=params['oscillation_period']
        )
        trajectories['linear_bidirectional'].append(traj)

    # Random diffusion
    trajectories['random_diffusion'] = []
    for i in range(n_trajectories):
        params = TRAJECTORY_PARAMS['random_diffusion']
        traj = generate_random_diffusion(
            n_points,
            step_size=params['step_size']
        )
        trajectories['random_diffusion'].append(traj)

    # Confined diffusion
    trajectories['confined_diffusion'] = []
    for i in range(n_trajectories):
        params = TRAJECTORY_PARAMS['confined_diffusion']
        traj = generate_confined_diffusion(
            n_points,
            step_size=params['step_size'],
            confinement_radius=params['confinement_radius']
        )
        trajectories['confined_diffusion'].append(traj)

    # Directed random
    trajectories['directed_random'] = []
    for i in range(n_trajectories):
        params = TRAJECTORY_PARAMS['directed_random']
        traj = generate_directed_random(
            n_points,
            step_size=params['step_size'],
            drift_magnitude=params['drift_magnitude']
        )
        trajectories['directed_random'].append(traj)

    return trajectories


def trajectories_to_dataframe(trajectories):
    """Convert trajectories dictionary to DataFrames for each motion type.

    Args:
        trajectories: Dictionary with motion types as keys and lists of trajectories as values

    Returns:
        Dictionary with motion types as keys and DataFrames as values
    """
    dataframes = {}

    for motion_type, trajs in trajectories.items():
        # Create a list to store all trajectory data
        all_data = []

        for i, traj in enumerate(trajs):
            # Create DataFrame for this trajectory
            df = pd.DataFrame(traj, columns=['x', 'y'])
            df['track_number'] = i + 1
            df['frame'] = np.arange(len(traj))
            df['motion_type'] = motion_type

            # Add to list
            all_data.append(df)

        # Concatenate all trajectories for this motion type
        dataframes[motion_type] = pd.concat(all_data, ignore_index=True)

    return dataframes
