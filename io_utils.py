"""
io_utils.py - CSV reading/writing and file processing utilities.

Handles reading trajectory CSV files, saving processed data with metrics,
generating split output files by classification, and collecting statistics.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from metrics import (
    get_radius_of_gyration_simple,
    get_trajectory_metrics_tensor,
    get_mean_step_length,
    get_scaled_rg,
)
from classification import classify_linear_motion_simple, classify_linear_motion
from config import resolve as _cfg


def validate_csv_columns(df, required_cols=None):
    """Validate that a DataFrame contains the required columns.

    Args:
        df: pandas DataFrame to validate
        required_cols: List of required column names.
            Defaults to ['track_number', 'frame', 'x', 'y']

    Returns:
        True if all required columns are present

    Raises:
        ValueError: If any required columns are missing
    """
    if required_cols is None:
        required_cols = ['track_number', 'frame', 'x', 'y']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return True


def process_csv_file(file_path, cutoff_len=None, sRg_cutoff=None,
                     linear_eig_ratio_cutoff=None, linear_step_align_cutoff=None,
                     method=None, include_shape_metrics=True,
                     include_linear_metrics=True,
                     linear_directionality_cutoff=None,
                     linear_perpendicular_cutoff=None):
    """Process a single CSV file to calculate trajectory metrics.

    Args:
        file_path: Path to CSV file
        (Parameters left as None take their defaults from config.py.)
        cutoff_len: Minimum trajectory length
        sRg_cutoff: Threshold for mobile/immobile classification
        linear_eig_ratio_cutoff: Threshold for eigenvalue ratio (tensor method)
        linear_step_align_cutoff: Threshold for step alignment (tensor method)
        method: Method to calculate Rg ('simple' or 'tensor')
        include_shape_metrics: Whether to include shape metrics
        include_linear_metrics: Whether to include linear motion metrics
        linear_directionality_cutoff: Threshold for directionality ratio (simple)
        linear_perpendicular_cutoff: Threshold for perpendicular distance (simple)

    Returns:
        DataFrame with original data plus metrics columns, or None on error
    """
    cutoff_len = _cfg(cutoff_len, 'CUTOFF_LEN')
    sRg_cutoff = _cfg(sRg_cutoff, 'SRG_CUTOFF')
    linear_eig_ratio_cutoff = _cfg(linear_eig_ratio_cutoff, 'LINEAR_EIGENVALUE_RATIO_CUTOFF')
    linear_step_align_cutoff = _cfg(linear_step_align_cutoff, 'LINEAR_STEP_ALIGNMENT_CUTOFF')
    method = _cfg(method, 'RG_METHOD')
    linear_directionality_cutoff = _cfg(linear_directionality_cutoff, 'LINEAR_DIRECTIONALITY_CUTOFF')
    linear_perpendicular_cutoff = _cfg(linear_perpendicular_cutoff, 'LINEAR_PERPENDICULAR_CUTOFF')
    print(f"Processing {file_path}")
    print(f"Using {method} method for Rg calculation")

    try:
        df = pd.read_csv(file_path)
        validate_csv_columns(df)

        tracks = df.groupby('track_number')
        track_lens = tracks.size()
        selected_tracks = track_lens[track_lens >= cutoff_len].index.tolist()

        print(f"Found {len(selected_tracks)} tracks with length >= {cutoff_len}")

        results = []
        for track_num in selected_tracks:
            track_data = tracks.get_group(track_num)
            track_data = track_data.sort_values('frame')

            frames = track_data['frame'].values
            min_frame = frames.min()
            max_frame = frames.max()

            traj_len = max_frame - min_frame + 1
            traj_matrix = np.full((traj_len, 2), np.nan)

            frame_indices = frames - min_frame
            traj_matrix[frame_indices, 0] = track_data['x'].values
            traj_matrix[frame_indices, 1] = track_data['y'].values

            mean_step = get_mean_step_length(traj_matrix)

            if method == 'simple':
                rg = get_radius_of_gyration_simple(traj_matrix)

                linear_result = classify_linear_motion_simple(
                    traj_matrix,
                    linear_directionality_cutoff,
                    linear_perpendicular_cutoff
                )

                metrics = {
                    'rg': rg,
                    'asymmetry': np.nan,
                    'skewness': np.nan,
                    'kurtosis': np.nan,
                    'eigenvalue_ratio': np.nan,
                    'step_alignment': np.nan,
                    'directionality_ratio': linear_result['directionality_ratio'],
                    'mean_perpendicular_distance': linear_result['mean_perpendicular_distance'],
                    'normalized_perpendicular_distance': linear_result['normalized_perpendicular_distance']
                }

                linear_classification = linear_result['classification']

            else:  # tensor method
                metrics = get_trajectory_metrics_tensor(traj_matrix)
                rg = metrics['rg']
                linear_classification = classify_linear_motion(
                    metrics['eigenvalue_ratio'],
                    metrics['step_alignment'],
                    metrics['directionality_ratio'],
                    eig_ratio_cutoff=linear_eig_ratio_cutoff,
                    step_align_cutoff=linear_step_align_cutoff
                )

            srg_value = get_scaled_rg(rg, mean_step)
            # sRg is undefined (NaN) for tracks with fewer than 2 usable localisations or
            # no movement at all; such tracks are not called mobile.
            if np.isnan(srg_value):
                mobility_classification = 'unclassified'
            else:
                mobility_classification = 'immobile' if srg_value < sRg_cutoff else 'mobile'

            result = {
                'track_number': track_num,
                'rg': rg,
                'mean_step_length': mean_step,
                'sRg': srg_value,
                'mobility_classification': mobility_classification,
                'linear_classification': linear_classification
            }

            if include_shape_metrics and method == 'tensor':
                result['asymmetry'] = metrics['asymmetry']
                result['skewness'] = metrics['skewness']
                result['kurtosis'] = metrics['kurtosis']

            if include_linear_metrics:
                if method == 'tensor':
                    result['eigenvalue_ratio'] = metrics['eigenvalue_ratio']
                    result['step_alignment'] = metrics['step_alignment']
                result['directionality_ratio'] = metrics['directionality_ratio']

                if method == 'simple':
                    result['mean_perpendicular_distance'] = metrics['mean_perpendicular_distance']
                    result['normalized_perpendicular_distance'] = metrics['normalized_perpendicular_distance']

            results.append(result)

        metrics_df = pd.DataFrame(results)
        if metrics_df.empty:
            # No tracks met the cutoff; return original df with NaN metric columns
            for col in ['rg', 'mean_step_length', 'sRg',
                        'mobility_classification', 'linear_classification']:
                df[col] = np.nan
            return df.sort_values(['track_number', 'frame'])
        merged_df = pd.merge(df, metrics_df, on='track_number', how='left')
        merged_df = merged_df.sort_values(['track_number', 'frame'])

        return merged_df

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def save_processed_data(df, file_path, suffix='_metrics', split_outputs=False,
                        mobile_only=False, output_dir=None, stats_data=None):
    """Save processed data to a new CSV file.

    Args:
        df: DataFrame with metrics
        file_path: Original file path
        suffix: Suffix for the output filename
        split_outputs: Whether to split output by classification
        mobile_only: Whether to only include mobile tracks in split outputs
        output_dir: Directory for output files (None = 'results' subdirectory)
        stats_data: Optional dictionary to collect statistics
    """
    path = Path(file_path)

    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = path.parent / 'results'

    output_path.mkdir(parents=True, exist_ok=True)

    new_filename = output_path / f"{path.stem}{suffix}{path.suffix}"
    df.to_csv(new_filename, index=False)
    print(f"Saved results to {new_filename}")

    if stats_data is not None and isinstance(stats_data, dict) and 'sRg' in df.columns:
        collect_file_statistics(df, path.stem, stats_data)

    if split_outputs and 'mobility_classification' in df.columns and 'linear_classification' in df.columns:
        _save_split_outputs(df, path, output_path, mobile_only)


def _save_split_outputs(df, path, output_path, mobile_only):
    """Save split output files by classification.

    Args:
        df: DataFrame with metrics
        path: Path object for original file
        output_path: Path object for output directory
        mobile_only: Whether to only include mobile tracks
    """
    mobile_df = df[df['mobility_classification'] == 'mobile'].copy()

    mobile_dir = output_path / 'mobile'
    mobile_dir.mkdir(parents=True, exist_ok=True)

    if not mobile_df.empty:
        mobile_filename = mobile_dir / f"{path.stem}_mobile{path.suffix}"
        mobile_df.to_csv(mobile_filename, index=False)
        print(f"Saved {len(mobile_df['track_number'].unique())} mobile tracks to {mobile_filename}")

    if mobile_only:
        df_split = mobile_df
        mobility_prefix = 'mobile'
    else:
        df_split = df.copy()
        mobility_prefix = 'all'

    linear_tracks = df_split[
        df_split['linear_classification'].isin(
            ['linear', 'linear_unidirectional', 'linear_bidirectional']
        )
    ]['track_number'].unique()
    nonlinear_tracks = df_split[
        df_split['linear_classification'] == 'non_linear'
    ]['track_number'].unique()

    if len(linear_tracks) > 0:
        linear_dir = output_path / f"{mobility_prefix}_linear"
        linear_dir.mkdir(parents=True, exist_ok=True)
        linear_df = df[df['track_number'].isin(linear_tracks)]
        linear_filename = linear_dir / f"{path.stem}_{mobility_prefix}_linear{path.suffix}"
        linear_df.to_csv(linear_filename, index=False)
        print(f"Saved {len(linear_tracks)} linear tracks to {linear_filename}")
    else:
        print(f"No linear tracks found for {path}")

    if len(nonlinear_tracks) > 0:
        nonlinear_dir = output_path / f"{mobility_prefix}_nonlinear"
        nonlinear_dir.mkdir(parents=True, exist_ok=True)
        nonlinear_df = df[df['track_number'].isin(nonlinear_tracks)]
        nonlinear_filename = nonlinear_dir / f"{path.stem}_{mobility_prefix}_nonlinear{path.suffix}"
        nonlinear_df.to_csv(nonlinear_filename, index=False)
        print(f"Saved {len(nonlinear_tracks)} non-linear tracks to {nonlinear_filename}")
    else:
        print(f"No non-linear tracks found for {path}")

    # Split linear into unidirectional and bidirectional
    uni_tracks = df_split[
        df_split['linear_classification'] == 'linear_unidirectional'
    ]['track_number'].unique()
    bidir_tracks = df_split[
        df_split['linear_classification'] == 'linear_bidirectional'
    ]['track_number'].unique()

    if len(uni_tracks) > 0:
        uni_dir = output_path / f"{mobility_prefix}_linear_unidirectional"
        uni_dir.mkdir(parents=True, exist_ok=True)
        uni_df = df[df['track_number'].isin(uni_tracks)]
        uni_filename = uni_dir / f"{path.stem}_{mobility_prefix}_linear_unidirectional{path.suffix}"
        uni_df.to_csv(uni_filename, index=False)
        print(f"Saved {len(uni_tracks)} unidirectional linear tracks to {uni_filename}")

    if len(bidir_tracks) > 0:
        bidir_dir = output_path / f"{mobility_prefix}_linear_bidirectional"
        bidir_dir.mkdir(parents=True, exist_ok=True)
        bidir_df = df[df['track_number'].isin(bidir_tracks)]
        bidir_filename = bidir_dir / f"{path.stem}_{mobility_prefix}_linear_bidirectional{path.suffix}"
        bidir_df.to_csv(bidir_filename, index=False)
        print(f"Saved {len(bidir_tracks)} bidirectional linear tracks to {bidir_filename}")


def collect_file_statistics(df, filename, stats_data):
    """Collect statistics for a processed file for later stats reporting.

    Args:
        df: DataFrame with metrics
        filename: Name of the file (without path)
        stats_data: Dictionary to store the statistics
    """
    if filename not in stats_data:
        stats_data[filename] = {}

    # total_tracks: every track in the file; analysed_tracks: those that met the
    # length cutoff and got metrics. Mobility percentages are of analysed tracks.
    total_tracks = df['track_number'].nunique()
    stats_data[filename]['total_tracks'] = total_tracks

    if 'mobility_classification' in df.columns:
        track_mobility = df.groupby('track_number')['mobility_classification'].first()
        analysed_tracks = int(track_mobility.notna().sum())
        stats_data[filename]['analysed_tracks'] = analysed_tracks
        stats_data[filename]['short_tracks'] = total_tracks - analysed_tracks
        mobile_tracks = (track_mobility == 'mobile').sum()
        immobile_tracks = (track_mobility == 'immobile').sum()
        pct_mobile = (mobile_tracks / analysed_tracks) * 100 if analysed_tracks > 0 else 0
        pct_immobile = (immobile_tracks / analysed_tracks) * 100 if analysed_tracks > 0 else 0
        stats_data[filename]['mobile_tracks'] = mobile_tracks
        stats_data[filename]['immobile_tracks'] = immobile_tracks
        stats_data[filename]['pct_mobile'] = pct_mobile
        stats_data[filename]['pct_immobile'] = pct_immobile

    if 'mobility_classification' in df.columns and 'linear_classification' in df.columns:
        mobile_df = df[df['mobility_classification'] == 'mobile']
        mobile_total = mobile_df['track_number'].nunique()
        track_linearity = mobile_df.groupby('track_number')['linear_classification'].first()

        linear_tracks = sum(
            track_linearity.isin(['linear', 'linear_unidirectional', 'linear_bidirectional'])
        )
        nonlinear_tracks = sum(track_linearity == 'non_linear')
        unidirectional_tracks = sum(track_linearity == 'linear_unidirectional')
        bidirectional_tracks = sum(track_linearity == 'linear_bidirectional')

        pct_linear = (linear_tracks / mobile_total) * 100 if mobile_total > 0 else 0
        pct_nonlinear = (nonlinear_tracks / mobile_total) * 100 if mobile_total > 0 else 0
        pct_unidirectional = (unidirectional_tracks / linear_tracks) * 100 if linear_tracks > 0 else 0
        pct_bidirectional = (bidirectional_tracks / linear_tracks) * 100 if linear_tracks > 0 else 0

        stats_data[filename].update({
            'linear_tracks': linear_tracks,
            'nonlinear_tracks': nonlinear_tracks,
            'unidirectional_tracks': unidirectional_tracks,
            'bidirectional_tracks': bidirectional_tracks,
            'pct_linear': pct_linear,
            'pct_nonlinear': pct_nonlinear,
            'pct_unidirectional': pct_unidirectional,
            'pct_bidirectional': pct_bidirectional,
        })

    # sRg statistics
    if 'sRg' in df.columns:
        all_srg = df.groupby('track_number')['sRg'].first().dropna()
        stats_data[filename]['mean_sRg_all'] = all_srg.mean()
        stats_data[filename]['se_sRg_all'] = all_srg.sem()

        if 'mobility_classification' in df.columns:
            mobile_ids = df[df['mobility_classification'] == 'mobile']['track_number'].unique()
            mobile_srg = df[df['track_number'].isin(mobile_ids)].groupby('track_number')['sRg'].first().dropna()
            stats_data[filename]['mean_sRg_mobile'] = mobile_srg.mean()
            stats_data[filename]['se_sRg_mobile'] = mobile_srg.sem()

    # Rg statistics
    if 'rg' in df.columns:
        all_rg = df.groupby('track_number')['rg'].first().dropna()
        stats_data[filename]['mean_rg_all'] = all_rg.mean()
        stats_data[filename]['se_rg_all'] = all_rg.sem()

        if 'mobility_classification' in df.columns:
            mobile_ids = df[df['mobility_classification'] == 'mobile']['track_number'].unique()
            mobile_rg = df[df['track_number'].isin(mobile_ids)].groupby('track_number')['rg'].first().dropna()
            stats_data[filename]['mean_rg_mobile'] = mobile_rg.mean()
            stats_data[filename]['se_rg_mobile'] = mobile_rg.sem()

    # Step length statistics
    if 'mean_step_length' in df.columns:
        all_step = df.groupby('track_number')['mean_step_length'].first().dropna()
        stats_data[filename]['mean_step_length_all'] = all_step.mean()
        stats_data[filename]['se_step_length_all'] = all_step.sem()

        if 'mobility_classification' in df.columns:
            mobile_ids = df[df['mobility_classification'] == 'mobile']['track_number'].unique()
            mobile_step = df[df['track_number'].isin(mobile_ids)].groupby('track_number')['mean_step_length'].first().dropna()
            stats_data[filename]['mean_step_length_mobile'] = mobile_step.mean()
            stats_data[filename]['se_step_length_mobile'] = mobile_step.sem()


def generate_stats_file(stats_data, directory_path, output_dir=None):
    """Generate a statistics summary file from collected data.

    Args:
        stats_data: Dictionary containing statistics data
        directory_path: Input directory; the file goes to its 'results' subdirectory
        output_dir: If given, the file goes here instead (next to the _metrics files)
    """
    if not stats_data:
        print("No statistics data available")
        return

    stats_df = pd.DataFrame.from_dict(stats_data, orient='index')

    aggregate_stats = {}
    numeric_columns = stats_df.select_dtypes(include=[np.number]).columns

    for col in numeric_columns:
        aggregate_stats[f'mean_{col}'] = stats_df[col].mean()
        aggregate_stats[f'se_{col}'] = stats_df[col].sem()

    stats_df.loc['AGGREGATE'] = pd.Series(aggregate_stats)

    output_path = Path(output_dir) if output_dir else Path(directory_path) / 'results'
    output_path.mkdir(parents=True, exist_ok=True)
    stats_file = output_path / 'experiment_statistics.csv'
    stats_df.to_csv(stats_file)
    print(f"Saved statistics summary to {stats_file}")


def process_directory(directory_path, cutoff_len=None, sRg_cutoff=None,
                      linear_eig_ratio_cutoff=None, linear_step_align_cutoff=None,
                      method=None, include_shape_metrics=True,
                      include_linear_metrics=True,
                      suffix='_metrics', split_outputs=False, mobile_only=False,
                      output_dir=None, generate_stats=False,
                      linear_directionality_cutoff=None,
                      linear_perpendicular_cutoff=None):
    """Process all CSV files in a directory.

    Args:
        directory_path: Path to directory containing CSV files
        cutoff_len: Minimum trajectory length
        sRg_cutoff: Threshold for mobile/immobile classification
        linear_eig_ratio_cutoff: Threshold for eigenvalue ratio
        linear_step_align_cutoff: Threshold for step alignment
        method: Method to calculate Rg ('simple' or 'tensor')
        include_shape_metrics: Whether to include shape metrics
        include_linear_metrics: Whether to include linear motion metrics
        suffix: Suffix for the output filenames
        split_outputs: Whether to split output by classification
        mobile_only: Whether to only include mobile tracks in split outputs
        output_dir: Directory for output files (None = same as input)
        generate_stats: Whether to generate a statistics summary file
        linear_directionality_cutoff: Threshold for directionality ratio (simple)
        linear_perpendicular_cutoff: Threshold for perpendicular distance (simple)
    """
    directory = Path(directory_path)

    if not directory.exists() or not directory.is_dir():
        print(f"Error: Directory {directory_path} does not exist")
        return

    csv_files = list(directory.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {directory_path}")
        return

    print(f"Found {len(csv_files)} CSV files in {directory_path}")

    stats_data = {} if generate_stats else None

    for file_path in csv_files:
        if suffix in file_path.stem:
            print(f"Skipping already processed file: {file_path}")
            continue

        df = process_csv_file(
            file_path, cutoff_len, sRg_cutoff,
            linear_eig_ratio_cutoff, linear_step_align_cutoff,
            method, include_shape_metrics, include_linear_metrics,
            linear_directionality_cutoff, linear_perpendicular_cutoff
        )

        if df is not None:
            save_processed_data(
                df, file_path, suffix, split_outputs,
                mobile_only, output_dir, stats_data
            )

    if generate_stats and stats_data:
        generate_stats_file(stats_data, directory_path, output_dir)

    print("Batch processing complete.")
