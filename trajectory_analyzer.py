#!/usr/bin/env python3
"""
trajectory_analyzer.py - Backward-compatible wrapper for trajectory analysis.

This module has been split into smaller modules for maintainability:
- config.py: Default configuration and parameters
- metrics.py: Rg, sRg, step length, tensor-based shape metrics
- classification.py: Linear/non-linear trajectory classification
- io_utils.py: CSV reading/writing, batch processing, statistics
- cli.py: Command-line interface and entry point

This file re-exports all public functions for backward compatibility.
For new code, import directly from the appropriate module.

Usage in Spyder:
  1. Edit config.py (or the CONFIG section below for legacy use)
  2. Run this script or cli.py

Command-line usage:
  python cli.py --file <path_to_csv>
  python cli.py --directory <path_to_dir>
  python trajectory_analyzer.py --file <path_to_csv>
"""

# Re-export all public functions for backward compatibility
from metrics import (
    get_radius_of_gyration_simple,
    get_trajectory_metrics_tensor,
    get_mean_step_length,
    get_scaled_rg,
    calculate_step_angle_variation,
)
from classification import (
    classify_linear_motion_simple,
    classify_linear_motion,
)
from io_utils import (
    validate_csv_columns,
    process_csv_file,
    save_processed_data,
    collect_file_statistics,
    generate_stats_file,
    process_directory,
)
from cli import run_script, main
from config import (
    USE_DIRECT_EXECUTION,
    EXECUTION_MODE,
    FILE_PATH,
    DIRECTORY_PATH,
    CUTOFF_LEN,
    SRG_CUTOFF,
    LINEAR_EIGENVALUE_RATIO_CUTOFF,
    LINEAR_STEP_ALIGNMENT_CUTOFF,
    LINEAR_DIRECTIONALITY_CUTOFF,
    LINEAR_PERPENDICULAR_CUTOFF,
    RG_METHOD,
    INCLUDE_SHAPE_METRICS,
    INCLUDE_LINEAR_METRICS,
    OUTPUT_SUFFIX,
    SPLIT_OUTPUTS,
    GENERATE_STATS,
    MOBILE_ONLY,
    OUTPUT_DIR,
)


if __name__ == "__main__":
    main()
