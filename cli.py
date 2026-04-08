"""
cli.py - Command-line interface and entry point for trajectory analysis.

Provides both argparse CLI and direct execution (Spyder/IDE) modes.
This is the main entry point that orchestrates the analysis pipeline.
"""

import os
import argparse

import config
from io_utils import process_csv_file, save_processed_data, process_directory, generate_stats_file


def run_script(mode, path, cutoff_len=3, sRg_cutoff=2.22236433588659,
               linear_eig_ratio_cutoff=5.0, linear_step_align_cutoff=0.7,
               method='tensor', include_shape_metrics=True,
               include_linear_metrics=True,
               suffix='_metrics', split_outputs=False, mobile_only=False,
               output_dir=None, generate_stats=False,
               linear_directionality_cutoff=0.8,
               linear_perpendicular_cutoff=0.15):
    """Run the script with the specified parameters.

    Args:
        mode: 'file' or 'directory'
        path: Path to file or directory
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
    if mode == 'file':
        df = process_csv_file(
            path, cutoff_len, sRg_cutoff,
            linear_eig_ratio_cutoff, linear_step_align_cutoff,
            method, include_shape_metrics, include_linear_metrics,
            linear_directionality_cutoff, linear_perpendicular_cutoff
        )
        if df is not None:
            stats_data = {} if generate_stats else None
            save_processed_data(
                df, path, suffix, split_outputs,
                mobile_only, output_dir, stats_data
            )
            if generate_stats and stats_data:
                generate_stats_file(stats_data, os.path.dirname(path))
    elif mode == 'directory':
        process_directory(
            path, cutoff_len, sRg_cutoff,
            linear_eig_ratio_cutoff, linear_step_align_cutoff,
            method, include_shape_metrics, include_linear_metrics,
            suffix, split_outputs, mobile_only, output_dir,
            generate_stats, linear_directionality_cutoff,
            linear_perpendicular_cutoff
        )
    else:
        print(f"Invalid mode: {mode}. Use 'file' or 'directory'.")


def main():
    """Main entry point: handles both direct execution and CLI modes."""
    if config.USE_DIRECT_EXECUTION:
        print("Running with direct execution settings from config")
        run_script(
            config.EXECUTION_MODE,
            config.FILE_PATH if config.EXECUTION_MODE == 'file' else config.DIRECTORY_PATH,
            config.CUTOFF_LEN,
            config.SRG_CUTOFF,
            config.LINEAR_EIGENVALUE_RATIO_CUTOFF,
            config.LINEAR_STEP_ALIGNMENT_CUTOFF,
            config.RG_METHOD,
            config.INCLUDE_SHAPE_METRICS,
            config.INCLUDE_LINEAR_METRICS,
            config.OUTPUT_SUFFIX,
            config.SPLIT_OUTPUTS,
            config.MOBILE_ONLY,
            config.OUTPUT_DIR,
            config.GENERATE_STATS,
            config.LINEAR_DIRECTIONALITY_CUTOFF,
            config.LINEAR_PERPENDICULAR_CUTOFF,
        )
        return

    parser = argparse.ArgumentParser(description="Calculate trajectory metrics")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file', help="Path to a CSV file")
    group.add_argument('--directory', help="Path to a directory of CSV files")

    parser.add_argument('--cutoff-len', type=int, default=3,
                        help="Minimum trajectory length (default: 3)")
    parser.add_argument('--srg-cutoff', type=float, default=2.22236433588659,
                        help="sRg threshold for mobile/immobile classification")
    parser.add_argument('--linear-eigenvalue-ratio', type=float, default=5.0,
                        help="Eigenvalue ratio threshold for linear classification")
    parser.add_argument('--linear-step-alignment', type=float, default=0.7,
                        help="Step alignment threshold for linear classification")
    parser.add_argument('--linear-directionality', type=float, default=0.8,
                        help="Directionality ratio threshold (simple method)")
    parser.add_argument('--linear-perpendicular', type=float, default=0.15,
                        help="Perpendicular distance threshold (simple method)")
    parser.add_argument('--method', choices=['simple', 'tensor'], default='tensor',
                        help="Method to calculate Rg (default: tensor)")
    parser.add_argument('--include-shape-metrics', action='store_true', default=True,
                        help="Include shape metrics in output")
    parser.add_argument('--no-shape-metrics', action='store_false',
                        dest='include_shape_metrics',
                        help="Do not include shape metrics in output")
    parser.add_argument('--include-linear-metrics', action='store_true', default=True,
                        help="Include linear motion metrics in output")
    parser.add_argument('--no-linear-metrics', action='store_false',
                        dest='include_linear_metrics',
                        help="Do not include linear motion metrics in output")
    parser.add_argument('--suffix', default='_metrics',
                        help="Suffix for output filenames (default: _metrics)")
    parser.add_argument('--split-outputs', action='store_true', default=False,
                        help="Create separate files for different classifications")
    parser.add_argument('--mobile-only', action='store_true', default=False,
                        help="Only include mobile tracks in split outputs")
    parser.add_argument('--output-dir',
                        help="Directory for output files (default: same as input)")
    parser.add_argument('--generate-stats', action='store_true', default=False,
                        help="Generate statistics summary file")

    args = parser.parse_args()

    if args.file:
        df = process_csv_file(
            args.file, args.cutoff_len, args.srg_cutoff,
            args.linear_eigenvalue_ratio, args.linear_step_alignment,
            args.method, args.include_shape_metrics, args.include_linear_metrics,
            args.linear_directionality, args.linear_perpendicular
        )
        if df is not None:
            save_processed_data(
                df, args.file, args.suffix, args.split_outputs,
                args.mobile_only, args.output_dir
            )
    elif args.directory:
        process_directory(
            args.directory, args.cutoff_len, args.srg_cutoff,
            args.linear_eigenvalue_ratio, args.linear_step_alignment,
            args.method, args.include_shape_metrics, args.include_linear_metrics,
            args.suffix, args.split_outputs, args.mobile_only, args.output_dir,
            args.generate_stats, args.linear_directionality, args.linear_perpendicular
        )


if __name__ == "__main__":
    main()
