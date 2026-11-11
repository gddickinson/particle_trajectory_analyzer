"""
cli.py - Command-line interface and entry point for trajectory analysis.

Provides both argparse CLI and direct execution (Spyder/IDE) modes.
This is the main entry point that orchestrates the analysis pipeline.
"""

import os
import sys
import argparse

import config
from io_utils import process_csv_file, save_processed_data, process_directory, generate_stats_file


def run_script(mode, path, cutoff_len=None, sRg_cutoff=None,
               linear_eig_ratio_cutoff=None, linear_step_align_cutoff=None,
               method=None, include_shape_metrics=True,
               include_linear_metrics=True,
               suffix='_metrics', split_outputs=False, mobile_only=False,
               output_dir=None, generate_stats=False,
               linear_directionality_cutoff=None,
               linear_perpendicular_cutoff=None):
    """Run the script with the specified parameters.

    Analysis parameters left as None take their defaults from config.py.

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
    if not path:
        print("No input path given: set FILE_PATH or DIRECTORY_PATH in config.py, "
              "or use --file / --directory on the command line.")
        return
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


def build_parser():
    """Command-line parser; analysis defaults come from config.py."""
    parser = argparse.ArgumentParser(description="Calculate trajectory metrics")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file', help="Path to a CSV file")
    group.add_argument('--directory', help="Path to a directory of CSV files")

    parser.add_argument('--cutoff-len', type=int, default=config.CUTOFF_LEN,
                        help=f"Minimum trajectory length (default: {config.CUTOFF_LEN})")
    parser.add_argument('--srg-cutoff', type=float, default=config.SRG_CUTOFF,
                        help=f"sRg threshold for mobile/immobile classification (default: {config.SRG_CUTOFF})")
    parser.add_argument('--linear-eigenvalue-ratio', type=float, default=config.LINEAR_EIGENVALUE_RATIO_CUTOFF,
                        help="Eigenvalue ratio threshold for linear classification, tensor method "
                             f"(default: {config.LINEAR_EIGENVALUE_RATIO_CUTOFF})")
    parser.add_argument('--linear-step-alignment', type=float, default=config.LINEAR_STEP_ALIGNMENT_CUTOFF,
                        help="Step alignment threshold for linear classification, tensor method "
                             f"(default: {config.LINEAR_STEP_ALIGNMENT_CUTOFF})")
    parser.add_argument('--linear-directionality', type=float, default=config.LINEAR_DIRECTIONALITY_CUTOFF,
                        help=f"Directionality ratio threshold, simple method (default: {config.LINEAR_DIRECTIONALITY_CUTOFF})")
    parser.add_argument('--linear-perpendicular', type=float, default=config.LINEAR_PERPENDICULAR_CUTOFF,
                        help=f"Perpendicular distance threshold, simple method (default: {config.LINEAR_PERPENDICULAR_CUTOFF})")
    parser.add_argument('--method', choices=['simple', 'tensor'], default=config.RG_METHOD,
                        help=f"Method to calculate Rg (default: {config.RG_METHOD})")
    parser.add_argument('--include-shape-metrics', action='store_true', default=config.INCLUDE_SHAPE_METRICS,
                        help=f"Include shape metrics in output (default: {config.INCLUDE_SHAPE_METRICS})")
    parser.add_argument('--no-shape-metrics', action='store_false',
                        dest='include_shape_metrics',
                        help="Do not include shape metrics in output")
    parser.add_argument('--include-linear-metrics', action='store_true', default=config.INCLUDE_LINEAR_METRICS,
                        help=f"Include linear motion metrics in output (default: {config.INCLUDE_LINEAR_METRICS})")
    parser.add_argument('--no-linear-metrics', action='store_false',
                        dest='include_linear_metrics',
                        help="Do not include linear motion metrics in output")
    parser.add_argument('--suffix', default=config.OUTPUT_SUFFIX,
                        help=f"Suffix for output filenames (default: {config.OUTPUT_SUFFIX})")
    parser.add_argument('--split-outputs', action='store_true', default=config.SPLIT_OUTPUTS,
                        help="Create separate files for different classifications")
    parser.add_argument('--mobile-only', action='store_true', default=config.MOBILE_ONLY,
                        help=f"Only include mobile tracks in split outputs (default: {config.MOBILE_ONLY})")
    parser.add_argument('--all-tracks', action='store_false', dest='mobile_only',
                        help="Include immobile tracks in split outputs")
    parser.add_argument('--output-dir',
                        help="Directory for output files (default: same as input)")
    parser.add_argument('--generate-stats', action='store_true', default=config.GENERATE_STATS,
                        help="Generate statistics summary file")
    return parser


def main(argv=None):
    """Main entry point.

    With command-line arguments, they are used (argparse). Without any, and
    with config.USE_DIRECT_EXECUTION set (e.g. running the file in Spyder),
    the settings in config.py are used.
    """
    argv = sys.argv[1:] if argv is None else list(argv)
    if config.USE_DIRECT_EXECUTION and not argv:
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

    args = build_parser().parse_args(argv)
    run_script(
        'file' if args.file else 'directory', args.file or args.directory,
        args.cutoff_len, args.srg_cutoff,
        args.linear_eigenvalue_ratio, args.linear_step_alignment,
        args.method, args.include_shape_metrics, args.include_linear_metrics,
        args.suffix, args.split_outputs, args.mobile_only, args.output_dir,
        args.generate_stats, args.linear_directionality, args.linear_perpendicular,
    )


if __name__ == "__main__":
    main()
