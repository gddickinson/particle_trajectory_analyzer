"""
config.py - Default configuration for trajectory analysis.

Contains all default parameters for trajectory metrics calculation,
classification thresholds, and output options.
"""

# Set to True to use direct execution in Spyder (False for command line)
USE_DIRECT_EXECUTION = True

# Choose single file or directory mode ('file' or 'directory')
EXECUTION_MODE = 'directory'

# Path to single file (for EXECUTION_MODE = 'file')
FILE_PATH = None

# Path to directory with CSV files (for EXECUTION_MODE = 'directory')
DIRECTORY_PATH = None

# Analysis parameters
CUTOFF_LEN = 3  # Minimum trajectory length
SRG_CUTOFF = 2.22236433588659  # Threshold for mobile/immobile classification

# Linear classification parameters for tensor method
LINEAR_EIGENVALUE_RATIO_CUTOFF = 20.0
LINEAR_STEP_ALIGNMENT_CUTOFF = 0.7

# Linear classification parameters for simple method
LINEAR_DIRECTIONALITY_CUTOFF = 0.8
LINEAR_PERPENDICULAR_CUTOFF = 0.15

# Analysis options
RG_METHOD = 'simple'
INCLUDE_SHAPE_METRICS = True   # asymmetry, skewness, kurtosis (tensor method only)
INCLUDE_LINEAR_METRICS = True  # directionality / perpendicular or eigenvalue metrics (README 5.1)

# Output options
OUTPUT_SUFFIX = '_metrics'
SPLIT_OUTPUTS = False
GENERATE_STATS = False
MOBILE_ONLY = True  # split outputs: mobile tracks only (manual section 7: mobile_linear/ ...)
OUTPUT_DIR = None


def resolve(value, name):
    """Return value, or this module's default `name` when value is None.

    config.py is the single source of the analysis defaults for the CLI,
    direct execution and io_utils.process_csv_file / process_directory.
    """
    return globals()[name] if value is None else value
