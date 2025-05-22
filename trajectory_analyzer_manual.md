# Trajectory Analyzer User Manual

## Table of Contents
1. [Introduction](#introduction)
2. [Installation and Setup](#installation-and-setup)
3. [Input Data Requirements](#input-data-requirements)
4. [Quick Start Guide](#quick-start-guide)
5. [Simple Method Analysis](#simple-method-analysis)
6. [Tensor-Based Analysis](#tensor-based-analysis)
7. [Output Files and Interpretation](#output-files-and-interpretation)
8. [Advanced Features](#advanced-features)
9. [Troubleshooting](#troubleshooting)
10. [Examples and Use Cases](#examples-and-use-cases)

---

## 1. Introduction

The Trajectory Analyzer is a Python tool for analyzing particle trajectories from single-particle tracking experiments. It calculates mobility metrics and classifies trajectories as:
- **Mobile vs Immobile** (based on scaled radius of gyration)
- **Linear vs Non-linear** (based on geometric or tensor analysis)
- **Unidirectional vs Bidirectional** (for linear trajectories)

### Key Features
- Two analysis methods: Simple (geometric) and Tensor-based
- Batch processing of multiple files
- Automatic trajectory classification
- Statistical summary generation
- Flexible output options

---

## 2. Installation and Setup

### Requirements
```bash
# Required Python packages
pip install numpy pandas scipy
```

### Download and Setup
1. Save `trajectory_analyzer.py` to your working directory
2. Make it executable (Linux/Mac):
   ```bash
   chmod +x trajectory_analyzer.py
   ```

### Testing Installation
```bash
python trajectory_analyzer.py --help
```

---

## 3. Input Data Requirements

### CSV File Format
Your input CSV must contain these columns:
- `track_number`: Unique identifier for each trajectory
- `frame`: Time point/frame number
- `x`: X-coordinate
- `y`: Y-coordinate

### Example Input
```csv
track_number,frame,x,y
1,0,10.5,20.3
1,1,10.7,20.1
1,2,11.0,19.8
2,0,30.2,40.5
2,1,30.5,40.2
```

### Important Notes
- Coordinates should be in consistent units (pixels, nm, μm)
- Frame numbers can have gaps (missing observations)
- Each track should have at least 3 points

---

## 4. Quick Start Guide

### For Spyder/IDE Users
1. Open `trajectory_analyzer.py`
2. Edit the CONFIG section:
   ```python
   USE_DIRECT_EXECUTION = True
   EXECUTION_MODE = 'file'  # or 'directory'
   FILE_PATH = r"path/to/your/data.csv"
   RG_METHOD = 'simple'  # or 'tensor'
   ```
3. Run the script (F5 in Spyder)

### For Command Line Users
```bash
# Analyze single file with simple method
python trajectory_analyzer.py --file data.csv --method simple

# Analyze directory with tensor method
python trajectory_analyzer.py --directory ./data --method tensor
```

---

## 5. Simple Method Analysis

### Overview
The simple method uses geometric calculations to classify trajectories without matrix operations. It's faster and ideal for large datasets.

### Key Parameters

#### In CONFIG Section:
```python
# Method selection
RG_METHOD = 'simple'

# Classification thresholds
SRG_CUTOFF = 2.22236433588659  # Mobile/immobile threshold
LINEAR_DIRECTIONALITY_CUTOFF = 0.8  # Directionality threshold (0-1)
LINEAR_PERPENDICULAR_CUTOFF = 0.15  # Straightness threshold

# Options
INCLUDE_LINEAR_METRICS = True  # Include directionality metrics
```

#### Command Line:
```bash
python trajectory_analyzer.py --file data.csv --method simple \
    --srg-cutoff 2.222 \
    --linear-directionality 0.8 \
    --linear-perpendicular 0.15
```

### How It Works

1. **Mobility Classification**:
   - Calculates scaled radius of gyration (sRg)
   - Compares to threshold: sRg < 2.222 → Immobile

2. **Linear Motion Classification**:
   - **Directionality Ratio**: net_displacement / total_path_length
     - High (≥0.8) = particle moved in one direction
     - Low (<0.8) = particle returned toward start
   
   - **Perpendicular Distance**: average deviation from straight line
     - Low (≤0.15) = trajectory is straight
     - High (>0.15) = trajectory is curved

3. **Classification Results**:
   - **Linear Unidirectional**: Straight path, one direction
   - **Linear Bidirectional**: Straight path, back-and-forth
   - **Non-linear**: Curved or random path

### Parameter Tuning Guide

| Parameter | Lower Value | Higher Value | Default |
|-----------|------------|--------------|---------|
| Directionality Cutoff | More tracks classified as bidirectional | More stringent unidirectional requirement | 0.8 |
| Perpendicular Cutoff | Only very straight tracks are linear | More curved tracks accepted as linear | 0.15 |

### When to Use Simple Method
- ✅ Large datasets (>10,000 trajectories)
- ✅ Real-time analysis needed
- ✅ Initial screening/exploration
- ✅ Limited computational resources
- ❌ Need shape metrics (asymmetry, skewness)
- ❌ Publishing detailed trajectory analysis

---

## 6. Tensor-Based Analysis

### Overview
The tensor method uses eigenvalue decomposition of the gyration tensor to characterize trajectory shape. It provides more metrics but is computationally intensive.

### Key Parameters

#### In CONFIG Section:
```python
# Method selection
RG_METHOD = 'tensor'

# Classification thresholds
SRG_CUTOFF = 2.22236433588659  # Mobile/immobile threshold
LINEAR_EIGENVALUE_RATIO_CUTOFF = 20.0  # Anisotropy threshold
LINEAR_STEP_ALIGNMENT_CUTOFF = 0.7  # Step alignment threshold

# Options
INCLUDE_SHAPE_METRICS = True  # Include asymmetry, skewness, kurtosis
INCLUDE_LINEAR_METRICS = True  # Include eigenvalue metrics
```

#### Command Line:
```bash
python trajectory_analyzer.py --file data.csv --method tensor \
    --srg-cutoff 2.222 \
    --linear-eigenvalue-ratio 20.0 \
    --linear-step-alignment 0.7 \
    --include-shape-metrics
```

### How It Works

1. **Gyration Tensor Calculation**:
   - Builds 2×2 tensor from trajectory coordinates
   - Extracts eigenvalues (λ₁ ≥ λ₂) and eigenvectors

2. **Metrics Calculated**:
   - **Radius of Gyration**: √(λ₁ + λ₂)
   - **Asymmetry**: Shape elongation measure
   - **Eigenvalue Ratio**: λ₁/λ₂ (trajectory anisotropy)
   - **Step Alignment**: How well steps align with principal axis
   - **Skewness/Kurtosis**: Distribution of steps along principal axis

3. **Classification Logic**:
   - **Linear**: High eigenvalue ratio AND high step alignment
   - **Directional**: Based on net displacement ratio

### Parameter Tuning Guide

| Parameter | Lower Value | Higher Value | Default |
|-----------|------------|--------------|---------|
| Eigenvalue Ratio Cutoff | More tracks classified as linear | More stringent linearity requirement | 20.0 |
| Step Alignment Cutoff | Accept more variable step directions | Require consistent direction | 0.7 |

### Additional Metrics Provided
- **Asymmetry**: 0 = circular, higher = more elongated
- **Skewness**: Asymmetry of motion along principal axis
- **Kurtosis**: Peakedness of step distribution

### When to Use Tensor Method
- ✅ Detailed shape analysis needed
- ✅ Publishing results
- ✅ Small to medium datasets
- ✅ Need theoretical metrics
- ❌ Very large datasets
- ❌ Real-time analysis

---

## 7. Output Files and Interpretation

### Default Output Structure
```
input_directory/
├── data.csv (original)
└── results/
    ├── data_metrics.csv (main output)
    ├── experiment_statistics.csv (if --generate-stats)
    └── mobile_linear/
        └── data_mobile_linear.csv (if --split-outputs)
```

### Main Output File (*_metrics.csv)
Contains all original data plus:

| Column | Description | Simple Method | Tensor Method |
|--------|-------------|---------------|---------------|
| rg | Radius of gyration | ✓ | ✓ |
| mean_step_length | Average step size | ✓ | ✓ |
| sRg | Scaled radius of gyration | ✓ | ✓ |
| mobility_classification | mobile/immobile | ✓ | ✓ |
| linear_classification | linear type or non_linear | ✓ | ✓ |
| directionality_ratio | Net/total displacement | ✓ | ✓ |
| normalized_perpendicular_distance | Deviation from straight | ✓ | ✗ |
| eigenvalue_ratio | λ₁/λ₂ | ✗ | ✓ |
| step_alignment | Alignment with principal axis | ✗ | ✓ |
| asymmetry | Shape elongation | ✗ | ✓ |
| skewness | Motion asymmetry | ✗ | ✓ |
| kurtosis | Step distribution shape | ✗ | ✓ |

### Split Output Files (--split-outputs)
Creates separate files for different classifications:
- `mobile/`: All mobile tracks
- `mobile_linear/`: Mobile tracks classified as linear
- `mobile_nonlinear/`: Mobile tracks classified as non-linear
- `mobile_linear_unidirectional/`: Unidirectional linear tracks
- `mobile_linear_bidirectional/`: Bidirectional linear tracks

### Statistics File (experiment_statistics.csv)
Summary statistics across all files:
- Track counts by classification
- Mean ± SE for all metrics
- Percentage breakdowns
- Aggregate statistics

---

## 8. Advanced Features

### Batch Processing
```bash
# Process entire directory
python trajectory_analyzer.py --directory ./data --method simple

# With custom output directory
python trajectory_analyzer.py --directory ./data --output-dir ./analysis
```

### Filtering Options
```python
# Minimum trajectory length
CUTOFF_LEN = 10  # Only analyze tracks with ≥10 points

# Mobile tracks only for split outputs
MOBILE_ONLY = True  # Don't create immobile track files
```

### Output Customization
```python
# Change output suffix
OUTPUT_SUFFIX = '_analyzed'  # Creates data_analyzed.csv

# Control what's included
INCLUDE_SHAPE_METRICS = False  # Faster, smaller output files
INCLUDE_LINEAR_METRICS = False  # Minimal output
```

### Command Line Options
```bash
# Full example with all options
python trajectory_analyzer.py \
    --directory ./experiments \
    --method tensor \
    --cutoff-len 5 \
    --srg-cutoff 2.0 \
    --linear-eigenvalue-ratio 15.0 \
    --linear-step-alignment 0.75 \
    --include-shape-metrics \
    --split-outputs \
    --mobile-only \
    --generate-stats \
    --output-dir ./results \
    --suffix _complete
```

---

## 9. Troubleshooting

### Common Issues and Solutions

#### "Missing required columns"
- **Problem**: Input file doesn't have track_number, frame, x, y
- **Solution**: Rename columns to match requirements

#### "No tracks with length >= cutoff"
- **Problem**: All trajectories shorter than CUTOFF_LEN
- **Solution**: Lower CUTOFF_LEN or check data quality

#### Memory Error with Large Files
- **Problem**: Too many trajectories for available RAM
- **Solution**: 
  - Use simple method instead of tensor
  - Process in batches
  - Increase system memory

#### All Tracks Classified as Immobile
- **Problem**: sRg threshold too high for your data
- **Solution**: 
  - Check coordinate units (should be consistent)
  - Adjust SRG_CUTOFF based on your system
  - Verify tracking quality

#### No Linear Tracks Found
- **Problem**: Classification thresholds too stringent
- **Solution**:
  - Simple method: Increase LINEAR_PERPENDICULAR_CUTOFF
  - Tensor method: Decrease LINEAR_EIGENVALUE_RATIO_CUTOFF

### Performance Tips
1. **For faster processing**:
   - Use simple method
   - Set INCLUDE_SHAPE_METRICS = False
   - Process files in parallel (multiple instances)

2. **For better classification**:
   - Use longer trajectories (>20 frames)
   - Ensure consistent time intervals
   - Pre-filter noisy tracks

---

## 10. Examples and Use Cases

### Example 1: Quick Screening of Large Dataset
```python
# CONFIG settings for screening
USE_DIRECT_EXECUTION = True
EXECUTION_MODE = 'directory'
DIRECTORY_PATH = r"./screening_data"
RG_METHOD = 'simple'
CUTOFF_LEN = 5
SPLIT_OUTPUTS = False  # Just need metrics
GENERATE_STATS = True  # Get summary
```

### Example 2: Detailed Analysis for Publication
```python
# CONFIG settings for publication
RG_METHOD = 'tensor'
CUTOFF_LEN = 10  # Higher quality tracks only
INCLUDE_SHAPE_METRICS = True
INCLUDE_LINEAR_METRICS = True
SPLIT_OUTPUTS = True  # Separate files for figures
MOBILE_ONLY = True  # Focus on mobile fraction
```

### Example 3: Custom Thresholds for Specific System
```bash
# Microtubule transport (expect linear tracks)
python trajectory_analyzer.py --file MT_transport.csv \
    --method simple \
    --linear-directionality 0.7 \
    --linear-perpendicular 0.25

# Confined diffusion (expect low sRg)
python trajectory_analyzer.py --file confined.csv \
    --method tensor \
    --srg-cutoff 1.5 \
    --include-shape-metrics
```

### Example 4: Comparing Methods
```bash
# Run both methods on same data
python trajectory_analyzer.py --file data.csv --method simple --suffix _simple
python trajectory_analyzer.py --file data.csv --method tensor --suffix _tensor

# Compare results in output files
```

### Typical Workflow
1. **Initial exploration**: Use simple method with default parameters
2. **Parameter optimization**: Adjust thresholds based on known tracks
3. **Final analysis**: Use optimized parameters on full dataset
4. **Validation**: Manually check classification of representative tracks
5. **Visualization**: Use trajectory_viewer.py to inspect results

---

## Appendix: Quick Reference Card

### File Processing
```bash
# Single file
python trajectory_analyzer.py --file data.csv --method simple

# Directory
python trajectory_analyzer.py --directory ./data --method tensor

# With options
python trajectory_analyzer.py --file data.csv \
    --method simple \
    --split-outputs \
    --generate-stats
```

### Key Parameters
| Parameter | Simple Default | Tensor Default | Range |
|-----------|---------------|----------------|--------|
| sRg cutoff | 2.222 | 2.222 | 1.0-4.0 |
| Directionality | 0.8 | - | 0.5-0.95 |
| Perpendicular | 0.15 | - | 0.05-0.3 |
| Eigenvalue ratio | - | 20.0 | 5.0-50.0 |
| Step alignment | - | 0.7 | 0.5-0.9 |

### Output Files
- `*_metrics.csv`: Main results
- `*_mobile_linear.csv`: Linear mobile tracks
- `*_mobile_nonlinear.csv`: Non-linear mobile tracks
- `experiment_statistics.csv`: Summary stats

---

For questions or bug reports, please contact: [your contact info]