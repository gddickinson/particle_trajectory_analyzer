# Particle Trajectory Analyzer — Roadmap

## Current State
A single-file analysis tool (`trajectory_analyzer.py`) implementing geometric trajectory classification. Calculates radius of gyration (Rg), scaled Rg (sRg), directionality ratio, and normalized perpendicular distance to classify trajectories as linear unidirectional, linear bidirectional, or non-linear. Supports both simple geometric and tensor-based methods. Has a CONFIG section for in-script configuration plus argparse CLI. Includes reference data in `alfredo_sRg/`, test data in `testing/`, reference papers in `refs/`, and a manual (`trajectory_analyzer_manual.md`). Scientifically rigorous with clear mathematical documentation in the README.

## Short-term Improvements
- [ ] Split `trajectory_analyzer.py` into modules: `metrics.py` (Rg, sRg, directionality), `classification.py` (linear/nonlinear), `io_utils.py` (CSV reading/writing), `cli.py` (argparse), `config.py` (defaults)
- [ ] Add a `requirements.txt` (numpy, pandas, scipy — minimal dependencies)
- [ ] Add input validation for CSV columns — check that `track_number`, `frame`, `x`, `y` exist before processing
- [ ] Handle edge cases: tracks with <3 points, tracks with identical positions, NaN values in coordinates
- [ ] Add progress reporting for batch processing large directories
- [ ] Replace the CONFIG section with a proper config file (YAML or TOML) for cleaner separation

## Feature Enhancements
- [ ] Add MSD-based diffusion coefficient calculation alongside the geometric metrics
- [ ] Add visualization: plot trajectories colored by classification with a legend
- [ ] Add histogram/distribution plots of sRg, directionality ratio, and perpendicular distance
- [ ] Support additional input formats: TrackMate XML, trackpy HDF5, Icy XML
- [ ] Add threshold auto-tuning using known ground truth data (optimize F1 score)
- [ ] Add a simple GUI (tkinter) for loading files and adjusting thresholds interactively
- [ ] Support 3D trajectories (x, y, z) with appropriate 3D geometric calculations

## Long-term Vision
- [ ] Integrate as a plugin into the `particle_analysis` package for a unified workflow
- [ ] Add machine learning classification (random forest or SVM) trained on geometric features
- [ ] Implement Bayesian change-point detection for trajectories switching motion modes mid-track
- [ ] Build a benchmark suite comparing geometric method accuracy against tensor-based and ML methods
- [ ] Submit as a companion tool for a methods paper on geometric trajectory classification
- [ ] Publish on PyPI as a lightweight, dependency-minimal trajectory analysis library

## Technical Debt
- [ ] The single file likely exceeds 500 lines — modularization is the top priority
- [ ] Add unit tests using synthetic trajectories with known classifications (perfectly linear, perfectly random, confined)
- [ ] The `testing/` directory contents need documentation — what test cases exist and what they validate
- [ ] Add type hints to all functions, especially the metric calculation functions
- [ ] Document the `alfredo_sRg/` reference data — what is it and how was it generated?
- [ ] The manual (`trajectory_analyzer_manual.md`) may duplicate README content — consolidate
