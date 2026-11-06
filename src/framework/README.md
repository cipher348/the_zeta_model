# Framework scripts for the zeta model

This folder contains the analysis and plotting pipeline used to study the Riemann zeta zeros and the empirical calibration of the Duffing-style residual model. The scripts are exploratory and report-oriented: some generate scan data, some generate feature tables, and others produce publication-style figures.

The scripts here are not a single packaged app; they are a sequence of research workflows. In practice the typical order is:

1. Scan and validate zeta zeros with `grand_scan.py`
2. Create geometry feature tables with `grand_scan_v2.py`
3. Analyze geometry and calibration performance with `05_geometric_normalization.py`
4. Run bootstrap validation with `bootstrapping_v2.py`
5. Produce final summary figures with `S150.py`, `S152.py`, `S153.py`, and `figure_2.py`

Important note:
- Several scripts still use placeholder paths such as `your_path\...`.
- Before running them on a real dataset, replace those paths with the actual local paths to the input files on your machine.
- Some scripts write outputs to the current working directory, so it is best to run them from the folder that should contain the generated files.

## Script overview

### 1) `grand_scan.py`
Purpose:
- Large-scale scan of candidate zeta-zero neighborhoods from `t = 10` to `t = 50000`.
- Uses adaptive step sizes and optional validation against a zero list.
- Saves a CSV of scanned points and status flags.

Inputs:
- `zeros1.txt` in the working directory (optional; if missing, zero validation is skipped)
- No hardcoded data file beyond the zero list

Outputs:
- `grand_scan_10_50000.csv`

Key behavior:
- `load_zeros()` reads the zero list from `zeros1.txt`
- `measure_radius_fast()` estimates local radius for each `t`
- `check_zero_hit()` checks whether a sampled `t` falls near a known zero
- The script supports resuming from an existing CSV if a previous run was interrupted

Typical usage:
```bash
python src/framework/grand_scan.py
```

Notes:
- This is a high-cost scan and may take a while depending on the scan range.
- The output file is meant for exploratory search and zero validation rather than immediate model fitting.

### 2) `grand_scan_v2.py`
Purpose:
- Produces feature tables for zero analysis and downstream modelling.
- Reads a list of zeta zeros, evaluates each zero with the Duffing-style model, and computes geometry features.
- Builds a dataset for model fitting and gain analysis.

Inputs:
- `zeros1.txt` via `CONFIG['input_file']`
- File path defaults to a placeholder such as `your_path\zeros1.txt`

Outputs:
- `zeta_features_5000.csv`
- `zeta_features_5000.json`

Key generated columns include:
- `idx`, `set`, `t_known`, `t_detected`, `delta_actual`
- `R_prime`, `R_dbl`, `arg_DN`, `tan_angle`
- `hopping_frac`, `sync_index`, `response_magnitude`
- derived features such as `total_weighted`, `g_obs`, `g_obs_abs`, `log_g_obs`

Typical usage:
```bash
python src/framework/grand_scan_v2.py
```

Notes:
- The script explicitly copies the Duffing-Zeta geometry logic from `05_geometric_normalization.py`.
- It writes both a compact CSV and a richer JSON file for downstream analysis.
- The dataset is split into CAL and VAL subsets using `split_ratio`.

### 3) `05_geometric_normalization.py`
Purpose:
- Implements the full geometric normalization methodology.
- Separates sign prediction from magnitude calibration.
- Tests whether geometric quantities predict residual sign and magnitude.

Inputs:
- The known zero list used internally in `KNOWN_ZEROS`
- No external CSV is required for the core analysis

Outputs:
- `session123_geometric_normalization.png`

Key ideas:
- Uses a Duffing-Zeta simulation with parameters:
  - `ALPHA = 12.89`
  - `BETA = 370.92`
  - `GAMMA = 0.4811`
  - `LAM_OPT = 112.4`
  - `PHI_OPT = 2.035`
- Extracts geometry features such as:
  - `slope_R_prime`
  - `curvature_R_double_prime`
  - `complex_phase_arg_DN`
  - `tangent_angle`
  - `hopping_frac`
  - `sync_index`
- Uses a two-stage analysis:
  1. Predict the sign of `Δ`
  2. Fit magnitude only for correct-sign cases

Typical usage:
```bash
python src/framework/05_geometric_normalization.py
```

Notes:
- This script is more of a research experiment and diagnostic analysis than a reusable pipeline component.
- It saves a multi-panel summary figure to the current directory.

### 4) `bootstrapping_v2.py`
Purpose:
- Bootstrap validation of the S153-style calibration convention.
- Tests coverage, MAE, RMSE, parameter stability, and offset stability.

Inputs:
- `your_path\zeta_features_5000.csv` (placeholder path in the script)
- Expects a dataset with columns such as `total_weighted`, `delta_actual`, `s`, `align_state`, `g_norm`, and `set`

Outputs:
- `session152_bootstrap_final_results_5000.csv`
- `session152_bootstrap_s153_convention.png`
- `session152_bootstrap_s153_convention.pdf`
- `session152_summary_s153_convention.json`

Key behavior:
- Loads the dataset and derives alignment state and normalized gain values
- Fits the mean function by bootstrap resampling of the CAL subset
- Evaluates on the fixed VAL subset
- Produces coverage and error metrics for each bootstrap run

Typical usage:
```bash
python src/framework/bootstrapping_v2.py
```

Notes:
- This is the main robustness check in the framework.
- The script prints a verdict such as `VALIDATED FOR DEPLOYMENT` or `NEEDS REVIEW`.

### 5) `S150.py`
Purpose:
- Publishes a figure related to quantile-band calibration.
- Produces a publication-quality visualization of the calibration behavior.

Inputs:
- No input data file is required; it uses internally defined calibration parameters and simulated window data

Outputs:
- `figure1_calibration_s150.png`
- `figure1_calibration_s150.pdf`

Typical usage:
```bash
python src/framework/S150.py
```

Notes:
- This script is a report/figure generator rather than a data-processing step.
- It is useful when you want a clean visual summary of the calibration method.

### 6) `S152.py`
Purpose:
- Creates a report on bootstrap coverage stability.
- Reads the bootstrap CSV and produces a summary figure.

Inputs:
- `session152_bootstrap_final_results_5000.csv`

Outputs:
- `figure3_bootstrap_s152.png`
- `figure3_bootstrap_s152.pdf`

Typical usage:
```bash
python src/framework/S152.py
```

Notes:
- This script is intended to visualize the results produced by `bootstrapping_v2.py`.
- It uses summary statistics built into the script and plots histograms for coverage, error metrics, and offset behavior.

### 7) `S153.py`
Purpose:
- Creates a production/deployment-style summary figure.
- Simulates predictive behavior consistent with the reported deployment metrics.

Inputs:
- No real external dataset is required; the script generates synthetic data matching stated model metrics

Outputs:
- `figure4_deployment_s153.png`
- `figure4_deployment_s153.pdf`

Typical usage:
```bash
python src/framework/S153.py
```

Notes:
- This script is for presentation and communication of the final model performance, not for actual model execution.
- It is not needed for the core modeling pipeline.

### 8) `figure_2.py`
Purpose:
- Quick plot of the mean function `ĝ(S)` and its relative bootstrap behavior.

Inputs:
- Uses built-in baseline and bootstrap statistics embedded in the script

Outputs:
- `figure2_g_vs_S.png`

Typical usage:
```bash
python src/framework/figure_2.py
```

Notes:
- This is a light-weight visual summary useful for quickly checking the qualitative shape of the mean function.

## Typical execution order

For a complete pipeline, the usual sequence is:

```bash
python src/framework/grand_scan.py
python src/framework/grand_scan_v2.py
python src/framework/05_geometric_normalization.py
python src/framework/bootstrapping_v2.py
python src/framework/S152.py
python src/framework/S153.py
python src/framework/S150.py
python src/framework/figure_2.py
```

This ordering reflects the logical workflow:
- scan or collect data
- engineer features
- inspect geometry and sign/magnitude behavior
- validate by bootstrap
- render figures for reporting

## Files commonly involved in the pipeline

These are the main data artifacts referenced across the scripts:

- `zeros1.txt` — input zero list used for scanning and feature generation
- `grand_scan_10_50000.csv` — raw scan output
- `zeta_features_5000.csv` — zero-level feature table
- `zeta_features_5000.json` — full structured feature table
- `session152_bootstrap_final_results_5000.csv` — bootstrap statistics table
- `session152_summary_s153_convention.json` — summary JSON for the validated model

## Practical notes

- Replace placeholder paths like `your_path\...` before running scripts that expect data files.
- Some scripts write images to the current working directory; run them from a directory with write permissions.
- If you are using a different Python environment, make sure the required packages are installed, especially:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scipy`
  - `scikit-learn`
  - `tqdm`

## Summary

This folder contains a research pipeline for zeta-zero analysis and residual-model calibration. The scripts are organized into:

- data acquisition (`grand_scan.py`)
- feature generation (`grand_scan_v2.py`)
- geometry-based analysis (`05_geometric_normalization.py`)
- bootstrap validation (`bootstrapping_v2.py`)
- figure generation and reporting (`S150.py`, `S152.py`, `S153.py`, `figure_2.py`)

Together they form a workflow for exploring the zeta zero geometry, modeling residual behavior, and summarizing the final method in visual form.

