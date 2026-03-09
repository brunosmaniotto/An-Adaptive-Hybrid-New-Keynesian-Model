# Replication Guide

## An Adaptive-Hybrid New-Keynesian Model

**Author:** Bruno Cittolin Smaniotto
**Last Updated:** February 2026

This document provides complete instructions for replicating all results in "An Adaptive-Hybrid New-Keynesian Model."

---

## Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Data Sources](#data-sources)
5. [Full Replication](#full-replication)
6. [Replicating Figures](#replicating-figures)
7. [Replicating Tables](#replicating-tables)
8. [Replicating Empirical Analysis](#replicating-empirical-analysis)
9. [Model Documentation](#model-documentation)
10. [Expected Output](#expected-output)
11. [Troubleshooting](#troubleshooting)

---

## Overview

This replication package contains all code necessary to reproduce the figures, tables, and empirical analysis in the paper. The code is organized as follows:

```
├── code/
│   ├── models/            # Core model implementations
│   ├── simulations/       # Simulation scripts (organized by paper section)
│   ├── plotting/          # Figure generation scripts
│   ├── tables/            # Table generation scripts
│   ├── empirical/         # Empirical analysis scripts
│   └── run_all.py         # Master replication script
├── data/
│   └── raw/               # Raw data files (FRED CPI)
├── output/
│   ├── figures/           # Generated figures (PNG and PDF), by section
│   ├── tables/            # Generated tables (CSV), by section
│   ├── simulations/       # Cached simulation data (.pkl)
│   └── empirical/         # Empirical analysis output
├── manuscript/            # LaTeX source files
├── REPLICATION.md         # This file
└── README.md              # Repository overview
```

**Estimated replication time:**
- Full replication (all figures and tables): ~15-30 minutes
- Individual figures: ~10-60 seconds each
- Empirical analysis: ~2-5 minutes (requires FRED API or cached data)

---

## System Requirements

### Software
- **Python:** 3.9 or higher
- **Operating System:** Windows, macOS, or Linux
- **LaTeX:** For compiling the manuscript (optional)

### Hardware
- **RAM:** 4GB minimum, 8GB recommended
- **Disk Space:** ~500MB for code and output

---

## Installation

### Step 1: Clone or Download the Repository

```bash
git clone https://github.com/brunosmaniotto/An-Adaptive-Hybrid-New-Keynesian-Model.git
cd An-Adaptive-Hybrid-New-Keynesian-Model
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` includes:
- `numpy>=1.21.0` - Numerical computing
- `scipy>=1.7.0` - Scientific computing
- `pandas>=1.3.0` - Data manipulation
- `matplotlib>=3.4.0` - Plotting
- `seaborn>=0.11.0` - Visualization
- `statsmodels>=0.13.0` - Econometrics
- `fredapi>=0.5.0` - FRED data access

### Step 4: Configure FRED API Access (For Empirical Analysis)

The empirical scripts require a FRED API key. Get one free at: https://fred.stlouisfed.org/docs/api/api_key.html

**Option A:** Environment variable
```bash
export FRED_API_KEY="your_key_here"
```

**Option B:** Configuration file
Create `config.json` in the project root:
```json
{"FRED_API_KEY": "your_key_here"}
```

**Note:** Raw data is included in `data/raw/fred_data_raw.csv`, so empirical scripts can run without the API key using cached data.

---

## Data Sources

### Empirical Data

| Data | Source | File |
|------|--------|------|
| U.S. CPI Inflation | FRED (CPIAUCSL) | `data/raw/fred_data_raw.csv` |

The empirical analysis uses quarterly U.S. CPI inflation from 1960Q1 to 2024Q4.

### Simulated Data

All other figures use model simulations. No external data is required.

---

## Full Replication

The easiest way to replicate everything is via the master script:

```bash
python code/run_all.py
```

This runs four phases in order:

1. **Simulations** -- runs all model simulations, saves `.pkl` files to `output/simulations/`
2. **Empirical** -- runs persistence analysis, saves to `output/empirical/`
3. **Figures** -- generates all figures from cached data, saves to `output/figures/`
4. **Tables** -- generates all tables, saves to `output/tables/`

Options:

```bash
python code/run_all.py --figures  # Only regenerate figures (requires cached simulation data)
python code/run_all.py --tables   # Only regenerate tables (requires cached simulation data)
```

---

## Replicating Figures

### Individual Figures

Each figure consists of a simulation script (which produces data) and a plotting script (which reads data and produces the figure). Both can be run independently from the project root.

| Figure (Paper) | Simulation Script | Plotting Script | Output |
|--------|----------|---------|--------|
| Fig 1: Learning Mechanism | `code/simulations/section_2/learning_mechanism.py` | `code/plotting/section_2/figure_01_learning_mechanism.py` | `output/figures/section_2/figure_01_learning_mechanism.png` |
| Fig 2: Roles of k and e | `code/simulations/section_2/k_epsilon_roles.py` | `code/plotting/section_2/figure_02_k_epsilon_roles.py` | `output/figures/section_2/figure_02_k_epsilon_roles.png` |
| Fig 3: Lambda Comparison | `code/simulations/section_2/lambda_comparison.py` | `code/plotting/section_2/figure_03_lambda_comparison.py` | `output/figures/section_2/figure_03_lambda_comparison.png` |
| Fig 4: Shock Regimes | `code/simulations/section_2/shock_regimes.py` | `code/plotting/section_2/figure_04_shock_regimes.py` | `output/figures/section_2/figure_04_shock_regimes.png` |
| Fig 5: Persistence | `code/empirical/section_2/persistence_analysis.py` | `code/plotting/section_2/figure_05_persistence.py` | `output/figures/section_2/figure_05_persistence.png` |
| Fig 6: Oil Shocks | `code/simulations/section_3/oil_shocks.py` | `code/plotting/section_3/figure_06_oil_shocks.py` | `output/figures/section_3/figure_06_oil_shocks.png` |
| Fig 7: Great Moderation | `code/simulations/section_3/great_moderation.py` | `code/plotting/section_3/figure_07_great_moderation.py` | `output/figures/section_3/figure_07_great_moderation.png` |
| Fig 8: Missing Disinflation | `code/simulations/section_3/missing_disinflation.py` | `code/plotting/section_3/figure_08_missing_disinflation.py` | `output/figures/section_3/figure_08_missing_disinflation.png` |
| Fig 9: Post-Pandemic | `code/simulations/section_3/post_pandemic.py` | `code/plotting/section_3/figure_09_post_pandemic.py` | `output/figures/section_3/figure_09_post_pandemic.png` |
| Fig 10: Transmission Lags | `code/simulations/section_3/transmission_lags.py` | `code/plotting/section_3/figure_10_transmission_lags.py` | `output/figures/section_3/figure_10_transmission_lags.png` |
| Fig 11: Phillips Curve | `code/simulations/section_3/phillips_curve.py` | `code/plotting/section_3/figure_11_phillips_curve.py` | `output/figures/section_3/figure_11_phillips_curve.png` |
| Fig 12: Policy Asymmetry | `code/simulations/section_3/policy_asymmetry.py` | `code/plotting/section_3/figure_12_policy_asymmetry.py` | `output/figures/section_3/figure_12_policy_asymmetry.png` |
| Fig 13: Credibility Buffer | `code/simulations/section_3/credibility_buffer.py` | `code/plotting/section_3/figure_13_credibility_buffer.py` | `output/figures/section_3/figure_13_credibility_buffer.png` |
| Fig 14: Hyperinflation | `code/simulations/section_4/hyperinflation.py` | `code/plotting/section_4/figure_14_hyperinflation.py` | `output/figures/section_4/figure_14_hyperinflation.png` |
| Fig 15: Brazil | `code/simulations/section_4/brazil.py` | `code/plotting/section_4/figure_15_brazil.py` | `output/figures/section_4/figure_15_brazil.png` |
| Fig 16: Long Memory | `code/simulations/section_4/long_memory.py` | `code/plotting/section_4/figure_16_long_memory.py` | `output/figures/section_4/figure_16_long_memory.png` |
| Fig 17: Japan | `code/simulations/section_4/japan.py` | `code/plotting/section_4/figure_17_japan.py` | `output/figures/section_4/figure_17_japan.png` |
| Fig B1: Reanchoring | `code/simulations/appendix_B/reanchoring.py` | `code/plotting/appendix_B/figure_B1_reanchoring.py` | `output/figures/appendix_B/figure_B1_reanchoring.png` |
| Fig D1: Kalman | `code/empirical/appendix_D/kalman_comparison.py` | (generated by empirical script) | `output/figures/appendix_D/figure_D1_kalman.png` |
| Fig E1: MAB vs Bayesian | `code/simulations/appendix_E/mab_vs_bayesian.py` | `code/plotting/appendix_E/figure_E1_mab_vs_bayesian.py` | `output/figures/appendix_E/figure_E1_mab_vs_bayesian.png` |

**Output location:** `output/figures/` (both PNG and PDF formats, organized by section)

To generate a single figure, run the simulation first, then the plotting script:

```bash
python code/simulations/section_2/learning_mechanism.py
python code/plotting/section_2/figure_01_learning_mechanism.py
```

---

## Replicating Tables

### Table 1: Calibration Parameters

Table 1 is hardcoded in the paper based on values in `code/models/parameters.py`.

### Tables 2--3: Simulation Results

```bash
python code/tables/section_2/table_02_transitory_persistent.py
python code/tables/section_2/table_03_paradox_learning.py
```

**Output:** `output/tables/section_2/table_02_transitory_persistent.csv`, `output/tables/section_2/table_03_paradox_learning.csv`

### Table 4: Persistence by Regime

```bash
python code/empirical/section_2/persistence_analysis.py
```

**Output:** `output/empirical/section_2/persistence_estimates.csv`

### Tables 5--6: Policy Results

```bash
python code/tables/section_3/table_05_credibility_buffer.py
python code/tables/section_3/table_06_shock_gradual.py
```

**Output:** `output/tables/section_3/table_05.csv`, `output/tables/section_3/table_06.csv`

### Tables D1--D2: Robustness Checks

```bash
python code/empirical/appendix_D/kalman_comparison.py
python code/empirical/appendix_D/window_sensitivity.py
```

**Output:** `output/tables/appendix_D/table_D1.csv`, `output/tables/appendix_D/table_D2.csv`

---

## Replicating Empirical Analysis

The empirical analysis estimates time-varying inflation persistence using U.S. CPI data.

### Main Analysis (Figure 5, Table 4)

```bash
python code/empirical/section_2/persistence_analysis.py
```

**Methodology:**
1. Rolling 40-quarter AR(1) regressions on CPI inflation
2. Chow tests for structural breaks at candidate dates
3. Regime classification and summary statistics

**Output:**
- `output/figures/section_2/figure_05_persistence.png`
- `output/empirical/section_2/persistence_estimates.csv`

### Robustness: Kalman Filter (Figure D1, Table D1)

```bash
python code/empirical/appendix_D/kalman_comparison.py
```

**Methodology:**
- Time-varying parameter AR(1) model via state-space representation
- Maximum likelihood estimation of variance parameters
- Kalman filter and smoother for persistence path

**Output:**
- `output/figures/appendix_D/figure_D1_kalman.png`
- `output/tables/appendix_D/table_D1.csv`
- `output/empirical/appendix_D/kalman_persistence_estimates.csv`

### Robustness: Window Size Sensitivity (Table D2)

```bash
python code/empirical/appendix_D/window_sensitivity.py
```

**Output:** `output/tables/appendix_D/table_D2.csv`

---

## Model Documentation

### Core Models

| File | Description | Key Functions |
|------|-------------|---------------|
| `parameters.py` | All calibration parameters | `get_default_params()`, `get_structural_params()` |
| `mab_learning.py` | Multi-Armed Bandit learning | `update_theta()`, `compute_losses()` |
| `fire_solution.py` | FIRE rational expectations | `solve_fire()` |
| `toy_model.py` | Simple model (lambda=0) | `simulate()` |
| `full_model.py` | Full heterogeneous agent model | `simulate()`, `solve_sophisticated_fire()` |
| `three_arm_mab_learning.py` | Three-rule learning (CB, BL, TF) | `update_thetas()` |
| `three_arm_full_model.py` | Three-arm full model | `simulate()` |
| `long_memory_learning.py` | Exponentially discounted memory | `update_theta_long_memory()` |
| `bayesian_learning.py` | Bayesian alternative | `update_weights()` |
| `policy_experiments.py` | Policy experiment utilities | `run_experiment()` |
| `plot_utils.py` | Plotting utilities and style | `setup_style()`, `save_figure()` |

All model files are in `code/models/`.

### Parameter Values

All parameters are defined in `code/models/parameters.py`:

```python
# Structural Parameters
beta = 0.99          # Discount factor
sigma = 1.0          # Intertemporal elasticity
kappa = 0.024        # Phillips curve slope
phi_pi = 1.5         # Taylor rule: inflation
phi_y = 0.125        # Taylor rule: output
pi_star = 0.005      # Inflation target (quarterly, = 2% annual)

# Behavioral Parameters
lambda_fire = 0.35   # Fraction of FIRE agents
eta = 0.10           # Learning speed
k = 3                # Memory window (quarters)
epsilon = 1e-4       # Simplicity bias threshold
```

**Note on Calibration:** All model files follow a consistent quarterly calibration. The inflation target pi* = 0.005 corresponds to a 2% annual rate.

---

## Expected Output

### Figures

All figures appear in `output/figures/` in both PNG and PDF formats, organized by section:

| Directory | Figures |
|-----------|---------|
| `section_2/` | `figure_01_learning_mechanism`, `figure_02_k_epsilon_roles`, `figure_03_lambda_comparison`, `figure_04_shock_regimes`, `figure_05_persistence` |
| `section_3/` | `figure_06_oil_shocks` through `figure_13_credibility_buffer` |
| `section_4/` | `figure_14_hyperinflation` through `figure_17_japan` |
| `appendix_B/` | `figure_B1_reanchoring` |
| `appendix_D/` | `figure_D1_kalman` |
| `appendix_E/` | `figure_E1_mab_vs_bayesian` |

### Tables

CSV files in `output/tables/`, organized by section:

| Directory | Tables |
|-----------|--------|
| `section_2/` | `table_02_transitory_persistent.csv`, `table_03_paradox_learning.csv` |
| `section_3/` | `table_05.csv`, `table_06.csv` |
| `appendix_D/` | `table_D1.csv`, `table_D2.csv` |

### Key Numerical Values to Verify

| Claim | Expected Value | Source |
|-------|----------------|--------|
| Great Moderation correlation | 0.94 | `code/simulations/section_3/great_moderation.py` |
| Missing disinflation (NK) | -3.7% | `code/simulations/section_3/missing_disinflation.py` |
| Missing disinflation (Adaptive) | -0.6% | `code/simulations/section_3/missing_disinflation.py` |
| Post-pandemic peak (Adaptive) | 7.8% | `code/simulations/section_3/post_pandemic.py` |
| Post-pandemic peak (NK) | 16.7% | `code/simulations/section_3/post_pandemic.py` |
| Transmission lag difference | 40% | `code/simulations/section_3/transmission_lags.py` |
| Kalman vs rolling correlation | 0.86 | `code/empirical/appendix_D/kalman_comparison.py` |

---

## Troubleshooting

### Common Issues

**Issue:** `ModuleNotFoundError: No module named 'full_model'` (or similar)

**Solution:** Run scripts from the project root directory. Each script adds `code/models/` to the Python path automatically using relative paths.

---

**Issue:** FRED API errors

**Solution:**
1. Check that `FRED_API_KEY` is set correctly
2. Use cached data in `data/raw/fred_data_raw.csv`
3. The empirical scripts will fall back to cached data if the API fails

---

**Issue:** Figures look different from paper

**Solution:**
- Ensure matplotlib version >= 3.4.0
- Check that you're using the default matplotlib backend
- Random seeds are fixed in all scripts for reproducibility

---

**Issue:** Numerical values slightly different

**Solution:**
- Small differences (< 0.1%) are expected due to floating-point arithmetic
- Ensure NumPy/SciPy versions match requirements
- All random seeds are fixed; results should be deterministic

---

### Getting Help

If you encounter issues not covered here:

1. Verify your Python environment matches `requirements.txt`
2. Open an issue on the repository with:
   - Your Python version (`python --version`)
   - Your operating system
   - The complete error message
   - The command you ran

---

## Citation

If you use this code, please cite:

```bibtex
@article{smaniotto2026adaptive,
  title={An Adaptive-Hybrid New-Keynesian Model},
  author={Smaniotto, Bruno Cittolin},
  year={2026},
  journal={Working Paper}
}
```

---

## License

MIT License - see LICENSE file for details.
