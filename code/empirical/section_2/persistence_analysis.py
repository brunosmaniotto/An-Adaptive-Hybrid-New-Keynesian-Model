"""
Empirical Analysis: Persistence (Figure 5, Table 4)
====================================================

Estimates time-varying inflation persistence using rolling AR(1),
performs Chow structural break tests, and computes regime summary statistics.

Paper Section: Section 2
Dependencies:
    - Input: data/raw/fred_data_raw.csv (with FRED API fallback)
Output:
    - output/empirical/section_2/persistence_estimates.csv
    - output/empirical/section_2/persistence_data.pkl

The pickle contains all data needed for Figure 5 and Table 4:
    - dates, rho_rolling, rho_se: rolling AR(1) estimates
    - chow_results: Chow test F-stats and p-values at 4 candidate dates
    - regime_stats: summary statistics for 5 regimes
    - pi_star: inflation target (2.0)

Author: Bruno Cittolin Smaniotto
"""

import sys
import pickle
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from pathlib import Path
import warnings
import os
import json

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'models'))

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*divide by zero.*')
warnings.filterwarnings('ignore', message='.*invalid value.*')

# =============================================================================
# Configuration
# =============================================================================

ROLLING_WINDOW = 40  # quarters (10 years)
START_DATE = '1960-01-01'
END_DATE = '2024-12-31'

# Candidate break dates based on monetary history
CANDIDATE_BREAKS = {
    '1979-Q4': 'Volcker appointment',
    '1984-Q1': 'Volcker success / Great Moderation begins',
    '2008-Q4': 'Financial crisis',
    '2020-Q1': 'COVID pandemic',
}

# Regime definitions
REGIMES = {
    'Great Inflation (1968-1979)': ('1968-01-01', '1979-12-31'),
    'Volcker Era (1980-1984)': ('1980-01-01', '1984-12-31'),
    'Great Moderation (1985-2007)': ('1985-01-01', '2007-12-31'),
    'Post-Crisis (2008-2019)': ('2008-01-01', '2019-12-31'),
    'COVID Era (2020-2024)': ('2020-01-01', '2024-12-31'),
}

PI_STAR = 2.0  # inflation target


def get_fred_api_key(root_path: Path):
    """Load FRED API key from environment or config.json."""
    key = os.environ.get('FRED_API_KEY')
    if key:
        return key

    config_path = root_path / 'config.json'
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
                if 'FRED_API_KEY' in config:
                    return config['FRED_API_KEY']
        except Exception:
            pass
    return None


def format_quarter(date: pd.Timestamp) -> str:
    """Format a timestamp as YYYY-QN."""
    quarter = (date.month - 1) // 3 + 1
    return f"{date.year}-Q{quarter}"


# =============================================================================
# Data Loading
# =============================================================================

def load_inflation_data() -> pd.Series:
    """
    Load CPI data and compute quarterly annualized inflation.

    Tries local cache first, then FRED API as fallback.
    Returns annualized quarterly inflation: 400 * ln(P_t / P_{t-1}).
    """
    raw_path = PROJECT_ROOT / 'data' / 'raw' / 'fred_data_raw.csv'
    cpi = None
    inflation = None

    # 1. Try cached raw data file
    if raw_path.exists():
        print("  Loading from cached data...")
        df = pd.read_csv(raw_path)

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')

        if 'CPIAUCSL' in df.columns:
            cpi = df['CPIAUCSL']
        elif 'inflation_cpi' in df.columns:
            # Pre-computed quarterly inflation already available
            inflation = df['inflation_cpi'].dropna()
            print(f"  Loaded pre-computed inflation: {len(inflation)} observations")

    # 2. Try FRED API as fallback
    if cpi is None and inflation is None:
        key = get_fred_api_key(PROJECT_ROOT)
        if key:
            try:
                from fredapi import Fred
                print("  Fetching from FRED API...")
                fred = Fred(api_key=key)
                cpi = fred.get_series(
                    'CPIAUCSL',
                    observation_start=START_DATE,
                    observation_end=END_DATE,
                )
                print(f"  Downloaded {len(cpi)} monthly observations")
            except Exception as e:
                print(f"  FRED API failed: {e}")

    if cpi is None and inflation is None:
        print("  Error: No cached data and no FRED API key available.")
        sys.exit(1)

    # 3. Convert CPI levels to quarterly annualized inflation
    if inflation is None:
        cpi_q = cpi.resample('Q').last()
        inflation = 400 * np.log(cpi_q / cpi_q.shift(1))
        inflation = inflation.dropna()
        inflation.name = 'inflation'

    print(f"  Quarterly inflation: {len(inflation)} observations")
    print(f"  Date range: {format_quarter(inflation.index[0])} to "
          f"{format_quarter(inflation.index[-1])}")

    return inflation


# =============================================================================
# Rolling AR(1) Estimation
# =============================================================================

def rolling_ar1(series: pd.Series, window: int):
    """
    Estimate AR(1) coefficient using rolling windows.

    For each window, estimates: pi_t = c + rho * pi_{t-1} + e_t
    Returns the time series of rho estimates and standard errors.
    """
    rho_series = pd.Series(index=series.index, dtype=float)
    se_series = pd.Series(index=series.index, dtype=float)
    values = series.values

    for i in range(window, len(series)):
        y = values[i - window + 1:i + 1]
        y_lag = values[i - window:i]
        X = sm.add_constant(y_lag)
        model = sm.OLS(y, X).fit()
        rho_series.iloc[i] = model.params[1]
        se_series.iloc[i] = model.bse[1]

    return rho_series, se_series


# =============================================================================
# Chow Structural Break Tests
# =============================================================================

def chow_test(series: pd.Series, break_date: pd.Timestamp):
    """
    Perform Chow test for structural break at specified date.

    Tests whether the AR(1) relationship changes at the break date.
    Returns F-statistic and p-value.
    """
    # Split sample
    before = series[series.index < break_date].dropna()
    after = series[series.index >= break_date].dropna()

    if len(before) < 10 or len(after) < 10:
        return np.nan, np.nan

    # Estimate AR(1) on each subsample
    def ar1_ssr(s):
        y = s.values[1:]
        y_lag = s.values[:-1]
        X = sm.add_constant(y_lag)
        model = sm.OLS(y, X).fit()
        return model.ssr, len(y)

    ssr1, n1 = ar1_ssr(before)
    ssr2, n2 = ar1_ssr(after)

    # Pooled regression
    y = series.dropna().values[1:]
    y_lag = series.dropna().values[:-1]
    X = sm.add_constant(y_lag)
    model_pooled = sm.OLS(y, X).fit()
    ssr_pooled = model_pooled.ssr

    # Chow F-statistic
    k = 2  # number of parameters (constant + AR coefficient)
    F = ((ssr_pooled - ssr1 - ssr2) / k) / ((ssr1 + ssr2) / (n1 + n2 - 2 * k))

    # P-value from F distribution
    p_value = 1 - stats.f.cdf(F, k, n1 + n2 - 2 * k)

    return F, p_value


def run_chow_tests(inflation: pd.Series) -> pd.DataFrame:
    """
    Run Chow tests at all candidate break dates.

    Returns DataFrame with columns: date, label, F_stat, p_value, significant.
    """
    break_results = []
    for date_str, label in CANDIDATE_BREAKS.items():
        # Convert quarter string to timestamp
        year, q = date_str.split('-Q')
        break_date = pd.Timestamp(f"{year}-{(int(q) - 1) * 3 + 1:02d}-01")
        F_stat, p_val = chow_test(inflation, break_date)

        sig_str = ("***" if p_val < 0.01
                   else ("**" if p_val < 0.05
                         else ("*" if p_val < 0.1 else "")))
        print(f"    {date_str} ({label}): F = {F_stat:.2f}, "
              f"p = {p_val:.4f} {sig_str}")

        break_results.append({
            'date': date_str,
            'label': label,
            'F_stat': F_stat,
            'p_value': p_val,
            'significant': p_val < 0.05,
        })

    return pd.DataFrame(break_results)


# =============================================================================
# Regime Summary Statistics
# =============================================================================

def compute_regime_stats(rho: pd.Series) -> pd.DataFrame:
    """
    Compute summary statistics for each monetary policy regime.

    Returns DataFrame with columns: Regime, Mean rho, Std rho, N quarters.
    """
    regime_stats = []
    for regime_name, (start, end) in REGIMES.items():
        mask = (rho.index >= start) & (rho.index <= end)
        regime_rho = rho[mask]

        if len(regime_rho) > 0:
            stats_dict = {
                'Regime': regime_name,
                'Mean rho': regime_rho.mean(),
                'Std rho': regime_rho.std(),
                'N quarters': len(regime_rho),
            }
            regime_stats.append(stats_dict)
            print(f"    {regime_name:35s}: rho = {regime_rho.mean():.3f} "
                  f"(sd = {regime_rho.std():.3f}, n = {len(regime_rho)})")

    return pd.DataFrame(regime_stats)


# =============================================================================
# Main Analysis
# =============================================================================

def run_analysis():
    """Run the complete persistence analysis pipeline."""
    print("=" * 70)
    print("TIME-VARYING INFLATION PERSISTENCE ANALYSIS")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Step 1: Load data
    # -------------------------------------------------------------------------
    print("\n[1/4] Loading CPI data...")
    inflation = load_inflation_data()

    # -------------------------------------------------------------------------
    # Step 2: Rolling AR(1) estimation
    # -------------------------------------------------------------------------
    print("\n[2/4] Estimating rolling-window AR(1) persistence...")
    rho, rho_se = rolling_ar1(inflation, ROLLING_WINDOW)
    rho = rho.dropna()
    rho_se = rho_se.dropna()
    print(f"  Estimated {len(rho)} rolling AR(1) coefficients")
    print(f"  Window size: {ROLLING_WINDOW} quarters")

    # -------------------------------------------------------------------------
    # Step 3: Chow structural break tests
    # -------------------------------------------------------------------------
    print("\n[3/4] Testing for structural breaks (Chow tests)...")
    chow_results = run_chow_tests(inflation)

    # -------------------------------------------------------------------------
    # Step 4: Regime summary statistics
    # -------------------------------------------------------------------------
    print("\n[4/4] Computing regime summary statistics...")
    regime_stats = compute_regime_stats(rho)

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------
    output_dir = PROJECT_ROOT / 'output' / 'empirical' / 'section_2'
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV: persistence estimates (matches Current format with index=False)
    output_csv = pd.DataFrame({
        'date': rho.index,
        'rho_rolling': rho.values,
        'rho_se': rho_se.values,
    })
    csv_path = output_dir / 'persistence_estimates.csv'
    output_csv.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    # Pickle: all data for plotting (Figure 5) and tables (Table 4)
    pickle_data = {
        'dates': rho.index,
        'rho_rolling': rho.values,
        'rho_se': rho_se.values,
        'chow_results': chow_results,
        'regime_stats': regime_stats,
        'pi_star': PI_STAR,
        'candidate_breaks': CANDIDATE_BREAKS,
        'regimes': REGIMES,
        'rolling_window': ROLLING_WINDOW,
    }
    pkl_path = output_dir / 'persistence_data.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(pickle_data, f)
    print(f"  Saved: {pkl_path}")

    # -------------------------------------------------------------------------
    # Print summary for paper
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY FOR PAPER")
    print("=" * 70)

    print("\n1. PERSISTENCE BY REGIME:")
    for _, row in regime_stats.iterrows():
        print(f"   - {row['Regime']:35s}: rho = {row['Mean rho']:.3f}")

    print("\n2. CHOW TESTS (on raw inflation at candidate dates):")
    for _, row in chow_results.iterrows():
        sig = ("***" if row['p_value'] < 0.01
               else ("**" if row['p_value'] < 0.05
                     else ("*" if row['p_value'] < 0.1 else "")))
        print(f"   - {row['date']}: F = {row['F_stat']:.2f}, "
              f"p = {row['p_value']:.4f} {sig}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    return pickle_data


if __name__ == "__main__":
    run_analysis()
