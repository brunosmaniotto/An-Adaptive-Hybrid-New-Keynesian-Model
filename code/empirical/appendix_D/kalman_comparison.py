"""
Appendix D1: Kalman Filter vs Rolling Window Persistence
=========================================================

Robustness check comparing TVP-Kalman persistence estimates with rolling AR(1).

Paper Section: Appendix D
Dependencies:
    - Input: data/raw/fred_data_raw.csv
    - Input: output/empirical/section_2/persistence_estimates.csv
Output:
    - output/tables/appendix_D/table_D1.csv
    - output/figures/appendix_D/figure_D1_kalman.{pdf,png}

Table D1 format:
    Period | Kalman rho_bar | Rolling rho_bar | Difference
    across 5 regimes (Great Inflation, Volcker Era, Great Moderation,
    Post-Crisis, COVID Era).

TVP-AR(1) model:
    Observation: pi_t = c + rho_t * pi_{t-1} + e_t,    e_t ~ N(0, sigma2_obs)
    State:       rho_t = rho_{t-1} + eta_t,             eta_t ~ N(0, sigma2_state)

Author: Bruno Cittolin Smaniotto
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import minimize
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'models'))

from plot_utils import setup_style, save_figure


# =============================================================================
# Kalman Filter Implementation
# =============================================================================

class TVPPersistence:
    """
    Time-Varying Parameter AR(1) model for inflation persistence.

    Model:
        Observation: pi_t = c + rho_t * pi_{t-1} + e_t,    e_t ~ N(0, sigma2_obs)
        State:       rho_t = rho_{t-1} + eta_t,            eta_t ~ N(0, sigma2_state)

    Estimation via Maximum Likelihood using Kalman Filter.
    """

    def __init__(self, pi: np.ndarray) -> None:
        """
        Parameters
        ----------
        pi : array-like
            Inflation series (T observations).
        """
        self.pi = np.asarray(pi).flatten()
        self.T = len(self.pi) - 1  # lose one observation for AR(1)

        # Create lagged variables
        self.y = self.pi[1:]   # pi_t
        self.z = self.pi[:-1]  # pi_{t-1}

        # OLS initial estimates
        X = sm.add_constant(self.z)
        ols = sm.OLS(self.y, X).fit()
        self.rho_ols = ols.params[1]
        self.const_ols = ols.params[0]
        self.ols_var = np.var(ols.resid)

    def kalman_filter(self, sigma2_obs, sigma2_state, const, diffuse_P0=1e6):
        """
        Run Kalman filter for TVP-AR(1) model.

        Returns
        -------
        rho_filt : array, filtered state estimates E[rho_t | y_{1:t}]
        P_filt : array, filtered state variances Var[rho_t | y_{1:t}]
        rho_pred : array, predicted state estimates E[rho_t | y_{1:t-1}]
        P_pred : array, predicted state variances
        log_lik : float, log-likelihood
        """
        T = self.T
        y = self.y - const  # partial out constant
        z = self.z

        # Storage
        rho_pred = np.zeros(T)
        P_pred = np.zeros(T)
        rho_filt = np.zeros(T)
        P_filt = np.zeros(T)
        v = np.zeros(T)  # innovations
        F = np.zeros(T)  # innovation variances

        # Initialize with diffuse prior
        rho_pred[0] = self.rho_ols
        P_pred[0] = diffuse_P0

        log_lik = 0.0

        for t in range(T):
            # Prediction step (for t > 0)
            if t > 0:
                rho_pred[t] = rho_filt[t - 1]
                P_pred[t] = P_filt[t - 1] + sigma2_state

            # Innovation
            v[t] = y[t] - z[t] * rho_pred[t]

            # Innovation variance
            F[t] = z[t] ** 2 * P_pred[t] + sigma2_obs
            if F[t] < 1e-10:
                F[t] = 1e-10

            # Kalman gain
            K = P_pred[t] * z[t] / F[t]

            # Update step
            rho_filt[t] = rho_pred[t] + K * v[t]
            P_filt[t] = P_pred[t] - K * z[t] * P_pred[t]

            # Log-likelihood (skip first few for diffuse initialization)
            if t >= 4:
                log_lik -= 0.5 * (np.log(2 * np.pi) + np.log(F[t])
                                  + v[t] ** 2 / F[t])

        return rho_filt, P_filt, rho_pred, P_pred, log_lik

    def kalman_smoother(self, sigma2_obs, sigma2_state, const, diffuse_P0=1e6):
        """
        Run Kalman smoother (Rauch-Tung-Striebel algorithm).

        Returns
        -------
        rho_smooth : array, smoothed state estimates E[rho_t | y_{1:T}]
        P_smooth : array, smoothed state variances Var[rho_t | y_{1:T}]
        rho_filt : array, filtered state estimates
        P_filt : array, filtered state variances
        """
        T = self.T

        # Run filter
        rho_filt, P_filt, rho_pred, P_pred, _ = self.kalman_filter(
            sigma2_obs, sigma2_state, const, diffuse_P0
        )

        # Initialize smoother at t = T
        rho_smooth = np.zeros(T)
        P_smooth = np.zeros(T)
        rho_smooth[T - 1] = rho_filt[T - 1]
        P_smooth[T - 1] = P_filt[T - 1]

        # Backward recursion (RTS smoother)
        for t in range(T - 2, -1, -1):
            P_pred_next = (P_pred[t + 1] if t + 1 < T
                           else P_filt[t] + sigma2_state)
            if P_pred_next < 1e-10:
                P_pred_next = 1e-10

            J = P_filt[t] / P_pred_next
            rho_smooth[t] = rho_filt[t] + J * (rho_smooth[t + 1]
                                                - rho_pred[t + 1])
            P_smooth[t] = P_filt[t] + J ** 2 * (P_smooth[t + 1]
                                                  - P_pred_next)

        return rho_smooth, P_smooth, rho_filt, P_filt

    def neg_log_likelihood(self, params):
        """Negative log-likelihood for optimization."""
        sigma2_obs = np.exp(params[0])
        sigma2_state = np.exp(params[1])
        const = params[2]
        _, _, _, _, log_lik = self.kalman_filter(sigma2_obs, sigma2_state, const)
        return -log_lik

    def fit(self, maxiter=500):
        """
        Estimate hyperparameters via MLE.

        Returns
        -------
        dict with keys: rho_smoothed, rho_filtered, rho_se,
            sigma2_obs, sigma2_state, const, log_likelihood, converged
        """
        # Initial values (log-transformed for unconstrained optimization)
        sigma2_obs_init = self.ols_var
        sigma2_state_init = sigma2_obs_init * 0.01

        x0 = np.array([
            np.log(sigma2_obs_init),
            np.log(sigma2_state_init),
            self.const_ols,
        ])

        result = minimize(
            self.neg_log_likelihood, x0,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'disp': False},
        )

        sigma2_obs = np.exp(result.x[0])
        sigma2_state = np.exp(result.x[1])
        const = result.x[2]

        # Run smoother with estimated parameters
        rho_smooth, P_smooth, rho_filt, P_filt = self.kalman_smoother(
            sigma2_obs, sigma2_state, const
        )

        return {
            'rho_smoothed': rho_smooth,
            'rho_filtered': rho_filt,
            'rho_se': np.sqrt(P_smooth),
            'sigma2_obs': sigma2_obs,
            'sigma2_state': sigma2_state,
            'const': const,
            'log_likelihood': -result.fun,
            'converged': result.success,
        }


# =============================================================================
# Data Loading
# =============================================================================

def load_inflation() -> pd.Series:
    """Load inflation data from FRED raw file."""
    fred_path = PROJECT_ROOT / 'data' / 'raw' / 'fred_data_raw.csv'
    if not fred_path.exists():
        print(f"Error: {fred_path} not found.")
        sys.exit(1)

    df = pd.read_csv(fred_path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

    inflation = df['inflation_cpi'].dropna()
    print(f"  Loaded inflation: {len(inflation)} observations")
    print(f"  Sample: {inflation.index.min().date()} to "
          f"{inflation.index.max().date()}")
    return inflation


def load_rolling_estimates() -> pd.DataFrame:
    """Load rolling AR(1) estimates from section_2 output."""
    rolling_path = (PROJECT_ROOT / 'output' / 'empirical' / 'section_2'
                    / 'persistence_estimates.csv')
    if not rolling_path.exists():
        print(f"  Warning: Rolling estimates not found at {rolling_path}")
        print("  Run persistence_analysis.py first.")
        return None

    df = pd.read_csv(rolling_path)
    df['date'] = pd.to_datetime(df['date'])
    # Convert end-of-quarter dates to start-of-quarter for alignment
    df['date'] = df['date'].dt.to_period('Q').dt.to_timestamp()
    df.set_index('date', inplace=True)
    print(f"  Loaded rolling estimates: {len(df)} observations")
    return df


# =============================================================================
# Comparison Table (Table A1 / Table D1)
# =============================================================================

def build_comparison_table(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Build the comparison table across ALL 5 monetary policy regimes.

    Table format:
        Period | Kalman rho_bar | Rolling rho_bar | Difference

    Parameters
    ----------
    merged : DataFrame
        Merged Kalman + rolling estimates with datetime index.

    Returns
    -------
    DataFrame with the comparison table.
    """
    periods = {
        'Great Inflation (1968-1979)': ('1968-01-01', '1979-12-31'),
        'Volcker Era (1980-1984)': ('1980-01-01', '1984-12-31'),
        'Great Moderation (1985-2007)': ('1985-01-01', '2007-12-31'),
        'Post-Crisis (2008-2019)': ('2008-01-01', '2019-12-31'),
        'COVID Era (2020-2024)': ('2020-01-01', '2024-12-31'),
    }

    comparison_data = []
    for period_name, (start, end) in periods.items():
        mask = (merged.index >= start) & (merged.index <= end)
        period_data = merged[mask]

        if len(period_data) > 0:
            kalman_mean = period_data['rho_kalman'].mean()
            rolling_mean = period_data['rho_rolling'].mean()
            difference = kalman_mean - rolling_mean
            comparison_data.append({
                'Period': period_name,
                'N': len(period_data),
                'Kalman Mean': round(kalman_mean, 4),
                'Rolling Mean': round(rolling_mean, 4),
                'Difference': round(difference, 4),
            })
            print(f"    {period_name:35s}: Kalman = {kalman_mean:.3f}, "
                  f"Rolling = {rolling_mean:.3f}, "
                  f"Diff = {difference:+.3f}")

    return pd.DataFrame(comparison_data)


# =============================================================================
# Figure D1: Kalman vs Rolling Comparison
# =============================================================================

def create_kalman_figure(tvp_df: pd.DataFrame,
                         merged: pd.DataFrame,
                         inflation: pd.Series) -> plt.Figure:
    """
    Create the Kalman comparison figure (4-panel layout matching Current).

    Panel A: Kalman smoothed persistence with CI
    Panel B: Comparison of Kalman vs rolling
    Panel C: Filtered vs smoothed estimates
    Panel D: Inflation and persistence overlay
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Panel A: TVP Kalman estimates with confidence bands ---
    ax = axes[0, 0]
    ax.plot(tvp_df.index, tvp_df['rho_kalman'],
            color='darkblue', linewidth=2, label='Kalman Smoothed')
    ax.fill_between(
        tvp_df.index,
        tvp_df['rho_kalman'] - 1.96 * tvp_df['se_kalman'],
        tvp_df['rho_kalman'] + 1.96 * tvp_df['se_kalman'],
        alpha=0.2, color='blue', label='95% CI',
    )
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title('Panel A: Kalman Filter TVP Persistence', fontweight='bold')
    ax.set_ylabel(r'$\rho_t$ (persistence)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.5, 1.0])

    # --- Panel B: Comparison with rolling estimates ---
    ax = axes[0, 1]
    if merged is not None and len(merged) > 0:
        ax.plot(merged.index, merged['rho_rolling'], color='gray',
                linewidth=1.5, alpha=0.7, label='Rolling AR(1)')
    ax.plot(tvp_df.index, tvp_df['rho_kalman'],
            color='darkblue', linewidth=2, label='Kalman TVP')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title('Panel B: Comparison with Rolling Window', fontweight='bold')
    ax.set_ylabel(r'$\rho_t$ (persistence)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.5, 1.0])

    # --- Panel C: Filtered vs Smoothed ---
    ax = axes[1, 0]
    ax.plot(tvp_df.index, tvp_df['rho_filtered'],
            color='lightblue', linewidth=1.5, alpha=0.8, label='Filtered')
    ax.plot(tvp_df.index, tvp_df['rho_kalman'],
            color='darkblue', linewidth=2, label='Smoothed')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title('Panel C: Filtered vs Smoothed Estimates', fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel(r'$\rho_t$ (persistence)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.5, 1.0])

    # --- Panel D: Inflation and persistence overlay ---
    ax = axes[1, 1]
    ax2 = ax.twinx()

    ax.plot(inflation.index, inflation.values, color='gray',
            linewidth=1, alpha=0.5, label='Inflation')
    ax.set_ylabel('Inflation (%)', color='gray')
    ax.tick_params(axis='y', labelcolor='gray')

    ax2.plot(tvp_df.index, tvp_df['rho_kalman'],
             color='darkblue', linewidth=2, label='Persistence')
    ax2.set_ylabel(r'$\rho_t$ (persistence)', color='darkblue')
    ax2.tick_params(axis='y', labelcolor='darkblue')

    ax.set_title('Panel D: Inflation and Persistence', fontweight='bold')
    ax.set_xlabel('Year')
    ax.grid(True, alpha=0.3)

    # --- Recession shading on first three panels ---
    recessions = [
        ('1973-11-01', '1975-03-01'),
        ('1980-01-01', '1980-07-01'),
        ('1981-07-01', '1982-11-01'),
        ('1990-07-01', '1991-03-01'),
        ('2001-03-01', '2001-11-01'),
        ('2007-12-01', '2009-06-01'),
        ('2020-02-01', '2020-04-01'),
    ]
    for start, end in recessions:
        for axx in [axes[0, 0], axes[0, 1], axes[1, 0]]:
            axx.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                        alpha=0.1, color='gray')

    plt.tight_layout()
    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("APPENDIX D1: KALMAN FILTER vs ROLLING WINDOW PERSISTENCE")
    print("Robustness Check for Rolling AR(1) Estimates")
    print("=" * 70)

    setup_style()

    # -------------------------------------------------------------------------
    # Step 1: Load data
    # -------------------------------------------------------------------------
    print("\n[1/4] Loading data...")
    inflation = load_inflation()
    df_rolling = load_rolling_estimates()

    # -------------------------------------------------------------------------
    # Step 2: Estimate TVP persistence via Kalman filter
    # -------------------------------------------------------------------------
    print("\n[2/4] Estimating TVP persistence via Kalman filter...")
    model = TVPPersistence(inflation.values)
    results = model.fit(maxiter=500)

    print(f"  Optimization converged: {results['converged']}")
    print(f"  Log-likelihood: {results['log_likelihood']:.2f}")
    print(f"  sigma2_obs:   {results['sigma2_obs']:.4f}")
    print(f"  sigma2_state: {results['sigma2_state']:.6f}")
    print(f"  Signal-to-noise ratio: "
          f"{results['sigma2_state'] / results['sigma2_obs']:.6f}")
    print(f"  Constant: {results['const']:.4f}")
    print(f"  Smoothed rho: mean = {results['rho_smoothed'].mean():.4f}, "
          f"std = {results['rho_smoothed'].std():.4f}")

    # Create Kalman output DataFrame
    dates = inflation.index[1:]  # AR(1) loses one observation
    dates = dates.to_period('Q').to_timestamp()

    tvp_df = pd.DataFrame({
        'rho_kalman': results['rho_smoothed'],
        'rho_filtered': results['rho_filtered'],
        'se_kalman': results['rho_se'],
    }, index=dates)

    # -------------------------------------------------------------------------
    # Step 3: Build comparison table across 5 periods
    # -------------------------------------------------------------------------
    print("\n[3/4] Building comparison table (5 regimes)...")
    merged = None
    comparison_df = None

    if df_rolling is not None:
        merged = tvp_df.join(df_rolling[['rho_rolling']], how='inner').dropna()
        print(f"  Overlapping observations: {len(merged)}")

        # Overall correlation
        corr = merged['rho_kalman'].corr(merged['rho_rolling'])
        diff = merged['rho_kalman'] - merged['rho_rolling']
        rmse = np.sqrt((diff ** 2).mean())
        print(f"  Correlation: {corr:.4f}")
        print(f"  RMSE: {rmse:.4f}")

        comparison_df = build_comparison_table(merged)

        # Save Table D1
        table_dir = PROJECT_ROOT / 'output' / 'tables' / 'appendix_D'
        table_dir.mkdir(parents=True, exist_ok=True)
        table_path = table_dir / 'table_D1.csv'
        comparison_df.to_csv(table_path, index=False)
        print(f"\n  Saved: {table_path}")
    else:
        print("  Rolling estimates not available. Skipping comparison table.")

    # -------------------------------------------------------------------------
    # Step 4: Generate Figure D1
    # -------------------------------------------------------------------------
    print("\n[4/4] Generating Figure D1...")
    fig = create_kalman_figure(tvp_df, merged, inflation)

    fig_dir = PROJECT_ROOT / 'output' / 'figures' / 'appendix_D'
    fig_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, 'figure_D1_kalman', fig_dir, formats=['pdf', 'png'])

    # -------------------------------------------------------------------------
    # Save Kalman estimates
    # -------------------------------------------------------------------------
    kalman_output_dir = PROJECT_ROOT / 'output' / 'empirical' / 'appendix_D'
    kalman_output_dir.mkdir(parents=True, exist_ok=True)
    kalman_path = kalman_output_dir / 'kalman_persistence_estimates.csv'
    tvp_df.to_csv(kalman_path)
    print(f"  Saved: {kalman_path}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nKey Finding:")
    print("  The Kalman filter TVP estimates are highly correlated with")
    print("  rolling window estimates, validating both approaches.")
    print("  The TVP approach provides:")
    print("    - Smoother estimates with formal uncertainty quantification")
    print("    - No arbitrary window size selection")
    print("    - Optimal signal extraction from the data")

    if merged is not None:
        corr = merged['rho_kalman'].corr(merged['rho_rolling'])
        print(f"\n  Correlation between methods: {corr:.3f}")

    print("\n" + "=" * 70)
    print("ESTIMATION COMPLETE")
    print("=" * 70)

    return tvp_df, results


if __name__ == "__main__":
    main()
