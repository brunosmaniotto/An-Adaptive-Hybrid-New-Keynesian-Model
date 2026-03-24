"""
Plotting: Figure 5 - Time-Varying Inflation Persistence
========================================================

Reads: output/empirical/section_2/persistence_data.pkl
Writes: output/figures/section_2/figure_05_persistence.{pdf,png}

Figure includes:
    - Rolling 40-quarter AR(1) persistence with 95% confidence interval
    - Vertical lines at Chow test candidate dates:
        * Dashed (--) for statistically significant breaks (p < 0.05)
        * Dotted (:) for non-significant breaks (p >= 0.05)
    - Regime mean lines for key periods
    - NBER recession shading
"""

import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'models'))

from plot_utils import setup_style, save_figure


def plot_persistence(data: dict) -> plt.Figure:
    """
    Create the full persistence figure matching the paper's Figure 5.

    Parameters
    ----------
    data : dict
        Pickle data from persistence_analysis.py containing:
        dates, rho_rolling, rho_se, chow_results, regime_stats,
        candidate_breaks, regimes, pi_star, rolling_window.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    dates = data['dates']
    rho = pd.Series(data['rho_rolling'], index=dates)
    rho_se = pd.Series(data['rho_se'], index=dates)
    chow_results = data['chow_results']
    candidate_breaks = data['candidate_breaks']
    regimes = data['regimes']

    fig, ax = plt.subplots(figsize=(14, 5))

    # --- Rolling persistence with 95% CI ---
    ax.plot(dates, rho.values, linewidth=2, color='darkblue',
            label='Rolling AR(1) persistence')

    upper = rho + 1.96 * rho_se
    lower = rho - 1.96 * rho_se
    ax.fill_between(dates, lower.values, upper.values,
                    alpha=0.2, color='blue', label='95% CI')

    # --- Chow test candidate dates as vertical lines ---
    # Color mapping for each break date
    break_colors = {
        '1979-Q4': 'red',
        '1984-Q1': 'green',
        '2008-Q4': 'orange',
        '2020-Q1': 'purple',
    }

    for _, row in chow_results.iterrows():
        date_str = row['date']
        label_text = row['label'] if 'label' in row else date_str
        is_sig = row['significant']

        # Convert quarter string to timestamp
        year, q = date_str.split('-Q')
        break_date = pd.Timestamp(f"{year}-{(int(q) - 1) * 3 + 1:02d}-01")
        color = break_colors.get(date_str, 'gray')

        # Dashed for significant (p < 0.05), dotted for not significant
        linestyle = '--' if is_sig else ':'
        ax.axvline(break_date, color=color, linestyle=linestyle,
                   alpha=0.7, linewidth=1.5, label=f'{date_str}')

    # --- Regime average lines (key regimes only to avoid clutter) ---
    for regime_name, (start, end) in regimes.items():
        mask = (rho.index >= start) & (rho.index <= end)
        if rho[mask].notna().sum() > 0:
            avg = rho[mask].mean()
            if 'Great Inflation' in regime_name or 'Great Moderation' in regime_name:
                ax.hlines(avg, pd.Timestamp(start), pd.Timestamp(end),
                          colors='gray', linestyles=':', alpha=0.5)

    # --- Reference line at zero ---
    ax.axhline(0, color='black', linewidth=0.5, linestyle='-')

    # --- NBER recession shading ---
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
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   alpha=0.1, color='gray')

    # --- Labels and formatting ---
    ax.set_xlabel('Year')
    ax.set_ylabel(r'Persistence Coefficient ($\rho$)')
    window = data.get('rolling_window', 40)
    ax.set_title(
        f'Time-Varying Inflation Persistence (1970-2024)\n'
        f'{window}-Quarter Rolling AR(1) with Chow Test Candidate Dates',
        fontweight='bold', pad=15,
    )
    ax.legend(loc='upper right', framealpha=0.95, fontsize=9)
    ax.grid(True, alpha=0.3)

    # Data-driven y-limits with padding
    y_min = min(lower.min(), rho.min()) - 0.1
    y_max = max(upper.max(), rho.max()) + 0.1
    ax.set_ylim([y_min, y_max])
    ax.set_xlim([pd.Timestamp('1970-01-01'), pd.Timestamp('2024-12-31')])

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    setup_style()

    # Load pickle data from empirical analysis
    input_path = PROJECT_ROOT / 'output' / 'empirical' / 'section_2' / 'persistence_data.pkl'

    if not input_path.exists():
        # Fallback: try CSV-only mode (limited, no Chow tests)
        csv_path = PROJECT_ROOT / 'output' / 'empirical' / 'section_2' / 'persistence_estimates.csv'
        if csv_path.exists():
            print("Warning: pickle not found, falling back to CSV (no Chow tests in plot).")
            df = pd.read_csv(csv_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')

            # Build minimal data dict for plotting
            data = {
                'dates': df.index,
                'rho_rolling': df['rho_rolling'].values,
                'rho_se': df['rho_se'].values,
                'chow_results': pd.DataFrame(columns=['date', 'label', 'F_stat', 'p_value', 'significant']),
                'regime_stats': pd.DataFrame(),
                'candidate_breaks': {},
                'regimes': {
                    'Great Inflation (1968-1979)': ('1968-01-01', '1979-12-31'),
                    'Volcker Era (1980-1984)': ('1980-01-01', '1984-12-31'),
                    'Great Moderation (1985-2007)': ('1985-01-01', '2007-12-31'),
                    'Post-Crisis (2008-2019)': ('2008-01-01', '2019-12-31'),
                    'COVID Era (2020-2024)': ('2020-01-01', '2024-12-31'),
                },
                'pi_star': 2.0,
                'rolling_window': 40,
            }
        else:
            print(f"Error: Neither {input_path} nor {csv_path} found.")
            print("Run persistence_analysis.py first.")
            sys.exit(1)
    else:
        with open(input_path, 'rb') as f:
            data = pickle.load(f)

    print("Generating Figure 5: Time-Varying Inflation Persistence...")
    fig = plot_persistence(data)

    output_dir = PROJECT_ROOT / 'output' / 'figures' / 'section_2'
    output_dir.mkdir(parents=True, exist_ok=True)

    save_figure(fig, 'figure_05_persistence', output_dir, formats=['pdf', 'png'])
    print("Done!")
