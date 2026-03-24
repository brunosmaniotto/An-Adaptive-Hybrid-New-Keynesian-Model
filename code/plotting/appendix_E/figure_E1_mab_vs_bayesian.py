"""
Plotting: Figure E1 - MAB vs Bayesian
=====================================
3x3 grid: Rows = High/Medium/Low credibility, Cols = Credibility/Inflation/Output Gap.
With column headers, row labels, legends, percentage formatting, and correlation display.

Reads: output/simulations/appendix_E/mab_vs_bayesian.pkl
Writes: output/figures/appendix_E/figure_E1_mab_vs_bayesian.{pdf,png}
"""

import sys
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'models'))

from plot_utils import setup_style, save_figure, add_target_line, format_percent_axis
from parameters import get_default_params


def plot_mab_vs_bayesian(data: dict) -> plt.Figure:
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))

    params = get_default_params()
    pi_star = params['pi_star']

    scenarios = ['high_credibility', 'medium_credibility', 'low_credibility']
    scenario_labels = ['High Initial Credibility', 'Medium Initial Credibility',
                       'Low Initial Credibility']

    for row, (name, scenario_label) in enumerate(zip(scenarios, scenario_labels)):
        res = data[name]
        mab = res['mab']
        bayes = res['bayes']
        cfg = res['config']
        T = cfg['T']
        time = np.arange(T)

        # Column 0: Credibility State
        ax = axes[row, 0]
        ax.plot(time, mab['theta'], 'b-', linewidth=1.8, label=r'MAB ($\theta$)')
        ax.plot(time, bayes['weight'], 'r--', linewidth=1.8, label='Bayesian (w)')
        ax.axvline(cfg['shock_period'], color='gray', linestyle=':', alpha=0.7)

        if row == 0:
            ax.set_title('(a) Credibility State')
            ax.legend(loc='lower right', fontsize=8)
        ax.set_ylabel(scenario_label, fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        if row == 2:
            ax.set_xlabel('Quarters')

        # Correlation
        corr, _ = stats.pearsonr(mab['theta'][5:], bayes['weight'][5:])
        ax.text(0.95, 0.05, f'r = {corr:.3f}', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=9, style='italic')

        # Column 1: Inflation
        ax = axes[row, 1]
        ax.plot(time, mab['pi'], 'b-', linewidth=1.8, label='MAB')
        ax.plot(time, bayes['pi'], 'r--', linewidth=1.8, label='Bayesian')
        add_target_line(ax, pi_star)
        ax.axvline(cfg['shock_period'], color='gray', linestyle=':', alpha=0.7)

        if row == 0:
            ax.set_title('(b) Inflation (%, ann.)')
        format_percent_axis(ax, 'y', decimals=1, annualize=True)
        if row == 2:
            ax.set_xlabel('Quarters')

        # Column 2: Output Gap
        ax = axes[row, 2]
        ax.plot(time, mab['y'], 'b-', linewidth=1.8, label='MAB')
        ax.plot(time, bayes['y'], 'r--', linewidth=1.8, label='Bayesian')
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.axvline(cfg['shock_period'], color='gray', linestyle=':', alpha=0.7)

        if row == 0:
            ax.set_title('(c) Output Gap')
        format_percent_axis(ax, 'y', decimals=1)
        if row == 2:
            ax.set_xlabel('Quarters')

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    setup_style()
    input_path = PROJECT_ROOT / 'output' / 'simulations' / 'appendix_E' / 'mab_vs_bayesian.pkl'

    if not input_path.exists():
        print("Error: simulation file not found.")
        sys.exit(1)

    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    print("Generating figure...")
    fig = plot_mab_vs_bayesian(data)

    output_dir = PROJECT_ROOT / 'output' / 'figures' / 'appendix_E'
    output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, 'figure_E1_mab_vs_bayesian', output_dir, formats=['pdf', 'png'])
    print("Done!")
