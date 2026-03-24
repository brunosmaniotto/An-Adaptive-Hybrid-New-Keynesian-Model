"""
Plotting: Figure B1 - The Re-anchoring Problem
===============================================
2x2 layout: left column epsilon=0, right column epsilon=1e-4
Top row: inflation, bottom row: credibility

Reads: output/simulations/appendix_B/reanchoring.pkl
Writes: output/figures/appendix_B/figure_B1_reanchoring.{pdf,png}
"""

import sys
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'models'))

from plot_utils import setup_style, save_figure


def plot_reanchoring(data: dict) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(10, 4), sharex=True)

    color_inflation = 'steelblue'
    color_theta = 'firebrick'
    color_target = 'black'

    titles = [r'Without tie-breaking ($\varepsilon = 0$)',
              r'With tie-breaking ($\varepsilon = 10^{-4}$)']

    scenarios = ['epsilon_0', 'epsilon_pos']

    for col, (key, title) in enumerate(zip(scenarios, titles)):
        res = data[key]
        T = len(res['pi'])
        time = np.arange(T)
        pi_star_ann = res['pi_star'] * 400

        # Top row: Inflation
        ax_pi = axes[0, col]
        ax_pi.plot(time, res['pi'] * 400, color=color_inflation, linewidth=2, label='Inflation')
        ax_pi.axhline(y=pi_star_ann, color=color_target, linestyle='--',
                       linewidth=1.5, label=r'Target $\pi^*$')
        ax_pi.set_ylabel('Inflation (%, ann.)', fontsize=10)
        ax_pi.set_title(title, fontsize=11)
        ax_pi.grid(True, alpha=0.3)
        ax_pi.set_ylim(-2, 22)
        if col == 0:
            ax_pi.legend(loc='upper right', fontsize=8)

        # Bottom row: Theta
        ax_theta = axes[1, col]
        ax_theta.plot(time, res['theta'], color=color_theta, linewidth=2,
                       label=r'Credibility $\theta$')
        ax_theta.axhline(y=1.0, color=color_target, linestyle='--',
                          linewidth=1.5, label='Full anchoring')
        ax_theta.set_xlabel('Quarters', fontsize=10)
        ax_theta.set_ylabel(r'Credibility $\theta_t$', fontsize=10)
        ax_theta.grid(True, alpha=0.3)
        ax_theta.set_ylim(-0.05, 1.1)
        if col == 0:
            ax_theta.legend(loc='lower right', fontsize=8)

    # Panel labels
    panel_labels = ['(a)', '(b)', '(c)', '(d)']
    for idx, ax in enumerate(axes.flat):
        ax.text(-0.12, 1.05, panel_labels[idx], transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='bottom', ha='left')

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    setup_style()
    input_path = PROJECT_ROOT / 'output' / 'simulations' / 'appendix_B' / 'reanchoring.pkl'

    if not input_path.exists():
        print("Error: simulation file not found.")
        sys.exit(1)

    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    print("Generating figure...")
    fig = plot_reanchoring(data)

    output_dir = PROJECT_ROOT / 'output' / 'figures' / 'appendix_B'
    output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, 'figure_B1_reanchoring', output_dir, formats=['pdf', 'png'])
    print("Done!")
