"""
Plotting: Figure 06 - Oil Shocks and Initial Credibility
========================================================
Panel (a): Inflation paths under low vs high initial credibility
Panel (b): Credibility evolution

Reads: output/simulations/section_3/oil_shocks.pkl
Writes: output/figures/section_3/figure_06_oil_shocks.{pdf,png}
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


def plot_oil_shocks(data: dict) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    theta_values = sorted(data.keys())
    colors = {0.2: '#d62728', 1.0: '#1f77b4'}
    T = len(data[theta_values[0]]['pi'])

    # Panel (a): Inflation (annualized)
    ax1 = axes[0]
    for theta_0 in theta_values:
        pi = data[theta_0]['pi']
        ax1.plot(range(T), pi * 400, color=colors[theta_0],
                 linewidth=2.0, label=rf'$\theta_0 = {theta_0}$')
    ax1.axhline(y=2.0, color='gray', linestyle='--', linewidth=1, alpha=0.7, label=r'$\pi^*$')
    ax1.axvspan(0, 4, alpha=0.12, color='gray')
    ax1.set_xlabel('Quarters')
    ax1.set_ylabel('Inflation (%, ann.)')
    ax1.set_title('(a) Inflation')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_xlim(0, T)

    # Panel (b): Credibility
    ax2 = axes[1]
    for theta_0 in theta_values:
        theta = data[theta_0]['theta']
        ax2.plot(range(T), theta, color=colors[theta_0],
                 linewidth=2.0, label=rf'$\theta_0 = {theta_0}$')
    ax2.axvspan(0, 4, alpha=0.12, color='gray')
    ax2.set_xlabel('Quarters')
    ax2.set_ylabel(r'Credibility ($\theta_t$)')
    ax2.set_title('(b) Credibility')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.set_xlim(0, T)
    ax2.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    setup_style()
    input_path = PROJECT_ROOT / 'output' / 'simulations' / 'section_3' / 'oil_shocks.pkl'
    if not input_path.exists():
        print("Error: simulation file not found.")
        sys.exit(1)
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    fig = plot_oil_shocks(data)
    output_dir = PROJECT_ROOT / 'output' / 'figures' / 'section_3'
    output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, 'figure_06_oil_shocks', output_dir, formats=['pdf', 'png'])
    print("Done!")
