"""
Plotting: Figure 16 - Long Memory and Inflation Trauma
======================================================
Panel (a): Inflation response to identical shock across 3 scenarios
Panel (b): Credibility dynamics

Reads: output/simulations/section_4/long_memory.pkl
Writes: output/figures/section_4/figure_16_long_memory.{pdf,png}
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

T_SHOCK = 40


def plot_long_memory(results: dict) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_order = ['stable', 'distant_trauma', 'chronic_current']
    years = np.arange(T_SHOCK) / 4

    # Panel A: Inflation response
    ax = axes[0]
    for key in plot_order:
        r = results[key]
        label = f"{r['name']} (\u03b8\u2080={r['theta_final']:.2f})"
        ax.plot(years, r['pi_path'] * 400, color=r['color'],
                linestyle=r['linestyle'], linewidth=2.5, label=label)
    ax.axhline(2, color='gray', linestyle='--', alpha=0.5, label='Target')
    ax.set_xlabel('Years after shock')
    ax.set_ylabel('Inflation (% annualized)')
    ax.set_title('(a) Inflation Response to Identical Shock')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(0, T_SHOCK / 4)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # Panel B: Credibility dynamics
    ax = axes[1]
    for key in plot_order:
        r = results[key]
        ax.plot(years, r['theta_path'], color=r['color'],
                linestyle=r['linestyle'], linewidth=2.5, label=r['name'])
    ax.set_xlabel('Years after shock')
    ax.set_ylabel(r'Credibility ($\theta$)')
    ax.set_title('(b) Credibility Dynamics')
    ax.legend(loc='right', fontsize=8)
    ax.set_xlim(0, T_SHOCK / 4)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.suptitle(r'Long Memory: Same Shock, Different Outcomes ($\delta = 0.8$)', fontsize=11)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    setup_style()
    input_path = PROJECT_ROOT / 'output' / 'simulations' / 'section_4' / 'long_memory.pkl'

    if not input_path.exists():
        print("Error: simulation file not found.")
        sys.exit(1)

    with open(input_path, 'rb') as f:
        results = pickle.load(f)

    fig = plot_long_memory(results)
    output_dir = PROJECT_ROOT / 'output' / 'figures' / 'section_4'
    output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, 'figure_16_long_memory', output_dir, formats=['pdf', 'png'])
    print("Done!")
