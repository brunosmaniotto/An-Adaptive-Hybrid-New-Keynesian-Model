"""
Plotting: Figure 11 - Phillips Curve Estimation
===============================================
Panel (a): Standard vs Hybrid PC slope estimates across 3 credibility regimes
Panel (b): AR(1) inflation persistence by regime

Reads: output/simulations/section_3/phillips_curve.pkl
Writes: output/figures/section_3/figure_11_phillips_curve.{pdf,png}
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


def plot_phillips_curve(data: dict) -> plt.Figure:
    estimates = data['estimates']
    true_kappa = data['true_kappa']

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    regimes = ['low', 'medium', 'high']
    regime_labels = {'low': 'Low \u03b8 (1970s)', 'medium': 'Medium \u03b8 (1990s)', 'high': 'High \u03b8 (2000s)'}
    colors = {'low': '#d62728', 'medium': '#2ca02c', 'high': '#1f77b4'}

    # Panel (a): Grouped bars - Standard vs Hybrid PC slope estimates
    ax_b = axes[0]
    x_pos = np.arange(len(regimes))
    bar_width = 0.35

    kappas_std = [estimates[r]['kappa_standard'] for r in regimes]
    kappas_hyb = [estimates[r]['kappa_hybrid'] for r in regimes]

    bars_std = ax_b.bar(x_pos - bar_width / 2, kappas_std, bar_width,
                        color=[colors[r] for r in regimes], edgecolor='black',
                        linewidth=1, label='Standard PC', alpha=0.9)
    bars_hyb = ax_b.bar(x_pos + bar_width / 2, kappas_hyb, bar_width,
                        color=[colors[r] for r in regimes], edgecolor='black',
                        linewidth=1, label='Hybrid PC', alpha=0.5, hatch='///')

    ax_b.axhline(true_kappa, color='black', linestyle='--', linewidth=1.5,
                 label=f'True \u03ba = {true_kappa:.3f}')

    for bar, val in zip(bars_std, kappas_std):
        ax_b.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.002,
                  f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    for bar, val in zip(bars_hyb, kappas_hyb):
        ax_b.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.002,
                  f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels([regime_labels[r] for r in regimes], fontsize=9)
    ax_b.set_ylabel(r'Estimated $\hat{\kappa}$')
    ax_b.set_title('(a) PC Slope Estimates: Standard vs Hybrid')
    ax_b.legend(loc='upper right', fontsize=8)
    ax_b.set_ylim(0, max(kappas_std) * 1.25)

    # Panel (b): Inflation persistence (AR(1))
    ax_c = axes[1]
    rhos = [estimates[r]['rho_ar1'] for r in regimes]

    bars = ax_c.bar(x_pos, rhos,
                    color=[colors[r] for r in regimes], edgecolor='black', linewidth=1, width=0.5)

    for bar, val in zip(bars, rhos):
        ax_c.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                  f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels([regime_labels[r] for r in regimes], fontsize=9)
    ax_c.set_ylabel(r'AR(1) Coefficient $\hat{\rho}$')
    ax_c.set_title('(b) Inflation Persistence')
    ax_c.set_ylim(0, 1.05)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    setup_style()
    input_path = PROJECT_ROOT / 'output' / 'simulations' / 'section_3' / 'phillips_curve.pkl'

    if not input_path.exists():
        print("Error: simulation file not found.")
        sys.exit(1)

    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    print("Generating figure...")
    fig = plot_phillips_curve(data)

    output_dir = PROJECT_ROOT / 'output' / 'figures' / 'section_3'
    output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, 'figure_11_phillips_curve', output_dir, formats=['pdf', 'png'])
    print("Done!")
