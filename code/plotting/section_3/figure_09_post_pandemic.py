"""
Plotting: Figure 09 - Post-Pandemic Inflation
==============================================
Panel 1: Inflation comparison (Adaptive vs NK)
Panel 2: Credibility collapse
Panel 3: Difference with light/dark shading phases and crossover

Reads: output/simulations/section_3/post_pandemic.pkl
Writes: output/figures/section_3/figure_09_post_pandemic.{pdf,png}
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


def plot_post_pandemic(data: dict) -> plt.Figure:
    T = len(data['ad_pi'])
    years = np.arange(T) / 4
    target = data['pi_star'] * 400
    diff = (data['ad_pi'] - data['nk_pi']) * 400

    peak_nk = max(data['nk_pi']) * 400
    peak_adaptive = max(data['ad_pi']) * 400
    dampening = peak_nk - peak_adaptive

    # Find crossover point
    crossover = None
    for t in range(5, T - 1):
        if diff[t] < 0 and diff[t + 1] >= 0:
            crossover = t
            break

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: Inflation comparison
    ax = axes[0]
    ax.plot(years, data['nk_pi'] * 400, 'r--', label='Standard NK', linewidth=2)
    ax.plot(years, data['ad_pi'] * 400, 'b-', label=r'Adaptive ($\theta_0$=0.60)', linewidth=2)
    ax.axhline(y=target, color='gray', linestyle=':', alpha=0.7, label='Target (2%)')
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Years')
    ax.set_ylabel('Inflation (% annual)')
    ax.set_title(f'Inflation: Dampening ({chr(8722)}{dampening:.1f}pp peak)')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(0, 15)

    # Panel 2: Credibility
    ax = axes[1]
    ax.plot(years, data['theta'], 'b-', linewidth=2)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Years')
    ax.set_ylabel(r'$\theta$ (Credibility)')
    ax.set_title('Credibility: Erodes During Shock')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 15)

    # Annotate min theta
    min_theta_idx = np.argmin(data['theta'])
    min_theta = min(data['theta'])
    ax.annotate(f'Min \u03b8 = {min_theta:.2f}',
                xy=(min_theta_idx / 4, min_theta),
                xytext=(min_theta_idx / 4 + 2, min_theta + 0.15),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='blue'))

    # Panel 3: Difference with green/red phases
    ax = axes[2]
    ax.plot(years, diff, 'purple', linewidth=2)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax.fill_between(years, diff, 0, where=(diff < 0), alpha=0.3, color='#999999',
                    label='Dampening (Adaptive < NK)')
    ax.fill_between(years, diff, 0, where=(diff >= 0), alpha=0.3, color='#333333',
                    label='Persistence (Adaptive > NK)')

    if crossover:
        ax.axvline(x=crossover / 4, color='purple', linestyle='--', alpha=0.7)
        ax.text(crossover / 4 + 0.2, ax.get_ylim()[0] + 0.5,
                f'Crossover\n({crossover / 4:.1f}y)',
                fontsize=9, color='purple', va='bottom')

    ax.set_xlabel('Years')
    ax.set_ylabel('Adaptive \u2212 NK (pp)')
    ax.set_title('Two Phases: Dampening \u2192 Persistence')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(0, 15)

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    setup_style()
    input_path = PROJECT_ROOT / 'output' / 'simulations' / 'section_3' / 'post_pandemic.pkl'
    if not input_path.exists():
        print("Error: simulation file not found.")
        sys.exit(1)
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    fig = plot_post_pandemic(data)
    output_dir = PROJECT_ROOT / 'output' / 'figures' / 'section_3'
    output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, 'figure_09_post_pandemic', output_dir, formats=['pdf', 'png'])
    print("Done!")
