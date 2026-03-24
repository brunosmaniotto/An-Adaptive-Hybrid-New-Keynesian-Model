"""
Plotting: Figure 02 - Roles of k and epsilon
=============================================
Panel (a): Re-anchoring problem with epsilon=0 (soft landing, BL always wins)
Panel (b): Memory window k=1 vs k=3 response to brief vs persistent shocks

Reads: output/simulations/section_2/k_epsilon_roles.pkl
Writes: output/figures/section_2/figure_02_k_epsilon_roles.{pdf,png}
"""

import sys
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'models'))

from plot_utils import setup_style, save_figure


def plot_k_epsilon_roles(data):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # =========================================================================
    # Panel (a): Re-anchoring problem - soft landing with epsilon = 0
    # =========================================================================
    ax = axes[0]
    d = data['panel_a']
    time = d['time']
    inflation = d['inflation']
    theta = d['theta']
    cb_wins = d['cb_wins']
    pi_star = d['pi_star']
    T = len(time)

    # Background shading based on which rule wins (white/light grey for B&W)
    for t in range(T - 1):
        if cb_wins[t]:
            ax.axvspan(t, t + 1, color='#ffffff', alpha=1.0, linewidth=0)
        else:
            ax.axvspan(t, t + 1, color='#999999', alpha=0.30, linewidth=0)

    # Lagged inflation for plotting
    inflation_lagged = np.zeros(T)
    inflation_lagged[0] = inflation[0]
    inflation_lagged[1:] = inflation[:-1]

    # Plot inflation and lagged inflation
    ax.plot(time, inflation, color='black', linewidth=2, label='Inflation')
    ax.plot(time, inflation_lagged, color='gray', linewidth=1.5, linestyle=':',
            alpha=0.7, label=r'Previous $\pi_{t-1}$')
    ax.axhline(pi_star, color='gray', linestyle='--', linewidth=1.5,
               label=r'Target $\pi^*$')

    ax.set_xlabel('Time (quarters)', fontsize=11)
    ax.set_ylabel('Inflation (%)', fontsize=11, color='black')
    ax.tick_params(axis='y', labelcolor='black')
    ax.set_xlim(0, T - 1)
    ax.set_ylim(0, 5)

    # Second y-axis for theta
    ax2 = ax.twinx()
    ax2.plot(time, theta, color='#4a90d9', linewidth=2.5, linestyle='--',
             alpha=0.7, label=r'Credibility $\theta$')
    ax2.set_ylabel(r'Credibility ($\theta$)', fontsize=11, color='#4a90d9')
    ax2.tick_params(axis='y', labelcolor='#4a90d9')
    ax2.set_ylim(0, 1.05)

    # Annotations
    ax.annotate('Stable:\nCB wins',
                xy=(5, 0.5), fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.annotate('Soft landing:\nBL always closer',
                xy=(40, 4.3), fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.annotate(r'$\theta$ stuck',
                xy=(52, 1.5), fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_title(r'(a) Without $\varepsilon$: Re-anchoring fails', fontsize=11)

    # Legend
    light_patch = Patch(facecolor='#ffffff', alpha=1.0, edgecolor='gray', label='CB wins')
    dark_patch = Patch(facecolor='#999999', alpha=0.30, label='BL wins')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2 + [light_patch, dark_patch],
               labels1 + labels2 + ['CB wins', 'BL wins'],
               loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=9)

    # =========================================================================
    # Panel (b): Memory window - k=1 vs k=3
    # =========================================================================
    ax = axes[1]
    d1 = data['panel_b_k1']
    d3 = data['panel_b_k3']
    time = d1['time']
    inflation = d1['inflation']
    pi_star = d1['pi_star']
    band_width = 1.0
    T = len(time)

    # Background shading (white/light grey for B&W)
    for t in range(T - 1):
        inside_band = abs(inflation[t] - pi_star) <= band_width
        if inside_band:
            ax.axvspan(t, t + 1, color='#ffffff', alpha=1.0, linewidth=0)
        else:
            ax.axvspan(t, t + 1, color='#999999', alpha=0.30, linewidth=0)

    # Plot inflation
    ax.plot(time, inflation, color='black', linewidth=2, label='Inflation')
    ax.axhline(pi_star, color='gray', linestyle='--', linewidth=1.5,
               label=r'Target $\pi^*$')

    ax.set_xlabel('Time (quarters)', fontsize=11)
    ax.set_ylabel('Inflation (%)', fontsize=11, color='black')
    ax.tick_params(axis='y', labelcolor='black')
    ax.set_xlim(0, T - 1)
    ax.set_ylim(0, 5)

    # Second y-axis for theta - k=3 solid, k=1 dashed
    ax2 = ax.twinx()
    ax2.plot(time, d3['theta'], color='#7fb3d5', linewidth=3, linestyle='-',
             alpha=0.9, label=r'$\theta$ ($k = 3$)')
    ax2.plot(time, d1['theta'], color='#1f4e79', linewidth=2, linestyle='--',
             alpha=0.9, label=r'$\theta$ ($k = 1$)')
    ax2.set_ylabel(r'Credibility ($\theta$)', fontsize=11, color='#1f4e79')
    ax2.tick_params(axis='y', labelcolor='#1f4e79')
    ax2.set_ylim(0, 1.05)

    # Annotations
    ax.annotate('Brief spike',
                xy=(13.5, 0.5), fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.annotate('Persistent\ndeviation',
                xy=(36, 0.5), fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_title(r'(b) Memory window $k$: Resilience to brief shocks', fontsize=11)

    # Legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2 + [light_patch, dark_patch],
               labels1 + labels2 + ['CB wins', 'BL wins'],
               loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=9)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20)

    return fig


if __name__ == '__main__':
    setup_style()
    input_path = PROJECT_ROOT / 'output' / 'simulations' / 'section_2' / 'k_epsilon_roles.pkl'

    if not input_path.exists():
        print("Error: simulation file not found.")
        sys.exit(1)

    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    fig = plot_k_epsilon_roles(data)
    output_dir = PROJECT_ROOT / 'output' / 'figures' / 'section_2'
    output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, 'figure_02_k_epsilon_roles', output_dir, formats=['pdf', 'png'])
    print("Done!")
