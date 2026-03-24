"""
Plotting: Figure 10 - Transmission Lags by Initial Credibility
==============================================================
Single panel with colored inflation IRFs and convergence annotations.

Reads: output/simulations/section_3/transmission_lags.pkl
Writes: output/figures/section_3/figure_10_transmission_lags.{pdf,png}
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


def plot_transmission_lags(data: dict) -> plt.Figure:
    results = data['results']
    lags = data['lags']
    pi_star = data['pi_star']
    shock_start = data.get('shock_start', 4)
    T = len(next(iter(results.values())))
    time = np.arange(T)

    colors = {0.2: '#d62728', 0.5: '#2ca02c', 0.8: '#1f77b4'}
    markers = {0.2: 's', 0.5: '^', 0.8: 'o'}
    regime_labels = {0.2: 'Low (1970s)', 0.5: 'Medium (1990s)', 0.8: 'High (2010s)'}
    y_offsets = {0.2: 1.2, 0.5: 0.6, 0.8: 0.0}

    fig, ax = plt.subplots(1, 1, figsize=(9, 3.5))

    for theta_0 in sorted(results.keys()):
        pi = results[theta_0]
        color = colors.get(theta_0, 'gray')
        regime = regime_labels.get(theta_0, f'\u03b8={theta_0:.1f}')
        marker = markers.get(theta_0, 'o')
        ax.plot(time, pi * 400, color=color, linewidth=2.5, label=regime,
                marker=marker, markevery=3, markersize=5)

        # Convergence annotation
        m = lags[theta_0]
        conv_q = m['convergence']
        if 0 < conv_q < 50:
            conv_time = shock_start + conv_q
            y_off = y_offsets.get(theta_0, 0.5)
            ax.axvline(conv_time, color=color, linestyle=':', alpha=0.5, linewidth=1)
            ax.annotate(f'{conv_q}Q',
                        xy=(conv_time, pi_star * 400),
                        xytext=(conv_time, pi_star * 400 + y_off + 0.3),
                        fontsize=11, color='white', fontweight='bold',
                        ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=color,
                                  edgecolor='none', alpha=0.9))

    ax.axhline(pi_star * 400, color='gray', linestyle='--', alpha=0.7, linewidth=1,
               label='Target (2%)')
    ax.set_xlabel('Quarters')
    ax.set_ylabel('Inflation (% annualized)')
    ax.set_xlim(0, 25)
    ax.legend(loc='upper right', fontsize=10)

    fig.tight_layout()
    return fig


if __name__ == '__main__':
    setup_style()
    input_path = PROJECT_ROOT / 'output' / 'simulations' / 'section_3' / 'transmission_lags.pkl'
    if not input_path.exists():
        print("Error: simulation file not found.")
        sys.exit(1)
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    fig = plot_transmission_lags(data)
    output_dir = PROJECT_ROOT / 'output' / 'figures' / 'section_3'
    output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, 'figure_10_transmission_lags', output_dir, formats=['pdf', 'png'])
    print("Done!")
