"""
Plotting: Figure 17 - Japan Below-Target Anchoring
===================================================
Panel (a): Credibility comparison across 3 scenarios
             (Japan 0%, Stable 2%, Mild undershoot 1%)
Panel (b): Loss difference dynamics for Japan scenario
             with epsilon threshold and light/dark shading

Reads: output/simulations/section_4/japan.pkl
Writes: output/figures/section_4/figure_17_japan.{pdf,png}
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


def plot_japan(data: dict) -> plt.Figure:
    """Create 2-panel Japan below-target anchoring figure."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    T = data['T']
    years = np.arange(T) / 4  # Convert quarters to years

    # ------------------------------------------------------------------
    # Panel (a): Credibility evolution comparison
    # ------------------------------------------------------------------
    ax = axes[0]

    scenarios = [
        ('japan_theta',  'blue',   '-',  'Japan (0%)'),
        ('stable_theta', 'black',  '--', 'Stable (2%)'),
        ('mild_theta',   '#666666', ':', 'Mild undershoot (1%)'),
    ]
    for key, color, style, label in scenarios:
        ax.plot(years, data[key], color=color, linestyle=style, linewidth=2, label=label)

    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Years')
    ax.set_ylabel(r'Credibility ($\theta$)')
    ax.set_title('(a) Credibility Evolution by Inflation Level')
    ax.legend(loc='lower left')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, T / 4)
    ax.grid(True, alpha=0.3)

    # Annotation about epsilon tolerance band
    ax.annotate(r'$\epsilon$ creates $\approx$1.6% tolerance band'
                '\naround target',
                xy=(T / 8, 0.95), fontsize=8, ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ------------------------------------------------------------------
    # Panel (b): Loss difference dynamics (Japan scenario)
    # ------------------------------------------------------------------
    ax = axes[1]
    loss_diff = data['japan_loss_diff']
    T_track = data.get('T_track', len(loss_diff))
    years_b = np.arange(T_track) / 4

    ax.plot(years_b, loss_diff, 'b-', linewidth=2)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
    ax.axhline(-1e-4, color='#333333', linestyle='--', alpha=0.5,
               label='Epsilon threshold')

    ax.set_xlabel('Years (after pre-fill)')
    ax.set_ylabel(r'$L_{CB} - L_{BL}$')
    ax.set_title('(b) Loss Difference (Japan: 0% inflation)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Shade region where CB wins vs BL wins (relative to epsilon threshold)
    ax.fill_between(years_b, loss_diff, -1e-4,
                    where=(loss_diff > -1e-4),
                    alpha=0.2, color='#999999', label='CB wins')
    ax.fill_between(years_b, loss_diff, -1e-4,
                    where=(loss_diff <= -1e-4),
                    alpha=0.2, color='#333333', label='BL wins')

    plt.suptitle('Japan: Below-Target Anchoring (Symmetric De-anchoring)\n'
                 'Long history of 0% inflation leads to expectations '
                 'anchored below target',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    setup_style()
    input_path = PROJECT_ROOT / 'output' / 'simulations' / 'section_4' / 'japan.pkl'

    if not input_path.exists():
        print(f"Error: simulation file not found at {input_path}")
        sys.exit(1)

    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    fig = plot_japan(data)
    output_dir = PROJECT_ROOT / 'output' / 'figures' / 'section_4'
    output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, 'figure_17_japan', output_dir, formats=['pdf', 'png'])
    print("Done!")
