"""
Plotting: Figure 07 - Great Moderation
=======================================
Single panel: Inflation paths for Adaptive vs Standard NK under low-volatility shocks.

Reads: output/simulations/section_3/great_moderation.pkl
Writes: output/figures/section_3/figure_07_great_moderation.{pdf,png}
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


def plot_great_moderation(data: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 2.8))
    years = np.arange(len(data['ad_pi'])) / 4
    target = data['pi_star'] * 400

    ax.plot(years, data['ad_pi'] * 400, 'b-',
            label='Adaptive', linewidth=1.5, alpha=0.9)
    ax.plot(years, data['nk_pi'] * 400, 'r--',
            label='Standard NK', linewidth=1.5, alpha=0.9)
    ax.axhline(y=target, color='gray', linestyle=':', alpha=0.7, label='Target (2%)')

    ax.set_xlabel('Years')
    ax.set_ylabel('Inflation (% annual)')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(1.0, 3.0)
    ax.set_xlim(0, 25)

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    setup_style()
    input_path = PROJECT_ROOT / 'output' / 'simulations' / 'section_3' / 'great_moderation.pkl'
    if not input_path.exists():
        print("Error: simulation file not found.")
        sys.exit(1)
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    fig = plot_great_moderation(data)
    output_dir = PROJECT_ROOT / 'output' / 'figures' / 'section_3'
    output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, 'figure_07_great_moderation', output_dir, formats=['pdf', 'png'])
    print("Done!")
