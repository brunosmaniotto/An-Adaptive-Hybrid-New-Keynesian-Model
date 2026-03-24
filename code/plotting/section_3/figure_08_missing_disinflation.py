"""
Plotting: Figure 08 - Missing Disinflation
============================================
Single panel: Inflation paths for Standard NK vs Adaptive with
arrow annotation showing reduced deflation.

Reads: output/simulations/section_3/missing_disinflation.pkl
Writes: output/figures/section_3/figure_08_missing_disinflation.{pdf,png}
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


def plot_missing_disinflation(data: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 2.8))
    T = len(data['ad_pi'])
    years = np.arange(T) / 4
    target = data['pi_star'] * 400

    ax.plot(years, data['nk_pi'] * 400, 'r--', label='Standard NK', linewidth=2)
    ax.plot(years, data['ad_pi'] * 400, 'b-',
            label=r'Adaptive ($\theta_0$=0.95)', linewidth=2)
    ax.axhline(y=target, color='gray', linestyle=':', alpha=0.7, label='Target (2%)')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.2)

    ax.set_xlabel('Years')
    ax.set_ylabel('Inflation (% annual)')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(0, 6)

    # Compute reduction percentage and add green arrow annotation
    min_nk = min(data['nk_pi']) * 400
    min_adaptive = min(data['ad_pi']) * 400
    deviation_nk = abs(min_nk - target)
    deviation_adaptive = abs(min_adaptive - target)
    reduction = (1 - deviation_adaptive / deviation_nk) * 100

    t_min = np.argmin(data['nk_pi'])
    arrow_x = t_min / 4 + 1.5
    ax.annotate('', xy=(arrow_x, min_adaptive), xytext=(arrow_x, min_nk),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax.text(arrow_x + 0.4, (min_nk + min_adaptive) / 2,
            f'{reduction:.0f}%\nless\ndeflation',
            fontsize=9, color='black', va='center')

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    setup_style()
    input_path = PROJECT_ROOT / 'output' / 'simulations' / 'section_3' / 'missing_disinflation.pkl'
    if not input_path.exists():
        print("Error: simulation file not found.")
        sys.exit(1)
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    fig = plot_missing_disinflation(data)
    output_dir = PROJECT_ROOT / 'output' / 'figures' / 'section_3'
    output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, 'figure_08_missing_disinflation', output_dir, formats=['pdf', 'png'])
    print("Done!")
