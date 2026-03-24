"""
Plotting: Figure 14 - Hyperinflation and the Three-Arm Model
=============================================================
Panel (a): Inflation paths for 3 policy stances (phi = 0.5, 1.0, 1.5)
Panel (b): Trend-following share (theta_TF) for 3 policy stances
Panel (c): Gear-shift stackplot (CB -> BL -> TF) for phi=0.5
Panel (d): Acceleration (delta_pi) for phi=0.5

Reads: output/simulations/section_4/hyperinflation.pkl
Writes: output/figures/section_4/figure_14_hyperinflation.{pdf,png}
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


def plot_hyperinflation(data: dict) -> plt.Figure:
    """Create 4-panel hyperinflation figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    phi_values = [0.5, 1.0, 1.5]
    colors = ['black', '#555555', '#999999']
    styles = ['-', '--', '-.']

    # ------------------------------------------------------------------
    # Panel (a): Inflation paths by policy stance
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    for phi, color, style in zip(phi_values, colors, styles):
        pi_ann = data[phi]['pi'] * 400
        ax.plot(pi_ann, color=color, linestyle=style, linewidth=2, label=f'$\\phi_\\pi = {phi}$')
    ax.axhline(2, color='gray', linestyle='--', alpha=0.5, label='Target')
    ax.set_xlabel('Quarter')
    ax.set_ylabel('Inflation (% annual)')
    ax.set_title('(a) Inflation by Policy Stance')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Panel (b): Trend-following share by policy stance
    # ------------------------------------------------------------------
    ax = axes[0, 1]
    for phi, color, style in zip(phi_values, colors, styles):
        ax.plot(data[phi]['theta_tf'], color=color, linestyle=style, linewidth=2,
                label=f'$\\phi_\\pi = {phi}$')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Quarter')
    ax.set_ylabel(r'Trend-Following Share ($\theta_{TF}$)')
    ax.set_title('(b) TF Activation Requires Passive Policy')
    ax.legend(loc='upper left')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Panel (c): Gear-shift stackplot for phi=0.3 (T=50)
    # ------------------------------------------------------------------
    ax = axes[1, 0]
    gear_phi = data.get('gear_shift_phi', 0.3)
    # Use dedicated gear_shift data if available (T=50), else fall back
    gd = data.get('gear_shift', data[gear_phi])
    T_gs = len(gd['theta_cb'])
    ax.stackplot(range(T_gs),
                 gd['theta_cb'], gd['theta_bl'], gd['theta_tf'],
                 labels=['CB-anchored', 'Backward-looking', 'Trend-following'],
                 colors=['#CCCCCC', '#888888', '#333333'], alpha=0.7)
    ax.set_xlabel('Quarter')
    ax.set_ylabel('Share')
    ax.set_title(f'(c) Gear Shift: CB $\\rightarrow$ BL $\\rightarrow$ TF '
                 f'($\\phi_\\pi = {gear_phi}$)')
    ax.legend(loc='center right')
    ax.set_ylim(0, 1)
    # Shock annotation
    ax.axvline(5, color='black', linestyle=':', alpha=0.7)
    ax.text(6, 0.95, 'Shock', fontsize=9)

    # ------------------------------------------------------------------
    # Panel (d): Acceleration (inflation level + delta_pi)
    # ------------------------------------------------------------------
    ax = axes[1, 1]
    pi_ann = gd['pi'] * 400
    delta_pi = gd.get('delta_pi', np.diff(gd['pi'])) * 400

    ax.plot(pi_ann, color='black', linestyle='-', linewidth=2, label='Inflation level')
    ax_twin = ax.twinx()
    ax_twin.plot(range(1, len(pi_ann)), delta_pi, color='#666666', linestyle='--', linewidth=1.5,
                 label=r'Acceleration ($\Delta\pi$)', alpha=0.7)
    ax_twin.axhline(0, color='gray', linestyle=':', alpha=0.5)

    ax.set_xlabel('Quarter')
    ax.set_ylabel('Inflation (% annual)')
    ax_twin.set_ylabel(r'$\Delta\pi$ (acceleration)')
    ax.set_title('(d) TF Wins When Inflation Accelerates')

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.suptitle('Hyperinflation and the Three-Arm Model\n'
                 'Trend-Following Creates Explosive Dynamics Under Fiscal Dominance',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    setup_style()
    input_path = PROJECT_ROOT / 'output' / 'simulations' / 'section_4' / 'hyperinflation.pkl'

    if not input_path.exists():
        print(f"Error: simulation file not found at {input_path}")
        sys.exit(1)

    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    fig = plot_hyperinflation(data)
    output_dir = PROJECT_ROOT / 'output' / 'figures' / 'section_4'
    output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, 'figure_14_hyperinflation', output_dir, formats=['pdf', 'png'])
    print("Done!")
