"""
Plotting: Figure 15 - Brazil Counterfactual vs Real Plan
========================================================
Panel (a): Inflation paths (counterfactual vs Real Plan)
Panel (b): URV expectation shares stackplot
Panel (c): Credibility dynamics

Reads: output/simulations/section_4/brazil.pkl
Writes: output/figures/section_4/figure_15_brazil.{pdf,png}
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


def plot_brazil(data: dict) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel A: Inflation paths
    ax = axes[0]
    t_cf = np.arange(len(data['cf_pi']))
    t_real = np.arange(len(data['real_pi']))
    ax.plot(t_cf, data['cf_pi'] * 400, color='#999999', linestyle='--', linewidth=2, label='Counterfactual')
    ax.plot(t_real, data['real_pi'] * 400, color='black', linestyle='-', linewidth=2, label='Real Plan')
    ax.axhline(2, color='gray', linestyle='--', alpha=0.5, label='Target')
    ax.set_xlabel('Quarter')
    ax.set_ylabel('Inflation (% annualized)')
    ax.set_title('(a) Same Shock, Different Outcomes')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Panel B: URV re-anchoring stackplot
    ax = axes[1]
    t_urv = np.arange(len(data['urv_theta_cb']))
    ax.stackplot(t_urv,
                 data['urv_theta_cb'], data['urv_theta_bl'], data['urv_theta_tf'],
                 labels=['CB-anchored', 'Backward-looking', 'Trend-following'],
                 colors=['#CCCCCC', '#888888', '#333333'], alpha=0.7)
    ax.set_xlabel('Quarters since URV')
    ax.set_ylabel('Expectation share')
    ax.set_title('(b) URV: Expectations Re-anchor')
    ax.legend(loc='center right', fontsize=9)
    ax.set_ylim(0, 1)

    # Panel C: Credibility dynamics
    ax = axes[2]
    ax.plot(t_cf, data['cf_theta_cb'], color='#999999', linestyle='--', linewidth=2, label='Counterfactual')
    ax.plot(t_real, data['real_theta_cb'], color='black', linestyle='-', linewidth=2, label='Real Plan')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Quarter')
    ax.set_ylabel(r'Credibility ($\theta_{CB}$)')
    ax.set_title('(c) Credibility Dynamics')
    ax.legend(loc='right')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    setup_style()
    input_path = PROJECT_ROOT / 'output' / 'simulations' / 'section_4' / 'brazil.pkl'

    if not input_path.exists():
        print("Error: simulation file not found.")
        sys.exit(1)

    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    fig = plot_brazil(data)
    output_dir = PROJECT_ROOT / 'output' / 'figures' / 'section_4'
    output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, 'figure_15_brazil', output_dir, formats=['pdf', 'png'])
    print("Done!")
