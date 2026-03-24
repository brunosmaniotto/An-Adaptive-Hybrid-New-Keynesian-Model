"""
Plotting: Figure 4 - Shock Regimes
==================================

Reads: output/simulations/section_2/shock_regimes.pkl
Writes: output/figures/section_2/figure_04_shock_regimes.pdf
"""

import sys
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Project imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'models'))

from plot_utils import setup_style, save_figure
from parameters import get_default_params

def plot_shock_regimes(results: dict) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    params = get_default_params()
    pi_star = params['pi_star']
    
    regimes = {
        'great_moderation': {'title': 'Great Moderation', 'subtitle': '(Moderate Volatility)', 'color': '#2ecc71'},
        'high_volatility': {'title': '1970s-Style', 'subtitle': '(High Volatility)', 'color': '#e74c3c'},
        'regime_switch': {'title': 'Credibility Cycle', 'subtitle': '(Calm -> Turbulent -> Calm)', 'color': '#3498db'}
    }
    order = ['great_moderation', 'high_volatility', 'regime_switch']
    
    T = len(results['great_moderation']['pi'])
    years = np.arange(T) / 4
    
    for col, regime in enumerate(order):
        res = results[regime]
        cfg = regimes[regime]
        color = cfg['color']
        
        # Inflation
        ax = axes[0, col]
        ax.plot(years, res['pi'] * 400, color=color, linewidth=1.5, alpha=0.9)
        ax.axhline(y=pi_star * 400, color='black', linestyle='--', linewidth=1, alpha=0.7)
        title_text = cfg['title'] + "\n" + cfg['subtitle']
        ax.set_title(title_text, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 16)
        if col == 0: ax.set_ylabel('Inflation (%, ann.)')
        
        if regime == 'regime_switch':
            ax.axvline(x=5, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
            ax.axvline(x=15, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
            ax.axvspan(5, 15, alpha=0.1, color='gray')

        # Credibility
        ax = axes[1, col]
        ax.plot(years, res['theta'], color=color, linewidth=1.5, alpha=0.9)
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(y=0.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Years')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        if col == 0: ax.set_ylabel('Credibility (theta)')

        if regime == 'regime_switch':
            ax.axvline(x=5, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
            ax.axvline(x=15, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
            ax.axvspan(5, 15, alpha=0.1, color='gray')
            
    # Labels
    axes[0, 2].text(1.05, 0.5, 'Inflation', transform=axes[0, 2].transAxes, fontsize=11, fontweight='bold', rotation=-90, va='center')
    axes[1, 2].text(1.05, 0.5, 'Credibility', transform=axes[1, 2].transAxes, fontsize=11, fontweight='bold', rotation=-90, va='center')
                    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    setup_style()
    input_path = PROJECT_ROOT / 'output' / 'simulations' / 'section_2' / 'shock_regimes.pkl'
    if not input_path.exists(): sys.exit(1)
    with open(input_path, 'rb') as f: data = pickle.load(f)
    fig = plot_shock_regimes(data)
    output_dir = PROJECT_ROOT / 'output' / 'figures' / 'section_2'
    output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, 'figure_04_shock_regimes', output_dir, formats=['pdf', 'png'])
