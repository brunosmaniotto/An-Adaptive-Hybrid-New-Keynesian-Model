"""
Plotting: Figure 3 - Lambda Comparison
======================================

Reads: output/simulations/section_2/lambda_comparison.pkl
Writes: output/figures/section_2/figure_03_lambda_comparison.pdf
"""

import sys
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.gridspec import GridSpec

# Project imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'models'))

from plot_utils import setup_style, save_figure, add_target_line, format_percent_axis
from parameters import get_default_params

def plot_lambda_comparison(data: dict) -> plt.Figure:
    config = data['config']
    results = data
    
    fig = plt.figure(figsize=(8.0, 4.375))
    gs = GridSpec(2, 2, height_ratios=[0.75, 0.50], hspace=0.3)
    axes = np.array([
        [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    ])
    
    params = get_default_params()
    pi_star = params['pi_star']
    T = config['T']
    time = np.arange(T)
    
    colors = ['red', 'blue', 'black']
    markers = ['s', '^', 'o']
    marker_every = 5
    lambda_values = config['lambda_values']
    
    for col_idx, scenario_name in enumerate(config['panel_order']):
        scenario_results = results[scenario_name]
        scenario_config = config['scenarios'][scenario_name]
        
        # Inflation
        ax_pi = axes[0, col_idx]
        for i, lam in enumerate(lambda_values):
            key = f'lambda_{lam}'
            label = f'$\lambda = {lam}$'
            if lam == 0: label += ' (All adaptive)'
            elif lam == 0.3: label += ' (Mixed)'
            elif lam == 1: label += ' (All FIRE)'
            
            ax_pi.plot(time, scenario_results[key]['pi'], color=colors[i],
                       linestyle='-', linewidth=1.5, marker=markers[i],
                       markersize=4, markevery=(i, marker_every), label=label)
                       
        add_target_line(ax_pi, pi_star, label=None)
        ax_pi.set_title(scenario_config['title'], fontsize=10)
        ax_pi.set_xlim(0, 25)
        format_percent_axis(ax_pi, axis='y', decimals=0, annualize=True)
        if col_idx == 0: ax_pi.set_ylabel('Inflation (%, ann.)')
        
        # Theta
        ax_theta = axes[1, col_idx]
        for i, lam in enumerate(lambda_values):
            key = f'lambda_{lam}'
            # No label for bottom row
            ax_theta.plot(time, scenario_results[key]['theta'], color=colors[i],
                          linestyle='-', linewidth=1.5, marker=markers[i],
                          markersize=4, markevery=(i, marker_every))
                          
        ax_theta.set_xlim(0, 25)
        ax_theta.set_ylim(-0.05, 1.05)
        ax_theta.set_xlabel('Quarters')
        if col_idx == 0: ax_theta.set_ylabel(r'Credibility ($	heta$)')
        
    # Scale Y axis
    max_pi_left = 0
    left_res = results[config['panel_order'][0]]
    for key in left_res:
        if key != 'config':
            max_pi_left = max(max_pi_left, np.max(left_res[key]['pi']))
    axes[0, 0].set_ylim(0, min(0.12, max_pi_left * 1.1))
    axes[0, 1].set_ylim(0, 0.025)
    
    axes[0, 0].legend(loc='upper right', fontsize=8)
    return fig

if __name__ == "__main__":
    setup_style()
    input_path = PROJECT_ROOT / 'output' / 'simulations' / 'section_2' / 'lambda_comparison.pkl'
    
    if not input_path.exists():
        print("Error: simulation file not found.")
        sys.exit(1)
        
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
        
    print("Generating figure...")
    fig = plot_lambda_comparison(data)
    
    output_dir = PROJECT_ROOT / 'output' / 'figures' / 'section_2'
    output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, 'figure_03_lambda_comparison', output_dir, formats=['pdf', 'png'])
    print("Done!")
