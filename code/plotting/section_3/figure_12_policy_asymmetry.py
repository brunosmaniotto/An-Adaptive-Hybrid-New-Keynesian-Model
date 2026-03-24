"""
Plotting: Figure 12 - Policy Asymmetry
======================================

Reads: output/simulations/section_3/policy_asymmetry.pkl
Writes: output/figures/section_3/figure_12_policy_asymmetry.pdf
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

def plot_policy_asymmetry(data: dict) -> plt.Figure:
    loss_matrix = data['loss_matrix']
    opt_phis = data['optimal_phis']
    thetas_plot = data['theta_plot']
    thetas_full = data['theta_full']
    phi_levels = data['phi_levels']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    colors = ['red', 'blue', 'green', 'purple']
    
    # Panel A
    for i, theta in enumerate(thetas_plot):
        loss = loss_matrix[i, :]
        ax1.plot(phi_levels, loss, color=colors[i], linewidth=2, label=rf'$\theta_0={theta}$')
        min_loss = np.min(loss)
        opt_phi = phi_levels[np.argmin(loss)]
        ax1.plot(opt_phi, min_loss, 'o', color=colors[i], markersize=7)
        
    ax1.axvline(1.0, color='gray', linestyle='--', label=r'$\phi_\pi=1$')
    ax1.set_xlabel(r'Policy Aggressiveness ($\phi_\pi$)')
    ax1.set_ylabel('Welfare Loss')
    ax1.set_title('(a) Loss Functions by Credibility')
    ax1.legend(loc='upper right', fontsize=9)
    
    # Panel B
    ax2.plot(thetas_full, opt_phis, 'o-', color='black', linewidth=2, markersize=4)
    ax2.set_xlabel(r'Initial Credibility ($\theta_0$)')
    ax2.set_ylabel(r'Optimal $\phi_\pi^*$')
    ax2.set_title('(b) Optimal Policy Coefficient')
    ax2.set_xlim(0, 1)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(1.0, color='gray', linestyle='--')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    setup_style()
    input_path = PROJECT_ROOT / 'output' / 'simulations' / 'section_3' / 'policy_asymmetry.pkl'
    
    if not input_path.exists():
        print("Error: simulation file not found.")
        sys.exit(1)
        
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
        
    print("Generating figure...")
    fig = plot_policy_asymmetry(data)
    
    output_dir = PROJECT_ROOT / 'output' / 'figures' / 'section_3'
    output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, 'figure_12_policy_asymmetry', output_dir, formats=['pdf', 'png'])
    print("Done!")
