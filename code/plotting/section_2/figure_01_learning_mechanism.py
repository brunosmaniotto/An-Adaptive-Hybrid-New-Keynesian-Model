"""
Plotting: Figure 1 - Learning Mechanism
=======================================

Reads: output/simulations/section_2/learning_mechanism.pkl
Writes: output/figures/section_2/figure_01_learning_mechanism.pdf
"""

import sys
import pickle
import matplotlib
matplotlib.use('Agg') # Non-interactive
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directories to path
# Path: code/plotting/section_2/figure_01.py
# Root is 3 levels up
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'models'))

from plot_utils import setup_style, save_figure

def plot_learning_mechanism(data: dict) -> plt.Figure:
    """Create the learning mechanism schematic figure."""
    time = data['time']
    inflation = data['inflation']
    theta = data['theta']
    pi_star = data['pi_star']
    band_width = data['band_width']
    T = len(time)

    # Create figure
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Shading
    for t in range(T - 1):
        inside_band = abs(inflation[t] - pi_star) <= band_width
        color = '#ffffff' if inside_band else '#999999'
        alpha = 1.0 if inside_band else 0.30
        ax1.axvspan(t, t + 1, color=color, alpha=alpha, linewidth=0)

    # Plot inflation
    ax1.plot(time, inflation, color='black', linewidth=2, label='Inflation')
    ax1.axhline(pi_star, color='#333333', linestyle='--', linewidth=2, label=r'Target $\pi^* $')
    ax1.axhline(pi_star + band_width, color='#555555', linestyle='-', linewidth=1.5, alpha=0.8)
    ax1.axhline(pi_star - band_width, color='#555555', linestyle='-', linewidth=1.5, alpha=0.8)
    ax1.fill_between(time, pi_star - band_width, pi_star + band_width, color='gray', alpha=0.15)

    ax1.set_xlabel('Time (quarters)', fontsize=11)
    ax1.set_ylabel('Inflation (%)', fontsize=11, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xlim(0, T - 1)
    ax1.set_ylim(0, 6)

    # Theta
    ax2 = ax1.twinx()
    ax2.plot(time, theta, color='#4a90d9', linewidth=2.5, linestyle='--', alpha=0.7, label=r'Credibility $\theta$')
    ax2.set_ylabel(r'Credibility ($	heta$)', fontsize=11, color='#4a90d9')
    ax2.tick_params(axis='y', labelcolor='#4a90d9')
    ax2.set_ylim(0, 1.05)

    # Legend
    from matplotlib.patches import Patch
    light_patch = Patch(facecolor='#ffffff', alpha=1.0, edgecolor='#cccccc', label='Inside band (CB wins)')
    dark_patch = Patch(facecolor='#999999', alpha=0.30, edgecolor='none', label='Outside band (BL wins)')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    ax2.legend(lines1 + lines2 + [light_patch, dark_patch],
               labels1 + labels2 + ['Inside band (CB wins)', 'Outside band (BL wins)'],
               loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fontsize=9)

    # Annotations
    ax1.annotate('Stable: CB rule\nforecasts well', xy=(7, 3.5), fontsize=9, ha='center',
                 bbox=dict(boxstyle='round', facecolor='#e0e0e0', edgecolor='#aaaaaa', alpha=0.8))
    ax1.annotate('Persistent deviation:\nBL rule wins', xy=(27, 5.3), fontsize=9, ha='center',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax1.annotate('Return to target:\nCredibility rebuilds', xy=(50, 5.3), fontsize=9, ha='center',
                 bbox=dict(boxstyle='round', facecolor='#e0e0e0', edgecolor='#aaaaaa', alpha=0.8))

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    return fig

if __name__ == "__main__":
    setup_style()
    
    # Load data
    input_path = PROJECT_ROOT / 'output' / 'simulations' / 'section_2' / 'learning_mechanism.pkl'
    if not input_path.exists():
        print(f"Error: {input_path} not found. Run simulation first.")
        sys.exit(1)
        
    print(f"Loading data from {input_path}...")
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
        
    print("Generating figure...")
    fig = plot_learning_mechanism(data)
    
    output_dir = PROJECT_ROOT / 'output' / 'figures' / 'section_2'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_figure(fig, 'figure_01_learning_mechanism', output_dir, formats=['pdf', 'png'])
    print("Done!")
