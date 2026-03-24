"""
Plotting: Figure 13 - Credibility Buffer
========================================

Reads: output/simulations/section_3/credibility_buffer.pkl
Writes: output/figures/section_3/figure_13_credibility_buffer.pdf
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

def plot_credibility_buffer(data: dict) -> plt.Figure:
    scenarios = [
        {'label': 'High Cred (Immediate)', 'color': '#1f77b4', 'style': '-'},
        {'label': 'High Cred (Delayed)', 'color': '#1f77b4', 'style': '--'},
        {'label': 'Low Cred (Immediate)', 'color': 'black', 'style': '-'},
        {'label': 'Low Cred (Delayed)', 'color': 'black', 'style': '--'}
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    
    # Panel A
    ax = axes[0]
    ax.axhline(2.0, color='gray', linestyle=':', linewidth=1)
    ax.axvspan(0, 8, color='gray', alpha=0.1, label='Delay')
    
    for s in scenarios:
        res = data[s['label']]
        ax.plot(res['pi']*400, color=s['color'], linestyle=s['style'], 
                label=s['label'], linewidth=2 if 'Immediate' in s['label'] else 1.5)
                
    ax.set_title("(a) Inflation Response")
    ax.set_ylabel("Inflation (%)")
    ax.set_xlabel("Quarters")
    ax.set_xlim(0, 30)
    ax.legend(loc='upper right', fontsize=8)
    
    # Panel B
    ax = axes[1]
    ax.axvspan(0, 8, color='gray', alpha=0.1)
    
    for s in scenarios:
        res = data[s['label']]
        pi_dev = (res['pi'] - get_default_params()['pi_star']) * 400
        y_dev = res['y'] * 100
        cum_loss = np.cumsum(pi_dev**2 + 0.25*y_dev**2)
        ax.plot(cum_loss, color=s['color'], linestyle=s['style'], linewidth=1.5, label=s['label'])
        
    ax.set_title("(b) Cumulative Welfare Loss")
    ax.set_ylabel("Loss Units")
    ax.set_xlabel("Quarters")
    ax.set_xlim(0, 30)
    ax.legend(loc='upper left', fontsize=8)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    setup_style()
    input_path = PROJECT_ROOT / 'output' / 'simulations' / 'section_3' / 'credibility_buffer.pkl'
    
    if not input_path.exists():
        print("Error: simulation file not found.")
        sys.exit(1)
        
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
        
    print("Generating figure...")
    fig = plot_credibility_buffer(data)
    
    output_dir = PROJECT_ROOT / 'output' / 'figures' / 'section_3'
    output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, 'figure_13_credibility_buffer', output_dir, formats=['pdf', 'png'])
    print("Done!")
