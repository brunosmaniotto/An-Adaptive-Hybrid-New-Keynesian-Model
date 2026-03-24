"""
Simulation: Learning Mechanism (Schematic)
==========================================

Generates stylized inflation and credibility paths for Figure 1.
"""

import sys
import pickle
import numpy as np
from pathlib import Path

# Add parent directories to path
# Path: code/simulations/section_2/learning_mechanism.py
# Root is 3 levels up
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'models'))

def generate_stylized_paths(T: int = 60) -> dict:
    """Generate stylized inflation and credibility paths."""
    np.random.seed(42)

    time = np.arange(T)
    pi_star = 2.0  # Target inflation (%)
    band_width = 1.0  # Band around target (±1%)
    eta = 0.10  # Learning rate

    inflation = np.zeros(T)

    # Phase 1: Stable
    for t in range(15):
        inflation[t] = pi_star + 0.3 * np.sin(t * 0.5) + np.random.normal(0, 0.2)

    # Phase 2: Deviation
    for t in range(15, 35):
        if t < 22:
            inflation[t] = pi_star + 0.5 + (t - 15) * 0.3 + np.random.normal(0, 0.15)
        else:
            inflation[t] = pi_star + 2.5 + np.random.normal(0, 0.3)

    # Phase 3: Recovery
    for t in range(35, T):
        decay = np.exp(-0.15 * (t - 35))
        inflation[t] = pi_star + 2.5 * decay + np.random.normal(0, 0.2)

    # Smooth
    from scipy.ndimage import uniform_filter1d
    inflation = uniform_filter1d(inflation, size=3)

    # Rules
    cb_wins = np.zeros(T, dtype=bool)
    for t in range(T):
        cb_wins[t] = abs(inflation[t] - pi_star) <= band_width

    # Theta
    theta = np.zeros(T)
    theta[0] = 1.0

    for t in range(1, T):
        if cb_wins[t-1]:
            theta[t] = (1 - eta) * theta[t-1] + eta * 1.0
        else:
            theta[t] = (1 - eta) * theta[t-1] + eta * 0.0

    return {
        'time': time,
        'inflation': inflation,
        'theta': theta,
        'cb_wins': cb_wins,
        'pi_star': pi_star,
        'band_width': band_width,
        'eta': eta
    }

if __name__ == "__main__":
    print("Running stylized simulation for Figure 1...")
    data = generate_stylized_paths(T=60)
    
    # Save output
    output_dir = PROJECT_ROOT / 'output' / 'simulations' / 'section_2'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'learning_mechanism.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
        
    print(f"Saved results to {output_path}")
