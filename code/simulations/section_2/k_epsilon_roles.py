"""
Simulation: Roles of k and epsilon (Figure 2)
=============================================

Generates data for:
(a) The re-anchoring problem (epsilon = 0)
(b) Memory window effect (k=1 vs k=3)
"""

import sys
import pickle
import numpy as np
from pathlib import Path

# Project imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'models'))

def simulate_soft_landing(T: int = 60) -> dict:
    """Simulate soft landing scenario (Panel A)."""
    pi_star = 2.0
    eta = 0.10

    inflation = np.zeros(T)
    for t in range(T):
        if t < 10: inflation[t] = pi_star
        elif t < 15: inflation[t] = pi_star + 0.4 * (t - 10)
        elif t < 20: inflation[t] = 4.0
        else:
            decay = np.exp(-0.06 * (t - 20))
            inflation[t] = pi_star + 2.0 * decay

    # Smooth
    from scipy.ndimage import uniform_filter1d
    inflation = uniform_filter1d(inflation, size=2)

    cb_wins = np.zeros(T, dtype=bool)
    cb_wins[0] = True

    for t in range(1, T):
        loss_cb = (inflation[t] - pi_star) ** 2
        loss_bl = (inflation[t] - inflation[t-1]) ** 2
        if loss_cb < 1e-6 and loss_bl < 1e-6:
            cb_wins[t] = True
        else:
            cb_wins[t] = loss_cb < loss_bl

    theta = np.zeros(T)
    theta[0] = 1.0

    for t in range(1, T):
        if cb_wins[t-1]:
            theta[t] = (1 - eta) * theta[t-1] + eta * 1.0
        else:
            theta[t] = (1 - eta) * theta[t-1] + eta * 0.0

    return {
        'time': np.arange(T),
        'inflation': inflation,
        'theta': theta,
        'cb_wins': cb_wins,
        'pi_star': pi_star
    }

def simulate_memory_effect(T: int = 60, k: int = 1) -> dict:
    """Simulate memory effect (Panel B)."""
    pi_star = 2.0
    band_width = 1.0
    eta = 0.10

    inflation = np.zeros(T)
    for t in range(T):
        if t < 12: inflation[t] = pi_star
        elif t == 12: inflation[t] = pi_star + 1.8
        elif t < 30: inflation[t] = pi_star
        elif t < 42: inflation[t] = pi_star + 1.8
        else: inflation[t] = pi_star

    cb_wins = np.zeros(T, dtype=bool)

    for t in range(T):
        start = max(0, t - k + 1)
        window_size = t - start + 1
        n_inside = sum(1 for s in range(start, t + 1)
                       if abs(inflation[s] - pi_star) <= band_width)
        cb_wins[t] = n_inside > window_size / 2

    theta = np.zeros(T)
    theta[0] = 1.0

    for t in range(1, T):
        if cb_wins[t-1]:
            theta[t] = (1 - eta) * theta[t-1] + eta * 1.0
        else:
            theta[t] = (1 - eta) * theta[t-1] + eta * 0.0

    return {
        'time': np.arange(T),
        'inflation': inflation,
        'theta': theta,
        'cb_wins': cb_wins,
        'pi_star': pi_star,
        'band_width': band_width
    }

if __name__ == "__main__":
    print("Running simulations for Figure 2...")
    
    # Panel A
    data_a = simulate_soft_landing(T=60)
    
    # Panel B
    np.random.seed(42)
    data_b_k1 = simulate_memory_effect(T=60, k=1)
    np.random.seed(42)
    data_b_k3 = simulate_memory_effect(T=60, k=3)
    
    results = {
        'panel_a': data_a,
        'panel_b_k1': data_b_k1,
        'panel_b_k3': data_b_k3
    }
    
    output_dir = PROJECT_ROOT / 'output' / 'simulations' / 'section_2'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'k_epsilon_roles.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
        
    print(f"Saved results to {output_path}")
