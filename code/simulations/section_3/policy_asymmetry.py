"""
Simulation: Policy Asymmetry (Figure 12)
========================================

Grid search for optimal policy.
"""

import sys
import pickle
import numpy as np
from pathlib import Path

# Project imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'models'))

from full_model import FullModel
from parameters import get_default_params, override_params

def compute_loss(result, pi_star=0.005, end=16):
    pi_dev = np.float64(result.pi[:end]) - np.float64(pi_star)
    y_dev = np.float64(result.y[:end])
    loss = np.sum(pi_dev**2, dtype=np.float64) + 0.25 * np.sum(y_dev**2, dtype=np.float64)
    return loss * np.float64(10000)

if __name__ == "__main__":
    print("Running simulations for Figure 12 (Policy Asymmetry)...")
    
    theta_levels_plot = [0.1, 0.3, 0.5, 0.7]
    theta_levels_full = np.linspace(0.05, 0.95, 19)
    phi_levels = np.linspace(0.5, 5.0, 46)
    
    params = get_default_params()
    params['gamma'] = 25000.0
    T = 16
    
    shock_path = np.zeros(T)
    shock_path[0:4] = 0.004
    rho_u = 0.8
    
    loss_matrix = np.zeros((len(theta_levels_plot), len(phi_levels)))
    optimal_phis_full = []
    
    # Panel A data
    for i, theta in enumerate(theta_levels_plot):
        for j, phi in enumerate(phi_levels):
            p = override_params(params, {'phi_pi': phi})
            m = FullModel(p)
            r = m.simulate(T, shock_path=shock_path, rho_u=rho_u, initial_theta=theta)
            loss_matrix[i, j] = compute_loss(r, pi_star=params["pi_star"])
            
    # Panel B data
    for theta in theta_levels_full:
        losses = []
        for phi in phi_levels:
            p = override_params(params, {'phi_pi': phi})
            m = FullModel(p)
            r = m.simulate(T, shock_path=shock_path, rho_u=rho_u, initial_theta=theta)
            losses.append(compute_loss(r, pi_star=params["pi_star"]))
        optimal_phis_full.append(phi_levels[np.argmin(losses)])
        
    results = {
        'loss_matrix': loss_matrix,
        'optimal_phis': optimal_phis_full,
        'theta_plot': theta_levels_plot,
        'theta_full': theta_levels_full,
        'phi_levels': phi_levels
    }
    
    output_dir = PROJECT_ROOT / 'output' / 'simulations' / 'section_3'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'policy_asymmetry.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
        
    print(f"Saved results to {output_path}")
