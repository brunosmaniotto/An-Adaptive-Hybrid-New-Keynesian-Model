"""
Simulation: Table 5 - The Credibility Buffer (Riding the Wave)
=============================================================

Paper Section: 3.1
Description: Compares Passive (phi_pi=1.1) vs Active (phi_pi=2.0) policy
under different initial credibility levels.
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

def compute_welfare(pi, y, pi_star, lambda_y=0.25):
    """Compute sum of squared deviations."""
    return np.sum((pi - pi_star)**2) + lambda_y * np.sum(y**2)

def run_scenario(theta_0, phi_pi):
    params = override_params(get_default_params(), {
        'phi_pi': phi_pi,
        'eta': 0.10,
        'k': 3,
        'epsilon': 1e-4,
    })
    model = FullModel(params)
    
    T = 16  # 4-year horizon as mentioned in manuscript
    pi_shock = 0.0125  # 5% annualized
    rho_u = 0.75
    
    shock_path = np.zeros(T)
    shock_path[0] = pi_shock
    
    result = model.simulate(T, shock_path=shock_path, rho_u=rho_u, initial_theta=theta_0)
    
    welfare = compute_welfare(result.pi, result.y, params['pi_star'])
    peak_pi = np.max(result.pi) * 400 # Annualized %
    
    return {
        'peak_pi': peak_pi,
        'welfare': welfare,
        'final_theta': result.theta[-1]
    }

if __name__ == "__main__":
    print("Running simulations for Table 5 (Credibility Buffer)...")
    
    scenarios = [
        {'label': 'High θ, Passive', 'theta': 0.95, 'phi': 1.1},
        {'label': 'High θ, Active',  'theta': 0.95, 'phi': 2.0},
        {'label': 'Low θ, Passive',  'theta': 0.15, 'phi': 1.1},
        {'label': 'Low θ, Active',   'theta': 0.15, 'phi': 2.0}
    ]
    
    results = {}
    for s in scenarios:
        results[s['label']] = run_scenario(s['theta'], s['phi'])
        
    output_dir = PROJECT_ROOT / 'output' / 'simulations' / 'section_3'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'table_05_data.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
        
    print(f"Saved results to {output_path}")
