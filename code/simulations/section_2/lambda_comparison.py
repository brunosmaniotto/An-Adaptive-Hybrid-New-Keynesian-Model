"""
Simulation: Lambda Comparison (Figure 3)
========================================

Simulates model under different FIRE fractions (lambda).
"""

import sys
import pickle
import numpy as np
from pathlib import Path

# Project imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'models'))

from full_model import FullModel
from parameters import get_default_params, get_figure3_config, override_params

def run_scenario(scenario_config: dict, lambda_values: list, eta: float,
                 T: int, shock_period: int, initial_theta: float = 1.0,
                 phi_pi: float = 1.5) -> dict:
    
    shock_size = scenario_config['shock_size']
    rho_u = scenario_config['rho_u']
    duration = scenario_config.get('duration', 1)

    shock_path = np.zeros(T)
    for t in range(shock_period, min(shock_period + duration, T)):
        shock_path[t] = shock_size

    results = {}
    for lam in lambda_values:
        params = override_params(get_default_params(),
            {'lambda_fire': lam, 'eta': eta, 'phi_pi': phi_pi})
        model = FullModel(params)
        
        # Use MIT shock (max_iter=-1)
        res = model.simulate(T=T, shock_path=shock_path, rho_u=rho_u,
                             initial_theta=initial_theta, max_iter=-1)
        
        # Store essential data (arrays)
        results[f'lambda_{lam}'] = {
            'pi': res.pi,
            'theta': res.theta,
            'y': res.y,
            'i': res.i
        }
    return results

if __name__ == "__main__":
    print("Running simulations for Figure 3 (Lambda Comparison)...")
    
    config = get_figure3_config()
    config['initial_theta'] = 1.0
    
    all_results = {}
    for name, scen_config in config['scenarios'].items():
        print(f"  Scenario: {name}")
        all_results[name] = run_scenario(
            scenario_config=scen_config,
            lambda_values=config['lambda_values'],
            eta=config['eta'],
            T=config['T'],
            shock_period=config['shock_period'],
            initial_theta=config['initial_theta'],
            phi_pi=config.get('phi_pi', 1.5)
        )
        
    # Add config for plotting
    all_results['config'] = config
    
    output_dir = PROJECT_ROOT / 'output' / 'simulations' / 'section_2'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'lambda_comparison.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(all_results, f)
        
    print(f"Saved results to {output_path}")
