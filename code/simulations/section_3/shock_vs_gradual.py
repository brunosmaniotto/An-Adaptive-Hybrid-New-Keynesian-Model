"""
Simulation: Shock Therapy vs Gradualism
========================================

Paper Section: 3.2
Description: Compares three policy strategies for a credibility crisis:
1. Shock therapy: aggressive from day one
2. Gradualism: slow ramp-up
3. Constant moderate: never get aggressive
"""

import sys
import pickle
import numpy as np
from pathlib import Path

# Project imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'models'))

from policy_experiments import (
    simulate_policy_experiment,
    constant_policy,
    gradual_policy,
    compute_welfare,
    find_recovery_time,
)
from parameters import get_default_params, override_params

CONFIG = {
    'T': 80,
    'initial_inflation': 0.035,
    'shock_size': 0.006,
    'rho_shock': 0.8,
    'initial_theta': 0.05,
    'phi_pi_aggressive': 3.5,
    'phi_pi_moderate': 1.5,
    'ramp_duration': 20,
    'eta': 0.10,
    'k': 3,
    'epsilon': 1e-4,
    'gamma': 25000.0,
    'lambda_y': 0.25,
}

def run_experiment(config: dict) -> dict:
    params = override_params(get_default_params(), {
        'eta': config['eta'],
        'k': config['k'],
        'epsilon': config['epsilon'],
        'gamma': config['gamma'],
    })

    T = config['T']
    shock_path = np.zeros(T)
    shock_path[0:4] = config['shock_size']

    strategies = {
        'Shock therapy': constant_policy(config['phi_pi_aggressive']),
        'Gradualism': gradual_policy(
            config['phi_pi_moderate'],
            config['phi_pi_aggressive'],
            config['ramp_duration']
        ),
        'Constant moderate': constant_policy(config['phi_pi_moderate']),
    }

    results = {}
    for name, phi_func in strategies.items():
        result = simulate_policy_experiment(
            params, T, shock_path,
            rho_u=config['rho_shock'],
            initial_theta=config['initial_theta'],
            initial_inflation=config['initial_inflation'],
            phi_pi_func=phi_func,
        )

        welfare = compute_welfare(result, params['pi_star'], config['lambda_y'])
        recovery = find_recovery_time(result.theta, 0.5)

        results[name] = {
            'pi': result.pi,
            'theta': result.theta,
            'phi_pi_path': result.phi_pi_path,
            'welfare': welfare,
            'recovery': recovery,
            'final_theta': result.theta[-1],
        }

    return results

if __name__ == "__main__":
    print("Running Shock Therapy vs Gradualism Simulation...")
    results = run_experiment(CONFIG)
    
    output_dir = PROJECT_ROOT / 'output' / 'simulations' / 'section_3'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'shock_vs_gradual.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
        
    print(f"Saved results to {output_path}")
