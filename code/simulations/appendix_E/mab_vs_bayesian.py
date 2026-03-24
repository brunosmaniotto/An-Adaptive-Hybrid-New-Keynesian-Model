"""
Simulation: MAB vs Bayesian (Figure E1)
=======================================

Comparison of learning mechanisms.
"""

import sys
import pickle
import numpy as np
from pathlib import Path

# Project imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'models'))

from toy_model import ToyModel
from bayesian_learning import BayesianToyModel, calibrate_initial_prior
from parameters import get_default_params, override_params

def run_comparison(config):
    T = config['T']
    shock_path = np.zeros(T)
    shock_path[config['shock_period']] = config['shock_size']
    rho_u = config['rho_u']
    
    # MAB
    mab_params = override_params(get_default_params(), {'eta': config['eta']})
    mab_model = ToyModel(mab_params)
    mab_res = mab_model.simulate(T, shock_path=shock_path, rho_u=rho_u, initial_theta=config['initial_theta'])
    
    # Bayesian
    a0, b0 = calibrate_initial_prior(config['initial_theta'], config['bayesian_confidence'])
    bayes_params = get_default_params()
    bayes_params['alpha_0'] = a0; bayes_params['beta_0'] = b0
    bayes_model = BayesianToyModel(bayes_params)
    bayes_res = bayes_model.simulate(T, shock_path=shock_path, rho_u=rho_u, initial_weight=config['initial_theta'])
    
    return {
        'mab': {'pi': mab_res.pi, 'theta': mab_res.theta, 'y': mab_res.y},
        'bayes': {'pi': bayes_res['pi'], 'weight': bayes_res['weight'], 'y': bayes_res['y']}
    }

if __name__ == "__main__":
    print("Running simulations for Figure E1 (MAB vs Bayesian)...")
    
    base_config = {
        'T': 80, 'shock_period': 5, 'shock_size': 0.02, 'rho_u': 0.6,
        'eta': 0.10, 'bayesian_confidence': 10.0
    }
    
    scenarios = {
        'high_credibility': {'initial_theta': 0.9},
        'medium_credibility': {'initial_theta': 0.5},
        'low_credibility': {'initial_theta': 0.1}
    }
    
    results = {}
    for name, cfg in scenarios.items():
        c = base_config.copy()
        c.update(cfg)
        results[name] = run_comparison(c)
        results[name]['config'] = c
        
    output_dir = PROJECT_ROOT / 'output' / 'simulations' / 'appendix_E'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'mab_vs_bayesian.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
        
    print(f"Saved results to {output_path}")
