"""
Simulation: Reanchoring Problem (Figure B1)
============================================
Demonstrates why epsilon is essential for re-anchoring.
Uses ToyModel with T=120, shock at t=5.

Writes: output/simulations/appendix_B/reanchoring.pkl
"""

import sys
import pickle
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'models'))

from toy_model import ToyModel
from parameters import get_default_params, override_params


def run_simulation(epsilon, eta=0.10, T=120, shock_size=0.02, rho_u=0.8, shock_period=5):
    """Run simulation with specified epsilon value."""
    params = override_params(get_default_params(), {
        'eta': eta,
        'epsilon': epsilon,
    })
    model = ToyModel(params)
    shock_path = np.zeros(T)
    shock_path[shock_period] = shock_size
    result = model.simulate(T=T, shock_path=shock_path, rho_u=rho_u, initial_theta=1.0)
    return {
        'pi': result.pi,
        'theta': result.theta,
        'pi_star': params['pi_star'],
    }


if __name__ == "__main__":
    print("Running simulations for Figure B1 (Reanchoring)...")
    results = {
        'epsilon_0': run_simulation(0.0),
        'epsilon_pos': run_simulation(1e-4),
    }
    output_dir = PROJECT_ROOT / 'output' / 'simulations' / 'appendix_B'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'reanchoring.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved results to {output_path}")
