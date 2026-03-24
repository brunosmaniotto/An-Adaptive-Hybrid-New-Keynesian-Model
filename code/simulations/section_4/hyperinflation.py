"""
Simulation: Hyperinflation and the Three-Arm Model
===================================================
Runs 3 policy stances (phi_pi = 0.5, 1.0, 1.5) to compare
hyperinflation dynamics under different monetary regimes.

Also saves detailed results for the gear-shift stackplot (phi=0.5).

Uses baseline parameters throughout:
- lambda_fire=0.35, eta=0.10, kappa=0.024
- epsilon_bl=1e-4, epsilon_tf=2e-4
- Shock: shock[5]=0.08, T=60, rho_u=0.90

Writes: output/simulations/section_4/hyperinflation.pkl
"""

import sys
import pickle
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'models'))

from three_arm_full_model import ThreeArmFullModel
from parameters import get_default_params


def get_hyperinflation_params(phi_pi: float) -> dict:
    """Return parameters for a hyperinflation experiment at given phi_pi."""
    params = get_default_params()
    params['phi_pi'] = phi_pi
    params['epsilon_bl'] = 1e-4   # Baseline complexity cost
    params['epsilon_tf'] = 2e-4   # Higher cost for trend-following
    return params


def run_single_experiment(phi_pi: float, T: int = 60,
                          shock_size: float = 0.08, rho_u: float = 0.90):
    """Run a single hyperinflation experiment and return the result object."""
    params = get_hyperinflation_params(phi_pi)
    model = ThreeArmFullModel(params)
    shocks = np.zeros(T)
    shocks[5] = shock_size
    result = model.simulate(T, shock_path=shocks, rho_u=rho_u)
    return result


if __name__ == '__main__':
    print("Running hyperinflation simulations...")

    phi_values = [0.5, 1.0, 1.5]
    T = 60

    data = {}

    for phi in phi_values:
        print(f"  phi_pi = {phi:.1f} ...")
        result = run_single_experiment(phi, T=T)
        data[phi] = {
            'pi': result.pi,
            'theta_cb': result.theta_cb,
            'theta_bl': result.theta_bl,
            'theta_tf': result.theta_tf,
        }
        peak = np.max(result.pi) * 400
        max_tf = np.max(result.theta_tf)
        print(f"    Peak inflation = {peak:.1f}%, Max TF share = {max_tf:.2f}")

    # Run separate gear-shift illustration with T=50 for phi_pi=0.5
    print("  Gear shift illustration (phi_pi=0.5, T=50) ...")
    gs_result = run_single_experiment(0.5, T=50)
    delta_pi = np.diff(gs_result.pi)
    data['gear_shift'] = {
        'pi': gs_result.pi,
        'theta_cb': gs_result.theta_cb,
        'theta_bl': gs_result.theta_bl,
        'theta_tf': gs_result.theta_tf,
        'delta_pi': delta_pi,
    }
    data['gear_shift_phi'] = 0.5

    out = PROJECT_ROOT / 'output' / 'simulations' / 'section_4' / 'hyperinflation.pkl'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved to {out}")
