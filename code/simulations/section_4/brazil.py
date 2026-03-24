"""
Simulation: Brazil - Counterfactual vs Real Plan
=================================================
Runs three phases:
1. Counterfactual: active policy from hyperinflation without URV
2. URV phase: expectations re-anchor via ThreeArmMABLearning
3. Real Plan: NK simulation from post-URV anchored state

Writes: output/simulations/section_4/brazil.pkl
"""

import sys
import pickle
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'models'))

from three_arm_full_model import ThreeArmFullModel
from three_arm_mab_learning import ThreeArmMABLearning
from parameters import get_default_params

# Pre-URV snapshot (Brazil 1993-94)
PRE_URV_SNAPSHOT = {
    'monthly_inflation': 0.50,
    'quarterly_inflation': (1.50 ** 3) - 1,  # ~237.5%
    'theta_cb': 0.05,
    'theta_bl': 0.15,
    'theta_tf': 0.80,
}


def get_brazil_params():
    params = get_default_params()
    params['phi_pi'] = 1.5
    params['lambda_fire'] = 0.0
    params['eta'] = 0.10
    params['kappa'] = 0.024
    params['epsilon_bl'] = 1e-4
    params['epsilon_tf'] = 2e-4
    return params


def run_counterfactual(T=80):
    """Counterfactual: active policy directly from hyperinflation."""
    params = get_brazil_params()
    model = ThreeArmFullModel(params)
    shocks = np.zeros(T)
    shocks[0:4] = 0.02
    result = model.simulate(
        T, shock_path=shocks, rho_u=0.8,
        initial_theta_cb=PRE_URV_SNAPSHOT['theta_cb'],
        initial_theta_bl=PRE_URV_SNAPSHOT['theta_bl'],
        initial_pi=PRE_URV_SNAPSHOT['quarterly_inflation'],
        verbose=False
    )
    return result


def run_urv_phase(T_urv=4):
    """URV phase: expectations re-anchor while Cruzeiro inflation continues."""
    mab = ThreeArmMABLearning(
        k=3, eta=0.10, pi_star=0.005,
        epsilon_cb=0.0, epsilon_bl=1e-4, epsilon_tf=2e-4
    )
    theta_cb = PRE_URV_SNAPSHOT['theta_cb']
    theta_bl = PRE_URV_SNAPSHOT['theta_bl']

    theta_cb_path = [theta_cb]
    theta_bl_path = [theta_bl]
    theta_tf_path = [1 - theta_cb - theta_bl]

    # Pre-fill with hyperinflation history
    for pi in [2.0, 2.2, 2.4, 2.6, 2.8]:
        mab.add_observation(pi)

    # URV period: inflation held constant at target
    for t in range(T_urv):
        mab.add_observation(0.005)
        theta_cb, theta_bl, theta_tf = mab.update_theta(theta_cb, theta_bl)
        theta_cb_path.append(theta_cb)
        theta_bl_path.append(theta_bl)
        theta_tf_path.append(theta_tf)

    return {
        'theta_cb': np.array(theta_cb_path),
        'theta_bl': np.array(theta_bl_path),
        'theta_tf': np.array(theta_tf_path),
        'final_theta_cb': theta_cb_path[-1],
        'final_theta_bl': theta_bl_path[-1],
    }


def run_real_plan(urv_result, T=80):
    """Real Plan: NK simulation from post-URV anchored state."""
    params = get_brazil_params()
    model = ThreeArmFullModel(params)
    shocks = np.zeros(T)
    shocks[0:4] = 0.02
    result = model.simulate(
        T, shock_path=shocks, rho_u=0.8,
        initial_theta_cb=urv_result['final_theta_cb'],
        initial_theta_bl=urv_result['final_theta_bl'],
        verbose=False
    )
    return result


if __name__ == '__main__':
    print("Running Brazil simulations...")
    cf_result = run_counterfactual(T=80)
    urv = run_urv_phase(T_urv=4)
    real_result = run_real_plan(urv, T=80)

    data = {
        'cf_pi': cf_result.pi,
        'cf_theta_cb': cf_result.theta_cb,
        'real_pi': real_result.pi,
        'real_theta_cb': real_result.theta_cb,
        'urv_theta_cb': urv['theta_cb'],
        'urv_theta_bl': urv['theta_bl'],
        'urv_theta_tf': urv['theta_tf'],
    }

    out = PROJECT_ROOT / 'output' / 'simulations' / 'section_4' / 'brazil.pkl'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved to {out}")
