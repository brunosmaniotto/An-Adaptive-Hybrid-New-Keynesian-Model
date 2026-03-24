"""
Simulation: Long Memory and Inflation Trauma
=============================================
Runs 20-year histories through LongMemoryMABLearning to establish different
credibility states, then hits all countries with identical new shock.

3 scenarios: stable, distant_trauma, chronic_current

Writes: output/simulations/section_4/long_memory.pkl
"""

import sys
import pickle
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'models'))

from long_memory_learning import LongMemoryMABLearning
from full_model import FullModel
from parameters import get_default_params, override_params

# Calibrated inflation rates (quarterly)
PI_HYPER = 2.375      # 50% monthly -> 237.5% quarterly
PI_MODERATE = 0.018   # 7.5% annual -> 1.8% quarterly
PI_STABLE = 0.005     # 2% annual -> 0.5% quarterly

# Durations (in quarters)
T_HISTORY = 80        # 20 years of history
HYPER_DURATION = 20   # 5 years
MODERATE_DURATION = 40  # 10 years

# New shock parameters
T_SHOCK = 40
SHOCK_SIZE = 0.015
SHOCK_DURATION = 4


def run_history(inflation_history, delta=0.8):
    """Run a country through its history, return final theta."""
    mab = LongMemoryMABLearning(
        delta=delta, eta=0.10, epsilon=1e-4, pi_star=PI_STABLE
    )
    theta = 0.9
    for pi in inflation_history:
        mab.add_observation(pi)
        theta = mab.update_theta(theta)
    return theta


def run_new_shock(theta_initial, params):
    """Run NK simulation with a new shock from given credibility."""
    model = FullModel(params)
    shock_path = np.zeros(T_SHOCK)
    shock_path[:SHOCK_DURATION] = SHOCK_SIZE
    result = model.simulate(T=T_SHOCK, shock_path=shock_path, rho_u=0.8, initial_theta=theta_initial)
    return result.pi, result.theta


if __name__ == '__main__':
    params = override_params(get_default_params(), {
        'phi_pi': 1.5,
        'lambda_fire': 0.0,
        'eta': 0.10,
    })

    # Create inflation histories
    scenarios = {
        'stable': np.full(T_HISTORY, PI_STABLE),
        'distant_trauma': np.concatenate([
            np.full(HYPER_DURATION, PI_HYPER),
            np.full(T_HISTORY - HYPER_DURATION, PI_STABLE)
        ]),
        'chronic_current': np.concatenate([
            np.full(T_HISTORY - MODERATE_DURATION, PI_STABLE),
            np.full(MODERATE_DURATION, PI_MODERATE)
        ]),
    }

    scenario_meta = {
        'stable': {'name': 'Stable', 'color': 'black', 'linestyle': '-'},
        'distant_trauma': {'name': 'Distant trauma', 'color': '#666666', 'linestyle': '--'},
        'chronic_current': {'name': 'Chronic (current)', 'color': '#999999', 'linestyle': '-.'},
    }

    results = {}
    for key, history in scenarios.items():
        theta_final = run_history(history)
        pi_path, theta_path = run_new_shock(theta_final, params)
        results[key] = {
            'pi_path': pi_path,
            'theta_path': theta_path,
            'theta_final': theta_final,
            'name': scenario_meta[key]['name'],
            'color': scenario_meta[key]['color'],
            'linestyle': scenario_meta[key]['linestyle'],
        }
        print(f"  {scenario_meta[key]['name']:<20}: theta_final={theta_final:.3f}, peak_pi={np.max(pi_path)*400:.1f}%")

    out = PROJECT_ROOT / 'output' / 'simulations' / 'section_4' / 'long_memory.pkl'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved to {out}")
