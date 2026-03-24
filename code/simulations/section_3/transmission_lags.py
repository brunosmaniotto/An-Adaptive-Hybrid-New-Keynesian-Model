"""
Simulation: Transmission Lags by Initial Credibility
=====================================================
Runs identical cost-push shock under different initial credibility levels.
Computes convergence lags for annotations.

Writes: output/simulations/section_3/transmission_lags.pkl
"""

import sys
import pickle
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'models'))

from full_model import FullModel
from parameters import get_default_params, override_params


def compute_transmission_lag(pi_path, pi_star, threshold, shock_period):
    """Quarters until inflation returns to within threshold of target (sustained)."""
    T = len(pi_path)
    for t in range(shock_period + 1, T - 4):
        if abs(pi_path[t] - pi_star) < threshold:
            stays_within = True
            for s in range(t, min(t + 4, T)):
                if abs(pi_path[s] - pi_star) >= threshold:
                    stays_within = False
                    break
            if stays_within:
                return t - shock_period
    return -1


if __name__ == '__main__':
    T = 100
    shock_start = 4
    shock_size = 0.0075
    rho_u = 0.75
    threshold = 0.001

    shock_path = np.zeros(T)
    shock_path[shock_start] = shock_size

    base_params = get_default_params()
    pi_star = base_params['pi_star']

    theta_values = [0.2, 0.5, 0.8]
    results = {}
    lags = {}

    for t0 in theta_values:
        params = override_params(base_params, {'eta': 0.10, 'phi_pi': 1.5})
        model = FullModel(params)
        res = model.simulate(T=T, shock_path=shock_path, rho_u=rho_u, initial_theta=t0)
        results[t0] = res.pi
        conv = compute_transmission_lag(res.pi, pi_star, threshold, shock_start)
        peak = np.max(res.pi[shock_start:shock_start + 20]) * 400
        lags[t0] = {'convergence': conv, 'peak': peak}
        print(f"  theta_0={t0}: convergence={conv}Q, peak={peak:.1f}%")

    data = {
        'results': results,
        'lags': lags,
        'pi_star': pi_star,
        'shock_start': shock_start,
    }
    out = PROJECT_ROOT / 'output' / 'simulations' / 'section_3' / 'transmission_lags.pkl'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved to {out}")
