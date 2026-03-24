"""
Simulation: Phillips Curve Estimation on Model-Generated Data
=============================================================
Runs simulations under 3 credibility regimes (high/medium/low) with demand shocks.
Estimates standard PC, hybrid PC, and AR(1) persistence for each regime.

Reads: parameters, FullModel
Writes: output/simulations/section_3/phillips_curve.pkl
"""

import sys
import pickle
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'models'))

from full_model import FullModel
from parameters import get_default_params, override_params


def estimate_standard_pc(pi, y):
    """Estimate pi_t = c + kappa * y_t + e_t. Returns (kappa, se, r2)."""
    X = np.column_stack([np.ones(len(y)), y])
    beta = np.linalg.lstsq(X, pi, rcond=None)[0]
    kappa_hat = beta[1]
    y_pred = X @ beta
    ss_res = np.sum((pi - y_pred) ** 2)
    ss_tot = np.sum((pi - np.mean(pi)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    n = len(pi)
    mse = ss_res / (n - 2)
    se_kappa = np.sqrt(mse * np.linalg.inv(X.T @ X)[1, 1])
    return kappa_hat, se_kappa, r2


def estimate_hybrid_pc(pi, y):
    """Estimate pi_t = c + gamma_b * pi_{t-1} + kappa * y_t + e_t. Returns (gamma_b, kappa, r2)."""
    pi_lag = pi[:-1]
    pi_curr = pi[1:]
    y_curr = y[1:]
    X = np.column_stack([np.ones(len(pi_lag)), pi_lag, y_curr])
    beta = np.linalg.lstsq(X, pi_curr, rcond=None)[0]
    gamma_b = beta[1]
    kappa_hat = beta[2]
    y_pred = X @ beta
    ss_res = np.sum((pi_curr - y_pred) ** 2)
    ss_tot = np.sum((pi_curr - np.mean(pi_curr)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return gamma_b, kappa_hat, r2


def estimate_inflation_persistence(pi):
    """Estimate AR(1): pi_t = c + rho * pi_{t-1} + e_t. Returns rho."""
    pi_lag = pi[:-1]
    pi_curr = pi[1:]
    X = np.column_stack([np.ones(len(pi_lag)), pi_lag])
    beta = np.linalg.lstsq(X, pi_curr, rcond=None)[0]
    return beta[1]


if __name__ == '__main__':
    np.random.seed(42)
    T = 200
    shock_std = 0.002

    # Generate demand shocks (natural rate shocks)
    innovations = np.random.normal(0, shock_std * 5, T)
    rn_shocks = np.zeros(T)
    for t in range(1, T):
        rn_shocks[t] = 0.3 * rn_shocks[t - 1] + innovations[t]

    # eta=0.001 (near-zero) holds theta approximately constant at its initial value,
    # allowing us to estimate PC coefficients under fixed credibility regimes.
    # Setting eta=0 is rejected by parameter validation; 0.001 is functionally equivalent.
    params = override_params(get_default_params(), {'eta': 0.001})
    model = FullModel(params)
    true_kappa = params['kappa']

    # Run 3 credibility regimes
    regimes = {'high': 0.95, 'medium': 0.5, 'low': 0.2}
    estimates = {}

    for name, theta0 in regimes.items():
        res = model.simulate(T=T, rn_path=rn_shocks.copy(), rho_u=0.0, initial_theta=theta0)

        kappa_std, se_std, r2_std = estimate_standard_pc(res.pi, res.y)
        gamma_b, kappa_hyb, r2_hyb = estimate_hybrid_pc(res.pi, res.y)
        rho_ar1 = estimate_inflation_persistence(res.pi)

        estimates[name] = {
            'kappa_standard': kappa_std,
            'se_standard': se_std,
            'r2_standard': r2_std,
            'gamma_b': gamma_b,
            'kappa_hybrid': kappa_hyb,
            'r2_hybrid': r2_hyb,
            'rho_ar1': rho_ar1,
        }
        print(f"  {name:6s}: kappa_std={kappa_std:+.4f}, rho_AR1={rho_ar1:.3f}, gamma_b={gamma_b:.3f}")

    data = {'estimates': estimates, 'true_kappa': true_kappa}
    out = PROJECT_ROOT / 'output' / 'simulations' / 'section_3' / 'phillips_curve.pkl'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved to {out}")
