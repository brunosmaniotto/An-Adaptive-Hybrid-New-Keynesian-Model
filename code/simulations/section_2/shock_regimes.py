"""
Simulation: Shock Regimes (Figure 4)
====================================

Simulates model under different shock regimes:
1. Great Moderation
2. 1970s-style (High Volatility)
3. Regime Switch (Credibility Cycle)
"""

import sys
import pickle
import numpy as np
from pathlib import Path

# Project imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'models'))

from full_model import FullModel
from parameters import get_default_params

def generate_asymmetric_shocks(n: int, mean: float, std: float, skew_factor: float = 2.0) -> np.ndarray:
    sigma_ln = np.sqrt(np.log(1 + (std / (mean + std)) ** 2 * skew_factor))
    mu_ln = np.log(mean + std) - sigma_ln ** 2 / 2
    raw = np.exp(np.random.normal(mu_ln, sigma_ln, n))
    shocks = raw - np.exp(mu_ln + sigma_ln ** 2 / 2) + mean
    return shocks

def generate_shock_sequence(T: int, regime: str, seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    shocks = np.zeros(T)

    if regime == 'great_moderation':
        shocks = generate_asymmetric_shocks(n=T, mean=0.0003, std=0.0006, skew_factor=1.5)
    elif regime == 'high_volatility':
        shocks = generate_asymmetric_shocks(n=T, mean=0.003, std=0.002, skew_factor=2.5)
        for t in [10, 20, 32, 45]:
            if t < T: shocks[t] += 0.008
    elif regime == 'regime_switch':
        T_start, T_end = T // 6, T // 2
        shocks[:T_start] = generate_asymmetric_shocks(n=T_start, mean=0.0003, std=0.0006, skew_factor=1.5)
        shocks[T_start:T_end] = generate_asymmetric_shocks(n=T_end-T_start, mean=0.005, std=0.004, skew_factor=3.0)
        shocks[T_start] += 0.008
        shocks[T_end:] = generate_asymmetric_shocks(n=T-T_end, mean=0.0003, std=0.0006, skew_factor=1.5)
        
    return shocks

def simulate_regime(T: int, regime: str, params: dict) -> dict:
    shocks = generate_shock_sequence(T, regime)
    
    if regime == 'regime_switch': init_theta = 1.0; rho_u = 0.95
    elif regime == 'great_moderation': init_theta = 0.5; rho_u = 0.7
    else: init_theta = 0.9; rho_u = 0.94
    
    model = FullModel(params)
    res = model.simulate(T=T, shock_path=shocks, rho_u=rho_u, initial_theta=init_theta)
    
    return {'pi': res.pi, 'theta': res.theta, 'shocks': shocks}

if __name__ == "__main__":
    print("Running simulations for Figure 4 (Shock Regimes)...")
    
    params = get_default_params()
    # Baseline calibration overrides
    params['lambda_fire'] = 0.35
    params['eta'] = 0.10
    params['kappa'] = 0.024
    params['epsilon'] = 1e-4
    
    T = 120
    regimes = ['great_moderation', 'high_volatility', 'regime_switch']
    
    all_results = {}
    for r in regimes:
        print(f"  Simulating: {r}")
        all_results[r] = simulate_regime(T, r, params)
        
    output_dir = PROJECT_ROOT / 'output' / 'simulations' / 'section_2'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'shock_regimes.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(all_results, f)
        
    print(f"Saved results to {output_path}")
