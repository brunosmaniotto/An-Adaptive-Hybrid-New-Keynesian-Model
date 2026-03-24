"""
Table 2: Transitory vs Persistent Shocks
========================================

Shows that for transitory shocks (ρ=0), adaptive = FIRE.
Only persistent shocks trigger de-anchoring.

Note: Uses ToyModel (λ=0, pure adaptive) intentionally — the table compares
the pure adaptive mechanism against a FIRE benchmark to isolate the learning
dynamics, before λ is introduced in the calibration discussion.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Project imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'models'))

from toy_model import ToyModel, FIREBenchmark
from parameters import get_default_params, override_params

def annualize(rate): return rate * 4 * 100

def generate_table():
    print("Generating Table 2 (Transitory vs Persistent)...")
    
    T = 60
    shock_period = 5
    shock_size = 0.01125
    eta = 0.10
    phi_pi = 1.5
    rho_values = [0.0, 0.4, 0.85]
    
    results = []
    
    for rho in rho_values:
        shock = np.zeros(T)
        shock[shock_period] = shock_size
        
        # FIRE
        fire_params = get_default_params()
        fire_params['phi_pi'] = phi_pi
        fire = FIREBenchmark(fire_params)
        res_fire = fire.simulate(T=T, shock_path=shock, rho_u=rho)
        peak_fire = np.max(res_fire.pi)
        
        # Adaptive
        params = get_default_params()
        params['eta'] = eta
        params['phi_pi'] = phi_pi
        params['k'] = 3
        params['epsilon'] = 1e-4
        
        model = ToyModel(params)
        res_ad = model.simulate(T=T, shock_path=shock, rho_u=rho, initial_theta=1.0)
        peak_ad = np.max(res_ad.pi)
        min_theta = np.min(res_ad.theta)
        
        diff = peak_ad - peak_fire
        
        results.append({
            'rho': rho,
            'peak_fire': annualize(peak_fire - params['pi_star']),
            'peak_adaptive': annualize(peak_ad - params['pi_star']),
            'diff': annualize(diff),
            'min_theta': min_theta
        })
        
    df = pd.DataFrame(results)
    
    # Save
    output_dir = PROJECT_ROOT / 'output' / 'tables' / 'section_2'
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / 'table_02_transitory_persistent.csv', index=False)
    print(f"Saved to {output_dir / 'table_02_transitory_persistent.csv'}")
    
    # Print LaTeX
    print("\n% LaTeX Table")
    print(df.to_latex(index=False, float_format="%.3f"))

if __name__ == "__main__":
    generate_table()
