"""
Table 3: The Paradox of Faster Learning
=======================================

Shows that faster learning (higher η) leads to worse outcomes.

Note: Uses ToyModel (λ=0, pure adaptive) intentionally — the table isolates
the effect of learning speed on the pure adaptive mechanism, before λ is
introduced in the calibration discussion.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Project imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'models'))

from toy_model import ToyModel
from parameters import get_default_params

def annualize(rate): return rate * 4 * 100

def generate_table():
    print("Generating Table 3 (Paradox of Learning)...")
    
    T = 120
    shock_period = 5
    shock_size = 0.02
    rho = 0.85
    phi_pi = 1.5
    eta_values = [0.10, 0.20, 0.30]
    pi_star = 0.005
    tolerance = 0.0005
    
    results = []
    
    for eta in eta_values:
        shock = np.zeros(T)
        shock[shock_period] = shock_size
        
        params = get_default_params()
        params['eta'] = eta
        params['phi_pi'] = phi_pi
        params['k'] = 3
        params['epsilon'] = 1e-4
        
        model = ToyModel(params)
        res = model.simulate(T=T, shock_path=shock, rho_u=rho, initial_theta=1.0)
        
        peak_pi = np.max(res.pi)
        min_theta = np.min(res.theta)
        
        q_target = T
        peak_idx = np.argmax(res.pi)
        for t in range(peak_idx, T):
            if abs(res.pi[t] - pi_star) < tolerance:
                q_target = t - shock_period
                break
                
        results.append({
            'eta': eta,
            'peak_inflation': annualize(peak_pi - pi_star),
            'min_theta': min_theta,
            'quarters_to_target': q_target if q_target < T else f">{T-shock_period}",
            'final_theta': res.theta[-1]
        })
        
    df = pd.DataFrame(results)
    
    # Save
    output_dir = PROJECT_ROOT / 'output' / 'tables' / 'section_2'
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / 'table_03_paradox_learning.csv', index=False)
    print(f"Saved to {output_dir / 'table_03_paradox_learning.csv'}")
    
    # Print LaTeX
    print("\n% LaTeX Table")
    print(df.to_latex(index=False, float_format="%.3f"))

if __name__ == "__main__":
    generate_table()
