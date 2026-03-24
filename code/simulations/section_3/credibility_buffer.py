"""
Simulation: Credibility Buffer (Figure 13)
==========================================

Immediate vs Delayed policy response.
"""

import sys
import pickle
import numpy as np
from pathlib import Path

# Project imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'models'))

from full_model import FullModel
from mab_learning import MABLearning
from parameters import get_default_params

def run_scenario(theta_0, delay):
    params = get_default_params()
    params['kappa'] = 0.024
    model = FullModel(params)
    mab = MABLearning(k=params['k'], eta=params['eta'], epsilon=params['epsilon'])
    
    T = 60
    pi, y, i, theta = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)
    
    pi[0] = params['pi_star']
    theta[0] = theta_0
    i[0] = params['rn_bar'] + params['pi_star']
    mab.reset(); mab.add_observation(pi[0])
    
    rho_u = 0.95
    u = np.zeros(T)
    u[0] = 0.005
    for t in range(1, T): u[t] = rho_u * u[t-1]
    rn = np.full(T, params['rn_bar'])
    
    for t in range(1, T):
        if t <= delay:
            model.phi_pi = 0.0; model.phi_y = 0.0; v_t = 0.0
        else:
            model.phi_pi = params['phi_pi']; model.phi_y = params['phi_y']; v_t = 0.0
            
        E_pi_br = theta[t-1] * model.pi_star + (1 - theta[t-1]) * pi[t-1]
        E_y_br = 0.0
        
        sp = {'rho_u': rho_u, 'rho_r': 0.0, 'rho_v': 0.0}
        E_pi_fire, E_y_fire = model.fire_solver.solve_expectations(u[t], rn[t], v_t, sp)
        
        E_pi = model.lam * E_pi_fire + (1 - model.lam) * E_pi_br
        E_y = model.lam * E_y_fire + (1 - model.lam) * E_y_br
        
        pi[t], y[t] = model._solve_nk_system(E_pi, E_y, u[t], rn[t], v_t)
        
        if t <= delay:
            i[t] = params['rn_bar'] + params['pi_star']
        else:
            i[t] = (params['rn_bar'] + params['pi_star'] + 
                    params['phi_pi'] * (pi[t] - params['pi_star']) + 
                    params['phi_y'] * y[t])
                    
        mab.add_observation(pi[t])
        theta[t] = mab.update_theta(theta[t-1])
        
    return {'pi': pi, 'y': y}

if __name__ == "__main__":
    print("Running simulations for Figure 13 (Credibility Buffer)...")
    
    scenarios = [
        {'label': 'High Cred (Immediate)', 'theta': 0.95, 'delay': 0},
        {'label': 'High Cred (Delayed)', 'theta': 0.95, 'delay': 8},
        {'label': 'Low Cred (Immediate)', 'theta': 0.50, 'delay': 0},
        {'label': 'Low Cred (Delayed)', 'theta': 0.50, 'delay': 8}
    ]
    
    results = {}
    for s in scenarios:
        results[s['label']] = run_scenario(s['theta'], s['delay'])
        
    output_dir = PROJECT_ROOT / 'output' / 'simulations' / 'section_3'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'credibility_buffer.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
        
    print(f"Saved results to {output_path}")
