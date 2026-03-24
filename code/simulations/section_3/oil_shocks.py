import sys, pickle, numpy as np
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'models'))
from full_model import FullModel
from parameters import get_default_params, override_params

def run_simulation(theta_0, params, T=25):
    model = FullModel(params)
    shock_path = np.zeros(T); shock_path[0] = 0.01
    res = model.simulate(T=T, shock_path=shock_path, rho_u=0.7, initial_theta=theta_0)
    return {'pi': res.pi, 'theta': res.theta}

if __name__ == '__main__':
    base_params = get_default_params()
    params = override_params(base_params, {'phi_pi': 1.5, 'lambda_fire': 0.35, 'epsilon': 1e-4})
    results = {t: run_simulation(t, params) for t in [0.2, 1.0]}
    out = PROJECT_ROOT / 'output/simulations/section_3/oil_shocks.pkl'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'wb') as f: pickle.dump(results, f)
