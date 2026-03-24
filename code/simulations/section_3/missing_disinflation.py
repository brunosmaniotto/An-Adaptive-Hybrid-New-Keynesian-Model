import sys, pickle, numpy as np
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'models'))
from full_model import FullModel
from parameters import get_default_params

if __name__ == '__main__':
    T = 60; shock = np.zeros(T); shock[4] = -0.005
    params = get_default_params()
    params['lambda_fire'] = 0.35
    model_ad = FullModel(params)
    res_ad = model_ad.simulate(T=T, shock_path=shock, rho_u=0.7, initial_theta=0.95)
    params_nk = params.copy(); params_nk['lambda_fire'] = 1.0
    model_nk = FullModel(params_nk)
    res_nk = model_nk.simulate(T=T, shock_path=shock, rho_u=0.7)
    results = {'ad_pi': res_ad.pi, 'nk_pi': res_nk.pi, 'pi_star': params['pi_star']}
    out = PROJECT_ROOT / 'output/simulations/section_3/missing_disinflation.pkl'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'wb') as f: pickle.dump(results, f)
