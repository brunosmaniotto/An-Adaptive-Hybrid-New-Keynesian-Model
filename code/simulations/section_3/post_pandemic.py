import sys, pickle, numpy as np
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'models'))
from full_model import FullModel
from parameters import get_default_params

if __name__ == '__main__':
    T = 80; shock = np.zeros(T); shock[4]=0.007; shock[6]=0.005; shock[8]=0.006
    params = get_default_params(); params['phi_pi']=1.1
    model_ad = FullModel(params)
    res_ad = model_ad.simulate(T=T, shock_path=shock, rho_u=0.84, initial_theta=0.60)
    params_nk = params.copy(); params_nk['lambda_fire']=1.0
    model_nk = FullModel(params_nk)
    res_nk = model_nk.simulate(T=T, shock_path=shock, rho_u=0.84)
    results = {'ad_pi': res_ad.pi, 'nk_pi': res_nk.pi, 'theta': res_ad.theta, 'pi_star': params['pi_star']}
    out = PROJECT_ROOT / 'output/simulations/section_3/post_pandemic.pkl'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'wb') as f: pickle.dump(results, f)
