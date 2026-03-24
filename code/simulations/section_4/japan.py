"""
Simulation: Japan - Below-Target Anchoring
==========================================
Runs 3 scenarios demonstrating symmetric de-anchoring:
1. Japan (0% inflation) - full de-anchoring below target
2. Stable (2% inflation at target) - credibility maintained
3. Mild undershoot (1% inflation) - partial de-anchoring

Uses LongMemoryMABLearning with delta=0.8, eta=0.10, epsilon=1e-4,
pi_star=0.005.  All start with theta_0=0.9, run 160 quarters (40 years).

For the Japan scenario, does a 40Q pre-fill, then tracks loss_diff
for 120 subsequent quarters (matching Current implementation).

Writes: output/simulations/section_4/japan.pkl
"""

import sys
import pickle
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'models'))

from long_memory_learning import LongMemoryMABLearning


def run_scenario(inflation_rate: float, T: int = 160, theta_0: float = 0.9):
    """Run a single scenario: track theta for T quarters."""
    mab = LongMemoryMABLearning(
        delta=0.8, eta=0.10, epsilon=1e-4, pi_star=0.005
    )
    theta = theta_0
    theta_path = []

    for _ in range(T):
        mab.add_observation(inflation_rate)
        theta = mab.update_theta(theta)
        theta_path.append(theta)

    return np.array(theta_path)


def run_japan_with_prefill(T_prefill: int = 40, T_track: int = 120,
                           theta_0: float = 0.9):
    """
    Run Japan scenario with separate prefill + tracking phases.

    Matches Current's run_japan_scenario():
    1. Pre-fill T_prefill quarters of 0% inflation history
    2. Track loss_diff for T_track additional quarters

    Returns theta_path (160Q total) and loss_diff_path (120Q tracked).
    """
    mab = LongMemoryMABLearning(
        delta=0.8, eta=0.10, epsilon=1e-4, pi_star=0.005
    )

    # Phase 1: Pre-fill (add observations + update theta)
    theta = theta_0
    theta_path_prefill = []
    for _ in range(T_prefill):
        mab.add_observation(0.0)
        theta = mab.update_theta(theta)
        theta_path_prefill.append(theta)

    # Phase 2: Track with loss_diff
    theta_path_track = []
    loss_diff_path = []
    for _ in range(T_track):
        mab.add_observation(0.0)
        theta = mab.update_theta(theta)
        theta_path_track.append(theta)
        loss_diff_path.append(mab.get_loss_difference())

    theta_path = np.array(theta_path_prefill + theta_path_track)
    loss_diff_path = np.array(loss_diff_path)

    return theta_path, loss_diff_path, T_track


if __name__ == '__main__':
    print("Running Japan simulations...")

    # Scenario 1: Japan (0% inflation) with pre-fill
    japan_theta, japan_loss_diff, T_track = run_japan_with_prefill()
    print(f"  Japan (0%):    final theta = {japan_theta[-1]:.4f}")

    # Scenario 2: Stable at target (2% annual = 0.005 quarterly)
    stable_theta = run_scenario(0.005, T=160)
    print(f"  Stable (2%):   final theta = {stable_theta[-1]:.4f}")

    # Scenario 3: Mild undershoot (1% annual = 0.0025 quarterly)
    mild_theta = run_scenario(0.0025, T=160)
    print(f"  Mild (1%):     final theta = {mild_theta[-1]:.4f}")

    data = {
        'japan_theta': japan_theta,
        'stable_theta': stable_theta,
        'mild_theta': mild_theta,
        'japan_loss_diff': japan_loss_diff,
        'T': 160,           # Total quarters for theta paths (panel a)
        'T_track': T_track, # Quarters tracked for loss_diff (panel b)
    }

    out = PROJECT_ROOT / 'output' / 'simulations' / 'section_4' / 'japan.pkl'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved to {out}")
