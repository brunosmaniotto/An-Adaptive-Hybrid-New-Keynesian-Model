"""
Policy Experiments Module
=========================

Provides specialized simulation capabilities for policy experiments:
1. Time-varying policy (phi_pi as function of time)
2. Separate anchor vs policy targets (for undershooting experiments)

This module extends the base model capabilities without modifying FullModel,
keeping the core simulation code clean while enabling policy analysis.

Author: Bruno Cittolin Smaniotto
"""

import numpy as np
from typing import Dict, Optional, Callable, Tuple
from dataclasses import dataclass

from mab_learning import MABLearning
from parameters import get_default_params, override_params


@dataclass
class PolicySimResult:
    """Container for policy experiment results."""
    pi: np.ndarray          # Inflation path
    y: np.ndarray           # Output gap path
    i: np.ndarray           # Interest rate path
    theta: np.ndarray       # CB credibility path
    phi_pi_path: np.ndarray # Policy aggressiveness path

    @property
    def T(self) -> int:
        return len(self.pi)


def simulate_policy_experiment(
    params: Dict,
    T: int,
    shock_path: Optional[np.ndarray] = None,
    rho_u: float = 0.5,
    initial_theta: float = 1.0,
    initial_inflation: Optional[float] = None,
    phi_pi_func: Optional[Callable[[int], float]] = None,
    pi_anchor: Optional[float] = None,
    pi_policy: Optional[float] = None,
) -> PolicySimResult:
    """
    Simulate with time-varying policy and/or separate anchor/policy targets.

    This is a simplified BR-only simulation (lambda=0) designed for policy
    experiments where the focus is on credibility dynamics rather than
    FIRE/BR interaction.

    Parameters
    ----------
    params : dict
        Model parameters
    T : int
        Number of periods
    shock_path : np.ndarray, optional
        Cost-push shock path
    rho_u : float
        Shock persistence
    initial_theta : float
        Initial credibility
    initial_inflation : float, optional
        Initial inflation level
    phi_pi_func : callable, optional
        Function returning phi_pi at time t. If None, uses constant params['phi_pi'].
    pi_anchor : float, optional
        Target that agents anchor to. Defaults to params['pi_star'].
    pi_policy : float, optional
        Target for Taylor rule. Defaults to params['pi_star'].

    Returns
    -------
    PolicySimResult
        Simulation results with phi_pi_path
    """
    # Extract parameters
    beta = params['beta']
    kappa = params['kappa']
    sigma = params['sigma']
    phi_y = params['phi_y']
    pi_star = params['pi_star']
    rn_bar = params['rn_bar']
    eta = params['eta']
    k = params['k']
    epsilon = params['epsilon']
    gamma = params.get('gamma', 25000.0)

    # Defaults
    if phi_pi_func is None:
        phi_pi_const = params['phi_pi']
        phi_pi_func = lambda t: phi_pi_const
    if pi_anchor is None:
        pi_anchor = pi_star
    if pi_policy is None:
        pi_policy = pi_star

    # Build shock path with AR(1) propagation
    u = np.zeros(T)
    if shock_path is not None:
        u[:len(shock_path)] = shock_path[:T]
        for t in range(1, T):
            if t >= len(shock_path) or shock_path[t] == 0:
                u[t] = rho_u * u[t-1]

    # Initialize arrays
    pi = np.zeros(T)
    y = np.zeros(T)
    i = np.zeros(T)
    theta = np.zeros(T)
    phi_pi_path = np.zeros(T)

    # Initial conditions
    pi[0] = initial_inflation if initial_inflation is not None else pi_anchor
    y[0] = 0.0
    theta[0] = initial_theta
    phi_pi_path[0] = phi_pi_func(0)
    i[0] = rn_bar + pi_policy + phi_pi_path[0] * (pi[0] - pi_policy)

    # MAB learning
    mab = MABLearning(
        k=k,
        eta=eta,
        epsilon=epsilon,
        pi_star=pi_anchor,  # Agents anchor to announced target
        gamma=gamma
    )
    mab.add_observation(pi[0])

    for t in range(1, T):
        # Get current policy aggressiveness
        phi_pi = phi_pi_func(t)
        phi_pi_path[t] = phi_pi

        # BR expectations (anchor to ANNOUNCED target)
        E_pi_br = theta[t-1] * pi_anchor + (1 - theta[t-1]) * pi[t-1]

        # Solve NK system
        # PC: pi = beta * E_pi + kappa * y + u
        # IS + TR combined
        A = beta * E_pi_br + u[t]
        B = -sigma * (1 - phi_pi) * pi_policy + sigma * E_pi_br
        C = 1 + sigma * phi_y
        D = sigma * phi_pi

        pi[t] = (A * C + kappa * B) / (C + kappa * D)
        y[t] = (B - D * pi[t]) / C

        # Taylor rule
        i[t] = rn_bar + pi_policy + phi_pi * (pi[t] - pi_policy) + phi_y * y[t]

        # MAB learning
        mab.add_observation(pi[t])
        theta[t] = mab.update_theta(theta[t-1])

    return PolicySimResult(
        pi=pi, y=y, i=i, theta=theta, phi_pi_path=phi_pi_path
    )


# =============================================================================
# Policy path factories
# =============================================================================

def constant_policy(phi_pi: float) -> Callable[[int], float]:
    """Create constant policy function."""
    return lambda t: phi_pi


def shock_therapy_policy(phi_pi_aggressive: float) -> Callable[[int], float]:
    """Create shock therapy policy: aggressive from day one."""
    return lambda t: phi_pi_aggressive


def gradual_policy(
    phi_pi_start: float,
    phi_pi_end: float,
    ramp_duration: int
) -> Callable[[int], float]:
    """
    Create gradual ramp-up policy.

    Linear increase from phi_pi_start to phi_pi_end over ramp_duration quarters.
    """
    def policy(t):
        if t >= ramp_duration:
            return phi_pi_end
        return phi_pi_start + (phi_pi_end - phi_pi_start) * (t / ramp_duration)
    return policy


# =============================================================================
# Analysis utilities
# =============================================================================

def compute_welfare(
    result: PolicySimResult,
    pi_target: float,
    lambda_y: float = 0.25
) -> float:
    """
    Compute welfare loss.

    Parameters
    ----------
    result : PolicySimResult
        Simulation output
    pi_target : float
        Target inflation for welfare evaluation
    lambda_y : float
        Weight on output gap in loss function

    Returns
    -------
    loss : float
        Welfare loss (sum of squared deviations)
    """
    pi_dev = result.pi - pi_target
    return np.sum(pi_dev**2) + lambda_y * np.sum(result.y**2)


def find_recovery_time(theta: np.ndarray, threshold: float = 0.5) -> Optional[int]:
    """
    Find first quarter when credibility exceeds threshold.

    Returns None if threshold never reached.
    """
    crossings = np.where(theta >= threshold)[0]
    if len(crossings) > 0:
        return int(crossings[0])
    return None


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("Testing Policy Experiments Module")
    print("=" * 60)

    params = override_params(get_default_params(), {
        'eta': 0.10,
        'k': 3,
        'epsilon': 1e-4,
        'gamma': 25000.0,
    })

    # Test 1: Shock therapy vs gradualism
    print("\n--- Test: Shock therapy vs gradualism ---")
    T = 80
    shock_path = np.zeros(T)
    shock_path[0:4] = 0.006

    result_shock = simulate_policy_experiment(
        params, T, shock_path, rho_u=0.8,
        initial_theta=0.05, initial_inflation=0.035,
        phi_pi_func=constant_policy(3.5)
    )

    result_gradual = simulate_policy_experiment(
        params, T, shock_path, rho_u=0.8,
        initial_theta=0.05, initial_inflation=0.035,
        phi_pi_func=gradual_policy(1.5, 3.5, 20)
    )

    print(f"Shock therapy: Welfare={compute_welfare(result_shock, 0.005):.4f}, "
          f"Recovery={find_recovery_time(result_shock.theta)}")
    print(f"Gradualism: Welfare={compute_welfare(result_gradual, 0.005):.4f}, "
          f"Recovery={find_recovery_time(result_gradual.theta)}")

    # Test 2: Undershooting
    print("\n--- Test: Undershooting ---")
    T = 80
    shock_path = np.zeros(T)
    shock_path[0:4] = 0.003

    result_standard = simulate_policy_experiment(
        params, T, shock_path, rho_u=0.7,
        initial_theta=0.1, initial_inflation=0.025,
        pi_anchor=0.005, pi_policy=0.005
    )

    result_undershoot = simulate_policy_experiment(
        params, T, shock_path, rho_u=0.7,
        initial_theta=0.1, initial_inflation=0.025,
        pi_anchor=0.005, pi_policy=0.0
    )

    print(f"Standard (2%): Welfare={compute_welfare(result_standard, 0.005):.4f}, "
          f"Recovery={find_recovery_time(result_standard.theta)}")
    print(f"Undershoot (0%): Welfare={compute_welfare(result_undershoot, 0.005):.4f}, "
          f"Recovery={find_recovery_time(result_undershoot.theta)}")

    print("\n" + "=" * 60)
    print("All tests completed!")
