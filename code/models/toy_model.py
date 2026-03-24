"""
Toy Model (λ=0)
===============

Simplified adaptive New Keynesian model where all agents are boundedly rational.
This corresponds to λ=0 in the full model, eliminating FIRE agents entirely.

Model equations:
    Phillips Curve: π_t = β·E_t[π_{t+1}] + κ·y_t + u_t
    IS Curve:       y_t = E_t[y_{t+1}] - σ·(i_t - E_t[π_{t+1}] - r^n_t)
    Taylor Rule:    i_t = r^n + π* + φ_π·(π_t - π*) + φ_y·y_t

Aggregate expectations (all agents boundedly rational):
    E_t[π_{t+1}] = θ_t·π* + (1-θ_t)·π_{t-1}
    E_t[y_{t+1}] = FIRE output expectation (isolates credibility to inflation only)

Learning dynamics:
    θ_{t+1} = (1-η)·θ_t + η·𝟙[L^CB_t ≤ L^BL_t + ε]

The model is solved as a simultaneous equation system each period.
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

from mab_learning import MABLearning
from parameters import get_default_params, validate_params


@dataclass
class SimulationResult:
    """Container for simulation results."""
    pi: np.ndarray      # Inflation path
    y: np.ndarray       # Output gap path
    i: np.ndarray       # Interest rate path
    theta: np.ndarray   # CB credibility path

    @property
    def T(self) -> int:
        """Simulation length."""
        return len(self.pi)


class ToyModel:
    """
    Toy adaptive NK model with λ=0 (all agents boundedly rational).

    This model provides a clean laboratory for studying how MAB learning
    affects inflation dynamics without the complication of FIRE agents.

    Attributes
    ----------
    params : dict
        Model parameters
    beta, sigma, kappa : float
        Structural parameters
    phi_pi, phi_y : float
        Taylor rule coefficients
    pi_star, rn_bar : float
        Policy targets
    eta, epsilon, k : float
        Learning parameters
    mab : MABLearning
        MAB learning algorithm instance
    """

    def __init__(self, params: Optional[Dict[str, float]] = None):
        """
        Initialize the toy model.

        Parameters
        ----------
        params : dict, optional
            Model parameters. If None, uses default calibration.
        """
        if params is None:
            params = get_default_params()

        # Validate parameters
        validate_params(params)
        self.params = params

        # Structural parameters
        self.beta = params['beta']
        self.sigma = params['sigma']
        self.kappa = params['kappa']

        # Policy parameters
        self.phi_pi = params['phi_pi']
        self.phi_y = params['phi_y']
        self.pi_star = params['pi_star']
        self.rn_bar = params['rn_bar']

        # Learning parameters
        self.k = params['k']
        self.eta = params['eta']
        self.epsilon = params['epsilon']

        # Shock persistence
        self.rho_u = params.get('rho_u', 0.5)

        # Initialize MAB learning module
        self.mab = MABLearning(
            k=self.k,
            eta=self.eta,
            epsilon=self.epsilon,
            pi_star=self.pi_star
        )

    def _compute_fire_E_y(self, u_t: float, rho_u: float) -> float:
        """
        Compute FIRE expected output gap for next period.

        Under rational expectations with AR(1) shocks:
            E[y_{t+1}] = a_y * rho_u * u_t
        where a_y = -sigma * phi_pi / denom

        This isolates the credibility mechanism to inflation expectations only,
        providing a cleaner identification of the credibility channel.

        Parameters
        ----------
        u_t : float
            Current cost-push shock
        rho_u : float
            Shock persistence

        Returns
        -------
        E_y_fire : float
            FIRE expected output gap
        """
        # Compute RE solution coefficient
        denom = (1 - self.beta * rho_u
                 + self.kappa * self.sigma * self.phi_pi
                 + self.kappa * self.sigma * self.phi_y)
        a_y = -self.sigma * self.phi_pi / denom

        # E[y_{t+1}] under FIRE = a_y * E[u_{t+1}] = a_y * rho_u * u_t
        return a_y * rho_u * u_t

    def _compute_expectations(self, theta_t: float,
                               pi_lag: float,
                               u_t: float = 0.0,
                               rho_u: float = 0.0) -> tuple:
        """
        Compute aggregate expectations given current state.

        Parameters
        ----------
        theta_t : float
            Current fraction using CB-anchored rule
        pi_lag : float
            Lagged inflation (for backward-looking agents)
        u_t : float
            Current cost-push shock (for FIRE output expectations)
        rho_u : float
            Shock persistence (for FIRE output expectations)

        Returns
        -------
        E_pi : float
            Expected inflation
        E_y : float
            Expected output gap (FIRE expectation)
        """
        E_pi = theta_t * self.pi_star + (1 - theta_t) * pi_lag
        # Use FIRE output expectations - isolates credibility to inflation only
        E_y = self._compute_fire_E_y(u_t, rho_u)
        return E_pi, E_y

    def _solve_period(self, E_pi: float, E_y: float,
                      u_t: float, rn_t: float) -> tuple:
        """
        Solve for current period outcomes given expectations.

        Solves the 2x2 system:
            [1, -κ] [π_t]   [β·E[π] + u_t]
            [σφ_π, 1+σφ_y] [y_t] = [E[y] - σ·(r̄ + π*(1-φ_π) - E[π] - r^n)]

        Parameters
        ----------
        E_pi : float
            Expected next-period inflation
        E_y : float
            Expected next-period output
        u_t : float
            Cost-push shock
        rn_t : float
            Natural rate of interest

        Returns
        -------
        pi_t : float
            Current inflation
        y_t : float
            Current output gap
        """
        # System matrix A
        A = np.array([
            [1, -self.kappa],
            [self.sigma * self.phi_pi, 1 + self.sigma * self.phi_y]
        ])

        # RHS vector B
        # Note: (1-beta)*pi_star ensures steady state inflation = pi_star when E[pi] = pi_star
        B1 = (1 - self.beta) * self.pi_star + self.beta * E_pi + u_t
        B2 = (E_y
              - self.sigma * self.rn_bar
              - self.sigma * self.pi_star * (1 - self.phi_pi)
              + self.sigma * E_pi
              + self.sigma * rn_t)

        B = np.array([B1, B2])

        # Solve system
        solution = np.linalg.solve(A, B)
        return solution[0], solution[1]

    def simulate(self,
                 T: int,
                 shock_path: Optional[np.ndarray] = None,
                 rho_u: Optional[float] = None,
                 initial_theta: float = 1.0) -> SimulationResult:
        """
        Simulate the model for T periods.

        Parameters
        ----------
        T : int
            Number of periods to simulate
        shock_path : np.ndarray, optional
            Exogenous cost-push shock path. If None, no shocks.
        rho_u : float, optional
            Shock persistence. Overrides params if provided.
        initial_theta : float, default=1.0
            Initial credibility (fraction using CB rule)

        Returns
        -------
        SimulationResult
            Container with pi, y, i, theta paths
        """
        # Use provided rho_u or fall back to params
        rho = rho_u if rho_u is not None else self.rho_u

        # Initialize storage
        pi = np.zeros(T)
        y = np.zeros(T)
        i = np.zeros(T)
        theta = np.zeros(T)

        # Shock path: propagate AR(1) if initial shock provided
        u = np.zeros(T)
        if shock_path is not None:
            u[:len(shock_path)] = shock_path[:T]
            # Propagate AR(1) dynamics
            for t in range(1, T):
                if t >= len(shock_path) or shock_path[t] == 0:
                    u[t] = rho * u[t-1]

        # Initial conditions
        pi[0] = self.pi_star
        y[0] = 0.0
        theta[0] = initial_theta
        i[0] = self.rn_bar + self.pi_star

        # Reset MAB learning
        self.mab.reset()
        self.mab.add_observation(pi[0])

        # Main simulation loop
        for t in range(1, T):
            # Form expectations based on t-1 state
            # Pass shock info for FIRE output expectations
            E_pi, E_y = self._compute_expectations(theta[t-1], pi[t-1], u[t], rho)

            # Solve for current outcomes
            pi[t], y[t] = self._solve_period(
                E_pi, E_y, u[t], self.rn_bar
            )

            # Compute interest rate
            i[t] = (self.rn_bar + self.pi_star
                    + self.phi_pi * (pi[t] - self.pi_star)
                    + self.phi_y * y[t])

            # Update learning
            self.mab.add_observation(pi[t])
            theta[t] = self.mab.update_theta(theta[t-1])

        return SimulationResult(pi=pi, y=y, i=i, theta=theta)


class FIREBenchmark:
    """
    FIRE (Full-Information Rational Expectations) benchmark model.

    Standard NK model where all agents have rational expectations.
    Used as a comparison benchmark for the adaptive model.
    """

    def __init__(self, params: Optional[Dict[str, float]] = None):
        """
        Initialize FIRE benchmark.

        Parameters
        ----------
        params : dict, optional
            Model parameters. If None, uses default calibration.
        """
        if params is None:
            params = get_default_params()

        self.params = params
        self.beta = params['beta']
        self.sigma = params['sigma']
        self.kappa = params['kappa']
        self.phi_pi = params['phi_pi']
        self.phi_y = params['phi_y']
        self.pi_star = params['pi_star']
        self.rn_bar = params['rn_bar']

        # Compute RE solution coefficients
        self._compute_re_coefficients()

    def _compute_re_coefficients(self):
        """
        Compute rational expectations solution coefficients.

        Under RE, the equilibrium response to a cost-push shock u_t is:
            π_t = a_π · u_t
            y_t = a_y · u_t

        where coefficients depend on structural parameters.
        """
        # Denominator from solving RE system
        denom = (1 - self.beta * self.params.get('rho_u', 0.5)
                 + self.kappa * self.sigma * self.phi_pi
                 + self.kappa * self.sigma * self.phi_y)

        # Store base coefficients (will adjust for rho in simulate)
        self.base_denom_factor = (self.kappa * self.sigma * self.phi_pi
                                   + self.kappa * self.sigma * self.phi_y)

    def simulate(self,
                 T: int,
                 shock_path: Optional[np.ndarray] = None,
                 rho_u: float = 0.0) -> SimulationResult:
        """
        Simulate FIRE model for T periods.

        Parameters
        ----------
        T : int
            Number of periods
        shock_path : np.ndarray, optional
            Cost-push shock path
        rho_u : float
            Shock persistence

        Returns
        -------
        SimulationResult
            Simulation results (theta is always 1.0 for FIRE)
        """
        # Initialize
        pi = np.full(T, self.pi_star)
        y = np.zeros(T)
        i = np.full(T, self.rn_bar + self.pi_star)
        theta = np.ones(T)  # Full credibility always

        # If no shocks, return steady state
        if shock_path is None:
            return SimulationResult(pi=pi, y=y, i=i, theta=theta)

        # Build shock path with AR(1) propagation
        u = np.zeros(T)
        u[:len(shock_path)] = shock_path[:T]
        for t in range(1, T):
            if t >= len(shock_path) or shock_path[t] == 0:
                u[t] = rho_u * u[t-1]

        # RE solution coefficients (depend on rho)
        denom = 1 - self.beta * rho_u + self.base_denom_factor

        a_pi = 1 / denom
        a_y = -self.sigma * self.phi_pi / denom

        # Compute paths
        for t in range(T):
            pi[t] = self.pi_star + a_pi * u[t]
            y[t] = a_y * u[t]
            i[t] = (self.rn_bar + self.pi_star
                    + self.phi_pi * (pi[t] - self.pi_star)
                    + self.phi_y * y[t])

        return SimulationResult(pi=pi, y=y, i=i, theta=theta)


# =============================================================================
# Testing
# =============================================================================

def test_toy_model():
    """Basic test of the toy model."""
    print("Testing Toy Model")
    print("=" * 60)

    model = ToyModel()

    # Test steady state (no shocks)
    print("\nTest 1: Steady state (no shocks)")
    result = model.simulate(T=20)
    print(f"  Final inflation: {result.pi[-1]:.4f} (target: {model.pi_star})")
    print(f"  Final theta: {result.theta[-1]:.4f}")

    # Test transitory shock
    print("\nTest 2: Transitory cost-push shock")
    model2 = ToyModel()
    shock = np.zeros(20)
    shock[5] = 0.01
    result2 = model2.simulate(T=20, shock_path=shock, rho_u=0.0)
    print(f"  Pre-shock (t=4): pi={result2.pi[4]:.4f}")
    print(f"  Shock (t=5): pi={result2.pi[5]:.4f}")
    print(f"  Post-shock (t=10): pi={result2.pi[10]:.4f}")

    # Test FIRE benchmark
    print("\nTest 3: FIRE benchmark")
    fire = FIREBenchmark()
    result_fire = fire.simulate(T=20, shock_path=shock, rho_u=0.0)
    print(f"  Shock (t=5): pi={result_fire.pi[5]:.4f}")
    print(f"  Post-shock (t=10): pi={result_fire.pi[10]:.4f}")

    print("\n" + "=" * 60)
    print("Tests completed")


if __name__ == "__main__":
    test_toy_model()
