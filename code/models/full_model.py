"""
Full Adaptive-Hybrid New Keynesian Model with Sophisticated FIRE
================================================================

Extends the toy model by introducing heterogeneous agents:
- Fraction lambda of FIRE (rational expectations) agents
- Fraction (1-lambda) of boundedly rational agents using MAB learning

Key feature: Sophisticated FIRE
-------------------------------
Unlike "naive" FIRE agents who assume lambda=1, sophisticated FIRE agents
observe the credibility state theta_t and solve forward taking its evolution
into account. This creates a fixed-point problem:

1. FIRE expectations depend on the path of theta
2. The path of theta depends on inflation outcomes
3. Inflation depends on FIRE expectations

Algorithm:
----------
1. Solve naive model as warm start
2. Iterate until convergence:
   a. Forward simulate given current FIRE expectations
   b. Backward solve for sophisticated FIRE expectations given theta path
   c. Check convergence
   d. Update with dampening

Model equations:
----------------
Phillips Curve: pi_t = beta * E_t[pi_{t+1}] + kappa * y_t + u_t
IS Curve:       y_t = E_t[y_{t+1}] - sigma * (i_t - E_t[pi_{t+1}] - rn_t)
Taylor Rule:    i_t = rn_bar + pi* + phi_pi * (pi_t - pi*) + phi_y * y_t + v_t

Aggregate expectations:
-----------------------
E_t[pi_{t+1}] = lambda * E^FIRE_t[pi_{t+1}] + (1-lambda) * [theta_t * pi* + (1-theta_t) * pi_{t-1}]
E_t[y_{t+1}]  = E^FIRE_t[y_{t+1}]  (BR agents use FIRE output expectations)

Learning dynamics:
------------------
theta_{t+1} = (1-eta) * theta_t + eta * 1[L^CB_t <= L^BL_t + epsilon]

Special cases:
--------------
- lambda = 0: Equivalent to toy model (all agents boundedly rational)
- lambda = 1: Standard NK model with FIRE (theta irrelevant)
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from mab_learning import MABLearning
from fire_solution import FIRESolver
from parameters import get_default_params, validate_params


@dataclass
class SimulationResult:
    """Container for simulation results."""
    pi: np.ndarray      # Inflation path
    y: np.ndarray       # Output gap path
    i: np.ndarray       # Interest rate path
    theta: np.ndarray   # CB credibility path

    # Optional: convergence info for sophisticated FIRE
    converged: bool = True
    iterations: int = 1

    @property
    def T(self) -> int:
        """Simulation length."""
        return len(self.pi)


class FullModel:
    """
    Full adaptive-hybrid New Keynesian model with sophisticated FIRE.

    This model nests both the toy model (lambda=0) and the standard NK model
    with FIRE (lambda=1) as special cases.

    Parameters
    ----------
    params : dict, optional
        Model parameters. If None, uses default calibration.
        Key parameters:
        - lambda_fire: Fraction of FIRE agents (default 0.35)
        - eta: Learning speed for MAB algorithm
        - epsilon: Tie-breaking threshold
        - k: Memory window for loss evaluation

    Attributes
    ----------
    lam : float
        Fraction of FIRE agents (named 'lam' to avoid Python keyword)
    """

    def __init__(self, params: Optional[Dict[str, float]] = None):
        """Initialize the full model."""
        if params is None:
            params = get_default_params()

        # Add lambda if not present (for backward compatibility)
        if 'lambda_fire' not in params:
            params['lambda_fire'] = 0.35

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

        # Heterogeneity parameter
        self.lam = params['lambda_fire']

        # Learning parameters
        self.k = params['k']
        self.eta = params['eta']
        self.epsilon = params['epsilon']
        self.gamma = params.get('gamma', None)

        # Shock persistence
        self.rho_u = params.get('rho_u', 0.5)
        self.rho_r = params.get('rho_r', 0.9)
        self.rho_v = params.get('rho_v', 0.0)

        # Initialize FIRE solver (for naive warm start)
        self.fire_solver = FIRESolver(params)

    def _solve_nk_system(
        self,
        E_pi: float,
        E_y: float,
        u: float,
        rn: float = None,
        v: float = 0.0
    ) -> Tuple[float, float]:
        """
        Solve the NK system for (pi, y) given expectations and shocks.

        System:
        - Phillips Curve: pi = (1-beta)pi* + beta E[pi'] + kappa*y + u
        - IS + Taylor: y = E[y'] - sigma*(i - E[pi'] - rn)
                       i = rn_bar + pi* + phi_pi*(pi - pi*) + phi_y*y + v
        """
        if rn is None:
            rn = self.rn_bar

        # System matrix A @ [pi, y]' = B
        A = np.array([
            [1.0, -self.kappa],
            [self.sigma * self.phi_pi, 1.0 + self.sigma * self.phi_y]
        ])

        # RHS
        B1 = (1 - self.beta) * self.pi_star + self.beta * E_pi + u
        B2 = (E_y
              - self.sigma * self.rn_bar
              - self.sigma * self.pi_star * (1 - self.phi_pi)
              - self.sigma * v
              + self.sigma * E_pi
              + self.sigma * rn)

        B = np.array([B1, B2])
        solution = np.linalg.solve(A, B)
        return solution[0], solution[1]

    def _simulate_naive(
        self,
        T: int,
        u: np.ndarray,
        rn: np.ndarray,
        v: np.ndarray,
        initial_theta: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate naive FIRE model (used as warm start).

        Returns: (pi, y, theta, E_fire)
        """
        pi = np.zeros(T)
        y = np.zeros(T)
        theta = np.zeros(T)
        E_fire = np.zeros(T)

        # Initial conditions
        pi[0] = self.pi_star
        y[0] = 0.0
        theta[0] = initial_theta
        E_fire[0] = self.pi_star

        # MAB for theta updates
        mab = MABLearning(
            k=self.k,
            eta=self.eta,
            epsilon=self.epsilon,
            pi_star=self.pi_star,
            gamma=self.gamma
        )
        mab.add_observation(pi[0])

        shock_persistence = {
            'rho_u': self.rho_u,
            'rho_r': self.rho_r,
            'rho_v': self.rho_v
        }

        for t in range(1, T):
            # Naive FIRE expectations (assume lambda=1)
            E_pi_fire, E_y_fire = self.fire_solver.solve_expectations(
                u[t], rn[t], v[t], shock_persistence
            )
            E_fire[t] = E_pi_fire

            # BR expectations
            E_pi_br = theta[t-1] * self.pi_star + (1 - theta[t-1]) * pi[t-1]

            # Aggregate expectations
            E_pi = self.lam * E_pi_fire + (1 - self.lam) * E_pi_br
            E_y = E_y_fire  # BR use FIRE output expectations

            # Solve NK system
            pi[t], y[t] = self._solve_nk_system(E_pi, E_y, u[t], rn[t], v[t])

            # Update theta via MAB
            mab.add_observation(pi[t])
            theta[t] = mab.update_theta(theta[t-1])

        return pi, y, theta, E_fire

    def _forward_simulate(
        self,
        T: int,
        u: np.ndarray,
        rn: np.ndarray,
        v: np.ndarray,
        E_fire: np.ndarray,
        initial_theta: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward simulate given FIRE expectations.

        Returns: (pi, y, theta)
        """
        pi = np.zeros(T)
        y = np.zeros(T)
        theta = np.zeros(T)

        pi[0] = self.pi_star
        y[0] = 0.0
        theta[0] = initial_theta

        # MAB for theta updates
        mab = MABLearning(
            k=self.k,
            eta=self.eta,
            epsilon=self.epsilon,
            pi_star=self.pi_star,
            gamma=self.gamma
        )
        mab.add_observation(pi[0])

        for t in range(1, T):
            # BR expectations
            E_pi_br = theta[t-1] * self.pi_star + (1 - theta[t-1]) * pi[t-1]

            # Aggregate expectations (using provided FIRE expectations)
            E_pi = self.lam * E_fire[t] + (1 - self.lam) * E_pi_br

            # For output, use steady state expectation
            E_y = 0.0

            # Solve NK system
            pi[t], y[t] = self._solve_nk_system(E_pi, E_y, u[t], rn[t], v[t])

            # Update theta via MAB
            mab.add_observation(pi[t])
            theta[t] = mab.update_theta(theta[t-1])

        return pi, y, theta

    def _backward_solve_fire(
        self,
        T: int,
        u: np.ndarray,
        rn: np.ndarray,
        v: np.ndarray,
        theta: np.ndarray,
        pi: np.ndarray
    ) -> np.ndarray:
        """
        Backward solve for sophisticated FIRE expectations.

        Given the path of theta and pi from forward simulation, compute
        rational expectations by solving the NK system backward from
        terminal conditions.
        """
        E_fire_pi = np.zeros(T)
        E_fire_y = np.zeros(T)

        # Terminal conditions: expect steady state
        E_fire_pi[T-1] = self.pi_star
        E_fire_y[T-1] = 0.0

        if T > 1:
            E_fire_pi[T-2] = self.pi_star
            E_fire_y[T-2] = 0.0

        # Backward recursion
        for t in range(T-3, 0, -1):
            # At time t+1, what will aggregate expectations be?
            theta_t_plus_1 = theta[t+1]
            pi_t = pi[t]

            # BR expectation at t+1 about t+2
            E_pi_br = theta_t_plus_1 * self.pi_star + (1 - theta_t_plus_1) * pi_t

            # FIRE expectation at t+1 about t+2 (from backward recursion)
            E_fire_pi_t_plus_2 = E_fire_pi[t+1]

            # Aggregate expectation at t+1
            E_pi_agg = self.lam * E_fire_pi_t_plus_2 + (1 - self.lam) * E_pi_br

            # Expected output
            E_y_agg = E_fire_y[t+1]

            # Shock at t+1
            u_t_plus_1 = u[t+1]
            rn_t_plus_1 = rn[t+1]
            v_t_plus_1 = v[t+1]

            # Solve NK system at t+1
            pi_t_plus_1, y_t_plus_1 = self._solve_nk_system(
                E_pi_agg, E_y_agg, u_t_plus_1, rn_t_plus_1, v_t_plus_1
            )

            E_fire_pi[t] = pi_t_plus_1
            E_fire_y[t] = y_t_plus_1

        return E_fire_pi

    def _simulate_mit_sophisticated(
        self,
        T: int,
        u: np.ndarray,
        rn: np.ndarray,
        v: np.ndarray,
        initial_theta: float,
        tol: float = 1e-8,
        dampening: float = 0.3,
        verbose: bool = False
    ) -> SimulationResult:
        """
        MIT shock simulation with sophisticated FIRE.

        FIRE agents are aware of BR agents (account for theta dynamics) but
        don't anticipate future shocks. At each period, they solve forward
        based on current shock and its AR(1) persistence.

        This gives realistic dynamics where:
        - Steady state before shock (no anticipation)
        - Gradual adjustment after shock (FIRE accounts for BR expectations)
        """
        pi = np.zeros(T)
        y = np.zeros(T)
        theta = np.zeros(T)

        # Initial conditions
        pi[0] = self.pi_star
        y[0] = 0.0
        theta[0] = initial_theta

        # MAB for theta updates
        mab = MABLearning(
            k=self.k,
            eta=self.eta,
            epsilon=self.epsilon,
            pi_star=self.pi_star,
            gamma=self.gamma
        )
        mab.add_observation(pi[0])

        for t in range(1, T):
            # Current shock and expected future path (AR(1) decay from current)
            current_u = u[t]

            # Sophisticated FIRE: solve for expectations accounting for BR agents
            # This is a simplified version - FIRE expects theta to stay roughly constant
            # and solves for the implied inflation given current shock persistence

            # BR expectations (based on t-1 state)
            E_pi_br = theta[t-1] * self.pi_star + (1 - theta[t-1]) * pi[t-1]

            # FIRE expectations: solve for E[pi] consistent with the system
            # Under AR(1) shock, FIRE expects future inflation to decay toward target
            # as the shock decays. They account for the BR agents' influence.

            # Iterative solve for this period's FIRE expectation
            E_pi_fire = self.pi_star  # Initial guess
            for _ in range(50):  # Inner iteration for this period
                # Aggregate expectation given current FIRE guess
                E_pi_agg = self.lam * E_pi_fire + (1 - self.lam) * E_pi_br

                # Solve NK system
                pi_implied, y_implied = self._solve_nk_system(
                    E_pi_agg, 0.0, current_u, rn[t], v[t]
                )

                # FIRE expectation should be consistent with implied inflation
                # accounting for shock persistence
                # E[pi_{t+1}] = weighted average of target and decay toward it
                expected_future_u = self.rho_u * current_u

                # Simple approximation: FIRE expects inflation to decay toward target
                # as shock decays, weighted by how much BR agents anchor
                decay_factor = self.rho_u * (1 - self.lam * theta[t-1])
                E_pi_fire_new = self.pi_star + decay_factor * (pi_implied - self.pi_star)

                if abs(E_pi_fire_new - E_pi_fire) < tol:
                    break
                E_pi_fire = dampening * E_pi_fire_new + (1 - dampening) * E_pi_fire

            # Final solve with converged expectations
            E_pi_agg = self.lam * E_pi_fire + (1 - self.lam) * E_pi_br
            pi[t], y[t] = self._solve_nk_system(E_pi_agg, 0.0, current_u, rn[t], v[t])

            # Update theta via MAB
            mab.add_observation(pi[t])
            theta[t] = mab.update_theta(theta[t-1])

        # Compute interest rates
        i = np.zeros(T)
        for t in range(T):
            i[t] = (self.rn_bar + self.pi_star
                    + self.phi_pi * (pi[t] - self.pi_star)
                    + self.phi_y * y[t]
                    + v[t])

        return SimulationResult(
            pi=pi, y=y, i=i, theta=theta,
            converged=True, iterations=-1
        )

    def simulate(
        self,
        T: int,
        shock_path: Optional[np.ndarray] = None,
        rn_path: Optional[np.ndarray] = None,
        v_path: Optional[np.ndarray] = None,
        rho_u: Optional[float] = None,
        initial_theta: float = 1.0,
        max_iter: int = 200,
        tol: float = 1e-8,
        dampening: float = 0.3,
        verbose: bool = False
    ) -> SimulationResult:
        """
        Simulate the model for T periods using sophisticated FIRE.

        Parameters
        ----------
        T : int
            Number of periods to simulate
        shock_path : np.ndarray, optional
            Cost-push shock path (u_t). If None, no cost-push shocks.
        rn_path : np.ndarray, optional
            Natural rate path. If None, constant at rn_bar.
        v_path : np.ndarray, optional
            Monetary policy shock path. If None, no policy shocks.
        rho_u : float, optional
            Cost-push shock persistence. Overrides params if provided.
        initial_theta : float, default=1.0
            Initial credibility (fraction using CB rule)
        max_iter : int, default=200
            Maximum iterations for fixed-point
        tol : float, default=1e-8
            Convergence tolerance
        dampening : float, default=0.3
            Dampening parameter for stability
        verbose : bool, default=False
            Print iteration progress

        Returns
        -------
        SimulationResult
            Container with pi, y, i, theta paths
        """
        # Handle pure FIRE case (no iteration needed)
        if self.lam >= 1.0:
            return self._simulate_pure_fire(T, shock_path, rn_path, v_path, rho_u)

        # Use provided rho_u or fall back to params
        rho = rho_u if rho_u is not None else self.rho_u

        # Build shock paths with AR(1) propagation
        u = np.zeros(T)
        if shock_path is not None:
            u[:len(shock_path)] = shock_path[:T]
            for t in range(1, T):
                if t >= len(shock_path) or shock_path[t] == 0:
                    u[t] = rho * u[t-1]

        # Natural rate path
        rn = np.full(T, self.rn_bar)
        if rn_path is not None:
            rn[:len(rn_path)] = rn_path[:T]

        # Monetary policy shock path
        v = np.zeros(T)
        if v_path is not None:
            v[:len(v_path)] = v_path[:T]

        # Step 0: Warm start from naive model
        if verbose:
            print("Computing naive solution as warm start...")

        pi_naive, y_naive, theta_naive, E_fire = self._simulate_naive(
            T, u, rn, v, initial_theta
        )

        # Handle lambda = 0 case (no FIRE agents, no iteration needed)
        if self.lam <= 0.0:
            i = np.zeros(T)
            for t in range(T):
                i[t] = (self.rn_bar + self.pi_star
                        + self.phi_pi * (pi_naive[t] - self.pi_star)
                        + self.phi_y * y_naive[t]
                        + v[t])
            return SimulationResult(
                pi=pi_naive, y=y_naive, i=i, theta=theta_naive,
                converged=True, iterations=1
            )

        # Handle naive FIRE case (no backward solve, MIT shock compatible)
        if max_iter == 0:
            # Use naive E_fire directly - no anticipation of future shocks
            pi, y, theta = self._forward_simulate(T, u, rn, v, E_fire, initial_theta)
            i = np.zeros(T)
            for t in range(T):
                i[t] = (self.rn_bar + self.pi_star
                        + self.phi_pi * (pi[t] - self.pi_star)
                        + self.phi_y * y[t]
                        + v[t])
            return SimulationResult(
                pi=pi, y=y, i=i, theta=theta,
                converged=True, iterations=0
            )

        # Handle MIT shock with sophisticated FIRE (max_iter == -1)
        # FIRE agents are aware of BR agents but don't anticipate future shocks
        if max_iter == -1:
            return self._simulate_mit_sophisticated(
                T, u, rn, v, initial_theta, tol, dampening, verbose
            )

        # Fixed-point iteration for sophisticated FIRE
        if verbose:
            print(f"Fixed-point iteration (max_iter={max_iter}, tol={tol})")

        converged = False
        for iteration in range(max_iter):
            # Forward simulate given current E_fire
            pi, y, theta = self._forward_simulate(
                T, u, rn, v, E_fire, initial_theta
            )

            # Backward solve for sophisticated FIRE expectations
            E_fire_new = self._backward_solve_fire(T, u, rn, v, theta, pi)

            # Check convergence
            diff = np.max(np.abs(E_fire_new - E_fire))

            if verbose and (iteration < 5 or iteration % 10 == 0):
                print(f"  Iteration {iteration}: max|dE_fire| = {diff:.2e}")

            if diff < tol:
                converged = True
                if verbose:
                    print(f"  Converged at iteration {iteration}!")
                break

            # Update with dampening
            E_fire = dampening * E_fire_new + (1 - dampening) * E_fire

        if not converged and verbose:
            print(f"  Warning: Did not converge after {max_iter} iterations")

        # Final forward simulation with converged expectations
        pi, y, theta = self._forward_simulate(T, u, rn, v, E_fire, initial_theta)

        # Compute interest rates
        i = np.zeros(T)
        for t in range(T):
            i[t] = (self.rn_bar + self.pi_star
                    + self.phi_pi * (pi[t] - self.pi_star)
                    + self.phi_y * y[t]
                    + v[t])

        return SimulationResult(
            pi=pi, y=y, i=i, theta=theta,
            converged=converged,
            iterations=iteration + 1
        )

    def _simulate_pure_fire(
        self,
        T: int,
        shock_path: Optional[np.ndarray],
        rn_path: Optional[np.ndarray],
        v_path: Optional[np.ndarray],
        rho_u: Optional[float]
    ) -> SimulationResult:
        """Simulate pure FIRE model (lambda=1)."""
        rho = rho_u if rho_u is not None else self.rho_u

        # Build shock paths
        u = np.zeros(T)
        if shock_path is not None:
            u[:len(shock_path)] = shock_path[:T]
            for t in range(1, T):
                if t >= len(shock_path) or shock_path[t] == 0:
                    u[t] = rho * u[t-1]

        rn = np.full(T, self.rn_bar)
        if rn_path is not None:
            rn[:len(rn_path)] = rn_path[:T]

        v = np.zeros(T)
        if v_path is not None:
            v[:len(v_path)] = v_path[:T]

        pi = np.zeros(T)
        y = np.zeros(T)
        i = np.zeros(T)
        theta = np.ones(T)

        pi[0] = self.pi_star
        y[0] = 0.0
        i[0] = self.rn_bar + self.pi_star

        shock_persistence = {
            'rho_u': rho,
            'rho_r': self.rho_r,
            'rho_v': self.rho_v
        }

        for t in range(1, T):
            E_pi, E_y = self.fire_solver.solve_expectations(
                u[t], rn[t], v[t], shock_persistence
            )
            pi[t], y[t] = self._solve_nk_system(E_pi, E_y, u[t], rn[t], v[t])
            i[t] = (self.rn_bar + self.pi_star
                    + self.phi_pi * (pi[t] - self.pi_star)
                    + self.phi_y * y[t]
                    + v[t])

        return SimulationResult(pi=pi, y=y, i=i, theta=theta,
                                converged=True, iterations=1)


# =============================================================================
# Testing
# =============================================================================

def test_full_model():
    """Test the full model with basic scenarios."""
    print("Testing Full Adaptive-Hybrid NK Model (Sophisticated FIRE)")
    print("=" * 60)

    # Test 1: lambda = 0 should behave like toy model
    print("\n--- Test 1: lambda = 0 (toy model equivalent) ---")
    params_toy = get_default_params()
    params_toy['lambda_fire'] = 0.0

    model_toy = FullModel(params_toy)
    result_toy = model_toy.simulate(T=20)

    print(f"Initial: pi={result_toy.pi[0]:.4f}, theta={result_toy.theta[0]:.4f}")
    print(f"Final:   pi={result_toy.pi[-1]:.4f}, theta={result_toy.theta[-1]:.4f}")
    print(f"Converged: {result_toy.converged}, Iterations: {result_toy.iterations}")

    # Test 2: lambda = 1 should be pure FIRE
    print("\n--- Test 2: lambda = 1 (pure FIRE) ---")
    params_fire = get_default_params()
    params_fire['lambda_fire'] = 1.0

    model_fire = FullModel(params_fire)
    result_fire = model_fire.simulate(T=20)

    print(f"Theta always 1: {np.all(result_fire.theta == 1.0)}")

    # Test 3: Intermediate lambda with shock
    print("\n--- Test 3: lambda = 0.35 with cost-push shock ---")
    params_mid = get_default_params()
    params_mid['lambda_fire'] = 0.35

    model_mid = FullModel(params_mid)
    shock = np.zeros(40)
    shock[5] = 0.02  # 2% shock at t=5

    result_mid = model_mid.simulate(T=40, shock_path=shock, rho_u=0.5,
                                     initial_theta=0.8, verbose=True)

    print(f"\nPre-shock (t=4):  pi={result_mid.pi[4]:.4f}, theta={result_mid.theta[4]:.4f}")
    print(f"Shock (t=5):      pi={result_mid.pi[5]:.4f}, theta={result_mid.theta[5]:.4f}")
    print(f"Post-shock (t=15): pi={result_mid.pi[15]:.4f}, theta={result_mid.theta[15]:.4f}")
    print(f"Final (t=39):     pi={result_mid.pi[-1]:.4f}, theta={result_mid.theta[-1]:.4f}")

    print("\n" + "=" * 60)
    print("All tests completed!")


if __name__ == "__main__":
    test_full_model()
