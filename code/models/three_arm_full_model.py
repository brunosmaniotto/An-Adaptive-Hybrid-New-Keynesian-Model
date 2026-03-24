"""
Three-Arm Full Adaptive-Hybrid New Keynesian Model
===================================================

Extends the full model to include three forecasting rules for BR agents:
1. CB-anchored:      E[pi_{t+1}] = pi*
2. Backward-looking: E[pi_{t+1}] = pi_{t-1}
3. Trend-following:  E[pi_{t+1}] = pi_{t-1} + (pi_{t-1} - pi_{t-2})

This uses the same proper NK model structure as the baseline:
- Forward-looking IS curve
- New Keynesian Phillips Curve
- Taylor rule
- Sophisticated FIRE expectations (FIRE agents observe theta and solve forward)
- Heterogeneous agents (FIRE + BR)

The key innovation is that BR agents now choose among THREE rules,
allowing the model to generate both persistence (BL) and amplification (TF).

Uses the epsilon framework for complexity costs:
- epsilon_cb = 0 (CB rule is cognitively free)
- epsilon_bl = complexity cost for backward-looking
- epsilon_tf = complexity cost for trend-following (higher than BL)

Sophisticated FIRE Implementation:
- FIRE agents observe theta_t and solve forward taking its evolution into account
- Uses fixed-point iteration with adaptive dampening for numerical stability
- Falls back to best approximate solution if exact convergence not achieved
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

from fire_solution import FIRESolver
from parameters import get_default_params, validate_params
from three_arm_mab_learning import ThreeArmMABLearning


@dataclass
class ThreeArmSimulationResult:
    """Container for three-arm model simulation results."""
    pi: np.ndarray           # Inflation path
    y: np.ndarray            # Output gap path
    i: np.ndarray            # Interest rate path
    theta_cb: np.ndarray     # CB-anchored share
    theta_bl: np.ndarray     # Backward-looking share
    theta_tf: np.ndarray     # Trend-following share
    expectations: np.ndarray # Aggregate inflation expectations
    shocks: np.ndarray       # Shock path
    n_iterations: int = 0    # Number of iterations for sophisticated FIRE
    converged: bool = True   # Whether fixed-point iteration converged
    final_residual: float = 0.0  # Final residual if not converged

    @property
    def T(self) -> int:
        return len(self.pi)

    @property
    def theta(self) -> np.ndarray:
        """For compatibility: return CB share as 'credibility'."""
        return self.theta_cb


class ThreeArmFullModel:
    """
    Three-arm adaptive-hybrid NK model with sophisticated FIRE.

    This extends the full model to include a trend-following forecasting rule
    while maintaining the rigorous NK model structure (IS curve, NKPC, Taylor rule).

    FIRE agents are "sophisticated": they observe theta_t and solve forward
    taking its evolution into account. This requires solving a fixed-point
    problem via iteration.

    Parameters
    ----------
    params : dict, optional
        Model parameters. If None, uses default calibration.
        Additional three-arm parameters:
        - epsilon_cb, epsilon_bl, epsilon_tf: Complexity costs for each rule
    """

    def __init__(self, params: Optional[Dict[str, float]] = None):
        if not params:
            params = get_default_params()

        # Add defaults for three-arm specific parameters (epsilon framework)
        params.setdefault('lambda_fire', 0.35)
        params.setdefault('epsilon_cb', 0.0)      # CB rule is cognitively free
        params.setdefault('epsilon_bl', 1e-4)     # Same as two-arm epsilon
        params.setdefault('epsilon_tf', 2e-4)     # Higher cost for trend-following

        # Validate and store
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

        # Heterogeneity
        self.lam = params['lambda_fire']

        # Learning parameters
        self.k = params['k']
        self.eta = params['eta']

        # Epsilon framework costs
        self.epsilon_cb = params['epsilon_cb']
        self.epsilon_bl = params['epsilon_bl']
        self.epsilon_tf = params['epsilon_tf']

        # Shock persistence
        self.rho_u = params.get('rho_u', 0.5)
        self.rho_r = params.get('rho_r', 0.9)
        self.rho_v = params.get('rho_v', 0.0)

        # Initialize FIRE solver (for naive warm start)
        self.fire_solver = FIRESolver(params)

    def _create_mab(self) -> ThreeArmMABLearning:
        """Create a fresh MAB learner instance."""
        return ThreeArmMABLearning(
            k=self.k,
            eta=self.eta,
            pi_star=self.pi_star,
            epsilon_cb=self.epsilon_cb,
            epsilon_bl=self.epsilon_bl,
            epsilon_tf=self.epsilon_tf
        )

    def _solve_nk_system(
        self,
        E_pi: float,
        E_y: float,
        u_t: float,
        rn_t: float,
        v_t: float
    ) -> Tuple[float, float]:
        """
        Solve for current period outcomes given expectations.

        Uses the proper 2x2 NK system (NKPC + IS with Taylor rule).
        """
        A = np.array([
            [1, -self.kappa],
            [self.sigma * self.phi_pi, 1 + self.sigma * self.phi_y]
        ])

        B1 = (1 - self.beta) * self.pi_star + self.beta * E_pi + u_t
        B2 = (E_y
              - self.sigma * self.rn_bar
              - self.sigma * self.pi_star * (1 - self.phi_pi)
              - self.sigma * v_t
              + self.sigma * E_pi
              + self.sigma * rn_t)

        B = np.array([B1, B2])
        solution = np.linalg.solve(A, B)
        return solution[0], solution[1]

    def _get_br_expectation(
        self,
        theta_cb: float,
        theta_bl: float,
        pi_lag: float,
        pi_lag2: float
    ) -> float:
        """Compute aggregate BR expectation from three rules."""
        theta_tf = 1.0 - theta_cb - theta_bl
        E_cb = self.pi_star
        E_bl = pi_lag
        E_tf = 2 * pi_lag - pi_lag2  # pi_{t-1} + (pi_{t-1} - pi_{t-2})
        return theta_cb * E_cb + theta_bl * E_bl + theta_tf * E_tf

    def _forward_simulate(
        self,
        T: int,
        u: np.ndarray,
        rn: np.ndarray,
        v: np.ndarray,
        E_fire: np.ndarray,
        initial_theta_cb: float,
        initial_theta_bl: float,
        initial_pi: float,
        zlb: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward simulate given fixed FIRE expectations.

        Returns: (pi, y, i, theta_cb, theta_bl, theta_tf)
        """
        pi = np.zeros(T)
        y = np.zeros(T)
        i = np.zeros(T)
        theta_cb = np.zeros(T)
        theta_bl = np.zeros(T)
        theta_tf = np.zeros(T)

        pi[0] = initial_pi
        y[0] = 0.0
        i[0] = self.rn_bar + self.pi_star
        theta_cb[0] = initial_theta_cb
        theta_bl[0] = initial_theta_bl
        theta_tf[0] = 1.0 - initial_theta_cb - initial_theta_bl

        mab = self._create_mab()
        mab.add_observation(self.pi_star)
        mab.add_observation(initial_pi)

        for t in range(1, T):
            # BR expectations
            pi_lag = pi[t-1]
            pi_lag2 = pi[t-2] if t >= 2 else self.pi_star
            E_pi_br = self._get_br_expectation(
                theta_cb[t-1], theta_bl[t-1], pi_lag, pi_lag2
            )

            # Aggregate expectations with sophisticated FIRE
            E_pi = self.lam * E_fire[t] + (1 - self.lam) * E_pi_br
            E_y = 0.0  # Simplified

            # Solve NK system
            pi[t], y[t] = self._solve_nk_system(E_pi, E_y, u[t], rn[t], v[t])

            # Interest rate (with optional ZLB)
            i_taylor = (self.rn_bar + self.pi_star
                       + self.phi_pi * (pi[t] - self.pi_star)
                       + self.phi_y * y[t]
                       + v[t])

            if zlb is not None:
                i[t] = max(i_taylor, zlb)
                if i[t] > i_taylor:
                    real_rate_gap = (i[t] - E_pi) - (i_taylor - E_pi)
                    y[t] = y[t] - self.sigma * real_rate_gap
                    pi[t] = self.beta * E_pi + self.kappa * y[t] + u[t]
            else:
                i[t] = i_taylor

            # Update MAB learning
            mab.add_observation(pi[t])
            if self.lam < 1.0:
                theta_cb[t], theta_bl[t], theta_tf[t] = mab.update_theta(
                    theta_cb[t-1], theta_bl[t-1]
                )
            else:
                theta_cb[t] = 1.0
                theta_bl[t] = 0.0
                theta_tf[t] = 0.0

        return pi, y, i, theta_cb, theta_bl, theta_tf

    def _backward_solve_fire(
        self,
        T: int,
        u: np.ndarray,
        rn: np.ndarray,
        v: np.ndarray,
        theta_cb: np.ndarray,
        theta_bl: np.ndarray,
        pi: np.ndarray
    ) -> np.ndarray:
        """
        Backward solve for sophisticated FIRE expectations.

        Given theta and pi paths from forward simulation, compute
        FIRE expectations that account for BR behavior.
        """
        E_fire_pi = np.full(T, self.pi_star)

        # Backward recursion from terminal condition
        for t in range(T-3, 0, -1):
            theta_cb_tp1 = theta_cb[t+1]
            theta_bl_tp1 = theta_bl[t+1]
            pi_t = pi[t]
            pi_tm1 = pi[t-1] if t >= 1 else self.pi_star

            # BR expectation at t+1 about t+2
            E_pi_br = self._get_br_expectation(
                theta_cb_tp1, theta_bl_tp1, pi_t, pi_tm1
            )

            # FIRE expectation at t+1 (from recursion)
            E_fire_tp1 = E_fire_pi[t+1]

            # Aggregate expectation at t+1
            E_pi_agg = self.lam * E_fire_tp1 + (1 - self.lam) * E_pi_br
            E_y_agg = 0.0

            # Solve NK at t+1 to get what FIRE expects at t
            pi_tp1, _ = self._solve_nk_system(
                E_pi_agg, E_y_agg, u[t+1], rn[t+1], v[t+1]
            )

            # Bound to prevent numerical explosion
            max_pi = max(0.5, np.max(np.abs(u)) * 20 + self.pi_star)
            E_fire_pi[t] = np.clip(pi_tp1, -max_pi, max_pi)

        return E_fire_pi

    def simulate(
        self,
        T: int,
        shock_path: Optional[np.ndarray] = None,
        rn_path: Optional[np.ndarray] = None,
        v_path: Optional[np.ndarray] = None,
        rho_u: Optional[float] = None,
        initial_theta_cb: float = 1.0,
        initial_theta_bl: float = 0.0,
        initial_pi: Optional[float] = None,
        zlb: Optional[float] = None,
        max_iter: int = 500,
        tol: float = 1e-7,
        verbose: bool = False
    ) -> ThreeArmSimulationResult:
        """
        Simulate the three-arm model for T periods with sophisticated FIRE.

        Parameters
        ----------
        T : int
            Number of periods
        shock_path : np.ndarray, optional
            Cost-push shock path
        rn_path : np.ndarray, optional
            Natural rate path
        v_path : np.ndarray, optional
            Monetary policy shock path
        rho_u : float, optional
            Shock persistence (overrides params)
        initial_theta_cb : float
            Initial CB share (default 1.0 = fully anchored)
        initial_theta_bl : float
            Initial BL share (default 0.0)
        initial_pi : float, optional
            Initial inflation (default pi_star)
        zlb : float, optional
            Zero lower bound on nominal rate (None = no ZLB)
        max_iter : int
            Maximum iterations for sophisticated FIRE fixed-point
        tol : float
            Convergence tolerance
        verbose : bool
            Print iteration progress

        Returns
        -------
        ThreeArmSimulationResult
            Container with all simulation paths
        """
        rho = rho_u if rho_u is not None else self.rho_u
        pi_init = initial_pi if initial_pi is not None else self.pi_star

        # Build shock paths with AR(1)
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

        # Initialize E_fire with naive solution (warm start)
        shock_persistence = {'rho_u': rho, 'rho_r': self.rho_r, 'rho_v': self.rho_v}
        E_fire = np.zeros(T)
        for t in range(T):
            E_pi_naive, _ = self.fire_solver.solve_expectations(
                u[t], rn[t], v[t], shock_persistence
            )
            E_fire[t] = E_pi_naive

        # Handle pure FIRE case (no iteration needed)
        if self.lam >= 1.0:
            pi, y, i, theta_cb, theta_bl, theta_tf = self._forward_simulate(
                T, u, rn, v, E_fire, initial_theta_cb, initial_theta_bl, pi_init, zlb
            )
            expectations = E_fire.copy()
            return ThreeArmSimulationResult(
                pi=pi, y=y, i=i,
                theta_cb=theta_cb, theta_bl=theta_bl, theta_tf=theta_tf,
                expectations=expectations, shocks=u,
                n_iterations=0, converged=True, final_residual=0.0
            )

        # Sophisticated FIRE: fixed-point iteration with adaptive dampening
        dampening = 0.05  # Start slow for stability
        prev_diff = float('inf')
        best_E_fire = E_fire.copy()
        best_diff = float('inf')
        oscillation_count = 0
        converged = False

        for iteration in range(max_iter):
            # Forward simulate with current E_fire
            pi, y, i, theta_cb, theta_bl, theta_tf = self._forward_simulate(
                T, u, rn, v, E_fire, initial_theta_cb, initial_theta_bl, pi_init, zlb
            )

            # Backward solve for sophisticated FIRE expectations
            E_fire_new = self._backward_solve_fire(
                T, u, rn, v, theta_cb, theta_bl, pi
            )

            # Check convergence
            diff = np.max(np.abs(E_fire_new - E_fire))

            # Track best solution
            if diff < best_diff:
                best_diff = diff
                best_E_fire = E_fire.copy()
                oscillation_count = 0
            else:
                oscillation_count += 1

            if verbose and iteration % 20 == 0:
                print(f"  Iter {iteration}: diff = {diff:.2e}, damp = {dampening:.3f}")

            if diff < tol:
                converged = True
                if verbose:
                    print(f"  Converged at iteration {iteration}")
                break

            # Adaptive dampening for stability
            if diff > prev_diff:
                dampening *= 0.5
                dampening = max(dampening, 0.001)
            elif oscillation_count > 10:
                dampening *= 0.7
                dampening = max(dampening, 0.001)
            elif diff < prev_diff * 0.9:
                dampening = min(dampening * 1.1, 0.2)

            prev_diff = diff

            # Dampened update
            E_fire = dampening * E_fire_new + (1 - dampening) * E_fire

            # Reset to best if stuck
            if oscillation_count > 50:
                E_fire = best_E_fire.copy()
                dampening = 0.01
                oscillation_count = 0

        # Final simulation with best E_fire
        pi, y, i, theta_cb, theta_bl, theta_tf = self._forward_simulate(
            T, u, rn, v, best_E_fire, initial_theta_cb, initial_theta_bl, pi_init, zlb
        )

        # Compute aggregate expectations for output
        expectations = np.zeros(T)
        expectations[0] = self.pi_star
        for t in range(1, T):
            pi_lag = pi[t-1]
            pi_lag2 = pi[t-2] if t >= 2 else self.pi_star
            E_pi_br = self._get_br_expectation(theta_cb[t-1], theta_bl[t-1], pi_lag, pi_lag2)
            expectations[t] = self.lam * best_E_fire[t] + (1 - self.lam) * E_pi_br

        return ThreeArmSimulationResult(
            pi=pi, y=y, i=i,
            theta_cb=theta_cb, theta_bl=theta_bl, theta_tf=theta_tf,
            expectations=expectations, shocks=u,
            n_iterations=iteration + 1,
            converged=converged,
            final_residual=best_diff
        )


# =============================================================================
# Testing
# =============================================================================

def test_three_arm_full_model() -> None:
    """Test three-arm full model with sophisticated FIRE."""
    print("Testing Three-Arm Full Model (Sophisticated FIRE)")
    print("=" * 60)

    # Test 1: Calm period - should behave like 2-arm (TF dormant)
    print("\n--- Test 1: Calm Period (TF should be dormant) ---")
    params = get_default_params()
    params['lambda_fire'] = 0.35

    model = ThreeArmFullModel(params)

    np.random.seed(42)
    T = 40
    shocks = np.random.normal(0, 0.001, T)

    result = model.simulate(T, shock_path=shocks, verbose=False)

    print(f"Mean TF share: {result.theta_tf.mean():.6f}")
    print(f"Max TF share: {result.theta_tf.max():.6f}")
    print(f"Mean CB share: {result.theta_cb.mean():.4f}")
    print(f"Converged: {result.converged}, Iterations: {result.n_iterations}")

    # Test 2: Large persistent shock
    print("\n--- Test 2: Large Persistent Shock ---")
    params = get_default_params()
    params['lambda_fire'] = 0.35
    params['eta'] = 0.15

    model = ThreeArmFullModel(params)

    shocks = np.zeros(60)
    shocks[5] = 0.02  # 2% shock

    result = model.simulate(T=60, shock_path=shocks, rho_u=0.8, verbose=False)

    print(f"Pre-shock theta_cb: {result.theta_cb[4]:.4f}")
    print(f"Peak inflation: {result.pi.max()*400:.1f}% (annual)")
    print(f"Final theta: CB={result.theta_cb[-1]:.4f}, BL={result.theta_bl[-1]:.4f}, "
          f"TF={result.theta_tf[-1]:.4f}")
    print(f"Converged: {result.converged}, Residual: {result.final_residual:.2e}")

    # Test 3: Passive policy (should allow TF activation)
    print("\n--- Test 3: Passive Policy (phi_pi < 1) ---")
    params = get_default_params()
    params['lambda_fire'] = 0.35
    params['phi_pi'] = 0.3  # Fiscal dominance
    params['eta'] = 0.15

    model = ThreeArmFullModel(params)

    shocks = np.zeros(80)
    shocks[5:10] = 0.02  # 5 quarters of 2% shocks

    result = model.simulate(T=80, shock_path=shocks, rho_u=0.8,
                           initial_theta_cb=0.8, verbose=False)

    print(f"Peak inflation: {result.pi.max()*400:.1f}% (annual)")
    print(f"Max TF share: {result.theta_tf.max():.4f}")
    print(f"Final theta: CB={result.theta_cb[-1]:.4f}, BL={result.theta_bl[-1]:.4f}, "
          f"TF={result.theta_tf[-1]:.4f}")
    print(f"Converged: {result.converged}, Residual: {result.final_residual:.2e}")

    # Test 4: Brazil scenarios
    print("\n--- Test 4: Brazil Cruzado (TF-dominated start) ---")
    params = get_default_params()
    params['lambda_fire'] = 0.35
    params['phi_pi'] = 1.5
    params['eta'] = 0.10

    model = ThreeArmFullModel(params)

    shocks = np.zeros(80)
    shocks[5] = 0.03

    result = model.simulate(T=80, shock_path=shocks, rho_u=0.8,
                           initial_theta_cb=0.05, initial_theta_bl=0.15, verbose=False)

    print(f"Peak inflation: {result.pi.max()*400:.1f}% (annual)")
    print(f"Min theta_cb: {result.theta_cb.min():.4f}")
    print(f"Converged: {result.converged}, Residual: {result.final_residual:.2e}")

    print("\n--- Test 5: Brazil Real (High credibility start) ---")
    result = model.simulate(T=80, shock_path=shocks, rho_u=0.8,
                           initial_theta_cb=0.85, initial_theta_bl=0.10, verbose=False)

    print(f"Peak inflation: {result.pi.max()*400:.1f}% (annual)")
    print(f"Min theta_cb: {result.theta_cb.min():.4f}")
    print(f"Converged: {result.converged}, Residual: {result.final_residual:.2e}")

    print("\n" + "=" * 60)
    print("All tests completed!")


if __name__ == "__main__":
    test_three_arm_full_model()
