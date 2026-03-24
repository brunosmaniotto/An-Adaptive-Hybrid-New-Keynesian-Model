"""
FIRE Solution Module (Blanchard-Kahn Method)
=============================================

Solves the standard 3-equation New Keynesian model for rational expectations
using the method of undetermined coefficients (analytically equivalent to
Blanchard-Kahn for this class of linear models).

This provides the benchmark rational expectations that FIRE agents in the
heterogeneous model use for forecasting.

Model:
------
Phillips Curve: pi_t = beta * E_t[pi_{t+1}] + kappa * y_t + u_t
IS Curve:       y_t = E_t[y_{t+1}] - sigma * (i_t - E_t[pi_{t+1}] - r^n_t)
Taylor Rule:    i_t = rn_bar + pi* + phi_pi * (pi_t - pi*) + phi_y * y_t + v_t

Shocks (all AR(1)):
-------------------
u_t:   Cost-push shock with persistence rho_u
r^n_t: Natural rate shock with persistence rho_r
v_t:   Monetary policy shock with persistence rho_v

References:
-----------
Blanchard, O. J., & Kahn, C. M. (1980). The solution of linear difference
    models under rational expectations. Econometrica, 1305-1311.
Gali, J. (2015). Monetary policy, inflation, and the business cycle.
    Princeton University Press.
"""

import numpy as np
from typing import Dict, Tuple, Optional


class FIRESolver:
    """
    Solves the standard New Keynesian model for FIRE expectations
    using the method of undetermined coefficients.

    The solution takes the form (in deviations from steady state):
        pi_tilde_t = a_pi_u * u_t + a_pi_r * rn_tilde_t + a_pi_v * v_t
        y_t = a_y_u * u_t + a_y_r * rn_tilde_t + a_y_v * v_t

    where pi_tilde = pi - pi* and rn_tilde = rn - rn_bar.

    Attributes
    ----------
    beta : float
        Discount factor
    sigma : float
        Intertemporal elasticity of substitution
    kappa : float
        Phillips curve slope
    phi_pi : float
        Taylor rule inflation coefficient
    phi_y : float
        Taylor rule output coefficient
    pi_star : float
        Inflation target
    rn_bar : float
        Steady-state natural rate
    """

    def __init__(self, params: Dict[str, float]):
        """
        Initialize the FIRE solver with model parameters.

        Parameters
        ----------
        params : dict
            Model parameters including:
            - beta: Discount factor
            - sigma: Intertemporal elasticity of substitution
            - kappa: Phillips curve slope
            - phi_pi: Taylor rule inflation coefficient
            - phi_y: Taylor rule output coefficient
            - pi_star: Inflation target
            - rn_bar: Steady-state natural rate
        """
        self.beta = params['beta']
        self.sigma = params['sigma']
        self.kappa = params['kappa']
        self.phi_pi = params['phi_pi']
        self.phi_y = params['phi_y']
        self.pi_star = params['pi_star']
        self.rn_bar = params['rn_bar']

        # Cache for policy coefficients (computed on demand)
        self._policy_coeffs: Optional[Dict] = None
        self._cached_persistence: Optional[Dict] = None

    def _compute_policy_coefficients(
        self,
        shock_persistence: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute policy function coefficients using undetermined coefficients.

        For each shock with persistence rho, we solve the 2x2 system:
            A(rho) @ [a_pi, a_y]' = b

        where A(rho) = [[1 - beta*rho,      -kappa           ],
                        [sigma*(phi_pi-rho), 1 - rho + sigma*phi_y]]

        Parameters
        ----------
        shock_persistence : dict
            AR(1) coefficients {'rho_u': ..., 'rho_r': ..., 'rho_v': ...}

        Returns
        -------
        dict
            Policy function coefficients for each shock
        """
        rho_u = shock_persistence.get('rho_u', 0.0)
        rho_r = shock_persistence.get('rho_r', 0.0)
        rho_v = shock_persistence.get('rho_v', 0.0)

        def solve_for_shock(rho: float, b: np.ndarray) -> np.ndarray:
            """Solve 2x2 system for a single shock type."""
            A = np.array([
                [1 - self.beta * rho, -self.kappa],
                [self.sigma * (self.phi_pi - rho), 1 - rho + self.sigma * self.phi_y]
            ])

            det = np.linalg.det(A)
            if np.abs(det) < 1e-12:
                raise ValueError(
                    f"Singular system for rho={rho}. Check Taylor principle: "
                    f"phi_pi + (1-beta)/kappa * phi_y > 1"
                )

            return np.linalg.solve(A, b)

        # RHS vectors for each shock type
        # Cost-push shock enters Phillips curve with coefficient 1
        b_u = np.array([1.0, 0.0])
        # Natural rate shock enters IS curve with coefficient +sigma
        b_r = np.array([0.0, self.sigma])
        # Monetary policy shock enters IS curve with coefficient -sigma
        b_v = np.array([0.0, -self.sigma])

        # Solve for each shock
        coeffs_u = solve_for_shock(rho_u, b_u)
        coeffs_r = solve_for_shock(rho_r, b_r)
        coeffs_v = solve_for_shock(rho_v, b_v)

        return {
            'a_pi_u': coeffs_u[0], 'a_y_u': coeffs_u[1],
            'a_pi_r': coeffs_r[0], 'a_y_r': coeffs_r[1],
            'a_pi_v': coeffs_v[0], 'a_y_v': coeffs_v[1],
        }

    def solve_expectations(
        self,
        u_t: float,
        rn_t: float,
        v_t: float,
        shock_persistence: Dict[str, float]
    ) -> Tuple[float, float]:
        """
        Compute FIRE expectations E_t[pi_{t+1}] and E_t[y_{t+1}].

        Parameters
        ----------
        u_t : float
            Current cost-push shock
        rn_t : float
            Current natural rate (level, not deviation)
        v_t : float
            Current monetary policy shock
        shock_persistence : dict
            AR(1) coefficients {'rho_u': ..., 'rho_r': ..., 'rho_v': ...}

        Returns
        -------
        E_pi_next : float
            E_t[pi_{t+1}]
        E_y_next : float
            E_t[y_{t+1}]
        """
        # Recompute coefficients if persistence changed
        if (self._policy_coeffs is None or
                self._cached_persistence != shock_persistence):
            self._policy_coeffs = self._compute_policy_coefficients(shock_persistence)
            self._cached_persistence = shock_persistence.copy()

        c = self._policy_coeffs
        rho_u = shock_persistence.get('rho_u', 0.0)
        rho_r = shock_persistence.get('rho_r', 0.0)
        rho_v = shock_persistence.get('rho_v', 0.0)

        # Convert natural rate to deviation from steady state
        rn_tilde = rn_t - self.rn_bar

        # Expected shocks next period (AR(1) forecasts)
        E_u_next = rho_u * u_t
        E_rn_tilde_next = rho_r * rn_tilde
        E_v_next = rho_v * v_t

        # Expected inflation deviation next period
        E_pi_tilde_next = (c['a_pi_u'] * E_u_next +
                          c['a_pi_r'] * E_rn_tilde_next +
                          c['a_pi_v'] * E_v_next)

        # Expected output gap next period
        E_y_next = (c['a_y_u'] * E_u_next +
                    c['a_y_r'] * E_rn_tilde_next +
                    c['a_y_v'] * E_v_next)

        # Convert inflation back to level
        E_pi_next = E_pi_tilde_next + self.pi_star

        return E_pi_next, E_y_next

    def get_current_values(
        self,
        u_t: float,
        rn_t: float,
        v_t: float,
        shock_persistence: Dict[str, float]
    ) -> Tuple[float, float]:
        """
        Compute current period FIRE equilibrium values pi_t and y_t.

        Parameters
        ----------
        u_t : float
            Current cost-push shock
        rn_t : float
            Current natural rate (level)
        v_t : float
            Current monetary policy shock
        shock_persistence : dict
            AR(1) coefficients

        Returns
        -------
        pi_t : float
            Current inflation
        y_t : float
            Current output gap
        """
        # Recompute coefficients if needed
        if (self._policy_coeffs is None or
                self._cached_persistence != shock_persistence):
            self._policy_coeffs = self._compute_policy_coefficients(shock_persistence)
            self._cached_persistence = shock_persistence.copy()

        c = self._policy_coeffs
        rn_tilde = rn_t - self.rn_bar

        # Current values from policy functions
        pi_tilde_t = (c['a_pi_u'] * u_t +
                      c['a_pi_r'] * rn_tilde +
                      c['a_pi_v'] * v_t)

        y_t = (c['a_y_u'] * u_t +
               c['a_y_r'] * rn_tilde +
               c['a_y_v'] * v_t)

        pi_t = pi_tilde_t + self.pi_star

        return pi_t, y_t

    def check_determinacy(self) -> Tuple[bool, np.ndarray]:
        """
        Check Blanchard-Kahn determinacy conditions.

        For 2 jump variables (pi, y) and 0 predetermined states,
        we need exactly 2 unstable eigenvalues (|lambda| > 1).

        Returns
        -------
        is_determinate : bool
            True if model is determinate
        eigenvalues : ndarray
            Eigenvalues of the system
        """
        A = np.array([
            [self.beta, 0],
            [self.sigma, 1]
        ])
        B = np.array([
            [1, -self.kappa],
            [self.sigma * self.phi_pi, 1 + self.sigma * self.phi_y]
        ])

        A_inv_B = np.linalg.solve(A, B)
        eigenvalues = np.linalg.eigvals(A_inv_B)

        n_unstable = np.sum(np.abs(eigenvalues) > 1)
        n_jump = 2  # pi and y are both jump variables

        is_determinate = bool(n_unstable == n_jump)

        return is_determinate, eigenvalues


# =============================================================================
# Testing
# =============================================================================

def test_fire_solver():
    """Test the FIRE solver with basic scenarios."""
    print("Testing FIRE Solver (Blanchard-Kahn)")
    print("=" * 60)

    params = {
        'beta': 0.99,
        'sigma': 1.0,
        'kappa': 0.024,
        'phi_pi': 1.5,
        'phi_y': 0.125,
        'pi_star': 0.02,
        'rn_bar': 0.02
    }

    solver = FIRESolver(params)

    # Check determinacy
    print("\n--- Determinacy Check ---")
    is_det, eigenvalues = solver.check_determinacy()
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Moduli: {np.abs(eigenvalues)}")
    print(f"Model is {'DETERMINATE' if is_det else 'INDETERMINATE'}")

    shock_persistence = {'rho_u': 0.5, 'rho_r': 0.9, 'rho_v': 0.0}

    # Test 1: No shocks - should be at steady state
    print("\n--- Test 1: No shocks ---")
    E_pi, E_y = solver.solve_expectations(
        u_t=0, rn_t=params['rn_bar'], v_t=0,
        shock_persistence=shock_persistence
    )
    print(f"E[pi] = {E_pi:.6f} (expected: {params['pi_star']})")
    print(f"E[y]  = {E_y:.6f} (expected: 0.0)")
    assert np.isclose(E_pi, params['pi_star']), "E_pi should equal pi_star"
    assert np.isclose(E_y, 0.0, atol=1e-10), "E_y should equal 0"
    print("PASSED")

    # Test 2: Positive cost-push shock
    print("\n--- Test 2: Cost-push shock (u=0.01) ---")
    E_pi, E_y = solver.solve_expectations(
        u_t=0.01, rn_t=params['rn_bar'], v_t=0,
        shock_persistence=shock_persistence
    )
    print(f"E[pi] = {E_pi:.6f} (expected > {params['pi_star']})")
    print(f"E[y]  = {E_y:.6f} (expected < 0)")
    assert E_pi > params['pi_star'], "Cost-push should raise inflation"
    assert E_y < 0, "Cost-push should reduce output"
    print("PASSED")

    # Test 3: Current values
    print("\n--- Test 3: Current period values ---")
    pi_t, y_t = solver.get_current_values(
        u_t=0.01, rn_t=params['rn_bar'], v_t=0,
        shock_persistence=shock_persistence
    )
    print(f"pi_t = {pi_t:.6f}")
    print(f"y_t  = {y_t:.6f}")
    print("PASSED")

    print("\n" + "=" * 60)
    print("All tests passed!")

    return solver


if __name__ == "__main__":
    test_fire_solver()
