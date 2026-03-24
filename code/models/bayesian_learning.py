"""
Bayesian Learning Module
========================

Alternative microfoundation for expectation formation using Beta-Bernoulli
updating. Provides robustness check showing that MAB and Bayesian approaches
generate qualitatively similar aggregate dynamics.

Key idea: Agents maintain Beta(α, β) prior over weight w on CB target.
Each period, they observe which forecast performed better and update accordingly.
The posterior mean w_t = α_t / (α_t + β_t) is analogous to θ_t in MAB.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class BayesianLearnerState:
    """State of the Bayesian learner."""
    alpha: float  # Beta distribution parameter (CB target "successes")
    beta: float   # Beta distribution parameter (backward-looking "successes")

    @property
    def weight(self) -> float:
        """Posterior mean weight on CB target (analogous to θ in MAB)."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def confidence(self) -> float:
        """Total evidence accumulated (higher = more confident, slower updates)."""
        return self.alpha + self.beta

    @property
    def variance(self) -> float:
        """Posterior variance - measure of uncertainty."""
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b)**2 * (a + b + 1))


class BayesianLearner:
    """
    Bayesian learning agent using Beta-Bernoulli updating.

    Each period:
    1. Agent forms expectation as weighted average: E[π] = w * π* + (1-w) * π_{t-1}
    2. Observes realized inflation π_t
    3. Computes squared forecast errors for each source
    4. Updates Beta prior: increment α if CB won, increment β if backward-looking won

    Parameters
    ----------
    alpha_0 : float
        Initial α parameter (prior "successes" for CB target)
    beta_0 : float
        Initial β parameter (prior "successes" for backward-looking)
    pi_star : float
        Central bank inflation target
    """

    def __init__(
        self,
        alpha_0: float = 10.0,
        beta_0: float = 10.0,
        pi_star: float = 0.005
    ):
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.pi_star = pi_star

        # Current state
        self.alpha = alpha_0
        self.beta = beta_0

        # History for analysis
        self.history = []

    def reset(self):
        """Reset to initial prior."""
        self.alpha = self.alpha_0
        self.beta = self.beta_0
        self.history = []

    def get_state(self) -> BayesianLearnerState:
        """Get current state."""
        return BayesianLearnerState(alpha=self.alpha, beta=self.beta)

    @property
    def weight(self) -> float:
        """Current posterior mean weight on CB target."""
        return self.alpha / (self.alpha + self.beta)

    def form_expectation(self, pi_lag: float) -> float:
        """
        Form inflation expectation as weighted average.

        Parameters
        ----------
        pi_lag : float
            Lagged inflation (π_{t-1})

        Returns
        -------
        E_pi : float
            Expected inflation
        """
        w = self.weight
        return w * self.pi_star + (1 - w) * pi_lag

    def update(self, pi_realized: float, pi_lag: float) -> float:
        """
        Update beliefs based on observed inflation.

        Uses Beta-Bernoulli conjugate updating:
        - If CB target forecast closer to realized: increment α
        - If backward-looking forecast closer: increment β

        Parameters
        ----------
        pi_realized : float
            Realized inflation this period
        pi_lag : float
            Lagged inflation (what backward-looking rule would predict)

        Returns
        -------
        new_weight : float
            Updated posterior mean weight
        """
        # Compute squared forecast errors
        L_CB = (pi_realized - self.pi_star) ** 2
        L_BL = (pi_realized - pi_lag) ** 2

        # Store pre-update state
        self.history.append({
            'alpha': self.alpha,
            'beta': self.beta,
            'weight': self.weight,
            'L_CB': L_CB,
            'L_BL': L_BL,
            'CB_wins': L_CB < L_BL
        })

        # Beta-Bernoulli update with epsilon tie-breaking (matching MAB)
        # When losses are similar, favor CB (entropy/parsimony argument)
        epsilon = 1e-4
        if L_CB <= L_BL + epsilon:
            # CB target performed better (or tie)
            self.alpha += 1
        else:
            # Backward-looking performed better
            self.beta += 1

        return self.weight


class BayesianToyModel:
    """
    Toy model with Bayesian learning (analogous to ToyModel with MAB).

    This is a simplified version for comparison purposes.
    """

    def __init__(self, params: dict):
        """
        Initialize model with parameters.

        Parameters
        ----------
        params : dict
            Model parameters including:
            - pi_star: inflation target
            - kappa: Phillips curve slope
            - sigma: IS curve slope
            - phi_pi: Taylor rule inflation coefficient
            - phi_y: Taylor rule output coefficient
            - rn_bar: natural real rate
            - alpha_0: initial Beta α
            - beta_0: initial Beta β
        """
        self.pi_star = params.get('pi_star', 0.005)
        self.beta = params.get('beta', 0.99)
        self.kappa = params.get('kappa', 0.024)
        self.sigma = params.get('sigma', 1.0)
        self.phi_pi = params.get('phi_pi', 1.5)
        self.phi_y = params.get('phi_y', 0.125)
        self.rn_bar = params.get('rn_bar', 0.005)

        # Bayesian learner
        alpha_0 = params.get('alpha_0', 10.0)
        beta_0 = params.get('beta_0', 10.0)
        self.learner = BayesianLearner(
            alpha_0=alpha_0,
            beta_0=beta_0,
            pi_star=self.pi_star
        )

    def _solve_period(
        self,
        E_pi: float,
        E_y: float,
        u_t: float,
        rn_t: float
    ) -> Tuple[float, float]:
        """
        Solve for current period inflation and output given expectations.

        Returns (pi_t, y_t)
        """
        # Solve via fixed-point iteration
        # From IS: y = E_y - sigma * (i - E_pi - rn)
        # From PC: pi = E_pi + kappa * y + u
        # From Taylor: i = rn + pi* + phi_pi * (pi - pi*) + phi_y * y
        y_t = 0.0
        for _ in range(50):
            pi_t = (1 - self.beta) * self.pi_star + self.beta * E_pi + self.kappa * y_t + u_t
            i_t = self.rn_bar + self.pi_star + self.phi_pi * (pi_t - self.pi_star) + self.phi_y * y_t
            y_new = E_y - self.sigma * (i_t - E_pi - rn_t)
            if abs(y_new - y_t) < 1e-10:
                break
            y_t = 0.5 * y_t + 0.5 * y_new

        pi_t = (1 - self.beta) * self.pi_star + self.beta * E_pi + self.kappa * y_t + u_t

        return pi_t, y_t

    def simulate(
        self,
        T: int,
        shock_path: np.ndarray,
        rho_u: float = 0.5,
        initial_weight: Optional[float] = None
    ) -> dict:
        """
        Simulate the model.

        Parameters
        ----------
        T : int
            Number of periods
        shock_path : np.ndarray
            Cost-push shock innovations
        rho_u : float
            Shock persistence
        initial_weight : float, optional
            If provided, set initial α, β to achieve this weight

        Returns
        -------
        result : dict
            Simulation results with pi, y, i, weight arrays
        """
        # Initialize arrays
        pi = np.zeros(T)
        y = np.zeros(T)
        i = np.zeros(T)
        weight = np.zeros(T)
        alpha = np.zeros(T)
        beta = np.zeros(T)

        # Reset learner
        self.learner.reset()

        # Set initial weight if specified
        if initial_weight is not None:
            # Set α, β to achieve desired weight with moderate confidence
            total = self.learner.alpha + self.learner.beta
            self.learner.alpha = initial_weight * total
            self.learner.beta = (1 - initial_weight) * total

        # Build shock path
        u = np.zeros(T)
        u[:len(shock_path)] = shock_path[:T]
        for t in range(1, T):
            if t >= len(shock_path) or shock_path[t] == 0:
                u[t] = rho_u * u[t-1]

        # Initial conditions
        pi[0] = self.pi_star
        y[0] = 0.0
        i[0] = self.rn_bar + self.pi_star
        weight[0] = self.learner.weight
        alpha[0] = self.learner.alpha
        beta[0] = self.learner.beta

        # Simulate
        for t in range(1, T):
            # Form expectation
            E_pi = self.learner.form_expectation(pi[t-1])
            E_y = 0.0

            # Solve for current period
            pi[t], y[t] = self._solve_period(E_pi, E_y, u[t], self.rn_bar)

            # Interest rate
            i[t] = (self.rn_bar + self.pi_star
                    + self.phi_pi * (pi[t] - self.pi_star)
                    + self.phi_y * y[t])

            # Update beliefs
            self.learner.update(pi[t], pi[t-1])

            # Store state
            weight[t] = self.learner.weight
            alpha[t] = self.learner.alpha
            beta[t] = self.learner.beta

        return {
            'pi': pi,
            'y': y,
            'i': i,
            'weight': weight,
            'alpha': alpha,
            'beta': beta,
            'u': u
        }


def calibrate_initial_prior(
    target_weight: float,
    confidence: float = 20.0
) -> Tuple[float, float]:
    """
    Calibrate initial (α₀, β₀) to achieve target weight with given confidence.

    Parameters
    ----------
    target_weight : float
        Desired initial w₀ = α₀/(α₀+β₀)
    confidence : float
        Total prior strength α₀ + β₀ (higher = slower learning)

    Returns
    -------
    alpha_0, beta_0 : tuple
        Initial Beta parameters
    """
    alpha_0 = target_weight * confidence
    beta_0 = (1 - target_weight) * confidence
    return alpha_0, beta_0
