"""
Long Memory MAB Learning Module
===============================

Extends the baseline MAB learning to incorporate exponentially discounted
memory, allowing the model to capture how agents weight historical inflation
experience when evaluating forecasting rules.

Key modification from baseline MAB:
- Instead of fixed k-period window: L = sum_{j=0}^{k-1} (pi_{t-j} - E[pi])^2
- Uses exponential discounting: L = sum_{s=0}^{T} delta^s (pi_{t-s} - E[pi])^2

The discount factor delta controls memory persistence:
- delta = 0.8 gives half-life ≈ 3 quarters (9 months)
- delta = 0.9 gives half-life ≈ 7 quarters (21 months)
- delta = 0.95 gives half-life ≈ 14 quarters (3.5 years)

Applications:
1. Inflation trauma: Countries with hyperinflation history maintain
   de-anchored expectations long after stabilization
2. Below-target anchoring: Japan's 20+ years of low inflation creates
   persistent expectations of undershooting (symmetric de-anchoring)
3. Path dependence: Same current inflation, different credibility
   depending on how countries arrived at current state

References:
    - See Section 4 of main paper for theoretical foundations
    - Malmendier & Nagel (2016) for empirical evidence on experience effects
"""

import numpy as np
from typing import Tuple, Optional
from collections import deque


class LongMemoryMABLearning:
    """
    MAB learning with exponentially discounted memory.

    This class extends the baseline MAB to allow agents to weight
    historical inflation experience with exponential decay. More recent
    observations have higher weight, but distant past still matters.

    Attributes
    ----------
    delta : float
        Discount factor for exponential weighting (0 < delta < 1)
    eta : float
        Updating probability (Calvo parameter)
    epsilon : float
        Tie-breaking threshold for rule selection
    pi_star : float
        Central bank inflation target
    max_memory : int
        Maximum number of periods to store in history
    inflation_history : deque
        Buffer storing inflation observations
    L_CB : float
        Current discounted loss for CB-anchored rule
    L_BL : float
        Current discounted loss for backward-looking rule
    """

    def __init__(
        self,
        delta: float = 0.8,
        eta: float = 0.10,
        epsilon: float = 1e-4,
        pi_star: float = 0.005,
        max_memory: int = 200
    ):
        """
        Initialize long memory MAB learning algorithm.

        Parameters
        ----------
        delta : float, default=0.8
            Discount factor for exponential weighting.
            Controls how quickly past observations decay in importance.
            Half-life = ln(0.5) / ln(delta) ≈ 3 quarters for delta=0.8.
        eta : float, default=0.10
            Updating probability (Calvo parameter).
            Fraction of agents who reconsider their rule each period.
        epsilon : float, default=1e-4
            Tie-breaking threshold favoring CB rule when losses similar.
        pi_star : float, default=0.005
            Central bank inflation target (0.5% quarterly ≈ 2% annual).
        max_memory : int, default=200
            Maximum periods to store (200 quarters = 50 years).
            Beyond this, oldest observations are dropped.
        """
        if not 0 < delta < 1:
            raise ValueError(f"delta must be in (0, 1), got {delta}")

        self.delta = delta
        self.eta = eta
        self.epsilon = epsilon
        self.pi_star = pi_star
        self.max_memory = max_memory

        # Memory buffer for inflation history
        self.inflation_history = deque(maxlen=max_memory)

        # Current loss values (updated by compute_losses)
        self.L_CB = 0.0
        self.L_BL = 0.0

    @property
    def half_life(self) -> float:
        """
        Return the half-life of memory in quarters.

        Half-life is the number of periods for weight to decay to 0.5.
        """
        return np.log(0.5) / np.log(self.delta)

    def add_observation(self, pi_t: float) -> None:
        """
        Add new inflation observation to history.

        Parameters
        ----------
        pi_t : float
            Current period inflation rate
        """
        self.inflation_history.append(pi_t)

    def compute_losses(self) -> Tuple[float, float]:
        """
        Compute exponentially discounted forecast errors for both rules.

        Loss functions evaluate how well each rule would have predicted
        historical inflation, with exponential discounting:

        - L^CB_t = sum_{s=0}^{T} delta^s (pi_{t-s} - pi*)^2
          CB-anchored rule: forecast was always pi*

        - L^BL_t = sum_{s=0}^{T-1} delta^s (pi_{t-s} - pi_{t-s-1})^2
          Backward-looking rule: forecast was previous period's inflation

        Returns
        -------
        L_CB : float
            Discounted cumulative loss for CB-anchored rule
        L_BL : float
            Discounted cumulative loss for backward-looking rule
        """
        n_obs = len(self.inflation_history)

        if n_obs < 2:
            return 0.0, 0.0

        history = list(self.inflation_history)

        # CB-anchored losses: delta^s * (pi_{t-s} - pi*)^2
        L_CB = sum(
            self.delta ** s * (history[-(s + 1)] - self.pi_star) ** 2
            for s in range(n_obs)
        )

        # Backward-looking losses: delta^s * (pi_{t-s} - pi_{t-s-1})^2
        L_BL = sum(
            self.delta ** s * (history[-(s + 1)] - history[-(s + 2)]) ** 2
            for s in range(n_obs - 1)
        )

        # Store for diagnostics
        self.L_CB = L_CB
        self.L_BL = L_BL

        return L_CB, L_BL

    def update_theta(self, theta_t: float) -> float:
        """
        Update fraction of agents using CB-anchored rule.

        Decision rule with epsilon tie-breaking:
        - If L^CB_t <= L^BL_t + epsilon: Choose CB rule
        - Otherwise: Choose BL rule

        Calvo updating:
        theta_{t+1} = (1-eta)*theta_t + eta * 1[L^CB_t <= L^BL_t + epsilon]

        Parameters
        ----------
        theta_t : float
            Current fraction using CB-anchored rule

        Returns
        -------
        theta_t_plus_1 : float
            Updated fraction using CB-anchored rule
        """
        L_CB, L_BL = self.compute_losses()

        # Epsilon tie-breaking favors CB rule when performance is similar
        choose_CB = float(L_CB <= L_BL + self.epsilon)

        # Calvo updating
        theta_t_plus_1 = (1 - self.eta) * theta_t + self.eta * choose_CB

        # Ensure theta stays in valid probability range [0, 1]
        return np.clip(theta_t_plus_1, 0.0, 1.0)

    def get_loss_difference(self) -> float:
        """
        Return current loss difference (L_CB - L_BL).

        Useful for diagnostics and understanding learning dynamics.
        Negative values indicate CB rule is performing better.

        Returns
        -------
        diff : float
            L_CB - L_BL (negative means CB rule outperforming)
        """
        return self.L_CB - self.L_BL

    def reset(self) -> None:
        """Reset the learning algorithm (clear history and losses)."""
        self.inflation_history.clear()
        self.L_CB = 0.0
        self.L_BL = 0.0

    def prefill_history(self, inflation_values: list) -> None:
        """
        Pre-fill inflation history with given values.

        Useful for initializing with historical inflation experience
        (e.g., 20 years of hyperinflation, or 10 years of low inflation).

        Parameters
        ----------
        inflation_values : list
            Inflation values to add to history (oldest first)
        """
        for pi in inflation_values:
            self.add_observation(pi)


# =============================================================================
# Testing
# =============================================================================

def test_long_memory_learning():
    """
    Test long memory MAB learning with three scenarios.

    Scenario 1: Stable inflation near target
    - CB-anchored rule should outperform
    - Theta should increase toward 1

    Scenario 2: Distant hyperinflation trauma
    - Start with 5 years of high inflation, then stabilize
    - Initial de-anchoring should fade with delta weighting

    Scenario 3: Below-target history (Japan scenario)
    - 10 years of near-zero inflation
    - Tests symmetric de-anchoring
    """
    np.random.seed(42)  # For reproducibility

    print("Testing Long Memory MAB Learning")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Report half-life for different delta values
    # -------------------------------------------------------------------------
    print("\nDiscount factor half-lives:")
    for d in [0.8, 0.85, 0.9, 0.95]:
        mab_temp = LongMemoryMABLearning(delta=d)
        print(f"  delta = {d}: half-life = {mab_temp.half_life:.1f} quarters "
              f"({mab_temp.half_life/4:.1f} years)")

    # -------------------------------------------------------------------------
    # Scenario 1: Stable inflation near target
    # -------------------------------------------------------------------------
    print("\nScenario 1: Stable inflation near target")
    print("-" * 60)
    print("Expected: CB rule outperforms, theta → 1")

    mab = LongMemoryMABLearning(delta=0.8, eta=0.10, epsilon=1e-4, pi_star=0.005)
    theta = 0.5

    # Pre-fill with stable history
    for _ in range(20):
        mab.add_observation(0.005)

    # Continue with stable inflation
    for t in range(10):
        pi = 0.005 + np.random.normal(0, 0.0001)
        mab.add_observation(pi)
        theta = mab.update_theta(theta)

    print(f"Final theta: {theta:.4f} (expected: high, near 1)")

    # -------------------------------------------------------------------------
    # Scenario 2: Distant hyperinflation trauma
    # -------------------------------------------------------------------------
    print("\nScenario 2: Distant hyperinflation (5y hyper, then 10y stable)")
    print("-" * 60)
    print("Expected: Initial de-anchoring, gradual recovery")

    mab = LongMemoryMABLearning(delta=0.8, eta=0.10, epsilon=1e-4, pi_star=0.005)
    theta = 0.9

    # 5 years (20 quarters) of hyperinflation
    for _ in range(20):
        mab.add_observation(0.50)  # 50% quarterly

    theta_after_hyper = mab.update_theta(theta)
    print(f"Theta after hyperinflation: {theta_after_hyper:.4f}")

    # 10 years (40 quarters) of stable inflation
    for t in range(40):
        mab.add_observation(0.005)
        theta = mab.update_theta(theta)
        if t % 10 == 9:
            print(f"  After {(t+1)//4} years stable: theta = {theta:.4f}")

    print(f"Final theta (10y after hyper): {theta:.4f}")

    # -------------------------------------------------------------------------
    # Scenario 3: Below-target history (Japan)
    # -------------------------------------------------------------------------
    print("\nScenario 3: Below-target history (Japan-style)")
    print("-" * 60)
    print("Expected: De-anchoring below target")

    mab = LongMemoryMABLearning(delta=0.8, eta=0.10, epsilon=1e-4, pi_star=0.005)

    # Pre-fill 10 years of near-zero inflation
    for _ in range(40):
        mab.add_observation(0.0)

    theta = 0.9
    # Track theta evolution
    for t in range(20):
        mab.add_observation(0.0)
        theta = mab.update_theta(theta)

    print(f"Final theta (20y below target): {theta:.4f}")
    print(f"Loss difference (L_CB - L_BL): {mab.get_loss_difference():.6f}")

    print("\n" + "=" * 60)
    print("Test completed successfully")

    return mab


if __name__ == "__main__":
    test_long_memory_learning()
