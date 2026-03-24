"""
Multi-Armed Bandit Learning Module
===================================

Implements the MAB learning algorithm for expectation formation in the
Adaptive-Hybrid New Keynesian Model.

Agents choose between two forecasting rules:
1. CB-anchored: E[pi_{t+1}] = pi* (central bank target)
2. Backward-looking: E[pi_{t+1}] = pi_{t-1} (naive extrapolation)

Selection is based on relative forecasting performance measured by
squared forecast errors over the last k periods.

Key features:
- Cumulative loss over k periods (default k=3)
- Epsilon tie-breaking favoring CB rule when performance similar
- Calvo-style updating: only fraction eta reconsiders each period

References:
    - See main paper for theoretical foundations
    - Brock & Hommes (1997) for related heterogeneous expectations models
"""

import numpy as np
from typing import Tuple, Optional
from collections import deque


class MABLearning:
    """
    Multi-Armed Bandit learning for expectation formation.

    This class implements the learning dynamics that determine how agents
    switch between CB-anchored and backward-looking forecasting rules
    based on their relative historical performance.

    Attributes
    ----------
    k : int
        Memory window (number of periods for loss calculation)
    eta : float
        Updating probability (Calvo parameter)
    epsilon : float
        Tie-breaking threshold for rule selection
    pi_star : float
        Central bank inflation target
    inflation_history : deque
        Buffer storing recent inflation observations
    L_CB : float
        Current cumulative loss for CB-anchored rule
    L_BL : float
        Current cumulative loss for backward-looking rule
    """

    def __init__(
        self,
        k: int = 3,
        eta: float = 0.10,
        epsilon: float = 1e-4,
        pi_star: float = 0.005,
        gamma: Optional[float] = None
    ):
        """
        Initialize MAB learning algorithm.

        Parameters
        ----------
        k : int, default=3
            Memory window (number of periods for loss calculation).
        eta : float, default=0.10
            Updating probability (Calvo parameter).
            Fraction of agents who reconsider their rule each period.
        epsilon : float, default=1e-4
            Tie-breaking threshold favoring CB rule when losses similar.
        pi_star : float, default=0.005
            Central bank inflation target (0.5% quarterly, equivalent to 2% annual).
        gamma : float, optional
            Inverse temperature for softmax (Boltzmann) switching. If None,
            uses hard indicator function (original model). If provided, uses
            softmax: P(CB) = 1 / (1 + exp(gamma * (L_CB - L_BL - epsilon))).
            Higher gamma values approach hard switching; lower values increase
            smoothing. Typical value for near-hard switching: 25000.
        """
        self.k = k
        self.eta = eta
        self.epsilon = epsilon
        self.pi_star = pi_star
        self.gamma = gamma

        # Memory buffer for inflation history
        # Need k+1 observations to compute k periods of backward-looking losses
        self.inflation_history = deque(maxlen=k + 1)

        # Current loss values (updated by compute_losses)
        self.L_CB = 0.0
        self.L_BL = 0.0

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
        Compute forecast errors for both rules over last k periods.

        Loss functions evaluate how well each rule would have predicted
        recent inflation:

        - L^CB_t = sum_{j=0}^{k-1} (pi_{t-j} - pi*)^2
          CB-anchored rule: forecast was always pi*

        - L^BL_t = sum_{j=0}^{k-1} (pi_{t-j} - pi_{t-j-1})^2
          Backward-looking rule: forecast was previous period's inflation

        Returns
        -------
        L_CB : float
            Cumulative squared forecast error for CB-anchored rule
        L_BL : float
            Cumulative squared forecast error for backward-looking rule
        """
        if len(self.inflation_history) < 2:
            # Not enough history to compute losses
            return 0.0, 0.0

        inflation_array = np.array(self.inflation_history)
        n_obs = len(inflation_array)

        # CB-anchored losses: (pi_t - pi*)^2
        # How badly did forecasting pi* miss the actual inflation?
        L_CB = 0.0
        for j in range(min(self.k, n_obs)):
            pi_t_minus_j = inflation_array[-(j + 1)]
            L_CB += (pi_t_minus_j - self.pi_star) ** 2

        # Backward-looking losses: (pi_t - pi_{t-1})^2
        # How badly did forecasting last period's inflation miss?
        L_BL = 0.0
        for j in range(min(self.k, n_obs - 1)):  # Need t and t-1
            pi_t_minus_j = inflation_array[-(j + 1)]
            pi_t_minus_j_minus_1 = inflation_array[-(j + 2)]
            L_BL += (pi_t_minus_j - pi_t_minus_j_minus_1) ** 2

        # Store for diagnostics
        self.L_CB = L_CB
        self.L_BL = L_BL

        return L_CB, L_BL

    def update_theta(self, theta_t: float) -> float:
        """
        Update fraction of agents using CB-anchored rule.

        Two modes depending on gamma parameter:

        1. Hard indicator (gamma=None, default):
           - If L^CB_t <= L^BL_t + epsilon: Choose CB rule
           - Otherwise: Choose BL rule
           - theta_{t+1} = (1-eta)*theta_t + eta * 1[L^CB_t <= L^BL_t + epsilon]

        2. Softmax/Boltzmann (gamma provided):
           - P(CB) = 1 / (1 + exp(gamma * (L_CB - L_BL - epsilon)))
           - theta_{t+1} = (1-eta)*theta_t + eta * P(CB)
           - Smooths the loss surface for optimization while preserving dynamics

        Parameters
        ----------
        theta_t : float
            Current fraction using CB-anchored rule

        Returns
        -------
        theta_t_plus_1 : float
            Updated fraction using CB-anchored rule
        """
        # Compute current losses for both rules
        L_CB, L_BL = self.compute_losses()

        if self.gamma is None:
            # Hard indicator (original model)
            # Epsilon tie-breaking favors CB rule when performance is similar
            prob_CB = float(L_CB <= L_BL + self.epsilon)
        else:
            # Softmax (Boltzmann) switching
            # Smooths transition while preserving tipping-point dynamics
            diff = L_CB - L_BL - self.epsilon
            # Clip exponent for numerical stability
            exponent = np.clip(self.gamma * diff, -50, 50)
            prob_CB = 1.0 / (1.0 + np.exp(exponent))

        # Calvo updating: fraction eta switch based on rule performance
        theta_t_plus_1 = (1 - self.eta) * theta_t + self.eta * prob_CB

        # Ensure theta stays in valid probability range [0, 1]
        theta_t_plus_1 = np.clip(theta_t_plus_1, 0.0, 1.0)

        return theta_t_plus_1

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


# =============================================================================
# Testing
# =============================================================================

def test_mab_learning():
    """
    Test MAB learning algorithm with two scenarios.

    Scenario 1: Stable inflation near target
    - CB-anchored rule should outperform
    - Theta should increase toward 1

    Scenario 2: Persistent high inflation
    - Backward-looking rule should outperform
    - Theta should decrease toward 0
    """
    print("Testing MAB Learning Algorithm")
    print("=" * 60)

    # Initialize with toy model defaults (pi_star=0.005 = 2% annual)
    mab = MABLearning(k=3, eta=0.10, epsilon=1e-4, pi_star=0.005)

    # -------------------------------------------------------------------------
    # Scenario 1: Stable inflation near target
    # -------------------------------------------------------------------------
    print("\nScenario 1: Stable inflation near target")
    print("-" * 60)
    print("Expected: CB rule outperforms, theta increases")

    theta = 0.5  # Start at 50-50
    inflation_path = [0.02, 0.021, 0.019, 0.020, 0.021, 0.02, 0.019]

    for t, pi in enumerate(inflation_path):
        mab.add_observation(pi)
        if t >= 1:  # Need at least 2 observations
            L_CB, L_BL = mab.compute_losses()
            theta = mab.update_theta(theta)
            print(f"t={t}: pi={pi:.4f}, L_CB={L_CB:.8f}, L_BL={L_BL:.8f}, "
                  f"theta={theta:.4f}")

    # -------------------------------------------------------------------------
    # Scenario 2: Persistent high inflation
    # -------------------------------------------------------------------------
    print("\nScenario 2: Persistent high inflation")
    print("-" * 60)
    print("Expected: BL rule outperforms, theta decreases")

    mab.reset()
    theta = 0.5
    inflation_path = [0.02, 0.03, 0.04, 0.045, 0.05, 0.055, 0.06]

    for t, pi in enumerate(inflation_path):
        mab.add_observation(pi)
        if t >= 1:
            L_CB, L_BL = mab.compute_losses()
            theta = mab.update_theta(theta)
            print(f"t={t}: pi={pi:.4f}, L_CB={L_CB:.8f}, L_BL={L_BL:.8f}, "
                  f"theta={theta:.4f}")

    print("\n" + "=" * 60)
    print("Test completed successfully")

    return mab


if __name__ == "__main__":
    test_mab_learning()
