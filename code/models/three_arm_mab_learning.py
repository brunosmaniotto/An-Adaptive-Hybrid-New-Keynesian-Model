"""
Three-Arm Multi-Armed Bandit Learning Module
=============================================

Extends the baseline two-arm MAB to include a third forecasting rule:
1. CB-anchored:      E[pi_{t+1}] = pi* (central bank target)
2. Backward-looking: E[pi_{t+1}] = pi_{t-1} (naive extrapolation)
3. Trend-following:  E[pi_{t+1}] = pi_{t-1} + (pi_{t-1} - pi_{t-2}) (linear extrapolation)

Uses consistent epsilon framework where each rule has a complexity cost:
- epsilon_cb = 0 (CB rule is cognitively free)
- epsilon_bl = simplicity cost for BL (same as two-arm epsilon)
- epsilon_tf = simplicity cost for TF (higher than BL)

Agents pick the rule with lowest adjusted loss: min(L_j + epsilon_j)

Key features:
- Cumulative loss over k periods (default k=3)
- Epsilon-based complexity costs creating natural hierarchy
- Calvo-style updating with speed eta
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import deque


class ThreeArmMABLearning:
    """
    Three-arm Multi-Armed Bandit learning for inflation expectations.

    Uses consistent epsilon framework where each rule has a complexity cost
    added to its loss. Agents pick the rule with lowest adjusted loss.

    Attributes
    ----------
    k : int
        Memory window (number of periods for loss calculation)
    eta : float
        Updating probability (Calvo parameter)
    pi_star : float
        Central bank inflation target
    epsilon_cb, epsilon_bl, epsilon_tf : float
        Complexity costs for each forecasting rule (epsilon_cb = 0 by design)
    inflation_history : deque
        Buffer storing recent inflation observations
    """

    def __init__(
        self,
        k: int = 3,
        eta: float = 0.10,
        pi_star: float = 0.005,
        epsilon_cb: float = 0.0,
        epsilon_bl: float = 1e-4,
        epsilon_tf: float = 2e-4
    ):
        """
        Initialize three-arm MAB learning algorithm.

        Parameters
        ----------
        k : int, default=3
            Memory window (periods for loss evaluation)
        eta : float, default=0.10
            Updating probability (Calvo parameter)
        pi_star : float, default=0.005
            Central bank inflation target (quarterly, ~2% annual)
        epsilon_cb : float, default=0.0
            Complexity cost for CB rule (should be 0 - anchoring is free)
        epsilon_bl : float, default=1e-4
            Complexity cost for backward-looking rule (same as two-arm epsilon)
        epsilon_tf : float, default=2e-4
            Complexity cost for trend-following rule (higher than BL)
        """
        self.k = k
        self.eta = eta
        self.pi_star = pi_star

        # Complexity costs (epsilon framework)
        self.epsilon_cb = epsilon_cb
        self.epsilon_bl = epsilon_bl
        self.epsilon_tf = epsilon_tf

        # Memory buffer: need k+2 observations for TF losses
        self.inflation_history = deque(maxlen=k + 2)

        # Current loss values (updated by compute_losses)
        self.L_CB = 0.0
        self.L_BL = 0.0
        self.L_TF = 0.0

    def reset(self) -> None:
        """Reset the inflation history buffer."""
        self.inflation_history.clear()
        self.L_CB = 0.0
        self.L_BL = 0.0
        self.L_TF = 0.0

    def add_observation(self, pi_t: float) -> None:
        """Add new inflation observation to history."""
        self.inflation_history.append(pi_t)

    def compute_losses(self) -> Tuple[float, float, float]:
        """
        Compute forecast errors for all three rules over last k periods.

        Loss functions evaluate how well each rule would have predicted
        recent inflation (squared errors only, complexity costs added later):

        - L^CB_t = sum_{j=0}^{k-1} (pi_{t-j} - pi*)^2
        - L^BL_t = sum_{j=0}^{k-1} (pi_{t-j} - pi_{t-j-1})^2
        - L^TF_t = sum_{j=0}^{k-1} (pi_{t-j} - [pi_{t-j-1} + (pi_{t-j-1} - pi_{t-j-2})])^2

        Returns
        -------
        L_CB, L_BL, L_TF : float
            Raw losses for each forecasting rule (before adding epsilon)
        """
        n_obs = len(self.inflation_history)

        if n_obs < 2:
            # Not enough history - return zeros
            return 0.0, 0.0, 0.0

        inflation_array = np.array(self.inflation_history)

        # CB-anchored losses: (pi_t - pi*)^2
        L_CB = 0.0
        for j in range(min(self.k, n_obs)):
            pi_t_minus_j = inflation_array[-(j + 1)]
            L_CB += (pi_t_minus_j - self.pi_star) ** 2

        # Backward-looking losses: (pi_t - pi_{t-1})^2
        L_BL = 0.0
        for j in range(min(self.k, n_obs - 1)):
            pi_t_minus_j = inflation_array[-(j + 1)]
            pi_t_minus_j_minus_1 = inflation_array[-(j + 2)]
            L_BL += (pi_t_minus_j - pi_t_minus_j_minus_1) ** 2

        # Trend-following losses: (pi_t - [pi_{t-1} + (pi_{t-1} - pi_{t-2})])^2
        L_TF = 0.0
        for j in range(min(self.k, n_obs - 2)):
            pi_t_minus_j = inflation_array[-(j + 1)]
            pi_t_minus_j_minus_1 = inflation_array[-(j + 2)]
            pi_t_minus_j_minus_2 = inflation_array[-(j + 3)]
            forecast_tf = pi_t_minus_j_minus_1 + (pi_t_minus_j_minus_1 - pi_t_minus_j_minus_2)
            L_TF += (pi_t_minus_j - forecast_tf) ** 2

        # Store for diagnostics
        self.L_CB = L_CB
        self.L_BL = L_BL
        self.L_TF = L_TF

        return L_CB, L_BL, L_TF

    def get_adjusted_losses(self) -> Tuple[float, float, float]:
        """
        Compute adjusted losses (raw loss + complexity cost).

        Returns
        -------
        adj_CB, adj_BL, adj_TF : float
            Adjusted losses for each rule
        """
        L_CB, L_BL, L_TF = self.compute_losses()

        # Add complexity costs (scaled by k for multi-period comparison)
        adj_CB = L_CB + self.k * self.epsilon_cb
        adj_BL = L_BL + self.k * self.epsilon_bl
        adj_TF = L_TF + self.k * self.epsilon_tf

        return adj_CB, adj_BL, adj_TF

    def get_winning_arm(self) -> int:
        """
        Determine which arm wins by finding minimum adjusted loss.

        Decision rule: Pick j* = argmin(L_j + k * epsilon_j)

        This is the key simplification: no separate tie-breaking logic,
        just pick the rule with lowest adjusted loss.

        Returns
        -------
        int
            Winning arm index: 0 = CB, 1 = BL, 2 = TF
        """
        adj_CB, adj_BL, adj_TF = self.get_adjusted_losses()
        losses = [adj_CB, adj_BL, adj_TF]
        return int(np.argmin(losses))

    def update_theta(
        self,
        theta_cb: float,
        theta_bl: float
    ) -> Tuple[float, float, float]:
        """
        Update arm shares based on winner-take-all with Calvo adjustment.

        The three theta values represent the fraction of BR agents using
        each forecasting rule. They must sum to 1.

        Calvo updating:
            theta_{t+1} = (1-eta)*theta_t + eta * target

        where target is [1,0,0], [0,1,0], or [0,0,1] depending on winner.

        Parameters
        ----------
        theta_cb : float
            Current fraction using CB-anchored rule
        theta_bl : float
            Current fraction using backward-looking rule
            (theta_tf = 1 - theta_cb - theta_bl implied)

        Returns
        -------
        theta_cb_new, theta_bl_new, theta_tf_new : float
            Updated fractions (sum to 1)
        """
        theta_tf = 1.0 - theta_cb - theta_bl

        winner = self.get_winning_arm()

        # Target shares (winner-take-all)
        target = np.array([0.0, 0.0, 0.0])
        target[winner] = 1.0

        # Current shares
        current = np.array([theta_cb, theta_bl, theta_tf])

        # Calvo updating
        new_shares = (1 - self.eta) * current + self.eta * target

        # Ensure they sum to 1 (numerical safety)
        new_shares = new_shares / new_shares.sum()

        return float(new_shares[0]), float(new_shares[1]), float(new_shares[2])

    def get_aggregate_expectation(
        self,
        theta_cb: float,
        theta_bl: float,
        pi_lag: float,
        pi_lag2: Optional[float] = None
    ) -> float:
        """
        Compute aggregate BR expectation as weighted average of rules.

        E^BR[pi] = theta_cb * pi* + theta_bl * pi_{t-1} + theta_tf * [pi_{t-1} + (pi_{t-1} - pi_{t-2})]

        Parameters
        ----------
        theta_cb, theta_bl : float
            Shares using CB and BL rules (theta_tf = 1 - theta_cb - theta_bl)
        pi_lag : float
            Last period's inflation (pi_{t-1})
        pi_lag2 : float, optional
            Two periods ago inflation (pi_{t-2}). If None, uses pi_lag.

        Returns
        -------
        E_pi_br : float
            Aggregate BR inflation expectation
        """
        theta_tf = 1.0 - theta_cb - theta_bl

        # Individual rule forecasts
        E_cb = self.pi_star
        E_bl = pi_lag
        # TF forecast: pi_{t-1} + (pi_{t-1} - pi_{t-2}) = 2*pi_{t-1} - pi_{t-2}
        E_tf = pi_lag + (pi_lag - (pi_lag2 if pi_lag2 is not None else pi_lag))

        # Weighted average
        return theta_cb * E_cb + theta_bl * E_bl + theta_tf * E_tf

    def get_diagnostics(self) -> Dict[str, float]:
        """Return diagnostic information about current state."""
        adj_CB, adj_BL, adj_TF = self.get_adjusted_losses()

        return {
            'L_CB_raw': self.L_CB,
            'L_BL_raw': self.L_BL,
            'L_TF_raw': self.L_TF,
            'L_CB_adj': adj_CB,
            'L_BL_adj': adj_BL,
            'L_TF_adj': adj_TF,
            'epsilon_cb': self.epsilon_cb,
            'epsilon_bl': self.epsilon_bl,
            'epsilon_tf': self.epsilon_tf,
            'winner': self.get_winning_arm()
        }

    def get_loss_differences(self) -> Tuple[float, float]:
        """
        Return loss differences for diagnostics.

        Returns
        -------
        diff_cb_bl : float
            L_CB - L_BL (negative means CB better than BL)
        diff_bl_tf : float
            L_BL - L_TF (negative means BL better than TF)
        """
        return (self.L_CB - self.L_BL, self.L_BL - self.L_TF)


# =============================================================================
# Testing
# =============================================================================

def test_three_arm_mab() -> None:
    """Test three-arm MAB learning with various scenarios."""
    print("Testing Three-Arm MAB Learning (Epsilon Framework)")
    print("=" * 60)

    mab = ThreeArmMABLearning(
        k=3,
        eta=0.10,
        pi_star=0.005,
        epsilon_cb=0.0,
        epsilon_bl=1e-4,
        epsilon_tf=2e-4
    )

    print(f"Parameters: epsilon_cb={mab.epsilon_cb}, epsilon_bl={mab.epsilon_bl}, "
          f"epsilon_tf={mab.epsilon_tf}")

    # Scenario 1: Stable inflation - CB should win
    print("\nScenario 1: Stable inflation near target")
    print("-" * 60)

    theta_cb, theta_bl = 0.5, 0.3
    inflation_path = [0.005, 0.0051, 0.0049, 0.0050, 0.0051, 0.0050]

    for t, pi in enumerate(inflation_path):
        mab.add_observation(pi)
        if t >= 2:
            diag = mab.get_diagnostics()
            theta_cb, theta_bl, theta_tf = mab.update_theta(theta_cb, theta_bl)
            print(f"t={t}: pi={pi*400:.2f}%, winner={['CB','BL','TF'][diag['winner']]}, "
                  f"theta=[{theta_cb:.3f},{theta_bl:.3f},{theta_tf:.3f}]")

    # Scenario 2: Trending inflation - TF should become attractive
    print("\nScenario 2: Trending inflation (rising)")
    print("-" * 60)

    mab.reset()
    theta_cb, theta_bl = 0.5, 0.3
    inflation_path = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035]

    for t, pi in enumerate(inflation_path):
        mab.add_observation(pi)
        if t >= 2:
            diag = mab.get_diagnostics()
            theta_cb, theta_bl, theta_tf = mab.update_theta(theta_cb, theta_bl)
            print(f"t={t}: pi={pi*400:.1f}%, L_adj=[{diag['L_CB_adj']:.6f},"
                  f"{diag['L_BL_adj']:.6f},{diag['L_TF_adj']:.6f}], "
                  f"winner={['CB','BL','TF'][diag['winner']]}")

    print("\n" + "=" * 60)
    print("Tests completed!")


if __name__ == "__main__":
    test_three_arm_mab()
