"""
Model Parameters Module
=======================

Centralized parameter definitions for the Adaptive-Hybrid NK Model.
All calibrations are based on standard quarterly values from Galí (2015).

This module provides:
- Default parameter dictionaries
- Parameter validation
- Easy parameter overrides for sensitivity analysis

Parameter Conventions
---------------------
- All rates are QUARTERLY (not annualized)
- pi_star = 0.005 means 0.5% per quarter = 2% annual inflation target
- rn_bar = 0.005 means 0.5% per quarter = 2% annual real rate
- Shocks are in same units (e.g., 0.01 = 1 percentage point quarterly shock)
- Phillips curve slope kappa is per quarter

To convert to annual rates, multiply by 4.
"""

from typing import Dict, Any


# =============================================================================
# Default Calibrations
# =============================================================================

def get_structural_params() -> Dict[str, float]:
    """
    Structural parameters from Galí (2015) calibration.

    Returns
    -------
    dict
        Structural model parameters
    """
    return {
        'beta': 0.99,       # Discount factor (quarterly)
        'sigma': 1.0,       # Intertemporal elasticity of substitution
        'kappa': 0.024,     # Phillips curve slope
    }


def get_policy_params() -> Dict[str, float]:
    """
    Monetary policy (Taylor rule) parameters.

    Returns
    -------
    dict
        Policy parameters
    """
    return {
        'phi_pi': 1.5,      # Taylor rule: inflation response
        'phi_y': 0.125,     # Taylor rule: output gap response
        'pi_star': 0.005,    # Inflation target (2% annual, quarterly rate)
        'rn_bar': 0.005,     # Steady-state natural real rate (2% annual)
    }


def get_learning_params() -> Dict[str, Any]:
    """
    MAB learning algorithm parameters.

    Returns
    -------
    dict
        Learning parameters
    """
    return {
        'k': 3,             # Memory window (periods for loss evaluation)
        'eta': 0.10,        # Calvo updating probability
        'epsilon': 1e-4,    # Tie-breaking threshold (favor CB rule when losses similar)
        'gamma': None,      # Softmax inverse temperature (None = hard indicator)
    }


def get_heterogeneity_params() -> Dict[str, float]:
    """
    Agent heterogeneity parameters for the full model.

    Returns
    -------
    dict
        Heterogeneity parameters
    """
    return {
        'lambda_fire': 0.35,  # Fraction of FIRE agents (Coibion & Gorodnichenko 2015)
    }


def get_shock_params() -> Dict[str, float]:
    """
    Shock process parameters (AR(1) persistence).

    Returns
    -------
    dict
        Shock persistence parameters
    """
    return {
        'rho_u': 0.5,       # Cost-push shock persistence
        'rho_r': 0.9,       # Natural rate shock persistence
        'rho_v': 0.0,       # Policy shock persistence (i.i.d.)
    }


def get_default_params() -> Dict[str, float]:
    """
    Complete default parameter set for the model.

    Combines all parameter categories into a single dictionary.

    Returns
    -------
    dict
        Complete parameter dictionary
    """
    params = {}
    params.update(get_structural_params())
    params.update(get_policy_params())
    params.update(get_learning_params())
    params.update(get_heterogeneity_params())
    params.update(get_shock_params())
    return params


def get_toy_model_params() -> Dict[str, float]:
    """
    Parameter set for the Toy Model (Section 2).

    The toy model uses simplified learning dynamics:
    - k = 1: Single-period loss evaluation (no memory window)
    - epsilon = 1e-4: Tie-breaking threshold favoring CB rule

    Returns
    -------
    dict
        Parameter dictionary for toy model
    """
    params = get_default_params()
    # Toy model simplifications (see manuscript Section 2)
    params['k'] = 1            # Single-period evaluation
    params['epsilon'] = 0.0    # No tie-breaking (strict inequality)
    return params


# =============================================================================
# Parameter Configurations for Figures
# =============================================================================

def get_figure1_config() -> Dict[str, Any]:
    """
    Configuration for Figure 1: Toy model inflation dynamics.

    Two shock scenarios (transitory vs persistent) comparing different eta values
    vs FIRE benchmark. Uses baseline phi_pi=1.5.

    Returns
    -------
    dict
        Figure configuration with shock scenarios and eta values
    """
    return {
        # Simulation settings
        'T': 60,                    # Simulation horizon (quarters)
        'shock_period': 5,          # When shock hits

        # Learning speeds to compare
        'eta_values': [0.1, 0.2, 0.3],

        # Policy parameter - use baseline
        'phi_pi': 1.5,

        # Three-panel layout: transitory, persistent+FIRE, persistent zoomed
        # Key insight: persistence matters more than magnitude
        'scenarios': {
            'transitory': {
                'title': 'Transitory Shock',
                'shock_size': 0.01125,
                'rho_u': 0.0,
            },
            'persistent': {
                'title': 'Persistent Shock',
                'shock_size': 0.01125,
                'rho_u': 0.8,
            },
        },

        # Panel layout
        'panel_order': ['transitory', 'persistent', 'persistent_zoomed'],
    }


def get_figure3_config() -> Dict[str, Any]:
    """
    Configuration for Figure 3: Effect of FIRE fraction (lambda) on dynamics.

    2x2 grid showing inflation and credibility dynamics.
    Left: Single transitory shock - shows FIRE attenuates impact
    Right: Long persistent shock - shows MAB increases persistence

    Returns
    -------
    dict
        Figure configuration with shock scenarios and lambda values
    """
    return {
        # Simulation settings
        'T': 60,                    # Simulation horizon (quarters)
        'shock_period': 5,          # When shock hits

        # Policy parameter - use baseline
        'phi_pi': 1.5,

        # Learning speed (fixed)
        'eta': 0.1,

        # FIRE fractions to compare - simplified to 3 values
        'lambda_values': [0.0, 0.3, 1.0],

        # Two shock scenarios
        'scenarios': {
            'single_shock': {
                'title': 'Single Shock (0.5%, ρ=0.8)',
                'shock_size': 0.005,    # 0.5% shock
                'rho_u': 0.8,           # Persistent
                'duration': 1,          # Single period
            },
            'long_shock': {
                'title': 'Long Shock (0.75% × 12 quarters)',
                'shock_size': 0.0075,   # 0.75% shock per quarter
                'rho_u': 0.0,           # No persistence
                'duration': 12,         # 12 quarters
            },
        },

        # Panel layout: columns are scenarios
        'panel_order': ['single_shock', 'long_shock'],
    }


# =============================================================================
# Parameter Validation
# =============================================================================

def validate_params(params: Dict[str, float]) -> bool:
    """
    Validate parameter values for economic consistency.

    Parameters
    ----------
    params : dict
        Parameter dictionary to validate

    Returns
    -------
    bool
        True if all parameters are valid

    Raises
    ------
    ValueError
        If any parameter is invalid
    """
    # Discount factor must be in (0, 1)
    if not 0 < params.get('beta', 0.99) < 1:
        raise ValueError("beta must be in (0, 1)")

    # Elasticity of substitution must be positive
    if params.get('sigma', 1.0) <= 0:
        raise ValueError("sigma must be positive")

    # Phillips curve slope must be positive
    if params.get('kappa', 0.024) <= 0:
        raise ValueError("kappa must be positive")

    # Taylor principle: phi_pi should exceed 1 for determinacy
    if params.get('phi_pi', 1.5) <= 1:
        print("Warning: phi_pi <= 1 may violate Taylor principle")

    # Learning parameters
    k = params.get('k', 3)
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")

    if not 0 < params.get('eta', 0.1) <= 1:
        raise ValueError("eta must be in (0, 1]")

    if params.get('epsilon', 1e-5) < 0:
        raise ValueError("epsilon must be non-negative")

    # Heterogeneity parameters
    lambda_fire = params.get('lambda_fire', 0.35)
    if not 0 <= lambda_fire <= 1:
        raise ValueError("lambda_fire must be in [0, 1]")

    # Shock persistence
    for rho in ['rho_u', 'rho_r', 'rho_v']:
        if not 0 <= params.get(rho, 0) < 1:
            raise ValueError(f"{rho} must be in [0, 1)")

    return True


# =============================================================================
# Parameter Override Utilities
# =============================================================================

def override_params(base: Dict[str, float],
                    overrides: Dict[str, float]) -> Dict[str, float]:
    """
    Create new parameter dict with overrides applied.

    Parameters
    ----------
    base : dict
        Base parameter dictionary
    overrides : dict
        Parameters to override

    Returns
    -------
    dict
        New dictionary with overrides applied
    """
    params = base.copy()
    params.update(overrides)
    return params
