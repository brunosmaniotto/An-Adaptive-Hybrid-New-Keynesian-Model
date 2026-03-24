"""
Plotting Utilities
==================

Professional black-and-white plotting utilities for academic papers.
Uses seaborn for consistent styling with publication-quality output.

Design principles:
- Grayscale-friendly (prints well in B&W)
- Clean, minimal aesthetic
- Consistent sizing for journal submission
- Distinguishable line styles without relying on color
"""

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Optional
from pathlib import Path


# =============================================================================
# Style Configuration
# =============================================================================

# Line styles for distinguishing series
LINE_STYLES = [
    {'linestyle': '-', 'linewidth': 1.5},      # Solid
    {'linestyle': '--', 'linewidth': 1.5},     # Dashed
    {'linestyle': '-.', 'linewidth': 1.5},     # Dash-dot
    {'linestyle': ':', 'linewidth': 2.0},      # Dotted (thicker for visibility)
]

# Basic colors for clarity
COLORS = ['black', 'blue', 'red', 'green', 'orange']

# Figure sizes (inches) - common journal formats
FIGURE_SIZES = {
    'single_column': (3.5, 2.8),      # Single column width
    'double_column': (7.0, 5.5),      # Full page width
    'square': (5.0, 5.0),             # Square figure
    '2x2_grid': (8.0, 7.0),           # 2x2 panel figure (larger for clarity)
}


def setup_style():
    """
    Configure matplotlib/seaborn for publication-quality B&W figures.

    Call this once at the start of any plotting script.
    """
    # Use seaborn's white style as base
    sns.set_style("whitegrid", {
        'grid.linestyle': ':',
        'grid.alpha': 0.5,
        'axes.edgecolor': '0.2',
        'axes.linewidth': 0.8,
    })

    # Set context for paper (larger fonts than default)
    sns.set_context("paper", font_scale=1.1)

    # Additional matplotlib settings
    plt.rcParams.update({
        # Font settings
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,

        # Figure settings
        'figure.dpi': 300,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,

        # Line settings
        'lines.linewidth': 1.5,

        # Legend settings
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',

        # Axis settings
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def get_line_style(index: int) -> dict:
    """
    Get line style for a given series index.

    Parameters
    ----------
    index : int
        Series index (0, 1, 2, ...)

    Returns
    -------
    dict
        Keyword arguments for plt.plot()
    """
    style = LINE_STYLES[index % len(LINE_STYLES)].copy()
    style['color'] = COLORS[index % len(COLORS)]
    return style


# =============================================================================
# Figure Creation
# =============================================================================

def create_figure(size: str = 'double_column',
                  nrows: int = 1,
                  ncols: int = 1,
                  **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a figure with consistent sizing.

    Parameters
    ----------
    size : str
        Figure size preset ('single_column', 'double_column', '2x2_grid')
    nrows, ncols : int
        Subplot grid dimensions
    **kwargs
        Additional arguments passed to plt.subplots()

    Returns
    -------
    fig : Figure
        Matplotlib figure
    axes : Axes or array of Axes
        Subplot axes
    """
    figsize = FIGURE_SIZES.get(size, FIGURE_SIZES['double_column'])

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        **kwargs
    )

    return fig, axes


def add_panel_labels(axes: List[plt.Axes],
                     labels: Optional[List[str]] = None,
                     loc: str = 'upper left'):
    """
    Add panel labels (a), (b), (c), etc. to subplots.

    Parameters
    ----------
    axes : list of Axes
        Subplot axes (flattened)
    labels : list of str, optional
        Custom labels. If None, uses (a), (b), (c), ...
    loc : str
        Label location ('upper left', 'upper right', etc.)
    """
    if labels is None:
        labels = [f'({chr(97+i)})' for i in range(len(axes))]

    # Location mapping
    loc_map = {
        'upper left': (0.02, 0.98),
        'upper right': (0.98, 0.98),
        'lower left': (0.02, 0.02),
        'lower right': (0.98, 0.02),
    }

    x, y = loc_map.get(loc, (0.02, 0.98))
    ha = 'left' if 'left' in loc else 'right'
    va = 'top' if 'upper' in loc else 'bottom'

    for ax, label in zip(axes, labels):
        ax.text(x, y, label,
                transform=ax.transAxes,
                fontsize=11,
                fontweight='bold',
                va=va, ha=ha)


# =============================================================================
# Common Plot Elements
# =============================================================================

def add_target_line(ax: plt.Axes,
                    target: float,
                    label: str = r'$\pi^*$',
                    **kwargs):
    """
    Add a horizontal line indicating the inflation target.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes
    target : float
        Target value
    label : str
        Label for legend
    **kwargs
        Additional line properties
    """
    defaults = {
        'color': '0.5',
        'linestyle': '--',
        'linewidth': 1.0,
        'alpha': 0.7,
        'zorder': 0,
    }
    defaults.update(kwargs)

    ax.axhline(target, label=label, **defaults)


def add_shock_indicator(ax: plt.Axes,
                        shock_period: int,
                        label: str = 'Shock',
                        **kwargs):
    """
    Add a vertical line indicating when a shock occurs.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes
    shock_period : int
        Period when shock occurs
    label : str
        Label for legend
    **kwargs
        Additional line properties
    """
    defaults = {
        'color': '0.7',
        'linestyle': ':',
        'linewidth': 1.0,
        'alpha': 0.7,
        'zorder': 0,
    }
    defaults.update(kwargs)

    ax.axvline(shock_period, label=label, **defaults)


def format_percent_axis(ax: plt.Axes,
                        axis: str = 'y',
                        decimals: int = 1,
                        annualize: bool = False):
    """
    Format axis labels as percentages.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes
    axis : str
        'x' or 'y'
    decimals : int
        Number of decimal places
    annualize : bool
        If True, multiply by 400 (for quarterly to annual conversion).
        If False, multiply by 100 (standard percentage).
    """
    from matplotlib.ticker import FuncFormatter

    multiplier = 400 if annualize else 100

    def pct_formatter(x, pos):
        return f'{x*multiplier:.{decimals}f}%'

    if axis == 'y':
        ax.yaxis.set_major_formatter(FuncFormatter(pct_formatter))
    else:
        ax.xaxis.set_major_formatter(FuncFormatter(pct_formatter))


# =============================================================================
# Save Utilities
# =============================================================================

def save_figure(fig: plt.Figure,
                filename: str,
                output_dir: Optional[Path] = None,
                formats: List[str] = ['pdf', 'png']):
    """
    Save figure in multiple formats.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure
    filename : str
        Base filename (without extension)
    output_dir : Path, optional
        Output directory. If None, uses current directory.
    formats : list of str
        File formats to save (e.g., ['pdf', 'png'])
    """
    if output_dir is None:
        output_dir = Path('.')
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        filepath = output_dir / f"{filename}.{fmt}"
        fig.savefig(filepath, format=fmt)
        print(f"Saved: {filepath}")


# =============================================================================
# Initialize on import
# =============================================================================

# Apply style settings when module is imported
setup_style()
