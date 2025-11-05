"""
Plotting utilities for DarkBottomLine framework.
"""

import matplotlib.pyplot as plt
try:
    import mplhep as hep
    _HAS_MPLHEP = True
except Exception:
    _HAS_MPLHEP = False
import matplotlib.patches as mpatches
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging


class CMSPlotStyle:
    """
    CMS-style plotting configuration.
    """

    def __init__(self):
        """Initialize CMS plot style."""
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }

        self.font_sizes = {
            'title': 16,
            'axis_label': 14,
            'tick_label': 12,
            'legend': 12,
            'annotation': 10
        }

        self.line_styles = {
            'solid': '-',
            'dashed': '--',
            'dotted': ':',
            'dashdot': '-.'
        }

        self.markers = {
            'circle': 'o',
            'square': 's',
            'triangle': '^',
            'diamond': 'D',
            'star': '*',
            'plus': '+',
            'cross': 'x'
        }

    def set_style(self):
        """Set matplotlib style to CMS-like appearance."""
        if _HAS_MPLHEP:
            plt.style.use(hep.style.CMS)
        else:
            plt.style.use('default')

        # Set font
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
        plt.rcParams['font.size'] = self.font_sizes['tick_label']

        # Set figure properties
        plt.rcParams['figure.figsize'] = (10, 8)
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['savefig.bbox'] = 'tight'

        # Set axis properties
        plt.rcParams['axes.linewidth'] = 1.2
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['grid.linewidth'] = 0.8

        # Set legend properties
        plt.rcParams['legend.frameon'] = True
        plt.rcParams['legend.fancybox'] = False
        plt.rcParams['legend.shadow'] = False
        plt.rcParams['legend.numpoints'] = 1
        plt.rcParams['legend.scatterpoints'] = 1

        # Set line properties
        plt.rcParams['lines.linewidth'] = 2.0
        plt.rcParams['lines.markersize'] = 6

        # Set tick properties
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['xtick.major.size'] = 6
        plt.rcParams['ytick.major.size'] = 6
        plt.rcParams['xtick.minor.size'] = 3
        plt.rcParams['ytick.minor.size'] = 3

        logging.info("CMS plot style applied" + (" (mplhep)" if _HAS_MPLHEP else " (matplotlib)"))

    def get_color(self, name: str) -> str:
        """Get color by name."""
        return self.colors.get(name, self.colors['primary'])

    def get_font_size(self, name: str) -> int:
        """Get font size by name."""
        return self.font_sizes.get(name, 12)

    def get_line_style(self, name: str) -> str:
        """Get line style by name."""
        return self.line_styles.get(name, '-')

    def get_marker(self, name: str) -> str:
        """Get marker by name."""
        return self.markers.get(name, 'o')


def get_process_colors() -> Dict[str, str]:
    """
    Get color scheme for different physics processes.

    Returns:
        Dictionary of process colors
    """
    return {
        'data': '#000000',
        'signal': '#ff0000',
        'ttbar': '#1f77b4',
        'wjets': '#ff7f0e',
        'zjets': '#2ca02c',
        'qcd': '#d62728',
        'st': '#9467bd',
        'diboson': '#8c564b',
        'triboson': '#e377c2',
        'other': '#7f7f7f'
    }


def get_process_labels() -> Dict[str, str]:
    """
    Get labels for different physics processes.

    Returns:
        Dictionary of process labels
    """
    return {
        'data': 'Data',
        'signal': 'Signal',
        'ttbar': 't#bar{t}',
        'wjets': 'W+jets',
        'zjets': 'Z+jets',
        'qcd': 'QCD',
        'st': 'Single top',
        'diboson': 'Diboson',
        'triboson': 'Triboson',
        'other': 'Other'
    }


def get_region_colors() -> Dict[str, str]:
    """
    Get color scheme for different analysis regions.

    Returns:
        Dictionary of region colors
    """
    return {
        'SR': '#ff0000',      # Red for signal region
        'TTbarCR': '#1f77b4', # Blue for ttbar CR
        'ZjetsCR': '#2ca02c', # Green for Z+jets CR
        'QCDCR': '#d62728',   # Red for QCD CR
        'VR_HighMET': '#ff7f0e', # Orange for validation regions
        'VR_LowBjet': '#9467bd'  # Purple for validation regions
    }


def get_region_labels() -> Dict[str, str]:
    """
    Get labels for different analysis regions.

    Returns:
        Dictionary of region labels
    """
    return {
        'SR': 'Signal Region',
        'TTbarCR': 't#bar{t} Control Region',
        'ZjetsCR': 'Z+jets Control Region',
        'QCDCR': 'QCD Control Region',
        'VR_HighMET': 'High MET Validation Region',
        'VR_LowBjet': 'Low B-jet Validation Region'
    }


def create_legend_elements(processes: List[str], colors: Dict[str, str],
                         labels: Dict[str, str]) -> List[mpatches.Patch]:
    """
    Create legend elements for processes.

    Args:
        processes: List of process names
        colors: Dictionary of process colors
        labels: Dictionary of process labels

    Returns:
        List of legend elements
    """
    elements = []

    for process in processes:
        color = colors.get(process, '#7f7f7f')
        label = labels.get(process, process)
        elements.append(mpatches.Patch(color=color, label=label))

    return elements


def add_cms_label(ax, year: str = "2022", lumi: float = 35.9,
                  preliminary: bool = True, x: float = 0.02, y: float = 0.98):
    """
    Add CMS label to plot.

    Args:
        ax: Matplotlib axis
        year: Data-taking year
        lumi: Luminosity in fb^-1
        preliminary: Whether to add "Preliminary" label
        x: X position (0-1)
        y: Y position (0-1)
    """
    if _HAS_MPLHEP:
        extra = "Preliminary" if preliminary else None
        try:
            hep.cms.label(extra=extra, data=True, lumi=lumi, year=year, ax=ax, loc=0)
        except Exception:
            hep.cms.text(extra=extra, ax=ax, loc=0)
            ax.text(x, y-0.05, f"{lumi:.1f} fb$^{-1}$ ({year})", transform=ax.transAxes,
                    fontsize=14, verticalalignment='top', horizontalalignment='left')
    else:
        cms_text = "CMS" + (" Preliminary" if preliminary else "")
        ax.text(x, y, cms_text, transform=ax.transAxes, fontsize=16, fontweight='bold',
                verticalalignment='top', horizontalalignment='left')
        ax.text(x, y-0.05, f"{lumi:.1f} fb$^{-1}$ ({year})", transform=ax.transAxes,
                fontsize=14, verticalalignment='top', horizontalalignment='left')


def add_ratio_plot(fig, ax_main, ax_ratio, data_values: np.ndarray,
                  mc_values: np.ndarray, bins: np.ndarray,
                  ylabel: str = "Data/MC") -> None:
    """
    Add ratio plot below main plot.

    Args:
        fig: Matplotlib figure
        ax_main: Main plot axis
        ax_ratio: Ratio plot axis
        data_values: Data histogram values
        mc_values: MC histogram values
        bins: Histogram bins
        ylabel: Y-axis label for ratio plot
    """
    # Calculate ratio
    ratio = np.divide(data_values, mc_values, out=np.zeros_like(data_values), where=mc_values!=0)

    # Plot ratio
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax_ratio.plot(bin_centers, ratio, 'ko', markersize=4)

    # Add horizontal line at 1
    ax_ratio.axhline(y=1, color='red', linestyle='--', alpha=0.7)

    # Set ratio plot properties
    ax_ratio.set_ylabel(ylabel)
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.grid(True, alpha=0.3)

    # Remove x-axis labels from main plot
    ax_main.set_xticklabels([])


def add_uncertainty_band(ax, bins: np.ndarray, values: np.ndarray,
                        uncertainties: np.ndarray, color: str = 'gray',
                        alpha: float = 0.3) -> None:
    """
    Add uncertainty band to plot.

    Args:
        ax: Matplotlib axis
        bins: Histogram bins
        values: Histogram values
        uncertainties: Uncertainty values
        color: Band color
        alpha: Band transparency
    """
    # Calculate band edges
    upper = values + uncertainties
    lower = values - uncertainties

    # Fill between upper and lower bounds
    ax.fill_between(bins[:-1], lower, upper, color=color, alpha=alpha, step='post')


def create_stacked_histogram(ax, histograms: Dict[str, np.ndarray], bins: np.ndarray,
                           colors: Dict[str, str], labels: Dict[str, str],
                           uncertainties: Optional[Dict[str, np.ndarray]] = None) -> None:
    """
    Create stacked histogram plot.

    Args:
        ax: Matplotlib axis
        histograms: Dictionary of histogram data
        bins: Histogram bins
        colors: Dictionary of process colors
        labels: Dictionary of process labels
        uncertainties: Dictionary of uncertainty data
    """
    # Sort processes by total yield
    process_totals = {process: np.sum(values) for process, values in histograms.items()}
    sorted_processes = sorted(process_totals.keys(), key=lambda x: process_totals[x], reverse=True)

    # Create stacked histogram
    bottom = np.zeros(len(bins) - 1)

    for process in sorted_processes:
        values = histograms[process]
        color = colors.get(process, '#7f7f7f')
        label = labels.get(process, process)

        ax.bar(bins[:-1], values, width=np.diff(bins), bottom=bottom,
               color=color, label=label, alpha=0.8)

        bottom += values

    # Add uncertainty band if provided
    if uncertainties:
        total_uncertainty = np.sqrt(sum(unc**2 for unc in uncertainties.values()))
        add_uncertainty_band(ax, bins, bottom, total_uncertainty)


def create_efficiency_plot(ax, signal_values: np.ndarray, background_values: np.ndarray,
                          bins: np.ndarray, xlabel: str, ylabel: str = "Efficiency") -> None:
    """
    Create efficiency plot.

    Args:
        ax: Matplotlib axis
        signal_values: Signal histogram values
        background_values: Background histogram values
        bins: Histogram bins
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    # Calculate efficiency
    signal_efficiency = signal_values / np.sum(signal_values) if np.sum(signal_values) > 0 else 0
    background_efficiency = background_values / np.sum(background_values) if np.sum(background_values) > 0 else 0

    # Plot efficiency
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax.plot(bin_centers, signal_efficiency, 'r-', label='Signal', linewidth=2)
    ax.plot(bin_centers, background_efficiency, 'b-', label='Background', linewidth=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)


def create_roc_curve(ax, signal_scores: np.ndarray, background_scores: np.ndarray,
                   xlabel: str = "Background Efficiency", ylabel: str = "Signal Efficiency") -> None:
    """
    Create ROC curve plot.

    Args:
        ax: Matplotlib axis
        signal_scores: Signal DNN scores
        background_scores: Background DNN scores
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    from sklearn.metrics import roc_curve

    # Combine scores and labels
    scores = np.concatenate([signal_scores, background_scores])
    labels = np.concatenate([np.ones(len(signal_scores)), np.zeros(len(background_scores))])

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(labels, scores)

    # Plot ROC curve
    ax.plot(fpr, tpr, 'b-', linewidth=2, label='ROC Curve')
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.7, label='Random')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)


def save_plot(fig, filename: str, dpi: int = 300, bbox_inches: str = 'tight') -> None:
    """
    Save plot with proper formatting.

    Args:
        fig: Matplotlib figure
        filename: Output filename
        dpi: DPI for saved plot
        bbox_inches: Bounding box for saved plot
    """
    fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
    logging.info(f"Plot saved to {filename}")


def create_subplot_grid(n_plots: int, n_cols: int = 3) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create subplot grid.

    Args:
        n_plots: Number of plots
        n_cols: Number of columns

    Returns:
        Tuple of (figure, axes array)
    """
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))

    if n_plots == 1:
        axes = np.array([axes])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    return fig, axes


def add_watermark(ax, text: str = "DarkBottomLine", x: float = 0.98, y: float = 0.02,
                 alpha: float = 0.3, rotation: float = 0) -> None:
    """
    Add watermark to plot.

    Args:
        ax: Matplotlib axis
        text: Watermark text
        x: X position (0-1)
        y: Y position (0-1)
        alpha: Text transparency
        rotation: Text rotation in degrees
    """
    ax.text(x, y, text, transform=ax.transAxes, fontsize=12, alpha=alpha,
            rotation=rotation, horizontalalignment='right', verticalalignment='bottom')
