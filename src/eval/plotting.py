"""
Plotting utilities for Phase I analysis.

All functions save figures to disk and optionally display them.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless servers
import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------------------------------- #
#  Style defaults
# --------------------------------------------------------------------------- #

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
plt.rcParams.update({
    "figure.figsize": (8, 5),
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 12,
})


def _save(fig: plt.Figure, path: Path, show: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Learning curves
# --------------------------------------------------------------------------- #

def plot_learning_curves(
    results: Dict[str, Dict[str, List[float]]],
    x_key: str = "step",
    y_key: str = "normalized_return",
    title: str = "Learning Curves",
    xlabel: str = "Training Steps",
    ylabel: str = "Normalized Return (D4RL)",
    save_path: str | Path = "plots/learning_curves.png",
    show: bool = False,
):
    """Plot learning curves for multiple runs / configurations.

    Parameters
    ----------
    results : dict
        Outer key = label (e.g. "CQL clean", "IQL k=30").
        Inner dict must contain *x_key* and *y_key* lists of equal length.
    """
    fig, ax = plt.subplots()
    for i, (label, data) in enumerate(results.items()):
        xs = data[x_key]
        ys = data[y_key]
        color = COLORS[i % len(COLORS)]
        ax.plot(xs, ys, label=label, color=color, linewidth=1.5)
        # If standard deviation data is available, shade it.
        std_key = y_key + "_std"
        if std_key in data:
            stds = np.array(data[std_key])
            ys_arr = np.array(ys)
            ax.fill_between(xs, ys_arr - stds, ys_arr + stds, color=color, alpha=0.15)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    _save(fig, Path(save_path), show)


# --------------------------------------------------------------------------- #
#  Q-value diagnostics
# --------------------------------------------------------------------------- #

def plot_q_value_diagnostics(
    steps: List[int],
    q_means: List[float],
    q_stds: List[float],
    title: str = "Q-Value Diagnostics",
    save_path: str | Path = "plots/q_diagnostics.png",
    show: bool = False,
):
    """Plot mean Q-value +/- 1 std over training."""
    fig, ax = plt.subplots()
    q_means_arr = np.array(q_means)
    q_stds_arr = np.array(q_stds)
    ax.plot(steps, q_means_arr, color=COLORS[0], label="Q mean")
    ax.fill_between(steps, q_means_arr - q_stds_arr, q_means_arr + q_stds_arr,
                    color=COLORS[0], alpha=0.2, label="+/- 1 std")
    ax.set_title(title)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Q-value")
    ax.legend()
    _save(fig, Path(save_path), show)


# --------------------------------------------------------------------------- #
#  Corruption comparison (bar chart)
# --------------------------------------------------------------------------- #

def plot_corruption_comparison(
    algo_names: List[str],
    corruption_levels: List[str],
    scores: Dict[str, List[float]],
    title: str = "Performance vs. Dataset Corruption",
    save_path: str | Path = "plots/corruption_comparison.png",
    show: bool = False,
):
    """Grouped bar chart: each algo gets one group, bars per corruption level.

    Parameters
    ----------
    algo_names : e.g. ["CQL", "IQL"]
    corruption_levels : e.g. ["clean", "k=30", "k=60"]
    scores : dict[algo_name -> list of scores aligned with corruption_levels]
    """
    fig, ax = plt.subplots()
    x = np.arange(len(algo_names))
    n_bars = len(corruption_levels)
    width = 0.8 / n_bars

    for i, level in enumerate(corruption_levels):
        values = [scores[algo][i] for algo in algo_names]
        offset = (i - n_bars / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=level, color=COLORS[i % len(COLORS)])
        # Value labels on bars.
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_title(title)
    ax.set_ylabel("Normalized Return")
    ax.set_xticks(x)
    ax.set_xticklabels(algo_names)
    ax.legend(title="Corruption")
    _save(fig, Path(save_path), show)


# --------------------------------------------------------------------------- #
#  Loss variance over time
# --------------------------------------------------------------------------- #

def plot_loss_variance(
    steps: List[int],
    variances: Dict[str, List[float]],
    title: str = "Critic Loss Variance (Rolling)",
    save_path: str | Path = "plots/loss_variance.png",
    show: bool = False,
):
    """Plot rolling variance of critic loss for multiple runs."""
    fig, ax = plt.subplots()
    for i, (label, var_vals) in enumerate(variances.items()):
        # Truncate steps to match variance length.
        xs = steps[:len(var_vals)]
        ax.plot(xs, var_vals, label=label, color=COLORS[i % len(COLORS)], linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Loss Variance")
    ax.legend()
    _save(fig, Path(save_path), show)
