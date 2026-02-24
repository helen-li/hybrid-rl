"""
Plotting utilities for Phase I and Phase II analysis.

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

    Uses a side-by-side subplot layout (one panel per algorithm) so each
    panel has at most one curve per corruption level — much easier to read
    than 6 overlapping lines on a single axis.

    Parameters
    ----------
    results : dict
        Outer key = label (e.g. "CQL clean", "IQL k=30").
        Inner dict must contain *x_key* and *y_key* lists of equal length.
    """
    _CORRUPTION_COLORS = {"clean": COLORS[0], "k=0": COLORS[0],
                          "k=30": COLORS[1], "k=60": COLORS[2]}
    _CORRUPTION_DASHES = {"clean": (1, 0), "k=0": (1, 0),    # solid
                          "k=30": (6, 2),                      # dashed
                          "k=60": (2, 2)}                      # dotted

    # Group labels by algorithm prefix.
    algos_seen: Dict[str, list] = {}  # algo -> [(label, data), ...]
    for label, data in results.items():
        algo_prefix = label.split()[0] if label.split() else "Other"
        algos_seen.setdefault(algo_prefix, []).append((label, data))

    n_panels = max(len(algos_seen), 1)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4.5), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, (algo, entries) in zip(axes, algos_seen.items()):
        for label, data in entries:
            xs = data[x_key]
            ys = data[y_key]
            corruption_tag = label.split()[-1] if len(label.split()) > 1 else ""
            color = _CORRUPTION_COLORS.get(corruption_tag, COLORS[0])
            dashes = _CORRUPTION_DASHES.get(corruption_tag, (1, 0))
            # Strip algo prefix from legend (panel title already says it).
            short_label = corruption_tag if corruption_tag else label
            ax.plot(xs, ys, label=short_label, color=color, linewidth=2,
                    dashes=dashes)
            std_key = y_key + "_std"
            if std_key in data:
                stds = np.array(data[std_key])
                ys_arr = np.array(ys)
                ax.fill_between(xs, ys_arr - stds, ys_arr + stds,
                                color=color, alpha=0.12)
        ax.set_title(algo, fontsize=13, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.legend(title="Corruption", fontsize=9, title_fontsize=9,
                  loc="lower right")
        ax.tick_params(axis="x", labelsize=9)
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k" if x >= 1e3 else f"{x:.0f}"))

    axes[0].set_ylabel(ylabel)
    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
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
    ax.plot(steps, q_means_arr, color=COLORS[0], label="Q mean", linewidth=1.5)
    ax.fill_between(steps, q_means_arr - q_stds_arr, q_means_arr + q_stds_arr,
                    color=COLORS[0], alpha=0.18, label="± 1 std")
    ax.set_title(title)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Q-value")
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k" if x >= 1e3 else f"{x:.0f}"))
    _save(fig, Path(save_path), show)


# --------------------------------------------------------------------------- #
#  Corruption comparison (bar chart)
# --------------------------------------------------------------------------- #

def plot_corruption_comparison(
    algo_names: List[str],
    corruption_levels: List[str],
    scores: Dict[str, List[float]],
    stds: Optional[Dict[str, List[float]]] = None,
    title: str = "Performance vs. Dataset Corruption",
    save_path: str | Path = "plots/corruption_comparison.png",
    show: bool = False,
):
    """Grouped bar chart: each algo gets one group, bars per corruption level.

    Parameters
    ----------
    algo_names : e.g. ["CQL", "IQL"]
    corruption_levels : e.g. ["clean", "k=30", "k=60"]
    scores : dict[algo_name -> list of mean scores aligned with corruption_levels]
    stds : dict[algo_name -> list of std scores] (optional, for error bars)
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(algo_names))
    n_bars = len(corruption_levels)
    width = 0.8 / n_bars

    all_vals = [scores[a][i] for a in algo_names for i in range(n_bars)]
    all_errs = [stds[a][i] for a in algo_names for i in range(n_bars)] if stds else [0] * len(all_vals)
    y_lo = max(0, min(v - e for v, e in zip(all_vals, all_errs)) - 5)

    for i, level in enumerate(corruption_levels):
        values = [scores[algo][i] for algo in algo_names]
        errs = [stds[algo][i] for algo in algo_names] if stds else None
        offset = (i - n_bars / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, values, width, yerr=errs, capsize=4,
            label=level, color=COLORS[i % len(COLORS)], alpha=0.85,
            edgecolor="white", linewidth=0.5,
        )
        # Value labels on bars.
        top_offset = max(all_errs) + 0.8 if stds else 0.5
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + top_offset,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylim(bottom=y_lo)
    ax.set_title(title)
    ax.set_ylabel("Normalized Return")
    ax.set_xticks(x)
    ax.set_xticklabels(algo_names, fontsize=12)
    ax.legend(title="Corruption", fontsize=10)
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
        ax.plot(xs, var_vals, label=label, color=COLORS[i % len(COLORS)],
                linewidth=1.0, alpha=0.85)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Loss Variance (log scale)")
    ax.legend(fontsize=9)
    _save(fig, Path(save_path), show)


# --------------------------------------------------------------------------- #
#  Phase II: Fine-tuning curves
# --------------------------------------------------------------------------- #

def plot_finetuning_curves(
    results: Dict[str, Dict[str, List[float]]],
    x_key: str = "step",
    y_key: str = "normalized_return",
    title: str = "Online Fine-tuning",
    xlabel: str = "Online Steps",
    ylabel: str = "Normalized Return (D4RL)",
    save_path: str | Path = "plots_phase2/finetuning.png",
    show: bool = False,
):
    """Plot fine-tuning learning curves comparing ensemble vs vanilla.

    Same format as ``plot_learning_curves`` but styled for Phase II
    with solid lines for ensemble and dashed lines for vanilla.
    """
    fig, ax = plt.subplots()
    for i, (label, data) in enumerate(results.items()):
        xs = data[x_key]
        ys = data[y_key]
        color = COLORS[i % len(COLORS)]
        linestyle = "--" if "vanilla" in label else "-"
        ax.plot(xs, ys, label=label, color=color, linewidth=1.5, linestyle=linestyle)
        std_key = y_key + "_std"
        if std_key in data:
            stds = np.array(data[std_key])
            ys_arr = np.array(ys)
            ax.fill_between(xs, ys_arr - stds, ys_arr + stds, color=color, alpha=0.15)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=9)
    _save(fig, Path(save_path), show)


# --------------------------------------------------------------------------- #
#  Phase II: Sample efficiency bar chart
# --------------------------------------------------------------------------- #

def plot_sample_efficiency(
    data: Dict[str, Dict[str, float]],
    title: str = "Sample Efficiency",
    save_path: str | Path = "plots_phase2/sample_efficiency.png",
    show: bool = False,
):
    """Bar chart showing online steps needed to reach a performance threshold.

    Parameters
    ----------
    data : dict
        Keys are labels (e.g. "CQL clean (ensemble)").
        Values are dicts with "mean" and "std" keys.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = list(data.keys())
    means = [data[l]["mean"] for l in labels]
    stds = [data[l]["std"] for l in labels]

    colors = [COLORS[0] if "ensemble" in l else COLORS[1] for l in labels]
    x = np.arange(len(labels))

    ax.bar(x, means, yerr=stds, capsize=3, color=colors, alpha=0.8)
    ax.set_title(title)
    ax.set_ylabel("Online Steps to Threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    # Legend for ensemble vs vanilla colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS[0], label="Ensemble bonus"),
        Patch(facecolor=COLORS[1], label="Vanilla (no bonus)"),
    ]
    ax.legend(handles=legend_elements)
    fig.tight_layout()
    _save(fig, Path(save_path), show)
