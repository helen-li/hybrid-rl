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

    One subplot per corruption level so each panel has only two lines
    (ensemble vs vanilla), making the comparison easy to read.
    """
    # Group results by corruption level.
    panels: Dict[str, Dict[str, Dict]] = {}  # corruption_tag -> {bonus_label -> data}
    for label, data in results.items():
        corruption_tag = label.split("(")[0].strip()
        panels.setdefault(corruption_tag, {})[label] = data

    n_panels = max(len(panels), 1)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.5), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, (corruption_tag, entries) in zip(axes, panels.items()):
        for label, data in entries.items():
            xs = data[x_key]
            ys = data[y_key]
            is_vanilla = "no bonus" in label.lower() or "vanilla" in label.lower()
            color = COLORS[1] if is_vanilla else COLORS[0]
            linestyle = "--" if is_vanilla else "-"
            short_label = "No Bonus" if is_vanilla else "Uncertainty Bonus"
            ax.plot(xs, ys, label=short_label, color=color, linewidth=1.8,
                    linestyle=linestyle)
            std_key = y_key + "_std"
            if std_key in data:
                stds = np.array(data[std_key])
                ys_arr = np.array(ys)
                ax.fill_between(xs, ys_arr - stds, ys_arr + stds,
                                color=color, alpha=0.15)
        ax.set_title(corruption_tag, fontsize=13, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.legend(fontsize=10, loc="lower right")
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k" if x >= 1e3 else f"{x:.0f}"))

    axes[0].set_ylabel(ylabel)
    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
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

    colors = [COLORS[0] if "bonus" in l.lower() and "no bonus" not in l.lower() else COLORS[1] for l in labels]
    x = np.arange(len(labels))

    ax.bar(x, means, yerr=stds, capsize=3, color=colors, alpha=0.8)
    ax.set_title(title)
    ax.set_ylabel("Online Steps to Threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    # Legend for bonus vs no-bonus colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS[0], label="Uncertainty Bonus"),
        Patch(facecolor=COLORS[1], label="No Bonus"),
    ]
    ax.legend(handles=legend_elements)
    fig.tight_layout()
    _save(fig, Path(save_path), show)


# --------------------------------------------------------------------------- #
#  Phase II: Bonus comparison bar chart
# --------------------------------------------------------------------------- #

def plot_bonus_comparison(
    algo_names: List[str],
    corruption_levels: List[str],
    ensemble_scores: Dict[str, List[float]],
    vanilla_scores: Dict[str, List[float]],
    ensemble_stds: Optional[Dict[str, List[float]]] = None,
    vanilla_stds: Optional[Dict[str, List[float]]] = None,
    title: str = "Ensemble vs Vanilla Fine-tuning",
    save_path: str | Path = "plots_phase2/bonus_comparison.png",
    show: bool = False,
):
    """Grouped bar chart comparing ensemble vs vanilla final performance.

    One subplot per algorithm, bars grouped by corruption level.
    """
    n_algos = max(len(algo_names), 1)
    fig, axes = plt.subplots(1, n_algos, figsize=(5 * n_algos, 5), sharey=True)
    if n_algos == 1:
        axes = [axes]

    x = np.arange(len(corruption_levels))
    width = 0.35

    for ax, algo in zip(axes, algo_names):
        ens_vals = ensemble_scores[algo]
        van_vals = vanilla_scores[algo]
        ens_err = ensemble_stds[algo] if ensemble_stds else None
        van_err = vanilla_stds[algo] if vanilla_stds else None

        ax.bar(x - width / 2, ens_vals, width, yerr=ens_err, capsize=4,
               label="Uncertainty Bonus", color=COLORS[0], alpha=0.85, edgecolor="white")
        ax.bar(x + width / 2, van_vals, width, yerr=van_err, capsize=4,
               label="No Bonus", color=COLORS[1], alpha=0.85, edgecolor="white")

        ax.set_title(algo, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(corruption_levels)
        ax.legend(fontsize=10)

    axes[0].set_ylabel("Final Normalized Return")
    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    _save(fig, Path(save_path), show)


# --------------------------------------------------------------------------- #
#  Phase II: Ensemble disagreement over training
# --------------------------------------------------------------------------- #

def plot_ensemble_disagreement(
    results: Dict[str, Dict[str, List[float]]],
    x_key: str = "train_step",
    y_key: str = "ensemble_disagreement",
    title: str = "Ensemble Disagreement",
    xlabel: str = "Training Steps",
    ylabel: str = "Mean Q-Ensemble Std",
    save_path: str | Path = "plots_phase2/disagreement.png",
    show: bool = False,
):
    """Plot ensemble disagreement over online training steps.

    One line per corruption level, showing how disagreement evolves.
    """
    _CORRUPTION_COLORS = {
        "clean": COLORS[0], "k=0": COLORS[0],
        "k=30": COLORS[1], "k=60": COLORS[2],
    }

    fig, ax = plt.subplots()
    for label, data in results.items():
        xs = data[x_key]
        ys = data[y_key]
        color = _CORRUPTION_COLORS.get(label, COLORS[3])
        ax.plot(xs, ys, label=label, color=color, linewidth=1.5)
        std_key = y_key + "_std"
        if std_key in data:
            stds = np.array(data[std_key])
            ys_arr = np.array(ys)
            ax.fill_between(xs, ys_arr - stds, ys_arr + stds, color=color, alpha=0.15)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=10)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k" if x >= 1e3 else f"{x:.0f}"))
    _save(fig, Path(save_path), show)
