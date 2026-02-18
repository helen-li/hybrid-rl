#!/usr/bin/env python3
"""
run_phase1.py -- Entry point for Phase I experiments.

Runs the full Phase I grid:
  algos    x corruption levels x seeds
  {CQL,IQL} x {clean, k=30, k=60} x {0,1,2}

Then generates comparison plots.

Usage
-----
  # Run everything (will take a while):
  python run_phase1.py

  # Quick smoke-test (fewer steps, single seed):
  python run_phase1.py --quick

  # Single run:
  python run_phase1.py --algo cql --env halfcheetah-medium-v2 --remove_top_k 30 --seed 0

  # Just plot from existing results:
  python run_phase1.py --plot_only
"""

from __future__ import annotations
import argparse
import json
from itertools import product
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.train import train, TrainConfig
from src.eval.plotting import (
    plot_learning_curves,
    plot_corruption_comparison,
    plot_loss_variance,
    plot_q_value_diagnostics,
)


# --------------------------------------------------------------------------- #
#  Experiment grid
# --------------------------------------------------------------------------- #

DEFAULT_ENVS = ["halfcheetah-medium-v2", "hopper-medium-v2"]
DEFAULT_ALGOS = ["cql", "iql"]
DEFAULT_CORRUPTION = [0.0, 30.0, 60.0]
DEFAULT_SEEDS = [0, 1, 2]

QUICK_STEPS = 20_000
FULL_STEPS = 500_000


def run_experiment_grid(
    envs: List[str],
    algos: List[str],
    corruptions: List[float],
    seeds: List[int],
    total_steps: int,
    device: str,
    save_dir: str,
):
    """Run the full grid of experiments."""
    for env, algo, k, seed in product(envs, algos, corruptions, seeds):
        print(f"\n{'='*60}")
        print(f"  {algo.upper()} | {env} | k={k} | seed={seed}")
        print(f"{'='*60}")
        cfg = TrainConfig(
            env_name=env,
            algo=algo,
            remove_top_k=k,
            seed=seed,
            total_steps=total_steps,
            device=device,
            save_dir=save_dir,
        )
        train(cfg)


# --------------------------------------------------------------------------- #
#  Plotting from saved results
# --------------------------------------------------------------------------- #

def _load_metrics(save_dir: str, algo: str, env: str, k: float, seed: int) -> dict | None:
    tag = f"k{int(k)}_noise0.0"
    run_name = f"{algo}_{env}_{tag}_s{seed}"
    path = Path(save_dir) / run_name / "metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def generate_plots(
    envs: List[str],
    algos: List[str],
    corruptions: List[float],
    seeds: List[int],
    save_dir: str,
    plot_dir: str = "plots",
):
    """Generate all Phase I analysis plots from saved metrics."""
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    for env in envs:
        # ---- Learning curves (one plot per env) ---------------------- #
        lc_data: Dict[str, Dict[str, list]] = {}
        for algo, k in product(algos, corruptions):
            all_returns = []
            steps = None
            for seed in seeds:
                m = _load_metrics(save_dir, algo, env, k, seed)
                if m is None:
                    continue
                all_returns.append(m["normalized_return"])
                steps = m["step"]

            if not all_returns or steps is None:
                continue

            # Average across seeds.
            arr = np.array(all_returns)
            mean = arr.mean(axis=0).tolist()
            std = arr.std(axis=0).tolist()

            label = f"{algo.upper()} k={int(k)}" if k > 0 else f"{algo.upper()} clean"
            lc_data[label] = {
                "step": steps,
                "normalized_return": mean,
                "normalized_return_std": std,
            }

        if lc_data:
            plot_learning_curves(
                lc_data,
                title=f"Phase I - {env}",
                save_path=f"{plot_dir}/learning_curves_{env}.png",
            )
            print(f"[plot] Saved learning curves for {env}")

        # ---- Corruption comparison bar chart ------------------------- #
        corruption_labels = [f"clean" if k == 0 else f"k={int(k)}" for k in corruptions]
        bar_scores: Dict[str, List[float]] = {}
        for algo in algos:
            scores = []
            for k in corruptions:
                seed_returns = []
                for seed in seeds:
                    m = _load_metrics(save_dir, algo, env, k, seed)
                    if m and m["normalized_return"]:
                        seed_returns.append(m["normalized_return"][-1])
                scores.append(float(np.mean(seed_returns)) if seed_returns else 0.0)
            bar_scores[algo.upper()] = scores

        if any(sum(v) > 0 for v in bar_scores.values()):
            plot_corruption_comparison(
                list(bar_scores.keys()),
                corruption_labels,
                bar_scores,
                title=f"Final Performance vs. Corruption - {env}",
                save_path=f"{plot_dir}/corruption_bar_{env}.png",
            )
            print(f"[plot] Saved corruption bar chart for {env}")

        # ---- Q-value diagnostics (one per algo, clean dataset) ------- #
        for algo in algos:
            m = _load_metrics(save_dir, algo, env, 0.0, seeds[0])
            if m is None or "q1_mean_avg" not in m:
                continue
            plot_q_value_diagnostics(
                steps=m["train_step"],
                q_means=m["q1_mean_avg"],
                q_stds=m["q1_mean_std"],
                title=f"Q-value - {algo.upper()} {env} (clean)",
                save_path=f"{plot_dir}/q_diag_{algo}_{env}.png",
            )

        # ---- Loss variance ------------------------------------------ #
        var_data: Dict[str, List[float]] = {}
        for algo, k in product(algos, corruptions):
            m = _load_metrics(save_dir, algo, env, k, seeds[0])
            if m is None or "critic_loss_rolling_var" not in m:
                continue
            label = f"{algo.upper()} k={int(k)}" if k > 0 else f"{algo.upper()} clean"
            var_data[label] = m["critic_loss_rolling_var"]

        if var_data:
            max_len = max(len(v) for v in var_data.values())
            plot_loss_variance(
                steps=list(range(max_len)),
                variances=var_data,
                title=f"Critic Loss Variance - {env}",
                save_path=f"{plot_dir}/loss_variance_{env}.png",
            )
            print(f"[plot] Saved loss variance plot for {env}")


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Phase I: Offline RL experiments")
    parser.add_argument("--quick", action="store_true", help="Quick smoke-test run")
    parser.add_argument("--plot_only", action="store_true", help="Skip training, just plot")
    # Single-run overrides
    parser.add_argument("--algo", type=str, default=None, choices=["cql", "iql"])
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--remove_top_k", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save_dir", type=str, default="results")
    args = parser.parse_args()

    envs = [args.env] if args.env else DEFAULT_ENVS
    algos = [args.algo] if args.algo else DEFAULT_ALGOS
    corruptions = [args.remove_top_k] if args.remove_top_k is not None else DEFAULT_CORRUPTION
    seeds = [args.seed] if args.seed is not None else DEFAULT_SEEDS

    if args.quick:
        steps = QUICK_STEPS
        seeds = [0]
    else:
        steps = args.total_steps or FULL_STEPS

    if not args.plot_only:
        run_experiment_grid(envs, algos, corruptions, seeds, steps, args.device, args.save_dir)

    generate_plots(envs, algos, corruptions, seeds, args.save_dir)
    print("\n[done] Phase I complete.")


if __name__ == "__main__":
    main()
