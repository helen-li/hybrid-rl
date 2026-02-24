#!/usr/bin/env python3
"""
run_phase1.py -- Entry point for Phase I experiments.

Runs the full Phase I plan (see PHASE1.md):
  - Train CQL and IQL on D4RL MuJoCo (HalfCheetah, Hopper) with top-k% corruption.
  - Evaluate normalized returns and stability (Q diagnostics, loss variance).
  - Produce learning curves, corruption bar chart, Q diagnostics, loss variance plots.

Experiment grid (full):
  envs x algos x corruption x seeds
  {halfcheetah-medium-v2, hopper-medium-v2} x {CQL, IQL} x {clean, k=30, k=60} x {0,1,2}
  = 36 runs. Optional Gaussian reward noise via --reward_noise_std (stretch goal).

Deliverables:
  - results/: one folder per run with metrics.json and checkpoint.pt
  - plots/: learning_curves_*, corruption_bar_*, q_diag_*, loss_variance_*

Usage
-----
  # Full Phase I grid (36 runs):
  python run_phase1.py --device auto

  # Quick smoke-test (20k steps, single seed):
  python run_phase1.py --quick
  python run_phase1.py --quick --algo cql --env halfcheetah-medium-v2

  # Single run:
  python run_phase1.py --algo cql --env halfcheetah-medium-v2 --remove_top_k 30 --seed 0

  # With Gaussian reward noise (stretch):
  python run_phase1.py --algo iql --env halfcheetah-medium-v2 --reward_noise_std 1.0

  # Regenerate plots from existing results:
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
    reward_noise_std: float = 0.0,
):
    """Run the full Phase I grid of experiments."""
    n_runs = len(envs) * len(algos) * len(corruptions) * len(seeds)
    noise_suffix = f", reward_noise_std={reward_noise_std}" if reward_noise_std > 0 else ""
    print(f"\nPhase I grid: {n_runs} runs (envs x algos x corruption x seeds){noise_suffix}")
    for env, algo, k, seed in product(envs, algos, corruptions, seeds):
        print(f"\n{'='*60}")
        print(f"  {algo.upper()} | {env} | k={k} | seed={seed}")
        print(f"{'='*60}")
        cfg = TrainConfig(
            env_name=env,
            algo=algo,
            remove_top_k=k,
            reward_noise_std=reward_noise_std,
            seed=seed,
            total_steps=total_steps,
            device=device,
            save_dir=save_dir,
        )
        train(cfg)


# --------------------------------------------------------------------------- #
#  Plotting from saved results
# --------------------------------------------------------------------------- #

def _load_metrics(
    save_dir: str,
    algo: str,
    env: str,
    k: float,
    seed: int,
    reward_noise_std: float = 0.0,
) -> dict | None:
    """Load metrics.json for a single run. Tag must match train.py run name."""
    tag = f"k{int(k)}_noise{float(reward_noise_std)}"
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
    reward_noise_std: float = 0.0,
):
    """Generate all Phase I analysis plots from saved metrics (see PHASE1.md §7)."""
    def load_m(save_dir: str, algo: str, env: str, k: float, seed: int) -> dict | None:
        return _load_metrics(save_dir, algo, env, k, seed, reward_noise_std)

    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    for env in envs:
        # ---- Learning curves (one plot per env) ---------------------- #
        lc_data: Dict[str, Dict[str, list]] = {}
        for algo, k in product(algos, corruptions):
            all_returns = []
            steps = None
            for seed in seeds:
                m = load_m(save_dir, algo, env, k, seed)
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
        bar_stds: Dict[str, List[float]] = {}
        for algo in algos:
            scores = []
            stds = []
            for k in corruptions:
                seed_returns = []
                for seed in seeds:
                    m = load_m(save_dir, algo, env, k, seed)
                    if m and m["normalized_return"]:
                        seed_returns.append(m["normalized_return"][-1])
                scores.append(float(np.mean(seed_returns)) if seed_returns else 0.0)
                stds.append(float(np.std(seed_returns)) if len(seed_returns) > 1 else 0.0)
            bar_scores[algo.upper()] = scores
            bar_stds[algo.upper()] = stds

        if bar_scores:
            plot_corruption_comparison(
                list(bar_scores.keys()),
                corruption_labels,
                bar_scores,
                stds=bar_stds,
                title=f"Final Performance vs. Corruption - {env}",
                save_path=f"{plot_dir}/corruption_bar_{env}.png",
            )
            print(f"[plot] Saved corruption bar chart for {env}")

        # ---- Q-value diagnostics (one per algo, clean dataset) ------- #
        for algo in algos:
            all_q_means = []
            train_steps = None
            for seed in seeds:
                m = load_m(save_dir, algo, env, 0.0, seed)
                if m is None or "q1_mean_avg" not in m:
                    continue
                all_q_means.append(m["q1_mean_avg"])
                train_steps = m["train_step"]
            if not all_q_means or train_steps is None:
                continue
            q_arr = np.array(all_q_means)
            plot_q_value_diagnostics(
                steps=train_steps,
                q_means=q_arr.mean(axis=0).tolist(),
                q_stds=q_arr.std(axis=0).tolist(),
                title=f"Q-value - {algo.upper()} {env} (clean)",
                save_path=f"{plot_dir}/q_diag_{algo}_{env}.png",
            )
            print(f"[plot] Saved Q-diagnostics for {algo.upper()} {env}")

        # ---- Loss variance ------------------------------------------ #
        var_data: Dict[str, List[float]] = {}
        for algo, k in product(algos, corruptions):
            seed_vars = []
            for seed in seeds:
                m = load_m(save_dir, algo, env, k, seed)
                if m is None or "critic_loss_rolling_var" not in m:
                    continue
                seed_vars.append(m["critic_loss_rolling_var"])
            if not seed_vars:
                continue
            # Average across seeds (truncate to shortest)
            min_len = min(len(v) for v in seed_vars)
            avg_var = np.mean([v[:min_len] for v in seed_vars], axis=0)
            # Downsample for readability (keep ~500 points)
            stride = max(1, min_len // 500)
            label = f"{algo.upper()} k={int(k)}" if k > 0 else f"{algo.upper()} clean"
            var_data[label] = avg_var[::stride].tolist()

        if var_data:
            # Build proper training-step x-axis from total_steps and rolling window offset
            max_len = max(len(v) for v in var_data.values())
            # Rolling variance starts after window (1000 steps) — map to training steps
            total_steps_cfg = 500_000  # fallback
            any_m = None
            for seed in seeds:
                any_m = load_m(save_dir, algos[0], env, corruptions[0], seed)
                if any_m is not None:
                    break
            if any_m and "critic_loss_rolling_var" in any_m:
                raw_len = len(any_m["critic_loss_rolling_var"])
                step_axis = np.linspace(1000, 1000 + raw_len - 1, max_len).tolist()
            else:
                step_axis = list(range(max_len))
            plot_loss_variance(
                steps=step_axis,
                variances=var_data,
                title=f"Critic Loss Variance - {env}",
                save_path=f"{plot_dir}/loss_variance_{env}.png",
            )
            print(f"[plot] Saved loss variance plot for {env}")


# --------------------------------------------------------------------------- #
#  CLI helpers
# --------------------------------------------------------------------------- #

def _detect_seeds(
    save_dir: str,
    algos: List[str],
    envs: List[str],
    corruptions: List[float],
    reward_noise_std: float = 0.0,
) -> List[int]:
    """Scan *save_dir* and return the sorted set of seed values that have results."""
    found: set[int] = set()
    for algo, env, k in product(algos, envs, corruptions):
        tag = f"k{int(k)}_noise{float(reward_noise_std)}"
        pattern = f"{algo}_{env}_{tag}_s*"
        for p in Path(save_dir).glob(pattern):
            metrics = p / "metrics.json"
            if metrics.exists():
                # Extract seed from directory name: ...s<N>
                try:
                    found.add(int(p.name.rsplit("_s", 1)[1]))
                except (IndexError, ValueError):
                    pass
    return sorted(found)


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Phase I: Characterizing failure modes in offline RL (see PHASE1.md)"
    )
    parser.add_argument("--quick", action="store_true", help="Quick run: 20k steps, single seed")
    parser.add_argument("--plot_only", action="store_true", help="Skip training; regenerate plots from save_dir")
    # Single-run overrides (if set, grid is reduced to one value each)
    parser.add_argument("--algo", type=str, default=None, choices=["cql", "iql"])
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--remove_top_k", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save_dir", type=str, default="results", help="Results directory (metrics, checkpoints)")
    parser.add_argument("--plot_dir", type=str, default="plots", help="Output directory for figures")
    parser.add_argument(
        "--reward_noise_std",
        type=float,
        default=0.0,
        help="Gaussian reward noise std (stretch goal); 0 = no noise",
    )
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
        run_experiment_grid(
            envs, algos, corruptions, seeds, steps, args.device, args.save_dir, args.reward_noise_std
        )

    # In plot-only mode with no explicit --seed, auto-detect available seeds
    # from the results directory so users don't have to specify them.
    if args.plot_only and args.seed is None:
        detected = _detect_seeds(args.save_dir, algos, envs, corruptions, args.reward_noise_std)
        if detected:
            seeds = detected
            print(f"[plot] Auto-detected seeds: {seeds}")

    generate_plots(
        envs, algos, corruptions, seeds, args.save_dir, args.plot_dir, args.reward_noise_std
    )
    print("\n[done] Phase I complete.")


if __name__ == "__main__":
    main()
