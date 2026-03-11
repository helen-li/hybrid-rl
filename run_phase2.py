#!/usr/bin/env python3
"""
run_phase2.py -- Entry point for Phase II experiments.

Fine-tunes Phase I offline checkpoints online with ensemble-based
exploration bonuses (see the Phase II section of PROPOSAL.md).

Experiment grid (full):
  envs x algos x corruption x bonus_type x seeds
  {halfcheetah, hopper} x {CQL, IQL} x {clean, k=30, k=60} x {ensemble, none} x {0,1,2}
  = 72 runs.

Usage
-----
  # Full Phase II grid:
  python run_phase2.py --device auto

  # Quick smoke-test (5k online steps, single seed):
  python run_phase2.py --quick

  # Single run:
  python run_phase2.py --algo cql --env halfcheetah-medium-v2 --remove_top_k 30 \\
      --bonus_type ensemble --seed 0

  # Vanilla fine-tuning baseline (no exploration bonus):
  python run_phase2.py --algo cql --env halfcheetah-medium-v2 --bonus_type none

  # Use 5 ensemble members instead of the default 3:
  python run_phase2.py --n_ensemble 5

  # Regenerate plots from existing results:
  python run_phase2.py --plot_only
"""

from __future__ import annotations
import argparse
import json
from itertools import product
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.finetune import finetune, FinetuneConfig
from src.eval.plotting import (
    plot_finetuning_curves,
    plot_bonus_comparison,
    plot_ensemble_disagreement,
    plot_sample_efficiency,
)


# --------------------------------------------------------------------------- #
#  Experiment grid defaults
# --------------------------------------------------------------------------- #

DEFAULT_ENVS = ["halfcheetah-medium-v2", "hopper-medium-v2"]
DEFAULT_ALGOS = ["cql", "iql"]
DEFAULT_CORRUPTION = [0.0, 30.0, 60.0]
DEFAULT_BONUS_TYPES = ["ensemble", "none"]
DEFAULT_SEEDS = [1, 2]

QUICK_STEPS = 5_000
FULL_STEPS = 250_000


# --------------------------------------------------------------------------- #
#  Checkpoint resolution
# --------------------------------------------------------------------------- #

def _resolve_checkpoint(
    checkpoint_dir: str,
    algo: str,
    env: str,
    k: float,
    seed: int,
    reward_noise_std: float = 0.0,
) -> str:
    """Build the path to a Phase I checkpoint."""
    tag = f"k{int(k)}_noise{float(reward_noise_std)}"
    run_name = f"{algo}_{env}_{tag}_s{seed}"
    path = Path(checkpoint_dir) / run_name / "checkpoint.pt"
    return str(path)


# --------------------------------------------------------------------------- #
#  Experiment grid runner
# --------------------------------------------------------------------------- #

def run_experiment_grid(
    envs: List[str],
    algos: List[str],
    corruptions: List[float],
    bonus_types: List[str],
    seeds: List[int],
    online_steps: int,
    device: str,
    checkpoint_dir: str,
    save_dir: str,
    n_ensemble: int = 3,
    bonus_coeff: float = 1.0,
    reward_noise_std: float = 0.0,
):
    """Run the full Phase II grid of fine-tuning experiments."""
    n_runs = len(envs) * len(algos) * len(corruptions) * len(bonus_types) * len(seeds)
    print(f"\nPhase II grid: {n_runs} runs "
          f"(envs x algos x corruption x bonus_type x seeds)")

    for env, algo, k, bonus_type, seed in product(envs, algos, corruptions, bonus_types, seeds):
        ckpt_path = _resolve_checkpoint(checkpoint_dir, algo, env, k, seed, reward_noise_std)

        if not Path(ckpt_path).exists():
            print(f"\n[skip] Checkpoint not found: {ckpt_path}")
            continue

        print(f"\n{'='*60}")
        print(f"  {algo.upper()} | {env} | k={k} | {bonus_type} | seed={seed}")
        print(f"{'='*60}")

        cfg = FinetuneConfig(
            checkpoint_path=ckpt_path,
            env_name=env,
            algo=algo,
            remove_top_k=k,
            reward_noise_std=reward_noise_std,
            n_ensemble=n_ensemble,
            bonus_coeff=bonus_coeff,
            bonus_type=bonus_type,
            online_steps=online_steps,
            seed=seed,
            device=device,
            save_dir=save_dir,
        )
        finetune(cfg)


# --------------------------------------------------------------------------- #
#  Plotting from saved results
# --------------------------------------------------------------------------- #

def _load_metrics(
    save_dir: str,
    algo: str,
    env: str,
    k: float,
    bonus_type: str,
    seed: int,
    reward_noise_std: float = 0.0,
) -> dict | None:
    """Load metrics.json for a single Phase II run."""
    bonus_tag = bonus_type if bonus_type != "none" else "vanilla"
    corruption_tag = f"k{int(k)}_noise{float(reward_noise_std)}"
    run_name = f"ft_{algo}_{env}_{corruption_tag}_{bonus_tag}_s{seed}"
    path = Path(save_dir) / run_name / "metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def generate_plots(
    envs: List[str],
    algos: List[str],
    corruptions: List[float],
    bonus_types: List[str],
    seeds: List[int],
    save_dir: str,
    plot_dir: str = "plots_phase2",
    reward_noise_std: float = 0.0,
):
    """Generate Phase II analysis plots from saved metrics."""
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    for env in envs:
        for algo in algos:
            # ---- Fine-tuning curves (one plot per algo+env) ---------- #
            ft_data: Dict[str, Dict[str, list]] = {}

            for k, bonus_type in product(corruptions, bonus_types):
                all_returns = []
                steps = None

                for seed in seeds:
                    m = _load_metrics(save_dir, algo, env, k, bonus_type, seed, reward_noise_std)
                    if m is None:
                        continue
                    all_returns.append(m["normalized_return"])
                    steps = m["step"]

                if not all_returns or steps is None:
                    continue

                arr = np.array(all_returns)
                mean = arr.mean(axis=0).tolist()
                std = arr.std(axis=0).tolist()

                bonus_label = "uncertainty bonus" if bonus_type == "ensemble" else "no bonus"
                k_label = f"k={int(k)}" if k > 0 else "clean"
                label = f"{k_label} ({bonus_label})"

                ft_data[label] = {
                    "step": steps,
                    "normalized_return": mean,
                    "normalized_return_std": std,
                }

            if ft_data:
                plot_finetuning_curves(
                    ft_data,
                    title=f"Phase II Fine-tuning - {algo.upper()} {env}",
                    save_path=f"{plot_dir}/finetune_{algo}_{env}.png",
                )
                print(f"[plot] Saved fine-tuning curves for {algo} {env}")

        # ---- Bonus comparison bar chart (one per env) ---------------- #
        corruption_labels = ["clean" if k == 0 else f"k={int(k)}" for k in corruptions]
        ensemble_scores: Dict[str, List[float]] = {}
        vanilla_scores: Dict[str, List[float]] = {}
        ensemble_stds: Dict[str, List[float]] = {}
        vanilla_stds: Dict[str, List[float]] = {}

        for algo in algos:
            ens_means, van_means = [], []
            ens_sds, van_sds = [], []
            for k in corruptions:
                for bonus_type, means_list, sds_list in [
                    ("ensemble", ens_means, ens_sds),
                    ("none", van_means, van_sds),
                ]:
                    seed_finals = []
                    for seed in seeds:
                        m = _load_metrics(save_dir, algo, env, k, bonus_type, seed, reward_noise_std)
                        if m and m["normalized_return"]:
                            seed_finals.append(m["normalized_return"][-1])
                    means_list.append(float(np.mean(seed_finals)) if seed_finals else 0.0)
                    sds_list.append(float(np.std(seed_finals)) if len(seed_finals) > 1 else 0.0)

            ensemble_scores[algo.upper()] = ens_means
            vanilla_scores[algo.upper()] = van_means
            ensemble_stds[algo.upper()] = ens_sds
            vanilla_stds[algo.upper()] = van_sds

        if any(ensemble_scores.values()) or any(vanilla_scores.values()):
            plot_bonus_comparison(
                list(ensemble_scores.keys()),
                corruption_labels,
                ensemble_scores,
                vanilla_scores,
                ensemble_stds,
                vanilla_stds,
                title=f"Uncertainty Bonus vs No Bonus - {env}",
                save_path=f"{plot_dir}/bonus_comparison_{env}.png",
            )
            print(f"[plot] Saved bonus comparison for {env}")

        # ---- Ensemble disagreement over training (one per algo+env) -- #
        for algo in algos:
            disagree_data: Dict[str, Dict[str, list]] = {}
            for k in corruptions:
                all_disagree = []
                train_steps = None
                for seed in seeds:
                    m = _load_metrics(save_dir, algo, env, k, "ensemble", seed, reward_noise_std)
                    if m is None or "ensemble_disagreement" not in m:
                        continue
                    all_disagree.append(m["ensemble_disagreement"])
                    train_steps = m.get("train_step", m["step"][1:])  # train_step or eval steps
                if not all_disagree or train_steps is None:
                    continue
                min_len = min(len(d) for d in all_disagree)
                arr = np.array([d[:min_len] for d in all_disagree])
                k_label = f"k={int(k)}" if k > 0 else "clean"
                disagree_data[k_label] = {
                    "train_step": train_steps[:min_len],
                    "ensemble_disagreement": arr.mean(axis=0).tolist(),
                    "ensemble_disagreement_std": arr.std(axis=0).tolist(),
                }
            if disagree_data:
                plot_ensemble_disagreement(
                    disagree_data,
                    title=f"Ensemble Disagreement - {algo.upper()} {env}",
                    save_path=f"{plot_dir}/disagreement_{algo}_{env}.png",
                )
                print(f"[plot] Saved disagreement plot for {algo} {env}")

        # ---- Sample efficiency (one per env) ------------------------- #
        # Measure steps to gain +10 normalized return from each run's starting
        # performance.  This is independent of absolute performance level and
        # fairly compares ensemble vs vanilla across all corruption levels.
        IMPROVEMENT_DELTA = 10.0
        efficiency_data: Dict[str, Dict[str, float]] = {}
        for algo, k in product(algos, corruptions):
            for bonus_type in bonus_types:
                steps_to_threshold = []
                for seed in seeds:
                    m = _load_metrics(save_dir, algo, env, k, bonus_type, seed, reward_noise_std)
                    if m is None or not m["normalized_return"]:
                        continue
                    start_perf = m["normalized_return"][0]
                    target = start_perf + IMPROVEMENT_DELTA
                    found = False
                    for step, ret in zip(m["step"], m["normalized_return"]):
                        if ret >= target:
                            steps_to_threshold.append(step)
                            found = True
                            break
                    if not found:
                        steps_to_threshold.append(m["step"][-1])

                if steps_to_threshold:
                    bonus_label = "uncertainty bonus" if bonus_type == "ensemble" else "no bonus"
                    k_label = f"k={int(k)}" if k > 0 else "clean"
                    label = f"{algo.upper()} {k_label} ({bonus_label})"
                    efficiency_data[label] = {
                        "mean": float(np.mean(steps_to_threshold)),
                        "std": float(np.std(steps_to_threshold)),
                    }

        if efficiency_data:
            plot_sample_efficiency(
                efficiency_data,
                title=f"Steps to +{int(IMPROVEMENT_DELTA)} Return - {env}",
                save_path=f"{plot_dir}/sample_efficiency_{env}.png",
            )
            print(f"[plot] Saved sample efficiency for {env}")


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Phase II: Uncertainty-driven online recovery"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Quick run: 5k online steps, single seed")
    parser.add_argument("--plot_only", action="store_true",
                        help="Skip training; regenerate plots from save_dir")
    # Single-run overrides
    parser.add_argument("--algo", type=str, default=None, choices=["cql", "iql"])
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--remove_top_k", type=float, default=None)
    parser.add_argument("--bonus_type", type=str, default=None, choices=["ensemble", "none"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--online_steps", type=int, default=None)
    # Ensemble config
    parser.add_argument("--n_ensemble", type=int, default=3,
                        help="Number of ensemble Q-network members (default: 3)")
    parser.add_argument("--bonus_coeff", type=float, default=1.0,
                        help="Exploration bonus coefficient lambda (default: 1.0)")
    # Directories
    parser.add_argument("--checkpoint_dir", type=str, default="results",
                        help="Directory containing Phase I checkpoints")
    parser.add_argument("--save_dir", type=str, default="results_phase2",
                        help="Output directory for Phase II results")
    parser.add_argument("--plot_dir", type=str, default="plots_phase2",
                        help="Output directory for Phase II figures")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--reward_noise_std", type=float, default=0.0,
                        help="Gaussian reward noise std (stretch goal); 0 = no noise")

    args = parser.parse_args()

    envs = [args.env] if args.env else DEFAULT_ENVS
    algos = [args.algo] if args.algo else DEFAULT_ALGOS
    corruptions = [args.remove_top_k] if args.remove_top_k is not None else DEFAULT_CORRUPTION
    bonus_types = [args.bonus_type] if args.bonus_type else DEFAULT_BONUS_TYPES
    seeds = [args.seed] if args.seed is not None else DEFAULT_SEEDS

    if args.quick:
        steps = QUICK_STEPS
        if args.seed is None:
            seeds = [seeds[0]]  # default to first available seed, don't override explicit --seed
    else:
        steps = args.online_steps or FULL_STEPS

    if not args.plot_only:
        run_experiment_grid(
            envs, algos, corruptions, bonus_types, seeds,
            steps, args.device, args.checkpoint_dir, args.save_dir,
            n_ensemble=args.n_ensemble,
            bonus_coeff=args.bonus_coeff,
            reward_noise_std=args.reward_noise_std,
        )

    generate_plots(
        envs, algos, corruptions, bonus_types, seeds,
        args.save_dir, args.plot_dir, args.reward_noise_std,
    )
    print("\n[done] Phase II complete.")


if __name__ == "__main__":
    main()