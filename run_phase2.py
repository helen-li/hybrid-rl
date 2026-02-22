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
from src.eval.plotting import plot_finetuning_curves, plot_sample_efficiency


# --------------------------------------------------------------------------- #
#  Experiment grid defaults
# --------------------------------------------------------------------------- #

DEFAULT_ENVS = ["halfcheetah-medium-v2", "hopper-medium-v2"]
DEFAULT_ALGOS = ["cql", "iql"]
DEFAULT_CORRUPTION = [0.0, 30.0, 60.0]
DEFAULT_BONUS_TYPES = ["ensemble", "none"]
DEFAULT_SEEDS = [0, 1, 2]

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

                bonus_label = "ensemble" if bonus_type == "ensemble" else "vanilla"
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

        # ---- Sample efficiency (one plot per env) -------------------- #
        # Compare ensemble vs vanilla across corruption levels
        efficiency_data: Dict[str, Dict[str, float]] = {}

        for algo, k in product(algos, corruptions):
            for bonus_type in bonus_types:
                steps_to_threshold = []
                for seed in seeds:
                    m = _load_metrics(save_dir, algo, env, k, bonus_type, seed, reward_noise_std)
                    if m is None:
                        continue
                    # Find first step where normalized return exceeds threshold.
                    # Use the clean-dataset offline performance as threshold.
                    clean_m = _load_metrics(save_dir, algo, env, 0.0, bonus_type, seed, reward_noise_std)
                    if clean_m is None or not clean_m["normalized_return"]:
                        continue
                    # Threshold: 80% of the clean offline starting performance
                    threshold = clean_m["normalized_return"][0] * 0.8
                    found = False
                    for step, ret in zip(m["step"], m["normalized_return"]):
                        if ret >= threshold:
                            steps_to_threshold.append(step)
                            found = True
                            break
                    if not found:
                        steps_to_threshold.append(m["step"][-1])

                if steps_to_threshold:
                    bonus_label = "ensemble" if bonus_type == "ensemble" else "vanilla"
                    k_label = f"k={int(k)}" if k > 0 else "clean"
                    label = f"{algo.upper()} {k_label} ({bonus_label})"
                    efficiency_data[label] = {
                        "mean": float(np.mean(steps_to_threshold)),
                        "std": float(np.std(steps_to_threshold)),
                    }

        if efficiency_data:
            plot_sample_efficiency(
                efficiency_data,
                title=f"Sample Efficiency - {env}",
                save_path=f"{plot_dir}/sample_efficiency_{env}.png",
            )
            print(f"[plot] Saved sample efficiency plot for {env}")


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
        seeds = [0]
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