"""
Unified offline training loop for Phase I.

Trains CQL or IQL on a (possibly corrupted) D4RL dataset and
periodically evaluates in the environment, logging all metrics.
"""

from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import trange

from src.data.loader import load_d4rl_dataset, get_d4rl_normalization_scores
from src.data.corruption import corrupt_dataset
from src.algos.cql import CQL, CQLConfig
from src.algos.iql import IQL, IQLConfig
from src.eval.metrics import (
    evaluate_policy,
    compute_normalized_return,
    compute_q_stats,
    rolling_loss_variance,
)


@dataclass
class TrainConfig:
    # Environment
    env_name: str = "halfcheetah-medium-v2"
    # Algorithm
    algo: str = "cql"                     # "cql" or "iql"
    # Corruption
    remove_top_k: float = 0.0            # 0 = clean, 30, 60
    reward_noise_std: float = 0.0        # 0 = no noise
    corruption_seed: int = 0
    # Training
    total_steps: int = 500_000
    batch_size: int = 256
    eval_interval: int = 10_000          # Evaluate every N gradient steps
    eval_episodes: int = 10
    log_interval: int = 1_000            # Log training metrics every N steps
    # Misc
    seed: int = 42
    device: str = "cpu"
    save_dir: str = "results"


def train(cfg: TrainConfig) -> Dict[str, list]:
    """Run a single offline training experiment.

    Returns
    -------
    log : dict
        All logged metrics, keyed by name, each value is a list over time.
    """
    # ------------------------------------------------------------------ #
    #  Setup
    # ------------------------------------------------------------------ #
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Determine device
    if cfg.device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = cfg.device
    print(f"[train] Using device: {device}")

    # Load data
    print(f"[train] Loading dataset: {cfg.env_name}")
    dataset, env = load_d4rl_dataset(cfg.env_name)
    print(f"[train] {dataset.summary()}")

    # Apply corruption
    if cfg.remove_top_k > 0 or cfg.reward_noise_std > 0:
        dataset = corrupt_dataset(
            dataset,
            remove_top_k=cfg.remove_top_k,
            reward_noise_std=cfg.reward_noise_std,
            seed=cfg.corruption_seed,
        )
        print(f"[train] After corruption: {dataset.summary()}")

    # Normalization scores
    random_score, expert_score = get_d4rl_normalization_scores(cfg.env_name)

    # Build agent
    obs_dim = dataset.obs_dim
    act_dim = dataset.act_dim

    if cfg.algo == "cql":
        algo_cfg = CQLConfig(batch_size=cfg.batch_size)
        agent = CQL(obs_dim, act_dim, algo_cfg, device=device)
    elif cfg.algo == "iql":
        algo_cfg = IQLConfig(batch_size=cfg.batch_size)
        agent = IQL(obs_dim, act_dim, algo_cfg, device=device)
    else:
        raise ValueError(f"Unknown algo: {cfg.algo}")

    print(f"[train] Algorithm: {cfg.algo.upper()}, obs_dim={obs_dim}, act_dim={act_dim}")

    # Output directory
    corruption_tag = f"k{int(cfg.remove_top_k)}_noise{cfg.reward_noise_std}"
    run_name = f"{cfg.algo}_{cfg.env_name}_{corruption_tag}_s{cfg.seed}"
    run_dir = Path(cfg.save_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # ------------------------------------------------------------------ #
    #  Training loop
    # ------------------------------------------------------------------ #
    log: Dict[str, list] = {
        "step": [],
        "normalized_return": [],
        "raw_return_mean": [],
        "raw_return_std": [],
        "ep_len_mean": [],
    }
    train_metrics_window: List[Dict[str, float]] = []
    critic_loss_history: List[float] = []

    start_time = time.time()

    for step in trange(1, cfg.total_steps + 1, desc=run_name):
        batch = dataset.sample(cfg.batch_size, device=device)
        metrics = agent.update(batch)

        train_metrics_window.append(metrics)
        critic_loss_history.append(metrics["critic_loss"])

        # ---- Periodic logging ---------------------------------------- #
        if step % cfg.log_interval == 0:
            q_stats = compute_q_stats(train_metrics_window[-cfg.log_interval:])
            # Store in log
            for k, v in q_stats.items():
                log.setdefault(k, []).append(v)
            log.setdefault("train_step", []).append(step)

        # ---- Periodic evaluation ------------------------------------- #
        if step % cfg.eval_interval == 0:
            eval_res = evaluate_policy(agent, env, n_episodes=cfg.eval_episodes, seed=cfg.seed)
            norm_ret = compute_normalized_return(eval_res["return_mean"], random_score, expert_score)

            log["step"].append(step)
            log["normalized_return"].append(norm_ret)
            log["raw_return_mean"].append(eval_res["return_mean"])
            log["raw_return_std"].append(eval_res["return_std"])
            log["ep_len_mean"].append(eval_res["ep_len_mean"])

            elapsed = time.time() - start_time
            print(
                f"\n[eval] step={step:,} | "
                f"norm_return={norm_ret:.1f} | "
                f"raw_return={eval_res['return_mean']:.1f}+/-{eval_res['return_std']:.1f} | "
                f"ep_len={eval_res['ep_len_mean']:.0f} | "
                f"critic_loss={metrics['critic_loss']:.4f} | "
                f"time={elapsed:.0f}s"
            )

    # ------------------------------------------------------------------ #
    #  Save results
    # ------------------------------------------------------------------ #
    # Rolling loss variance
    loss_var = rolling_loss_variance(critic_loss_history, window=1000)
    log["critic_loss_rolling_var"] = loss_var

    # Save log
    # Convert numpy types for JSON serialization
    def _convert(obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    serializable_log = {k: [_convert(x) for x in v] for k, v in log.items()}
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(serializable_log, f, indent=2)

    # Save model checkpoint
    torch.save(agent.state_dict(), run_dir / "checkpoint.pt")

    print(f"\n[train] Done. Results saved to {run_dir}")
    return log
