"""
Online fine-tuning loop for Phase II.

Loads a Phase I offline checkpoint, then fine-tunes online with:
  - Ensemble Q-networks for uncertainty estimation.
  - Exploration bonus from ensemble disagreement added to rewards.
  - Hybrid replay: mix of offline dataset + online replay buffer.
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import trange

from src.data.loader import load_d4rl_dataset, get_d4rl_normalization_scores
from src.data.corruption import corrupt_dataset
from src.data.dataset import ReplayBuffer
from src.algos.online_cql import OnlineCQL, OnlineCQLConfig
from src.algos.online_iql import OnlineIQL, OnlineIQLConfig
from src.eval.metrics import evaluate_policy, compute_normalized_return


@dataclass
class FinetuneConfig:
    # Phase I checkpoint
    checkpoint_path: str = ""
    # Environment: "halfcheetah-medium-v2" or "hopper-medium-v2"
    env_name: str = "halfcheetah-medium-v2"
    # Algorithm: "cql" or "iql"
    algo: str = "cql"
    # Corruption (to reload the same offline dataset for hybrid replay)
    remove_top_k: float = 0.0
    reward_noise_std: float = 0.0
    corruption_seed: int = 0
    # Ensemble / exploration
    n_ensemble: int = 3
    bonus_coeff: float = 1.0             # lambda for exploration bonus
    bonus_type: str = "ensemble"          # "ensemble" or "none"
    # Online training
    online_steps: int = 250_000
    online_ratio: float = 0.5            # fraction of batch from online buffer
    batch_size: int = 256
    eval_interval: int = 5_000
    eval_episodes: int = 10
    log_interval: int = 1_000
    # CQL-specific online overrides
    cql_alpha_online: float = 1.0        # reduced from 5.0 in offline phase
    # Misc
    seed: int = 0
    device: str = "cpu"
    save_dir: str = "results_phase2"


def finetune(cfg: FinetuneConfig) -> Dict[str, list]:
    """Run a single online fine-tuning experiment from a Phase I checkpoint.

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

    if cfg.device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = cfg.device
    print(f"[finetune] Using device: {device}")

    # Load the same offline dataset for hybrid replay
    print(f"[finetune] Loading offline dataset: {cfg.env_name}")
    offline_dataset, env = load_d4rl_dataset(cfg.env_name)
    if cfg.remove_top_k > 0 or cfg.reward_noise_std > 0:
        offline_dataset = corrupt_dataset(
            offline_dataset,
            remove_top_k=cfg.remove_top_k,
            reward_noise_std=cfg.reward_noise_std,
            seed=cfg.corruption_seed,
        )
    print(f"[finetune] Offline data: {offline_dataset.summary()}")

    # Normalization scores
    random_score, expert_score = get_d4rl_normalization_scores(cfg.env_name)

    obs_dim = offline_dataset.obs_dim
    act_dim = offline_dataset.act_dim

    # ------------------------------------------------------------------ #
    #  Build agent and load checkpoint
    # ------------------------------------------------------------------ #
    print(f"[finetune] Loading checkpoint: {cfg.checkpoint_path}")
    checkpoint = torch.load(cfg.checkpoint_path, map_location=device, weights_only=False)

    if cfg.algo == "cql":
        algo_cfg = OnlineCQLConfig(
            n_ensemble=cfg.n_ensemble,
            bonus_coeff=cfg.bonus_coeff,
            batch_size=cfg.batch_size,
            cql_alpha=cfg.cql_alpha_online,
        )
        agent = OnlineCQL(obs_dim, act_dim, algo_cfg, device=device)
    elif cfg.algo == "iql":
        algo_cfg = OnlineIQLConfig(
            n_ensemble=cfg.n_ensemble,
            bonus_coeff=cfg.bonus_coeff,
            batch_size=cfg.batch_size,
        )
        agent = OnlineIQL(obs_dim, act_dim, algo_cfg, device=device)
    else:
        raise ValueError(f"Unknown algo: {cfg.algo}")

    agent.load_offline_checkpoint(checkpoint)
    print(f"[finetune] {cfg.algo.upper()} loaded with {cfg.n_ensemble}-member ensemble")

    # Online replay buffer
    replay_buffer = ReplayBuffer(obs_dim, act_dim)

    # Output directory
    bonus_tag = cfg.bonus_type if cfg.bonus_type != "none" else "vanilla"
    corruption_tag = f"k{int(cfg.remove_top_k)}_noise{cfg.reward_noise_std}"
    run_name = f"ft_{cfg.algo}_{cfg.env_name}_{corruption_tag}_{bonus_tag}_s{cfg.seed}"
    run_dir = Path(cfg.save_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # ------------------------------------------------------------------ #
    #  Online fine-tuning loop
    # ------------------------------------------------------------------ #
    log: Dict[str, list] = {
        "step": [],
        "normalized_return": [],
        "raw_return_mean": [],
        "raw_return_std": [],
        "ep_len_mean": [],
    }
    train_metrics_window: List[Dict[str, float]] = []

    # Evaluate the offline checkpoint before any online training (step 0)
    eval_res = evaluate_policy(agent, env, n_episodes=cfg.eval_episodes, seed=cfg.seed)
    norm_ret = compute_normalized_return(eval_res["return_mean"], random_score, expert_score)
    log["step"].append(0)
    log["normalized_return"].append(norm_ret)
    log["raw_return_mean"].append(eval_res["return_mean"])
    log["raw_return_std"].append(eval_res["return_std"])
    log["ep_len_mean"].append(eval_res["ep_len_mean"])
    print(f"[finetune] Offline checkpoint performance: norm_return={norm_ret:.1f}")

    # Environment interaction state
    obs, _ = env.reset(seed=cfg.seed)
    ep_return = 0.0
    ep_len = 0
    avg_disagree = 0.0

    start_time = time.time()

    for step in trange(1, cfg.online_steps + 1, desc=run_name):
        # ---- Collect one transition ---------------------------------- #
        action = agent.select_action(np.asarray(obs, dtype=np.float32), deterministic=False)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Compute exploration bonus
        if cfg.bonus_type == "ensemble":
            bonus = agent.compute_exploration_bonus(
                np.asarray(obs, dtype=np.float32), action,
            )
        else:
            bonus = 0.0

        # Store transition with bonus-augmented reward in online buffer
        replay_buffer.add(obs, action, reward + bonus, next_obs, float(terminated))

        ep_return += reward
        ep_len += 1

        if done:
            obs, _ = env.reset()
            ep_return = 0.0
            ep_len = 0
        else:
            obs = next_obs

        # ---- Train with hybrid replay -------------------------------- #
        # Only start training once we have enough online transitions
        if replay_buffer.size >= cfg.batch_size:
            # Split batch: online_ratio from replay buffer, rest from offline
            n_online = max(1, int(cfg.batch_size * cfg.online_ratio))
            n_offline = cfg.batch_size - n_online

            online_batch = replay_buffer.sample(n_online, device=device)
            offline_batch = offline_dataset.sample(n_offline, device=device)

            # Merge the two batches
            batch = {
                k: torch.cat([online_batch[k], offline_batch[k]], dim=0)
                for k in online_batch.keys()
            }

            metrics = agent.update(batch)
            train_metrics_window.append(metrics)
        else:
            # Not enough online data yet -- train on offline only
            batch = offline_dataset.sample(cfg.batch_size, device=device)
            metrics = agent.update(batch)
            train_metrics_window.append(metrics)

        # ---- Periodic logging ---------------------------------------- #
        if step % cfg.log_interval == 0 and train_metrics_window:
            recent = train_metrics_window[-cfg.log_interval:]
            avg_disagree = np.mean([m.get("ensemble_disagreement", 0) for m in recent])
            avg_critic = np.mean([m["critic_loss"] for m in recent])
            log.setdefault("train_step", []).append(step)
            log.setdefault("ensemble_disagreement", []).append(float(avg_disagree))
            log.setdefault("critic_loss_avg", []).append(float(avg_critic))

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
            disagree_str = f" | disagree={avg_disagree:.3f}" if cfg.bonus_type == "ensemble" else ""
            print(
                f"\n[eval] step={step:,} | "
                f"norm_return={norm_ret:.1f} | "
                f"raw_return={eval_res['return_mean']:.1f}+/-{eval_res['return_std']:.1f}"
                f"{disagree_str} | "
                f"time={elapsed:.0f}s"
            )

    # ------------------------------------------------------------------ #
    #  Save results
    # ------------------------------------------------------------------ #
    def _convert(obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    serializable_log = {k: [_convert(x) for x in v] for k, v in log.items()}
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(serializable_log, f, indent=2)

    torch.save(agent.state_dict(), run_dir / "checkpoint.pt")

    print(f"\n[finetune] Done. Results saved to {run_dir}")
    return log