"""
Evaluation metrics for offline RL experiments.

Includes:
  - Normalized returns (D4RL convention)
  - Q-value statistics for stability analysis
  - Rolling loss variance
"""

from __future__ import annotations
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np


def evaluate_policy(
    agent,
    env: gym.Env,
    n_episodes: int = 10,
    deterministic: bool = True,
    seed: int | None = None,
) -> Dict[str, float]:
    """Roll out *agent* in *env* and return summary statistics.

    Returns
    -------
    dict with keys:
        return_mean, return_std, return_min, return_max, ep_len_mean
    """
    returns: List[float] = []
    lengths: List[int] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep if seed is not None else None)
        done = False
        ep_return = 0.0
        ep_len = 0
        while not done:
            action = agent.select_action(np.asarray(obs, dtype=np.float32), deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_return += reward
            ep_len += 1
        returns.append(ep_return)
        lengths.append(ep_len)

    returns_arr = np.array(returns)
    return {
        "return_mean": float(returns_arr.mean()),
        "return_std": float(returns_arr.std()),
        "return_min": float(returns_arr.min()),
        "return_max": float(returns_arr.max()),
        "ep_len_mean": float(np.mean(lengths)),
    }


def compute_normalized_return(
    raw_return: float,
    random_score: float,
    expert_score: float,
) -> float:
    """D4RL normalized score: 100 * (score - random) / (expert - random)."""
    return 100.0 * (raw_return - random_score) / max(expert_score - random_score, 1e-8)


def compute_q_stats(metrics_history: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate Q-value diagnostics over a window of training steps.

    Works for both Phase I (DoubleQNetwork with q1_mean/q2_mean) and
    Phase II (EnsembleQNetwork with q_mean/q_std).  Auto-detects which
    keys are present and returns the appropriate aggregates.

    Parameters
    ----------
    metrics_history : list of per-step metric dicts (from agent.update()).

    Returns
    -------
    dict with aggregated stats.
    Phase I keys: q1_mean_avg, q1_mean_std, q2_mean_avg, q2_mean_std
    Phase II keys: q_mean_avg, q_mean_std, ensemble_disagreement_avg,
                   ensemble_disagreement_std
    Always: critic_loss_avg, critic_loss_var
    """
    def _agg(key: str) -> Tuple[float, float]:
        vals = [m[key] for m in metrics_history if key in m]
        if not vals:
            return 0.0, 0.0
        arr = np.array(vals)
        return float(arr.mean()), float(arr.std())

    result: Dict[str, float] = {}

    # Phase I: DoubleQNetwork keys
    if any("q1_mean" in m for m in metrics_history):
        q1_mean, q1_std = _agg("q1_mean")
        q2_mean, q2_std = _agg("q2_mean")
        result.update({
            "q1_mean_avg": q1_mean,
            "q1_mean_std": q1_std,
            "q2_mean_avg": q2_mean,
            "q2_mean_std": q2_std,
        })

    # Phase II: EnsembleQNetwork keys
    if any("q_mean" in m for m in metrics_history):
        q_mean, q_std = _agg("q_mean")
        q_std_mean, q_std_std = _agg("q_std")
        disagree_mean, disagree_std = _agg("ensemble_disagreement")
        result.update({
            "q_mean_avg": q_mean,
            "q_mean_std": q_std,
            "q_std_avg": q_std_mean,
            "q_std_std": q_std_std,
            "ensemble_disagreement_avg": disagree_mean,
            "ensemble_disagreement_std": disagree_std,
        })

    cl_mean, cl_std = _agg("critic_loss")
    result.update({
        "critic_loss_avg": cl_mean,
        "critic_loss_var": cl_std ** 2,
    })

    return result


def rolling_loss_variance(
    loss_history: List[float],
    window: int = 1000,
) -> List[float]:
    """Compute rolling variance of a loss signal.

    Useful for detecting training instability / value collapse.
    """
    if len(loss_history) < window:
        return [float(np.var(loss_history))] if loss_history else [0.0]
    arr = np.array(loss_history)
    # Use a simple sliding window.
    variances = []
    for i in range(window, len(arr) + 1):
        variances.append(float(arr[i - window:i].var()))
    return variances
