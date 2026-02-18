"""
Synthetic dataset corruption utilities.

Implements the corruption strategies described in the proposal:
  1. Remove top-k% of trajectories ranked by episodic return.
  2. Add Gaussian noise to reward signals (stretch-goal).
"""

from __future__ import annotations
from typing import List, Tuple

import numpy as np

from src.data.dataset import OfflineDataset


# --------------------------------------------------------------------------- #
#  Internal helpers
# --------------------------------------------------------------------------- #

def _segment_into_trajectories(dataset: OfflineDataset) -> List[Tuple[int, int]]:
    """Return a list of (start_idx, end_idx) for each trajectory.

    A trajectory ends when *either* ``terminals[i]`` or ``timeouts[i]`` is 1.
    """
    boundaries: List[Tuple[int, int]] = []
    start = 0
    for i in range(dataset.size):
        if dataset.terminals[i] == 1.0 or dataset.timeouts[i] == 1.0:
            boundaries.append((start, i + 1))  # end is exclusive
            start = i + 1
    # Handle trailing transitions with no terminal flag.
    if start < dataset.size:
        boundaries.append((start, dataset.size))
    return boundaries


def _trajectory_returns(
    dataset: OfflineDataset,
    boundaries: List[Tuple[int, int]],
) -> np.ndarray:
    """Compute undiscounted episodic return for each trajectory."""
    returns = np.array(
        [dataset.rewards[s:e].sum() for s, e in boundaries],
        dtype=np.float64,
    )
    return returns


def _rebuild_dataset(
    dataset: OfflineDataset,
    keep_mask: np.ndarray,
) -> OfflineDataset:
    """Build a new ``OfflineDataset`` keeping only transitions where *keep_mask* is True."""
    return OfflineDataset(
        observations=dataset.observations[keep_mask].copy(),
        actions=dataset.actions[keep_mask].copy(),
        rewards=dataset.rewards[keep_mask].copy(),
        next_observations=dataset.next_observations[keep_mask].copy(),
        terminals=dataset.terminals[keep_mask].copy(),
        timeouts=dataset.timeouts[keep_mask].copy(),
    )


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #

def remove_top_k_trajectories(
    dataset: OfflineDataset,
    k_pct: float,
    seed: int = 0,
) -> OfflineDataset:
    """Remove the top-*k_pct*% trajectories ranked by episodic return.

    This simulates realistic coverage gaps where high-quality demonstrations
    are missing from the training data.

    Parameters
    ----------
    dataset : OfflineDataset
    k_pct : float
        Percentage of trajectories to remove, e.g. ``30`` or ``60``.
    seed : int
        RNG seed (used only for tie-breaking; removal is deterministic by rank).

    Returns
    -------
    OfflineDataset
        A new dataset with the top-performing trajectories removed.
    """
    if not (0 < k_pct < 100):
        raise ValueError(f"k_pct must be in (0, 100), got {k_pct}")

    boundaries = _segment_into_trajectories(dataset)
    returns = _trajectory_returns(dataset, boundaries)

    n_trajs = len(boundaries)
    n_remove = max(1, int(np.ceil(n_trajs * k_pct / 100)))

    # Indices of trajectories sorted by return (ascending).
    sorted_idxs = np.argsort(returns)
    keep_traj_idxs = set(sorted_idxs[:-n_remove].tolist())

    # Build a per-transition boolean mask.
    keep_mask = np.zeros(dataset.size, dtype=bool)
    for traj_idx, (s, e) in enumerate(boundaries):
        if traj_idx in keep_traj_idxs:
            keep_mask[s:e] = True

    new_ds = _rebuild_dataset(dataset, keep_mask)
    removed_pct = 100 * (1 - new_ds.size / dataset.size)
    print(
        f"[corruption] Removed top {k_pct}% trajs: "
        f"{n_remove}/{n_trajs} trajectories, "
        f"{dataset.size - new_ds.size} transitions ({removed_pct:.1f}% of data)"
    )
    return new_ds


def add_reward_noise(
    dataset: OfflineDataset,
    noise_std: float,
    seed: int = 0,
) -> OfflineDataset:
    """Add i.i.d. Gaussian noise to reward signals.

    Parameters
    ----------
    dataset : OfflineDataset
    noise_std : float
        Standard deviation of Gaussian noise.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    OfflineDataset
        A new dataset with noisy rewards (all other fields are shared, not copied).
    """
    rng = np.random.RandomState(seed)
    noise = rng.randn(dataset.size).astype(np.float32) * noise_std
    noisy_rewards = dataset.rewards + noise

    print(
        f"[corruption] Added Gaussian reward noise (std={noise_std:.3f}). "
        f"Reward shift: mean={noise.mean():.4f}, std={noise.std():.4f}"
    )

    return OfflineDataset(
        observations=dataset.observations,
        actions=dataset.actions,
        rewards=noisy_rewards,
        next_observations=dataset.next_observations,
        terminals=dataset.terminals,
        timeouts=dataset.timeouts,
    )


def corrupt_dataset(
    dataset: OfflineDataset,
    remove_top_k: float = 0.0,
    reward_noise_std: float = 0.0,
    seed: int = 0,
) -> OfflineDataset:
    """Apply a pipeline of corruption strategies.

    Convenience wrapper that chains trajectory removal and reward noise.
    """
    ds = dataset
    if remove_top_k > 0:
        ds = remove_top_k_trajectories(ds, k_pct=remove_top_k, seed=seed)
    if reward_noise_std > 0:
        ds = add_reward_noise(ds, noise_std=reward_noise_std, seed=seed)
    return ds
