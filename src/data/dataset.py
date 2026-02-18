"""
Offline RL dataset and replay buffer utilities.

Provides a simple numpy-backed dataset for offline training and a replay
buffer that can be used during online fine-tuning (Phase II).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch


@dataclass
class OfflineDataset:
    """Holds a complete offline dataset in flat numpy arrays.

    Fields
    ------
    observations : (N, obs_dim)
    actions      : (N, act_dim)
    rewards      : (N,)
    next_observations : (N, obs_dim)
    terminals    : (N,)   -- 1.0 if the episode truly ended
    timeouts     : (N,)   -- 1.0 if the episode was truncated
    """
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminals: np.ndarray
    timeouts: np.ndarray

    # ---- helpers ---------------------------------------------------------- #

    @property
    def size(self) -> int:
        return self.observations.shape[0]

    @property
    def obs_dim(self) -> int:
        return self.observations.shape[1]

    @property
    def act_dim(self) -> int:
        return self.actions.shape[1]

    def sample(self, batch_size: int, device: str = "cpu") -> Dict[str, torch.Tensor]:
        """Random mini-batch, returned as tensors on *device*."""
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "observations": torch.as_tensor(self.observations[idx], dtype=torch.float32, device=device),
            "actions": torch.as_tensor(self.actions[idx], dtype=torch.float32, device=device),
            "rewards": torch.as_tensor(self.rewards[idx], dtype=torch.float32, device=device),
            "next_observations": torch.as_tensor(self.next_observations[idx], dtype=torch.float32, device=device),
            "terminals": torch.as_tensor(self.terminals[idx], dtype=torch.float32, device=device),
        }

    def summary(self) -> str:
        """One-line human-readable summary."""
        return (
            f"OfflineDataset(size={self.size}, obs_dim={self.obs_dim}, "
            f"act_dim={self.act_dim}, reward in [{self.rewards.min():.2f}, {self.rewards.max():.2f}])"
        )


class ReplayBuffer:
    """Simple ring-buffer for online transitions (Phase II)."""

    def __init__(self, obs_dim: int, act_dim: int, capacity: int = 1_000_000):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.terminals = np.zeros(capacity, dtype=np.float32)

    def add(self, obs, action, reward, next_obs, terminal):
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_obs
        self.terminals[self.ptr] = terminal

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: str = "cpu") -> Dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "observations": torch.as_tensor(self.observations[idx], dtype=torch.float32, device=device),
            "actions": torch.as_tensor(self.actions[idx], dtype=torch.float32, device=device),
            "rewards": torch.as_tensor(self.rewards[idx], dtype=torch.float32, device=device),
            "next_observations": torch.as_tensor(self.next_observations[idx], dtype=torch.float32, device=device),
            "terminals": torch.as_tensor(self.terminals[idx], dtype=torch.float32, device=device),
        }
