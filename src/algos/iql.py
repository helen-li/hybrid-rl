"""
Implicit Q-Learning (IQL).

Reference: Kostrikov, Nair, Levine, "Offline Reinforcement Learning with
Implicit Q-Learning", ICLR 2022.  https://arxiv.org/abs/2110.06169

IQL avoids querying out-of-distribution actions entirely by:
  1. Learning V(s) via expectile regression on Q - V advantages.
  2. Training Q with standard Bellman targets using V (not max-Q).
  3. Extracting a policy via advantage-weighted regression (AWR).
"""

from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.algos.networks import (
    DoubleQNetwork,
    ValueNetwork,
    TanhGaussianPolicy,
    soft_update,
)


@dataclass
class IQLConfig:
    # Architecture
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    # Optimisation
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_value: float = 3e-4
    batch_size: int = 256
    discount: float = 0.99
    tau: float = 0.005
    # IQL-specific
    expectile: float = 0.7            # tau for expectile regression (0.5 = mean)
    temperature: float = 3.0          # beta for AWR policy extraction
    clip_score: float = 100.0         # Clip exp-advantage for stability


class IQL:
    """Implicit Q-Learning."""

    def __init__(self, obs_dim: int, act_dim: int, cfg: IQLConfig | None = None, device: str = "cpu"):
        self.cfg = cfg or IQLConfig()
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Networks ----------------------------------------------------------
        self.actor = TanhGaussianPolicy(obs_dim, act_dim, self.cfg.hidden_dims).to(device)
        self.critic = DoubleQNetwork(obs_dim, act_dim, self.cfg.hidden_dims).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        self.critic_target.requires_grad_(False)
        self.value = ValueNetwork(obs_dim, self.cfg.hidden_dims).to(device)

        # Optimisers --------------------------------------------------------
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.lr_critic)
        self.value_optim = torch.optim.Adam(self.value.parameters(), lr=self.cfg.lr_value)

        self._step = 0

    # ------------------------------------------------------------------ #
    #  Expectile loss
    # ------------------------------------------------------------------ #

    @staticmethod
    def _expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
        weight = torch.where(diff > 0, expectile, 1.0 - expectile)
        return (weight * diff.pow(2)).mean()

    # ------------------------------------------------------------------ #
    #  Core update
    # ------------------------------------------------------------------ #

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch["observations"]
        act = batch["actions"]
        rew = batch["rewards"]
        next_obs = batch["next_observations"]
        done = batch["terminals"]

        metrics: Dict[str, float] = {}

        # ----- Value loss (expectile regression) -------------------------- #
        with torch.no_grad():
            q1_target, q2_target = self.critic_target(obs, act)
            q_target = torch.min(q1_target, q2_target)

        v = self.value(obs)
        value_loss = self._expectile_loss(q_target - v, self.cfg.expectile)

        self.value_optim.zero_grad()
        value_loss.backward()
        self.value_optim.step()

        # ----- Critic loss (standard Bellman with V target) --------------- #
        with torch.no_grad():
            next_v = self.value(next_obs)
            target_q = rew + (1.0 - done) * self.cfg.discount * next_v

        q1, q2 = self.critic(obs, act)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # ----- Actor loss (advantage-weighted regression) ----------------- #
        with torch.no_grad():
            advantage = q_target - v
            exp_adv = torch.exp(advantage / self.cfg.temperature).clamp(max=self.cfg.clip_score)

        log_prob = self.actor.log_prob(obs, act)
        actor_loss = -(exp_adv * log_prob).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # ----- Target update ---------------------------------------------- #
        soft_update(self.critic_target, self.critic, self.cfg.tau)

        self._step += 1

        # ----- Diagnostics ------------------------------------------------ #
        metrics.update({
            "critic_loss": critic_loss.item(),
            "value_loss": value_loss.item(),
            "actor_loss": actor_loss.item(),
            "q1_mean": q1.mean().item(),
            "q2_mean": q2.mean().item(),
            "q1_std": q1.std().item(),
            "q2_std": q2.std().item(),
            "v_mean": v.mean().item(),
            "advantage_mean": advantage.mean().item(),
            "target_q_mean": target_q.mean().item(),
        })
        return metrics

    # ------------------------------------------------------------------ #
    #  Inference
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.actor.act(obs_t, deterministic=deterministic)
        return action.cpu().numpy().flatten()

    # ------------------------------------------------------------------ #
    #  Save / Load
    # ------------------------------------------------------------------ #

    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "value": self.value.state_dict(),
            "step": self._step,
        }

    def load_state_dict(self, d: dict):
        self.actor.load_state_dict(d["actor"])
        self.critic.load_state_dict(d["critic"])
        self.critic_target.load_state_dict(d["critic_target"])
        self.value.load_state_dict(d["value"])
        self._step = d["step"]
