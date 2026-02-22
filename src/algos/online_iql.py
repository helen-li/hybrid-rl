"""
Online IQL with ensemble critic for Phase II fine-tuning.

Extends the offline IQL agent with:
  - An ensemble of Q-networks (default 3) for uncertainty estimation.
  - Ensemble disagreement used as an exploration bonus.
"""

from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from src.algos.networks import (
    EnsembleQNetwork,
    DoubleQNetwork,
    ValueNetwork,
    TanhGaussianPolicy,
    soft_update,
)


@dataclass
class OnlineIQLConfig:
    # Architecture
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    # Ensemble
    n_ensemble: int = 3
    bonus_coeff: float = 1.0         # lambda for exploration bonus
    # Optimisation
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_value: float = 3e-4
    batch_size: int = 256
    discount: float = 0.99
    tau: float = 0.005
    # IQL-specific
    expectile: float = 0.7
    temperature: float = 3.0
    clip_score: float = 100.0


class OnlineIQL:
    """IQL with ensemble critic for online fine-tuning from offline checkpoints."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        cfg: OnlineIQLConfig | None = None,
        device: str = "cpu",
    ):
        self.cfg = cfg or OnlineIQLConfig()
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Actor (will be loaded from checkpoint) --------------------------------
        self.actor = TanhGaussianPolicy(obs_dim, act_dim, self.cfg.hidden_dims).to(device)

        # Ensemble critic -------------------------------------------------------
        self.critic = EnsembleQNetwork(
            obs_dim, act_dim, self.cfg.hidden_dims, n_members=self.cfg.n_ensemble,
        ).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        self.critic_target.requires_grad_(False)

        # Value network (same as offline IQL) -----------------------------------
        self.value = ValueNetwork(obs_dim, self.cfg.hidden_dims).to(device)

        # Optimisers ------------------------------------------------------------
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.lr_critic)
        self.value_optim = torch.optim.Adam(self.value.parameters(), lr=self.cfg.lr_value)

        self._step = 0

    # ------------------------------------------------------------------ #
    #  Load from Phase I checkpoint
    # ------------------------------------------------------------------ #

    def load_offline_checkpoint(self, checkpoint: dict):
        """Initialise from a Phase I IQL checkpoint.

        - Actor and value network weights are copied directly.
        - First 2 ensemble members are initialised from the saved DoubleQNetwork.
        - Remaining members keep random init for diversity.
        """
        # Actor
        self.actor.load_state_dict(checkpoint["actor"])

        # Value network
        self.value.load_state_dict(checkpoint["value"])

        # Critic: reconstruct a temporary DoubleQNetwork to load weights
        tmp_double_q = DoubleQNetwork(self.obs_dim, self.act_dim, self.cfg.hidden_dims)
        tmp_double_q.load_state_dict(checkpoint["critic"])
        self.critic.load_double_q(tmp_double_q)

        # Copy into target as well
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.critic_target.requires_grad_(False)

        # Rebuild optimisers so they track the new parameters
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.lr_critic)
        self.value_optim = torch.optim.Adam(self.value.parameters(), lr=self.cfg.lr_value)

        self._step = checkpoint.get("step", 0)

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

        # ----- Value loss (expectile regression against ensemble min-Q) --- #
        with torch.no_grad():
            q_target = self.critic_target.q_min(obs, act)

        v = self.value(obs)
        value_loss = self._expectile_loss(q_target - v, self.cfg.expectile)

        self.value_optim.zero_grad()
        value_loss.backward()
        self.value_optim.step()

        # ----- Critic loss (Bellman with V target, all ensemble members) -- #
        with torch.no_grad():
            next_v = self.value(next_obs)
            bellman_target = rew + (1.0 - done) * self.cfg.discount * next_v

        qs = self.critic(obs, act)  # list of N tensors
        critic_loss = sum(F.mse_loss(q, bellman_target) for q in qs)

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
        q_stack = torch.stack(qs)  # (N, batch)
        metrics.update({
            "critic_loss": critic_loss.item(),
            "value_loss": value_loss.item(),
            "actor_loss": actor_loss.item(),
            "q_mean": q_stack.mean().item(),
            "q_std": q_stack.std(dim=0).mean().item(),
            "v_mean": v.mean().item(),
            "advantage_mean": advantage.mean().item(),
            "target_q_mean": bellman_target.mean().item(),
            "ensemble_disagreement": q_stack.std(dim=0).mean().item(),
        })
        return metrics

    # ------------------------------------------------------------------ #
    #  Exploration bonus
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def compute_exploration_bonus(self, obs: np.ndarray, act: np.ndarray) -> float:
        """Return scalar exploration bonus: lambda * ensemble disagreement."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        act_t = torch.as_tensor(act, dtype=torch.float32, device=self.device).unsqueeze(0)
        disagreement = self.critic.disagreement(obs_t, act_t).item()
        return self.cfg.bonus_coeff * disagreement

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