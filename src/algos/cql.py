"""
Conservative Q-Learning (CQL).

Reference: Kumar et al., "Conservative Q-Learning for Offline Reinforcement
Learning", NeurIPS 2020.  https://arxiv.org/abs/2006.04779

This is the continuous-action variant (CQL(H)) that uses a SAC-style actor
and adds a conservative regulariser to the Q-loss.
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
    TanhGaussianPolicy,
    soft_update,
)


@dataclass
class CQLConfig:
    # Architecture
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    # Optimisation
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    batch_size: int = 256
    discount: float = 0.99
    tau: float = 0.005
    # SAC entropy
    init_alpha: float = 1.0
    learnable_alpha: bool = True
    lr_alpha: float = 3e-4
    # CQL-specific
    cql_alpha: float = 5.0           # Weight of the conservative penalty.
    cql_n_samples: int = 10          # Num action samples for log-sum-exp.
    cql_target_action_gap: float = -1.0  # <0 means fixed alpha, no Lagrange.
    cql_lagrange: bool = False       # Whether to use Lagrange dual for alpha.
    lr_cql_alpha: float = 3e-4


class CQL:
    """CQL(H) -- Conservative Q-Learning with SAC backbone."""

    def __init__(self, obs_dim: int, act_dim: int, cfg: CQLConfig | None = None, device: str = "cpu"):
        self.cfg = cfg or CQLConfig()
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Networks ----------------------------------------------------------
        self.actor = TanhGaussianPolicy(obs_dim, act_dim, self.cfg.hidden_dims).to(device)
        self.critic = DoubleQNetwork(obs_dim, act_dim, self.cfg.hidden_dims).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        self.critic_target.requires_grad_(False)

        # Optimisers --------------------------------------------------------
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.lr_critic)

        # SAC log-alpha (entropy coefficient) --------------------------------
        self.target_entropy = -float(act_dim)
        self.log_alpha = torch.tensor(
            np.log(self.cfg.init_alpha), dtype=torch.float32, device=device, requires_grad=self.cfg.learnable_alpha
        )
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.cfg.lr_alpha) if self.cfg.learnable_alpha else None

        # CQL log-alpha (conservative penalty weight) -----------------------
        if self.cfg.cql_lagrange:
            self.log_cql_alpha = torch.tensor(
                np.log(self.cfg.cql_alpha), dtype=torch.float32, device=device, requires_grad=True
            )
            self.cql_alpha_optim = torch.optim.Adam([self.log_cql_alpha], lr=self.cfg.lr_cql_alpha)
        else:
            self.log_cql_alpha = None
            self.cql_alpha_optim = None

        # Diagnostics -------------------------------------------------------
        self._step = 0

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

        # ----- Critic loss ------------------------------------------------ #
        with torch.no_grad():
            next_act, next_log_prob = self.actor.sample(next_obs)
            q1_next, q2_next = self.critic_target(next_obs, next_act)
            q_next = torch.min(q1_next, q2_next) - self.log_alpha.exp() * next_log_prob
            target_q = rew + (1.0 - done) * self.cfg.discount * q_next

        q1, q2 = self.critic(obs, act)
        bellman_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        # CQL conservative penalty -----------------------------------------
        cql_penalty = self._cql_penalty(obs, act, q1, q2)
        cql_alpha = self.log_cql_alpha.exp().detach() if self.log_cql_alpha is not None else self.cfg.cql_alpha

        critic_loss = bellman_loss + cql_alpha * cql_penalty

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Lagrange update for CQL alpha
        if self.cfg.cql_lagrange:
            cql_alpha_loss = -self.log_cql_alpha.exp() * (cql_penalty.detach() - self.cfg.cql_target_action_gap)
            self.cql_alpha_optim.zero_grad()
            cql_alpha_loss.backward()
            self.cql_alpha_optim.step()
            metrics["cql_alpha"] = self.log_cql_alpha.exp().item()

        # ----- Actor loss (SAC-style) ------------------------------------- #
        new_act, new_log_prob = self.actor.sample(obs)
        q1_new, q2_new = self.critic(obs, new_act)
        q_new = torch.min(q1_new, q2_new)
        alpha = self.log_alpha.exp().detach()

        actor_loss = (alpha * new_log_prob - q_new).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # ----- Alpha loss ------------------------------------------------- #
        if self.alpha_optim is not None:
            alpha_loss = -(self.log_alpha.exp() * (new_log_prob.detach() + self.target_entropy)).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            metrics["alpha"] = self.log_alpha.exp().item()

        # ----- Target update ---------------------------------------------- #
        soft_update(self.critic_target, self.critic, self.cfg.tau)

        self._step += 1

        # ----- Diagnostics ------------------------------------------------ #
        metrics.update({
            "critic_loss": critic_loss.item(),
            "bellman_loss": bellman_loss.item(),
            "cql_penalty": cql_penalty.item(),
            "actor_loss": actor_loss.item(),
            "q1_mean": q1.mean().item(),
            "q2_mean": q2.mean().item(),
            "q1_std": q1.std().item(),
            "q2_std": q2.std().item(),
            "target_q_mean": target_q.mean().item(),
        })
        return metrics

    # ------------------------------------------------------------------ #
    #  CQL conservative penalty
    # ------------------------------------------------------------------ #

    def _cql_penalty(
        self,
        obs: torch.Tensor,
        data_act: torch.Tensor,
        q1_data: torch.Tensor,
        q2_data: torch.Tensor,
    ) -> torch.Tensor:
        """CQL(H) penalty: log-sum-exp(Q) over random + policy actions minus Q on data."""
        B = obs.shape[0]
        N = self.cfg.cql_n_samples

        # Sample random uniform actions.
        rand_act = torch.FloatTensor(B * N, self.act_dim).uniform_(-1, 1).to(self.device)
        obs_rep = obs.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)

        # Sample actions from current policy.
        with torch.no_grad():
            policy_act, policy_log_prob = self.actor.sample(obs_rep)

        # Q-values for random actions.
        q1_rand, q2_rand = self.critic(obs_rep, rand_act)
        q1_rand = q1_rand.reshape(B, N)
        q2_rand = q2_rand.reshape(B, N)
        # Subtract log(uniform density) = log(1/(2^act_dim)) = act_dim * log(2)
        rand_density_correction = np.log(0.5 ** self.act_dim)

        # Q-values for policy actions.
        q1_pi, q2_pi = self.critic(obs_rep, policy_act)
        q1_pi = q1_pi.reshape(B, N)
        q2_pi = q2_pi.reshape(B, N)
        log_prob_pi = policy_log_prob.reshape(B, N)

        # log-sum-exp across both random and policy samples.
        cat_q1 = torch.cat([q1_rand - rand_density_correction, q1_pi - log_prob_pi.detach()], dim=1)
        cat_q2 = torch.cat([q2_rand - rand_density_correction, q2_pi - log_prob_pi.detach()], dim=1)

        cql_q1 = torch.logsumexp(cat_q1, dim=1).mean()
        cql_q2 = torch.logsumexp(cat_q2, dim=1).mean()

        # Subtract dataset Q-values.
        penalty = (cql_q1 - q1_data.mean()) + (cql_q2 - q2_data.mean())
        return penalty

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
            "log_alpha": self.log_alpha.detach().cpu(),
            "step": self._step,
        }

    def load_state_dict(self, d: dict):
        self.actor.load_state_dict(d["actor"])
        self.critic.load_state_dict(d["critic"])
        self.critic_target.load_state_dict(d["critic_target"])
        self.log_alpha.data.copy_(d["log_alpha"])
        self._step = d["step"]
