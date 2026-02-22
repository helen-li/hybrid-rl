"""
Online CQL with ensemble critic for Phase II fine-tuning.

Extends the offline CQL agent with:
  - An ensemble of Q-networks (default 3) for uncertainty estimation.
  - Ensemble disagreement used as an exploration bonus.
  - Reduced CQL conservative penalty for the online phase.
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
    TanhGaussianPolicy,
    soft_update,
)


@dataclass
class OnlineCQLConfig:
    # Architecture
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    # Ensemble
    n_ensemble: int = 3
    bonus_coeff: float = 1.0         # lambda for exploration bonus
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
    # CQL-specific (reduced for online phase)
    cql_alpha: float = 1.0           # Reduced from 5.0 in offline phase
    cql_n_samples: int = 10
    cql_lagrange: bool = False
    lr_cql_alpha: float = 3e-4
    cql_target_action_gap: float = -1.0


class OnlineCQL:
    """CQL with ensemble critic for online fine-tuning from offline checkpoints."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        cfg: OnlineCQLConfig | None = None,
        device: str = "cpu",
    ):
        self.cfg = cfg or OnlineCQLConfig()
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

        # Optimisers ------------------------------------------------------------
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.lr_critic)

        # SAC log-alpha (entropy coefficient) -----------------------------------
        self.target_entropy = -float(act_dim)
        self.log_alpha = torch.tensor(
            np.log(self.cfg.init_alpha), dtype=torch.float32, device=device,
            requires_grad=self.cfg.learnable_alpha,
        )
        self.alpha_optim = (
            torch.optim.Adam([self.log_alpha], lr=self.cfg.lr_alpha)
            if self.cfg.learnable_alpha else None
        )

        # CQL alpha -------------------------------------------------------------
        if self.cfg.cql_lagrange:
            self.log_cql_alpha = torch.tensor(
                np.log(self.cfg.cql_alpha), dtype=torch.float32, device=device,
                requires_grad=True,
            )
            self.cql_alpha_optim = torch.optim.Adam([self.log_cql_alpha], lr=self.cfg.lr_cql_alpha)
        else:
            self.log_cql_alpha = None
            self.cql_alpha_optim = None

        self._step = 0

    # ------------------------------------------------------------------ #
    #  Load from Phase I checkpoint
    # ------------------------------------------------------------------ #

    def load_offline_checkpoint(self, checkpoint: dict):
        """Initialise from a Phase I CQL checkpoint.

        - Actor weights are copied directly.
        - First 2 ensemble members are initialised from the saved DoubleQNetwork.
        - Remaining members keep random init for diversity.
        - SAC log-alpha is restored.
        """
        # Actor
        self.actor.load_state_dict(checkpoint["actor"])

        # Critic: reconstruct a temporary DoubleQNetwork to load weights
        tmp_double_q = DoubleQNetwork(self.obs_dim, self.act_dim, self.cfg.hidden_dims)
        tmp_double_q.load_state_dict(checkpoint["critic"])
        self.critic.load_double_q(tmp_double_q)

        # Copy into target as well
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.critic_target.requires_grad_(False)

        # SAC alpha
        if "log_alpha" in checkpoint:
            self.log_alpha.data.copy_(checkpoint["log_alpha"])

        # Rebuild optimisers so they track the new parameters
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.lr_critic)

        self._step = checkpoint.get("step", 0)

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
            # Pessimistic target using ensemble min
            q_next = self.critic_target.q_min(next_obs, next_act) - self.log_alpha.exp() * next_log_prob
            target_q = rew + (1.0 - done) * self.cfg.discount * q_next

        # Bellman loss: sum of MSE across all ensemble members
        qs = self.critic(obs, act)  # list of N tensors
        bellman_loss = sum(F.mse_loss(q, target_q) for q in qs)

        # CQL conservative penalty across all members
        cql_penalty = self._cql_penalty(obs, act, qs)
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
        q_new = self.critic.q_min(obs, new_act)
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
        q_stack = torch.stack(qs)  # (N, batch)
        metrics.update({
            "critic_loss": critic_loss.item(),
            "bellman_loss": bellman_loss.item(),
            "cql_penalty": cql_penalty.item(),
            "actor_loss": actor_loss.item(),
            "q_mean": q_stack.mean().item(),
            "q_std": q_stack.std(dim=0).mean().item(),
            "target_q_mean": target_q.mean().item(),
            "ensemble_disagreement": q_stack.std(dim=0).mean().item(),
        })
        return metrics

    # ------------------------------------------------------------------ #
    #  CQL conservative penalty (ensemble version)
    # ------------------------------------------------------------------ #

    def _cql_penalty(
        self,
        obs: torch.Tensor,
        data_act: torch.Tensor,
        qs_data: list[torch.Tensor],
    ) -> torch.Tensor:
        """CQL(H) penalty generalised to N ensemble members."""
        B = obs.shape[0]
        N = self.cfg.cql_n_samples

        # Random uniform actions
        rand_act = torch.FloatTensor(B * N, self.act_dim).uniform_(-1, 1).to(self.device)
        obs_rep = obs.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)

        # Policy actions
        with torch.no_grad():
            policy_act, policy_log_prob = self.actor.sample(obs_rep)

        rand_density_correction = np.log(0.5 ** self.act_dim)
        log_prob_pi = policy_log_prob.reshape(B, N)

        # Accumulate penalty across all ensemble members
        penalty = torch.tensor(0.0, device=self.device)
        for i, net in enumerate(self.critic.nets):
            sa_rand = torch.cat([obs_rep, rand_act], dim=-1)
            sa_pi = torch.cat([obs_rep, policy_act], dim=-1)

            q_rand = net(sa_rand).squeeze(-1).reshape(B, N)
            q_pi = net(sa_pi).squeeze(-1).reshape(B, N)

            cat_q = torch.cat(
                [q_rand - rand_density_correction, q_pi - log_prob_pi.detach()],
                dim=1,
            )
            lse = torch.logsumexp(cat_q, dim=1).mean()
            penalty = penalty + (lse - qs_data[i].mean())

        return penalty

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
            "log_alpha": self.log_alpha.detach().cpu(),
            "step": self._step,
        }

    def load_state_dict(self, d: dict):
        self.actor.load_state_dict(d["actor"])
        self.critic.load_state_dict(d["critic"])
        self.critic_target.load_state_dict(d["critic_target"])
        self.log_alpha.data.copy_(d["log_alpha"])
        self._step = d["step"]
