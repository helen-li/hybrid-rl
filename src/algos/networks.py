"""
Shared neural-network building blocks for CQL and IQL.

Both algorithms share the same Q-network and policy architectures;
only the losses differ.
"""

from __future__ import annotations
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


# --------------------------------------------------------------------------- #
#  MLP helper
# --------------------------------------------------------------------------- #

def mlp(dims: list[int], activation: type = nn.ReLU, output_activation: type = nn.Identity) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        act = activation if i < len(dims) - 2 else output_activation
        layers += [nn.Linear(dims[i], dims[i + 1]), act()]
    return nn.Sequential(*layers)


# --------------------------------------------------------------------------- #
#  Double Q-network  (for CQL and IQL critic)
# --------------------------------------------------------------------------- #

class DoubleQNetwork(nn.Module):
    """Two independent Q-networks with shared interface."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: list[int] = [256, 256]):
        super().__init__()
        self.q1 = mlp([obs_dim + act_dim] + hidden_dims + [1])
        self.q2 = mlp([obs_dim + act_dim] + hidden_dims + [1])

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([obs, act], dim=-1)
        return self.q1(sa).squeeze(-1), self.q2(sa).squeeze(-1)

    def q_min(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        q1, q2 = self(obs, act)
        return torch.min(q1, q2)


# --------------------------------------------------------------------------- #
#  Ensemble Q-network  (for Phase II online fine-tuning)
# --------------------------------------------------------------------------- #

class EnsembleQNetwork(nn.Module):
    """Ensemble of N independent Q-networks for uncertainty estimation.

    Generalises ``DoubleQNetwork`` to *n_members* heads.  Provides
    ``disagreement`` (std across members) used as an exploration bonus
    during online fine-tuning.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: list[int] = [256, 256],
        n_members: int = 3,
    ):
        super().__init__()
        self.n_members = n_members
        self.nets = nn.ModuleList(
            [mlp([obs_dim + act_dim] + hidden_dims + [1]) for _ in range(n_members)]
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> list[torch.Tensor]:
        """Return a list of *n_members* Q-value tensors, each shape ``(batch,)``."""
        sa = torch.cat([obs, act], dim=-1)
        return [net(sa).squeeze(-1) for net in self.nets]

    def q_min(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Pessimistic estimate: minimum Q across all ensemble members."""
        qs = self.forward(obs, act)
        return torch.stack(qs, dim=0).min(dim=0).values

    def disagreement(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Ensemble disagreement: std of Q-values across members ``(batch,)``."""
        qs = self.forward(obs, act)
        return torch.stack(qs, dim=0).std(dim=0)

    def load_double_q(self, double_q: DoubleQNetwork):
        """Initialise the first two members from a pretrained ``DoubleQNetwork``.

        Remaining members keep their random initialisation, providing
        diversity for meaningful disagreement estimates.
        """
        self.nets[0].load_state_dict(double_q.q1.state_dict())
        self.nets[1].load_state_dict(double_q.q2.state_dict())


# --------------------------------------------------------------------------- #
#  Value network  (used by IQL only)
# --------------------------------------------------------------------------- #

class ValueNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_dims: list[int] = [256, 256]):
        super().__init__()
        self.net = mlp([obs_dim] + hidden_dims + [1])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


# --------------------------------------------------------------------------- #
#  Tanh-squashed Gaussian policy  (continuous actions)
# --------------------------------------------------------------------------- #

class TanhGaussianPolicy(nn.Module):
    """Stochastic policy pi(a|s) parametrised as tanh-squashed Gaussian.

    Used by CQL (SAC-style actor) and can also be used as a stochastic
    policy for IQL (advantage-weighted regression variant).
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: list[int] = [256, 256]):
        super().__init__()
        self.trunk = mlp([obs_dim] + hidden_dims, activation=nn.ReLU, output_activation=nn.ReLU)
        self.mean_head = nn.Linear(hidden_dims[-1], act_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], act_dim)

    def forward(self, obs: torch.Tensor):
        h = self.trunk(obs)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (tanh-squashed action, log_prob)."""
        mean, log_std = self(obs)
        std = log_std.exp()
        dist = Normal(mean, std)
        # Reparametrised sample.
        x = dist.rsample()
        action = torch.tanh(x)
        # Log-prob with tanh correction.
        log_prob = dist.log_prob(x) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1)
        return action, log_prob

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Evaluate log pi(a|s) for given (obs, action) pairs."""
        mean, log_std = self(obs)
        std = log_std.exp()
        dist = Normal(mean, std)
        # Inverse tanh (atanh) to get pre-squash value.
        x = torch.atanh(action.clamp(-0.999, 0.999))
        log_prob = dist.log_prob(x) - torch.log(1.0 - action.pow(2) + 1e-6)
        return log_prob.sum(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mean, log_std = self(obs)
        if deterministic:
            return torch.tanh(mean)
        std = log_std.exp()
        dist = Normal(mean, std)
        return torch.tanh(dist.sample())


# --------------------------------------------------------------------------- #
#  Deterministic policy  (IQL extraction variant)
# --------------------------------------------------------------------------- #

class DeterministicPolicy(nn.Module):
    """Deterministic policy for advantage-weighted regression in IQL."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: list[int] = [256, 256]):
        super().__init__()
        self.net = mlp([obs_dim] + hidden_dims + [act_dim], output_activation=nn.Tanh)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        return self(obs)


# --------------------------------------------------------------------------- #
#  Soft-update helper
# --------------------------------------------------------------------------- #

@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float = 0.005):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.mul_(1 - tau).add_(sp.data, alpha=tau)
