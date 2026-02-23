"""Bid actor-critic network for PPO with residual backbone."""

from __future__ import annotations

import torch
import torch.nn as nn
from networks.features import BID_DIM


class ResBlock(nn.Module):
    """Pre-norm residual block: LayerNorm -> Linear -> ReLU -> Linear -> skip -> ReLU."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x + self.net(x))


class BidActorCritic(nn.Module):
    """Actor-critic for bidding phase.

    Input: BID_DIM features
    Actor output: 11 logits (bids 0-10)
    Critic output: scalar value
    """

    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(BID_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(*[ResBlock(hidden_dim) for _ in range(4)])
        self.actor_head = nn.Linear(hidden_dim, 11)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, legal_mask: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: [batch, BID_DIM]
            legal_mask: [batch, 11] binary mask of legal bids

        Returns:
            (log_probs, value) where log_probs is [batch, 11] masked log probs
            and value is [batch, 1]
        """
        h = self.input_proj(x)
        h = self.res_blocks(h)
        logits = self.actor_head(h)
        # Mask illegal actions
        logits = logits + (legal_mask.log().clamp(min=-1e8))
        log_probs = torch.log_softmax(logits, dim=-1)
        value = self.critic_head(h)
        return log_probs, value

    def get_action_and_value(self, x: torch.Tensor, legal_mask: torch.Tensor
                             ) -> tuple[int, float, float, float]:
        """Sample an action and return (action, log_prob, value, entropy)."""
        with torch.no_grad():
            device = next(self.parameters()).device
            log_probs, value = self.forward(x.unsqueeze(0).to(device), legal_mask.unsqueeze(0).to(device))
            log_probs = log_probs.squeeze(0)
            value = value.squeeze(0).item()
            probs = log_probs.exp()
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()
            return action.item(), log_probs[action].item(), value, dist.entropy().item()
