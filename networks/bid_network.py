"""Bid actor-critic network for PPO with learned card embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn

from networks.card_embedding import CardEmbedding, PlayerEmbedding
from networks.features import BID_SCALAR_DIM


class ResBlock(nn.Module):
    """Pre-norm residual block."""

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
    """Actor-critic for bidding phase with learned card embeddings.

    Input: dict with hand_ids, hand_mask, seen_ids, seen_mask, scalars
    Actor output: 11 logits (bids 0-10)
    Critic output: scalar value
    """

    def __init__(self, card_emb: CardEmbedding, player_emb: PlayerEmbedding,
                 hidden_dim: int = 256):
        super().__init__()
        self.card_emb = card_emb
        self.player_emb = player_emb
        # Input: hand_emb + seen_emb + scalars
        input_dim = card_emb.embed_dim * 2 + BID_SCALAR_DIM
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(*[ResBlock(hidden_dim) for _ in range(3)])
        self.actor_head = nn.Linear(hidden_dim, 11)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: dict, legal_mask: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            state: dict with batched tensors from encode_bid_state_v2
            legal_mask: [batch, 11] binary mask of legal bids

        Returns:
            (log_probs [batch, 11], value [batch, 1])
        """
        hand_emb = self.card_emb.embed_set(state["hand_ids"], state["hand_mask"])
        seen_emb = self.card_emb.embed_set(state["seen_ids"], state["seen_mask"])
        x = torch.cat([hand_emb, seen_emb, state["scalars"]], dim=-1)

        h = self.input_proj(x)
        h = self.res_blocks(h)

        logits = self.actor_head(h)
        logits = logits + (legal_mask.log().clamp(min=-1e8))
        log_probs = torch.log_softmax(logits, dim=-1)
        value = self.critic_head(h)
        return log_probs, value

    def get_action_and_value(self, state: dict, legal_mask: torch.Tensor
                             ) -> tuple[int, float, float, float]:
        """Sample an action and return (action, log_prob, value, entropy)."""
        with torch.no_grad():
            device = next(self.parameters()).device
            # Batch dimension
            state_b = {k: v.unsqueeze(0).to(device) for k, v in state.items()}
            mask_b = legal_mask.unsqueeze(0).to(device)

            log_probs, value = self.forward(state_b, mask_b)
            log_probs = log_probs.squeeze(0)
            value = value.squeeze(0).item()
            probs = log_probs.exp()
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()
            return action.item(), log_probs[action].item(), value, dist.entropy().item()
