"""Play actor-critic network for PPO with learned embeddings and LSTM trick history."""

from __future__ import annotations

import torch
import torch.nn as nn

from networks.card_embedding import CardEmbedding, PlayerEmbedding
from networks.features import PLAY_SCALAR_DIM, MAX_HAND_SIZE


LSTM_HIDDEN = 64


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


class PlayActorCritic(nn.Module):
    """Actor-critic for playing phase with learned embeddings.

    Input: dict state + dict trick_history
    Actor output: MAX_HAND_SIZE logits (play index 0-9)
    Critic output: scalar value
    """

    def __init__(self, card_emb: CardEmbedding, player_emb: PlayerEmbedding,
                 hidden_dim: int = 256):
        super().__init__()
        self.card_emb = card_emb
        self.player_emb = player_emb

        # LSTM input per trick: mean_card_emb + mean_player_emb + winner_emb
        lstm_input_dim = card_emb.embed_dim + player_emb.embed_dim * 2
        self.trick_lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=LSTM_HIDDEN,
            num_layers=1,
            batch_first=True,
        )

        # Main input: hand_emb + seen_emb + trick_cards_emb + scalars + lstm
        input_dim = card_emb.embed_dim * 3 + PLAY_SCALAR_DIM + LSTM_HIDDEN
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(*[ResBlock(hidden_dim) for _ in range(3)])

        # Per-card actor: project context to query, per-card embeddings to keys
        # logit[i] = dot(query, key[i]) — the network sees which card is at each slot
        self.actor_query = nn.Linear(hidden_dim, card_emb.embed_dim)
        self.actor_key = nn.Linear(card_emb.embed_dim, card_emb.embed_dim)

        self.critic_head = nn.Linear(hidden_dim, 1)

    def _encode_tricks(self, trick_history: dict | None, device: torch.device,
                       batch_size: int) -> torch.Tensor:
        """Encode trick history through LSTM.

        Args:
            trick_history: dict from encode_trick_history_v2 (batched) or None
            device: target device
            batch_size: for creating zeros if trick_history is None

        Returns:
            [batch, LSTM_HIDDEN]
        """
        if trick_history is None:
            return torch.zeros(batch_size, LSTM_HIDDEN, device=device)

        # card_ids: [batch, MAX_TRICKS, MAX_PLAYERS]
        card_ids = trick_history["card_ids"]
        card_masks = trick_history["card_masks"]
        player_ids = trick_history["player_ids"]
        player_masks = trick_history["player_masks"]
        winner_ids = trick_history["winner_ids"]
        trick_mask = trick_history["trick_mask"]

        # Embed cards per trick: mean-pool over players in each trick
        # [batch, MAX_TRICKS, MAX_PLAYERS, card_embed_dim]
        card_embs = self.card_emb.forward(card_ids)
        card_embs = card_embs * card_masks.unsqueeze(-1)
        card_count = card_masks.sum(dim=-1, keepdim=True).clamp(min=1)
        mean_card = card_embs.sum(dim=2) / card_count  # [batch, MAX_TRICKS, card_embed_dim]

        # Embed players per trick: mean-pool
        player_embs = self.player_emb.forward(player_ids)
        player_embs = player_embs * player_masks.unsqueeze(-1)
        player_count = player_masks.sum(dim=-1, keepdim=True).clamp(min=1)
        mean_player = player_embs.sum(dim=2) / player_count  # [batch, MAX_TRICKS, player_embed_dim]

        # Winner embedding
        winner_emb = self.player_emb.forward(winner_ids)  # [batch, MAX_TRICKS, player_embed_dim]

        # Concat per-trick features
        trick_features = torch.cat([mean_card, mean_player, winner_emb], dim=-1)
        # [batch, MAX_TRICKS, lstm_input_dim]

        # Zero out padding tricks
        trick_features = trick_features * trick_mask.unsqueeze(-1)

        _, (h_n, _) = self.trick_lstm(trick_features)
        return h_n.squeeze(0)  # [batch, LSTM_HIDDEN]

    def forward(self, state: dict, legal_mask: torch.Tensor,
                trick_history: dict | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            state: dict with batched tensors from encode_play_state_v2
            legal_mask: [batch, MAX_HAND_SIZE] binary mask
            trick_history: dict from encode_trick_history_v2 (batched) or None

        Returns:
            (log_probs [batch, MAX_HAND_SIZE], value [batch, 1])
        """
        batch_size = state["scalars"].shape[0]
        device = state["scalars"].device

        # Per-card embeddings for the actor (before pooling)
        # [batch, MAX_HAND_SIZE, card_embed_dim]
        per_card_emb = self.card_emb.forward(state["hand_ids"])

        hand_emb = self.card_emb.embed_set(state["hand_ids"], state["hand_mask"])
        seen_emb = self.card_emb.embed_set(state["seen_ids"], state["seen_mask"])
        trick_cards_emb = self.card_emb.embed_set(
            state["trick_card_ids"], state["trick_card_mask"])

        trick_enc = self._encode_tricks(trick_history, device, batch_size)

        x = torch.cat([hand_emb, seen_emb, trick_cards_emb,
                        state["scalars"], trick_enc], dim=-1)

        h = self.input_proj(x)
        h = self.res_blocks(h)

        # Per-card actor: dot(query, key) for each hand slot
        query = self.actor_query(h)               # [batch, card_embed_dim]
        keys = self.actor_key(per_card_emb)        # [batch, MAX_HAND_SIZE, card_embed_dim]
        logits = (keys * query.unsqueeze(1)).sum(dim=-1)  # [batch, MAX_HAND_SIZE]
        logits = logits + (legal_mask.log().clamp(min=-1e8))
        log_probs = torch.log_softmax(logits, dim=-1)
        value = self.critic_head(h)
        return log_probs, value

    def get_action_and_value(self, state: dict, legal_mask: torch.Tensor,
                             trick_history: dict | None = None
                             ) -> tuple[int, float, float, float]:
        """Sample an action and return (action, log_prob, value, entropy)."""
        with torch.no_grad():
            device = next(self.parameters()).device
            state_b = {k: v.unsqueeze(0).to(device) for k, v in state.items()}
            mask_b = legal_mask.unsqueeze(0).to(device)
            th_b = None
            if trick_history is not None:
                th_b = {k: v.unsqueeze(0).to(device) for k, v in trick_history.items()}

            log_probs, value = self.forward(state_b, mask_b, th_b)
            log_probs = log_probs.squeeze(0)
            value = value.squeeze(0).item()
            probs = log_probs.exp()
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()
            return action.item(), log_probs[action].item(), value, dist.entropy().item()
