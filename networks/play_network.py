"""Play actor-critic network for PPO with learned embeddings and LSTM trick history."""

from __future__ import annotations

import torch
import torch.nn as nn

from networks.card_embedding import CardEmbedding, PlayerEmbedding
from networks.features import PLAY_SCALAR_DIM, MAX_HAND_SIZE
from skull_king.cards import NUM_CARDS


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


class HandSelfAttention(nn.Module):
    """Multi-head self-attention over cards in hand, conditioned on game context.

    Each card sees every other card + a global context vector (bid progress,
    tricks remaining, etc.), learning relationships like:
    - "I'm a high trump and there are 2 others → I'm expendable"
    - "I'm the only card that can lose this trick → I'm the safe play"
    """

    def __init__(self, card_dim: int, context_dim: int, num_heads: int = 2):
        super().__init__()
        self.num_heads = num_heads
        # Project card + context into attention space
        self.card_context_proj = nn.Linear(card_dim + context_dim, card_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=card_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(card_dim)

    def forward(self, per_card_emb: torch.Tensor, context: torch.Tensor,
                hand_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            per_card_emb: [batch, MAX_HAND_SIZE, card_dim]
            context: [batch, context_dim] — global game state
            hand_mask: [batch, MAX_HAND_SIZE] — 1.0 where card exists

        Returns:
            [batch, MAX_HAND_SIZE, card_dim] — context-aware card representations
        """
        # Broadcast context to each card position
        ctx_expanded = context.unsqueeze(1).expand(-1, per_card_emb.shape[1], -1)
        card_ctx = torch.cat([per_card_emb, ctx_expanded], dim=-1)
        card_ctx = self.card_context_proj(card_ctx)

        # Attention mask: True = ignore (PyTorch convention)
        key_padding_mask = (hand_mask == 0)

        attn_out, _ = self.attn(card_ctx, card_ctx, card_ctx,
                                key_padding_mask=key_padding_mask)
        # Residual + norm
        out = self.norm(per_card_emb + attn_out)
        return out


class PlayActorCritic(nn.Module):
    """Actor-critic for playing phase with learned embeddings.

    Input: dict state + dict trick_history
    Actor output: MAX_HAND_SIZE logits (play index 0-9)
    Critic output: scalar value

    Cards in hand go through self-attention conditioned on game context,
    learning inter-card relationships (e.g. "this card is strong given
    what else I hold and how many tricks I still need").
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

        # Hand self-attention: cards see each other + game context
        # Context = hidden_dim (backbone output encodes bid, tricks, etc.)
        self.hand_attn = HandSelfAttention(
            card_dim=card_emb.embed_dim, context_dim=hidden_dim, num_heads=2)

        # Per-card actor uses context-aware card embeddings
        self.actor_query = nn.Linear(hidden_dim, card_emb.embed_dim)
        self.actor_key = nn.Linear(card_emb.embed_dim, card_emb.embed_dim)

        self.critic_head = nn.Linear(hidden_dim, 1)

        # Belief auxiliary head
        self.belief_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, NUM_CARDS),
        )

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

    def _backbone(self, state: dict, legal_mask: torch.Tensor,
                  trick_history: dict | None = None
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared backbone returning (log_probs, value, hidden).

        Returns:
            (log_probs [batch, MAX_HAND_SIZE], value [batch, 1], h [batch, hidden_dim])
        """
        batch_size = state["scalars"].shape[0]
        device = state["scalars"].device

        # Raw per-card embeddings [batch, MAX_HAND_SIZE, card_embed_dim]
        per_card_emb = self.card_emb.forward(state["hand_ids"])

        # Pooled embeddings for global context
        hand_emb = self.card_emb.embed_set(state["hand_ids"], state["hand_mask"])
        seen_emb = self.card_emb.embed_set(state["seen_ids"], state["seen_mask"])
        trick_cards_emb = self.card_emb.embed_set(
            state["trick_card_ids"], state["trick_card_mask"])

        trick_enc = self._encode_tricks(trick_history, device, batch_size)

        x = torch.cat([hand_emb, seen_emb, trick_cards_emb,
                        state["scalars"], trick_enc], dim=-1)

        h = self.input_proj(x)
        h = self.res_blocks(h)

        # Hand self-attention: cards attend to each other conditioned on
        # game context (h encodes bid, tricks won, tricks remaining, etc.)
        # This lets the network learn card relationships like "this strong
        # card is expendable because I have others" or "this is my only
        # way to lose a trick"
        context_cards = self.hand_attn(per_card_emb, h, state["hand_mask"])

        # Actor: dot(query from context, key from context-aware cards)
        query = self.actor_query(h)
        keys = self.actor_key(context_cards)
        logits = (keys * query.unsqueeze(1)).sum(dim=-1)
        logits = logits + (legal_mask.log().clamp(min=-1e8))
        log_probs = torch.log_softmax(logits, dim=-1)
        value = self.critic_head(h)
        return log_probs, value, h

    def forward(self, state: dict, legal_mask: torch.Tensor,
                trick_history: dict | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            (log_probs [batch, MAX_HAND_SIZE], value [batch, 1])
        """
        log_probs, value, _ = self._backbone(state, legal_mask, trick_history)
        return log_probs, value

    def forward_with_belief(self, state: dict, legal_mask: torch.Tensor,
                            trick_history: dict | None = None
                            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with belief head output.

        Returns:
            (log_probs [batch, MAX_HAND_SIZE], value [batch, 1],
             belief_logits [batch, NUM_CARDS])
        """
        log_probs, value, h = self._backbone(state, legal_mask, trick_history)
        belief_logits = self.belief_head(h)
        return log_probs, value, belief_logits

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
