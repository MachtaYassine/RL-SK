"""Shared card and player embedding modules for all networks."""

from __future__ import annotations

import torch
import torch.nn as nn

from skull_king.cards import NUM_CARDS, Suit, SpecialType, card_id_to_card

MAX_PLAYERS = 8
NUM_SPECIAL_TYPES = 5  # ESCAPE, PIRATE, MERMAID, SKULL_KING, TIGRESS
NUM_SUITS = 4
# Structural: suit_onehot(4) + rank/14(1) + is_special(1) + special_type_onehot(5) = 11
STRUCT_DIM = NUM_SUITS + 1 + 1 + NUM_SPECIAL_TYPES


class CardEmbedding(nn.Module):
    """Learned card embedding enriched with structural card features.

    Each card gets a learned 16-dim vector concatenated with an 11-dim
    structural descriptor (suit, rank, special type), projected back to
    embed_dim. This lets the network learn that cards sharing suit or
    rank are related, while still allowing arbitrary learned features.
    """

    def __init__(self, embed_dim: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.learned = nn.Embedding(NUM_CARDS, embed_dim)
        self.proj = nn.Linear(embed_dim + STRUCT_DIM, embed_dim)

        # Pre-compute structural features for all 70 cards
        struct = torch.zeros(NUM_CARDS, STRUCT_DIM)
        for cid in range(NUM_CARDS):
            card = card_id_to_card(cid)
            if card.suit is not None:
                struct[cid, card.suit.value] = 1.0                # suit one-hot [0:4]
                struct[cid, NUM_SUITS] = card.number / 14.0       # rank [4]
            else:
                struct[cid, NUM_SUITS + 1] = 1.0                  # is_special [5]
                if card.special is not None:
                    struct[cid, NUM_SUITS + 2 + card.special.value] = 1.0  # special_type [6:11]
        self.register_buffer("structural", struct)

    def forward(self, card_ids: torch.Tensor) -> torch.Tensor:
        """Embed card IDs.

        Args:
            card_ids: any shape of LongTensor with card IDs (0-69)

        Returns:
            (*card_ids.shape, embed_dim)
        """
        learned = self.learned(card_ids)
        struct = self.structural[card_ids]
        return self.proj(torch.cat([learned, struct], dim=-1))

    def embed_set(self, card_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Embed a variable-size set of cards via masked mean pooling.

        Args:
            card_ids: [batch, max_cards] LongTensor
            mask: [batch, max_cards] float binary mask

        Returns:
            [batch, embed_dim]
        """
        emb = self.forward(card_ids)  # [batch, max_cards, embed_dim]
        emb = emb * mask.unsqueeze(-1)
        count = mask.sum(dim=-1, keepdim=True).clamp(min=1)
        return emb.sum(dim=1) / count


class PlayerEmbedding(nn.Module):
    """Learned player identity embedding."""

    def __init__(self, max_players: int = MAX_PLAYERS, embed_dim: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(max_players, embed_dim)

    def forward(self, player_ids: torch.Tensor) -> torch.Tensor:
        """Embed player IDs.

        Args:
            player_ids: any shape of LongTensor with player IDs (0-7)

        Returns:
            (*player_ids.shape, embed_dim)
        """
        return self.embedding(player_ids)
