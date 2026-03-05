"""Neural agent wrapping trained PPO networks for inference."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from agents.base import Agent
from networks.card_embedding import CardEmbedding, PlayerEmbedding
from networks.bid_network import BidActorCritic
from networks.play_network import PlayActorCritic
from networks.features import (
    encode_bid_state_v2, encode_play_state_v2,
    get_legal_bid_mask, get_legal_play_mask,
    encode_trick_history_v2,
)
from skull_king.cards import SpecialType
from skull_king.game import GameState


class NeuralAgent(Agent):
    """Agent that uses trained neural networks for decisions."""

    def __init__(self, checkpoint_path: str, device: str = "cpu",
                 hidden_dim: int | None = None, card_embed_dim: int = 16,
                 player_embed_dim: int = 4):
        self.device = torch.device(device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)

        # Infer hidden_dim from checkpoint if not specified
        if hidden_dim is None:
            hidden_dim = ckpt["bid_net"]["input_proj.0.weight"].shape[0]

        card_emb = CardEmbedding(card_embed_dim)
        player_emb = PlayerEmbedding(embed_dim=player_embed_dim)

        self.bid_net = BidActorCritic(card_emb, player_emb, hidden_dim)
        self.play_net = PlayActorCritic(card_emb, player_emb, hidden_dim)
        self.bid_net.load_state_dict(ckpt["bid_net"])
        self.play_net.load_state_dict(ckpt["play_net"])
        self.bid_net.eval()
        self.play_net.eval()

        # Trick history tracking for LSTM
        self._trick_history: List[dict] = []
        self._current_trick_cards: List[int] = []
        self._current_trick_players: List[int] = []
        self._last_round: int = -1
        self._num_players: int = 4

    def reset_round(self) -> None:
        """Call at the start of each new round to clear trick history."""
        self._trick_history = []
        self._current_trick_cards = []
        self._current_trick_players = []

    def choose_bid(self, state: GameState) -> int:
        self._num_players = state.num_players
        if state.round_number != self._last_round:
            self._last_round = state.round_number
            self.reset_round()
        features = encode_bid_state_v2(state)
        mask = get_legal_bid_mask(state)
        action, _, _, _ = self.bid_net.get_action_and_value(features, mask)
        return action

    def choose_play(self, state: GameState) -> Tuple[int, Optional[bool]]:
        self._num_players = state.num_players
        if state.round_number != self._last_round:
            self._last_round = state.round_number
            self.reset_round()

        features = encode_play_state_v2(state)
        mask = get_legal_play_mask(state)
        trick_hist = encode_trick_history_v2(self._trick_history, self._num_players)
        action, _, _, _ = self.play_net.get_action_and_value(features, mask, trick_hist)

        tigress = None
        if action < len(state.hand) and state.hand[action].special == SpecialType.TIGRESS:
            need = state.all_bids[state.player_id] > state.all_tricks_won[state.player_id]
            tigress = need
        return action, tigress

    def observe_card_played(self, player_id: int, card_id: int) -> None:
        """Call after each card is played in a trick to update history."""
        self._current_trick_cards.append(card_id)
        self._current_trick_players.append(player_id)

    def observe_trick_complete(self, winner_id: int) -> None:
        """Call when a trick is completed to record it in history."""
        self._trick_history.append({
            "card_ids": list(self._current_trick_cards),
            "player_ids": list(self._current_trick_players),
            "winner_id": winner_id,
        })
        self._current_trick_cards = []
        self._current_trick_players = []
