"""Neural agent wrapping trained PPO networks for inference."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from agents.base import Agent
from networks.bid_network import BidActorCritic
from networks.play_network import PlayActorCritic
from networks.features import encode_bid_state, encode_play_state, get_legal_bid_mask, get_legal_play_mask
from skull_king.cards import SpecialType
from skull_king.game import GameState


class NeuralAgent(Agent):
    """Agent that uses trained neural networks for decisions."""

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.bid_net = BidActorCritic()
        self.play_net = PlayActorCritic()
        self.bid_net.load_state_dict(ckpt["bid_net"])
        self.play_net.load_state_dict(ckpt["play_net"])
        self.bid_net.eval()
        self.play_net.eval()

    def choose_bid(self, state: GameState) -> int:
        features = encode_bid_state(state)
        mask = get_legal_bid_mask(state)
        action, _, _, _ = self.bid_net.get_action_and_value(features, mask)
        return action

    def choose_play(self, state: GameState) -> Tuple[int, Optional[bool]]:
        features = encode_play_state(state)
        mask = get_legal_play_mask(state)
        action, _, _, _ = self.play_net.get_action_and_value(features, mask)
        tigress = None
        if action < len(state.hand) and state.hand[action].special == SpecialType.TIGRESS:
            need = state.all_bids[state.player_id] > state.all_tricks_won[state.player_id]
            tigress = need
        return action, tigress
