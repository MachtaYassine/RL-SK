"""Random agent that makes uniform random legal moves."""

from __future__ import annotations

import random
from typing import Optional, Tuple

from agents.base import Agent
from skull_king.cards import SpecialType
from skull_king.game import GameState


class RandomAgent(Agent):
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def choose_bid(self, state: GameState) -> int:
        return self.rng.randint(0, state.round_number)

    def choose_play(self, state: GameState) -> Tuple[int, Optional[bool]]:
        hand_idx = self.rng.choice(state.legal_actions)
        card = state.hand[hand_idx]
        tigress = None
        if card.special == SpecialType.TIGRESS:
            tigress = self.rng.choice([True, False])
        return hand_idx, tigress
