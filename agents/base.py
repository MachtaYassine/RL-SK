"""Abstract base agent for Skull King."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

from skull_king.game import GameState


class Agent(ABC):
    """Base class for all Skull King agents."""

    @abstractmethod
    def choose_bid(self, state: GameState) -> int:
        """Choose a bid given the game state.

        Returns:
            Bid value in range [0, round_number].
        """
        ...

    @abstractmethod
    def choose_play(self, state: GameState) -> Tuple[int, Optional[bool]]:
        """Choose a card to play given the game state.

        Returns:
            (hand_index, tigress_as_pirate) where hand_index is into state.hand
            and tigress_as_pirate is None unless playing Tigress.
        """
        ...

    def on_round_end(self, round_number: int, scores: list[int]) -> None:
        """Called at end of each round. Override for learning agents."""
        pass

    def on_game_end(self, final_scores: list[int]) -> None:
        """Called at end of game. Override for learning agents."""
        pass
