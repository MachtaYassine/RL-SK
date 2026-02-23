"""Wraps SkullKingGame + agents. Clean boundary for any renderer."""

from __future__ import annotations

from typing import List, Optional, Tuple

from skull_king.game import SkullKingGame, GameState, Phase
from skull_king.cards import Card, SpecialType
from agents.base import Agent


class GameManager:
    """Manages a Skull King game with one human (player 0) and N-1 AI agents."""

    def __init__(self, agents: List[Agent], seed: Optional[int] = None):
        self.num_players = len(agents) + 1
        self.agents = agents
        self.game = SkullKingGame(num_players=self.num_players, seed=seed)
        self.game.reset()
        self._prev_round = self.game.round_number
        self._round_just_ended = False
        self._last_round_scores: Optional[List[int]] = None

    def reset(self):
        self.game.reset()
        self._prev_round = self.game.round_number
        self._round_just_ended = False
        self._last_round_scores = None

    @property
    def phase(self) -> Phase:
        return self.game.phase

    @property
    def round_number(self) -> int:
        return self.game.round_number

    @property
    def max_rounds(self) -> int:
        return self.game.max_rounds

    @property
    def round_just_ended(self) -> bool:
        return self._round_just_ended

    @property
    def last_round_scores(self) -> Optional[List[int]]:
        return self._last_round_scores

    def clear_round_ended(self):
        """Called by app after showing round summary."""
        self._round_just_ended = False

    def get_state(self) -> GameState:
        return self.game.get_state(0)

    def get_scores(self) -> List[int]:
        return self.game.get_scores()

    def get_all_bids(self) -> List[int]:
        return [p.bid for p in self.game.players]

    def get_all_tricks_won(self) -> List[int]:
        return [p.tricks_won for p in self.game.players]

    def is_game_over(self) -> bool:
        return self.game.is_game_over()

    def get_winner(self) -> int:
        return self.game.get_winner()

    def is_human_turn(self) -> bool:
        if self.game.is_game_over():
            return False
        return self.game.get_current_player() == 0

    def human_bid(self, bid: int):
        self.game.step_bid(0, bid)

    def human_play(self, hand_index: int, tigress_as_pirate: Optional[bool] = None):
        old_round = self.game.round_number
        result = self.game.step_play(0, hand_index, tigress_as_pirate)
        self._check_round_change(old_round, result)
        return result

    def advance_ai(self) -> List[dict]:
        """Let all consecutive AI players act. Returns events."""
        events = []
        while not self.game.is_game_over():
            pid = self.game.get_current_player()
            if pid == 0:
                break
            agent = self.agents[pid - 1]
            state = self.game.get_state(pid)

            if self.game.phase == Phase.BIDDING:
                bid = agent.choose_bid(state)
                self.game.step_bid(pid, bid)
                events.append({"type": "ai_bid", "player": pid, "bid": bid})
            elif self.game.phase == Phase.PLAYING:
                hand_idx, tigress = agent.choose_play(state)
                card = state.hand[hand_idx]
                old_round = self.game.round_number
                result = self.game.step_play(pid, hand_idx, tigress)
                events.append({"type": "ai_play", "player": pid, "card": card})
                if result is not None:
                    events.append({"type": "trick_complete", "result": result})
                    if self._check_round_change(old_round, result):
                        events.append({"type": "round_over"})
                        break  # Pause for round summary
            else:
                break

        if self.game.is_game_over():
            events.append({"type": "game_over"})

        return events

    def _check_round_change(self, old_round, result) -> bool:
        """Detect if a round just ended by checking round_number change."""
        new_round = self.game.round_number
        if new_round != old_round or self.game.is_game_over():
            # Round ended — scores are stored in game.round_scores[old_round]
            if old_round in self.game.round_scores:
                self._last_round_scores = self.game.round_scores[old_round]
                self._round_just_ended = True
                return True
        return False

    def get_trick_cards(self) -> List[Tuple[int, Card]]:
        state = self.game.get_state(0)
        return list(zip(state.current_trick_players, state.current_trick_cards))

    def get_ai_hand_sizes(self) -> List[int]:
        return [len(self.game.players[i].hand) for i in range(1, self.num_players)]
