"""Human interactive agent for CLI play."""

from __future__ import annotations

from typing import Optional, Tuple

from agents.base import Agent
from skull_king.cards import SpecialType
from skull_king.game import GameState


class HumanAgent(Agent):
    """Interactive human player via CLI."""

    def choose_bid(self, state: GameState) -> int:
        print(f"\n=== Round {state.round_number} — Your turn to bid ===")
        print(f"Your hand: {', '.join(str(c) for c in state.hand)}")
        other_bids = [
            f"P{i}={b}" for i, b in enumerate(state.all_bids)
            if b >= 0 and i != state.player_id
        ]
        if other_bids:
            print(f"Bids so far: {', '.join(other_bids)}")
        print(f"Legal bids: 0-{state.round_number}")

        while True:
            try:
                bid = int(input("Your bid: "))
                if 0 <= bid <= state.round_number:
                    return bid
                print(f"Must be 0-{state.round_number}")
            except (ValueError, EOFError):
                print("Enter a number")

    def choose_play(self, state: GameState) -> Tuple[int, Optional[bool]]:
        print(f"\n=== Round {state.round_number}, Trick ===")
        print(f"Your bid: {state.all_bids[state.player_id]}, Won: {state.all_tricks_won[state.player_id]}")
        if state.current_trick_cards:
            trick_str = ", ".join(
                f"P{p}:{c}" for p, c in zip(state.current_trick_players, state.current_trick_cards)
            )
            print(f"Trick so far: {trick_str}")
        print(f"Your hand:")
        for i, card in enumerate(state.hand):
            legal = "  " if i in state.legal_actions else "X "
            print(f"  {legal}[{i}] {card}")

        while True:
            try:
                idx = int(input("Play card index: "))
                if idx in state.legal_actions:
                    card = state.hand[idx]
                    tigress = None
                    if card.special == SpecialType.TIGRESS:
                        choice = input("Tigress as (p)irate or (e)scape? ").lower()
                        tigress = choice.startswith("p")
                    return idx, tigress
                print(f"Not legal. Choose from {state.legal_actions}")
            except (ValueError, EOFError):
                print("Enter a number")
