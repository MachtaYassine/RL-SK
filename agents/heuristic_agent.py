"""Heuristic agent that uses simple rules to play Skull King."""

from __future__ import annotations

import random
from typing import Optional, Tuple

from agents.base import Agent
from skull_king.cards import Card, SpecialType, Suit
from skull_king.game import GameState


class HeuristicAgent(Agent):
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def choose_bid(self, state: GameState) -> int:
        """Bid based on number of strong cards in hand."""
        bid = 0
        for card in state.hand:
            if card.is_skull_king() or card.is_pirate():
                bid += 1
            elif card.is_mermaid():
                bid += 1
            elif card.is_trump() and card.number and card.number >= 10:
                bid += 1
            elif card.is_numbered() and card.number and card.number >= 13:
                bid += 1
        return min(bid, state.round_number)

    def choose_play(self, state: GameState) -> Tuple[int, Optional[bool]]:
        """Simple heuristic play strategy."""
        hand = state.hand
        legal = state.legal_actions
        legal_cards = [(i, hand[i]) for i in legal]

        # If leading the trick, play strongest card if we need tricks
        need_tricks = state.all_bids[state.player_id] > state.all_tricks_won[state.player_id]

        if need_tricks:
            # Play to win: prefer special cards first, then high cards
            best_idx = self._pick_strongest(legal_cards)
        else:
            # Play to lose: dump weakest card
            best_idx = self._pick_weakest(legal_cards, state.lead_suit)

        card = hand[best_idx]
        tigress = None
        if card.special == SpecialType.TIGRESS:
            tigress = need_tricks  # Pirate if we need tricks, Escape if not

        return best_idx, tigress

    def _pick_strongest(self, legal_cards: list[Tuple[int, Card]]) -> int:
        """Pick the strongest legal card."""
        # Priority: SK > Pirate > Mermaid > high trump > high suited
        best_score = -1
        best_idx = legal_cards[0][0]
        for idx, card in legal_cards:
            score = self._card_strength(card)
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx

    def _pick_weakest(self, legal_cards: list[Tuple[int, Card]], lead_suit: Optional[Suit]) -> int:
        """Pick the weakest legal card."""
        best_score = 10000
        best_idx = legal_cards[0][0]
        for idx, card in legal_cards:
            score = self._card_strength(card)
            if score < best_score:
                best_score = score
                best_idx = idx
        return best_idx

    def _card_strength(self, card: Card) -> int:
        if card.is_skull_king():
            return 200
        if card.is_pirate() or (card.is_tigress() and card.tigress_as_pirate):
            return 180
        if card.is_mermaid():
            return 170
        if card.is_escape() or (card.is_tigress() and card.tigress_as_pirate is False):
            return 0
        if card.is_tigress():
            return 90  # Unchosen tigress, moderate
        # Numbered card
        base = card.number or 0
        if card.is_trump():
            base += 100
        return base
