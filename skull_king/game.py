"""Skull King game engine.

Supports 2-8 players, 10 rounds (round N deals N cards).
Graybeard variant with special cards.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from skull_king.cards import Card, Suit, SpecialType, create_deck, NUM_CARDS
from skull_king.scoring import score_round
from skull_king.trick import TrickResult, resolve_trick


class Phase(Enum):
    BIDDING = auto()
    PLAYING = auto()
    ROUND_OVER = auto()
    GAME_OVER = auto()


@dataclass
class PlayerState:
    hand: List[Card] = field(default_factory=list)
    bid: int = -1  # -1 = not yet bid
    tricks_won: int = 0
    bonus_points: int = 0
    score: int = 0


@dataclass
class GameState:
    """Observable game state for an agent."""
    phase: Phase
    round_number: int  # 1-10
    num_players: int
    player_id: int  # Which player this state is for
    hand: List[Card]
    # Cards played so far this round (all tricks, for card tracking)
    cards_seen: List[int]  # card_ids of all cards played this round
    # Current trick
    current_trick_cards: List[Card]
    current_trick_players: List[int]
    # Bids and tricks won for all players (-1 if not yet bid)
    all_bids: List[int]
    all_tricks_won: List[int]
    all_scores: List[int]
    # Positional info
    bid_position: int  # 0-indexed position in bidding order
    position_in_trick: int  # How many cards already played this trick
    tricks_remaining: int
    # Lead suit of current trick (None if no numbered card yet)
    lead_suit: Optional[Suit]
    # Current trick winner info
    current_winner_id: int  # -1 if no cards played
    current_winning_card_id: int  # -1 if no cards played
    # Legal actions
    legal_actions: List[int]  # For bidding: [0..round_number], for playing: indices into hand


class SkullKingGame:
    """Full Skull King game engine."""

    def __init__(self, num_players: int = 4, seed: Optional[int] = None):
        assert 2 <= num_players <= 8
        self.num_players = num_players
        self.rng = random.Random(seed)
        self.players: List[PlayerState] = [PlayerState() for _ in range(num_players)]
        self.round_number = 0
        self.phase = Phase.GAME_OVER
        self.dealer = 0  # Rotates each round
        self.current_player = 0
        self.trick_leader = 0
        self.current_trick_cards: List[Card] = []
        self.current_trick_players: List[int] = []
        self.cards_seen: List[int] = []  # card_ids seen this round
        self.tricks_played_this_round = 0
        # Round scores for reward shaping
        self.round_scores: Dict[int, List[int]] = {}  # round -> [score per player]

    def reset(self) -> None:
        """Start a new game."""
        for p in self.players:
            p.score = 0
        self.round_number = 0
        self.dealer = self.rng.randint(0, self.num_players - 1)
        self.round_scores = {}
        self._start_next_round()

    @property
    def max_rounds(self) -> int:
        """Max rounds playable given deck size and player count."""
        return min(10, NUM_CARDS // self.num_players)

    def _start_next_round(self) -> None:
        """Deal cards and start bidding for next round."""
        self.round_number += 1
        if self.round_number > self.max_rounds:
            self.phase = Phase.GAME_OVER
            return

        deck = create_deck()
        self.rng.shuffle(deck)

        # Deal round_number cards to each player
        idx = 0
        for p in self.players:
            p.hand = deck[idx:idx + self.round_number]
            p.bid = -1
            p.tricks_won = 0
            p.bonus_points = 0
            idx += self.round_number

        self.cards_seen = []
        self.tricks_played_this_round = 0
        self.current_trick_cards = []
        self.current_trick_players = []

        # Bidding starts left of dealer
        self.dealer = (self.dealer + 1) % self.num_players
        self.current_player = (self.dealer + 1) % self.num_players
        self.trick_leader = self.current_player
        self.phase = Phase.BIDDING

    def get_state(self, player_id: int) -> GameState:
        """Get observable state for a specific player."""
        p = self.players[player_id]

        # Compute legal actions
        if self.phase == Phase.BIDDING:
            legal_actions = list(range(self.round_number + 1))  # 0 to round_number
            bid_position = self._bid_position(player_id)
        else:
            legal_actions = self._get_legal_play_indices(player_id)
            bid_position = 0

        # Lead suit
        lead_suit = self._get_lead_suit()

        # Current winner
        if self.current_trick_cards:
            result = resolve_trick(self.current_trick_cards)
            winner_idx = result.winner_index
            current_winner_id = self.current_trick_players[winner_idx]
            current_winning_card_id = result.winner_card.card_id
        else:
            current_winner_id = -1
            current_winning_card_id = -1

        return GameState(
            phase=self.phase,
            round_number=self.round_number,
            num_players=self.num_players,
            player_id=player_id,
            hand=list(p.hand),
            cards_seen=list(self.cards_seen),
            current_trick_cards=list(self.current_trick_cards),
            current_trick_players=list(self.current_trick_players),
            all_bids=[pl.bid for pl in self.players],
            all_tricks_won=[pl.tricks_won for pl in self.players],
            all_scores=[pl.score for pl in self.players],
            bid_position=bid_position,
            position_in_trick=len(self.current_trick_cards),
            tricks_remaining=self.round_number - self.tricks_played_this_round,
            lead_suit=lead_suit,
            current_winner_id=current_winner_id,
            current_winning_card_id=current_winning_card_id,
            legal_actions=legal_actions,
        )

    def _bid_position(self, player_id: int) -> int:
        """How many players have bid before this player (0-indexed)."""
        count = 0
        p = (self.dealer + 1) % self.num_players
        while p != player_id:
            if self.players[p].bid >= 0:
                count += 1
            p = (p + 1) % self.num_players
        return count

    def _get_lead_suit(self) -> Optional[Suit]:
        """Get the lead suit of the current trick."""
        for card in self.current_trick_cards:
            if card.is_numbered():
                return card.suit
        return None

    def _get_legal_play_indices(self, player_id: int) -> List[int]:
        """Get indices into the player's hand that are legal to play.

        Rules: Must follow lead suit if able (unless playing a special card).
        Special cards can always be played.
        If you have no cards of the lead suit, you can play anything.
        Tigress: always legal (choice of pirate/escape is separate).
        """
        hand = self.players[player_id].hand
        if not hand:
            return []

        lead_suit = self._get_lead_suit()

        # If no lead suit established or this is the lead, all cards are legal
        if lead_suit is None or len(self.current_trick_cards) == 0:
            return list(range(len(hand)))

        # Check if player has any cards of the lead suit
        has_lead_suit = any(c.is_numbered() and c.suit == lead_suit for c in hand)

        if not has_lead_suit:
            # Can play anything
            return list(range(len(hand)))

        # Must follow suit for numbered cards, but special cards are always legal
        legal = []
        for i, card in enumerate(hand):
            if card.is_special():
                legal.append(i)
            elif card.suit == lead_suit:
                legal.append(i)
            elif card.is_trump():
                # Can always play trump (Black) even if you have lead suit?
                # Standard Skull King: you MUST follow suit with numbered cards.
                # Black is treated as a regular suit for following purposes.
                pass
            # Off-suit numbered cards when you have lead suit: illegal
        return legal

    def step_bid(self, player_id: int, bid: int) -> Optional[Dict]:
        """Submit a bid for the current player.

        Returns dict with round info when all bids are in, else None.
        """
        assert self.phase == Phase.BIDDING
        assert player_id == self.current_player
        assert 0 <= bid <= self.round_number
        assert self.players[player_id].bid == -1

        self.players[player_id].bid = bid

        # Advance to next player who hasn't bid
        self.current_player = (self.current_player + 1) % self.num_players

        # Check if all players have bid
        if all(p.bid >= 0 for p in self.players):
            self.phase = Phase.PLAYING
            self.current_player = self.trick_leader
            return {"all_bids": [p.bid for p in self.players]}

        return None

    def step_play(self, player_id: int, hand_index: int,
                  tigress_as_pirate: Optional[bool] = None) -> Optional[TrickResult]:
        """Play a card from the player's hand.

        Args:
            player_id: Must be current_player.
            hand_index: Index into the player's hand.
            tigress_as_pirate: Required if playing Tigress.

        Returns:
            TrickResult when the trick is complete, else None.
        """
        assert self.phase == Phase.PLAYING
        assert player_id == self.current_player

        legal = self._get_legal_play_indices(player_id)
        assert hand_index in legal, f"Illegal play: {hand_index} not in {legal}"

        card = self.players[player_id].hand.pop(hand_index)

        # Handle Tigress choice
        if card.is_tigress():
            if tigress_as_pirate is None:
                # Default: play as escape if no choice given
                tigress_as_pirate = False
            card = card.with_tigress_choice(tigress_as_pirate)

        self.current_trick_cards.append(card)
        self.current_trick_players.append(player_id)
        self.cards_seen.append(card.card_id)

        # Advance to next player
        self.current_player = (self.current_player + 1) % self.num_players

        # Check if trick is complete
        if len(self.current_trick_cards) == self.num_players:
            result = resolve_trick(self.current_trick_cards)
            winner_player = self.current_trick_players[result.winner_index]
            self.players[winner_player].tricks_won += 1
            self.players[winner_player].bonus_points += result.bonus_points

            self.tricks_played_this_round += 1

            # Prepare for next trick or end round
            self.current_trick_cards = []
            self.current_trick_players = []

            if self.tricks_played_this_round >= self.round_number:
                # Round over - score it
                self._score_round()
                self._start_next_round()
                return result

            # Next trick: winner leads
            self.trick_leader = winner_player
            self.current_player = winner_player
            return result

        return None

    def _score_round(self) -> None:
        """Score the completed round."""
        round_scores = []
        for p in self.players:
            s = score_round(p.bid, p.tricks_won, p.bonus_points, self.round_number)
            p.score += s
            round_scores.append(s)
        self.round_scores[self.round_number] = round_scores

    def get_current_player(self) -> int:
        return self.current_player

    def is_game_over(self) -> bool:
        return self.phase == Phase.GAME_OVER

    def get_winner(self) -> int:
        """Return player_id with highest score."""
        return max(range(self.num_players), key=lambda i: self.players[i].score)

    def get_scores(self) -> List[int]:
        return [p.score for p in self.players]
