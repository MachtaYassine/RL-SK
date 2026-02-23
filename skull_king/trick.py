"""Trick resolution for Skull King.

Hierarchy (highest to lowest):
1. Skull King beats everything EXCEPT Mermaid
2. Mermaid beats Skull King (captures for bonus), loses to Pirate
3. Pirate beats all numbered cards and Mermaids
4. Black (trump) suited cards beat non-trump suited cards
5. Lead suit cards beat off-suit cards
6. Higher number beats lower number of same suit
7. Escape cards never win

Special interactions:
- If only Escapes are played, first Escape wins (someone must take the trick)
- Tigress played as Pirate acts as Pirate; as Escape acts as Escape
- If Mermaid captures Skull King: +50 bonus to Mermaid player
- If Pirate captures Skull King: no bonus (only Mermaid gets it... wait no)
  Actually in standard rules: capturing SK with any card = bonus to capturer
  Let me use Graybeard variant: Mermaid capturing SK = +50 bonus

Bonus points:
- Capturing Skull King (with Mermaid): +50
- Each Pirate captured in a trick by the winner: +30
- Pirate capturing a Mermaid: +20 per mermaid (some variants)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from skull_king.cards import Card, Suit


@dataclass
class TrickResult:
    winner_index: int  # Index into the played_cards list (position in trick)
    winner_card: Card
    bonus_points: int  # Bonus from special card captures


def resolve_trick(played_cards: List[Card]) -> TrickResult:
    """Resolve a trick and return the winner.

    Args:
        played_cards: Cards in order they were played. Index 0 = lead.

    Returns:
        TrickResult with winner index, card, and bonus points.
    """
    assert len(played_cards) >= 1

    if len(played_cards) == 1:
        return TrickResult(winner_index=0, winner_card=played_cards[0], bonus_points=0)

    # Determine lead suit (first non-escape, non-special numbered card)
    lead_suit: Optional[Suit] = None
    for card in played_cards:
        if card.is_numbered():
            lead_suit = card.suit
            break

    # Check for special card presence
    has_skull_king = any(c.is_skull_king() for c in played_cards)
    has_pirate = any(c.is_pirate() for c in played_cards)
    has_mermaid = any(c.is_mermaid() for c in played_cards)

    bonus = 0

    # Case 1: Skull King is in play
    if has_skull_king:
        if has_mermaid:
            # Mermaid captures Skull King! First mermaid wins.
            bonus += 50  # Bonus for capturing SK
            # Also count pirates captured
            bonus += sum(30 for c in played_cards if c.is_pirate())
            for i, card in enumerate(played_cards):
                if card.is_mermaid():
                    return TrickResult(winner_index=i, winner_card=card, bonus_points=bonus)
        else:
            # Skull King wins, captures all pirates for bonus
            bonus += sum(30 for c in played_cards if c.is_pirate())
            for i, card in enumerate(played_cards):
                if card.is_skull_king():
                    return TrickResult(winner_index=i, winner_card=card, bonus_points=bonus)

    # Case 2: No Skull King, but Pirates present
    if has_pirate:
        # First pirate wins; bonus for capturing mermaids
        bonus += sum(20 for c in played_cards if c.is_mermaid())
        for i, card in enumerate(played_cards):
            if card.is_pirate():
                return TrickResult(winner_index=i, winner_card=card, bonus_points=bonus)

    # Case 3: No Skull King, no Pirates, but Mermaids present
    if has_mermaid:
        # First mermaid wins
        for i, card in enumerate(played_cards):
            if card.is_mermaid():
                return TrickResult(winner_index=i, winner_card=card, bonus_points=0)

    # Case 4: Only numbered cards and escapes
    # Find highest numbered card considering trump and lead suit
    best_index = -1
    best_card: Optional[Card] = None

    for i, card in enumerate(played_cards):
        if card.is_escape() or card.is_special():
            continue  # Escapes never win among numbered cards

        if best_card is None:
            best_index = i
            best_card = card
            continue

        # Compare cards
        if _beats(card, best_card, lead_suit):
            best_index = i
            best_card = card

    # If no numbered card was found (all escapes), first player wins
    if best_index == -1:
        return TrickResult(winner_index=0, winner_card=played_cards[0], bonus_points=0)

    return TrickResult(winner_index=best_index, winner_card=best_card, bonus_points=bonus)


def _beats(challenger: Card, current_best: Card, lead_suit: Optional[Suit]) -> bool:
    """Does challenger beat current_best?"""
    assert challenger.is_numbered() and current_best.is_numbered()

    # Trump beats non-trump
    if challenger.is_trump() and not current_best.is_trump():
        return True
    if not challenger.is_trump() and current_best.is_trump():
        return False

    # Both trump or both non-trump
    if challenger.suit == current_best.suit:
        return challenger.number > current_best.number

    # Different non-trump suits: lead suit wins
    if lead_suit is not None:
        if challenger.suit == lead_suit and current_best.suit != lead_suit:
            return True
        if current_best.suit == lead_suit and challenger.suit != lead_suit:
            return False

    # Off-suit vs off-suit: first played (current_best) holds
    return False
