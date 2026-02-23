"""Scoring for Skull King (Graybeard variant).

Scoring rules:
- If bid == 0:
  - Success: + round_number * 10
  - Failure: - round_number * 10
- If bid > 0:
  - Success (tricks_won == bid): + bid * 20 + bonus_points
  - Failure: - abs(tricks_won - bid) * 10
"""

from __future__ import annotations


def score_round(bid: int, tricks_won: int, bonus_points: int, round_number: int) -> int:
    """Calculate score for a player's round.

    Args:
        bid: Player's bid for this round.
        tricks_won: Actual tricks won.
        bonus_points: Accumulated bonus from special card captures.
        round_number: Current round (1-10).

    Returns:
        Points earned (can be negative).
    """
    if bid == 0:
        if tricks_won == 0:
            return round_number * 10
        else:
            return -(round_number * 10)
    else:
        if tricks_won == bid:
            return bid * 20 + bonus_points
        else:
            return -(abs(tricks_won - bid) * 10)
