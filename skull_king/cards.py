"""Card definitions for Skull King.

70-card deck:
- 4 suits (Yellow, Green, Purple, Black/Jolly Roger) x 14 numbered cards (1-14)
- 5 Escape cards
- 5 Pirates
- 2 Mermaids
- 1 Skull King
- 1 Tigress (can be played as Escape or Pirate)
- Future: Kraken, White Whale, Loot cards
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional


class Suit(Enum):
    YELLOW = 0
    GREEN = 1
    PURPLE = 2
    BLACK = 3  # Trump suit (Jolly Roger)


class SpecialType(Enum):
    ESCAPE = 0
    PIRATE = 1
    MERMAID = 2
    SKULL_KING = 3
    TIGRESS = 4
    # Future expansions:
    # KRAKEN = 5
    # WHITE_WHALE = 6
    # LOOT = 7


@dataclass(frozen=True)
class Card:
    """A Skull King card.

    For numbered cards: suit and number are set, special is None.
    For special cards: special is set, suit and number are None.
    card_id is a unique 0-69 index used for binary card tracking.
    """
    card_id: int
    suit: Optional[Suit] = None
    number: Optional[int] = None
    special: Optional[SpecialType] = None
    # For Tigress: whether she's being played as pirate or escape
    tigress_as_pirate: Optional[bool] = None

    def is_numbered(self) -> bool:
        return self.suit is not None

    def is_special(self) -> bool:
        return self.special is not None

    def is_trump(self) -> bool:
        return self.suit == Suit.BLACK

    def is_escape(self) -> bool:
        return self.special == SpecialType.ESCAPE or (
            self.special == SpecialType.TIGRESS and self.tigress_as_pirate is False
        )

    def is_pirate(self) -> bool:
        return self.special == SpecialType.PIRATE or (
            self.special == SpecialType.TIGRESS and self.tigress_as_pirate is True
        )

    def is_mermaid(self) -> bool:
        return self.special == SpecialType.MERMAID

    def is_skull_king(self) -> bool:
        return self.special == SpecialType.SKULL_KING

    def is_tigress(self) -> bool:
        return self.special == SpecialType.TIGRESS

    def with_tigress_choice(self, as_pirate: bool) -> Card:
        """Return a new Card with the Tigress choice set."""
        assert self.special == SpecialType.TIGRESS
        return Card(
            card_id=self.card_id,
            special=SpecialType.TIGRESS,
            tigress_as_pirate=as_pirate,
        )

    @property
    def strength(self) -> int:
        """Numeric strength for comparison. Only meaningful for numbered cards."""
        if self.number is not None:
            return self.number
        return 0

    def __repr__(self) -> str:
        if self.is_numbered():
            return f"{self.suit.name[0]}{self.number}"
        if self.is_tigress():
            mode = "P" if self.tigress_as_pirate else ("E" if self.tigress_as_pirate is False else "?")
            return f"Tigress({mode})"
        return self.special.name

    def __hash__(self) -> int:
        return hash(self.card_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return self.card_id == other.card_id


# Card ID layout:
# 0-13:  Yellow 1-14
# 14-27: Green 1-14
# 28-41: Purple 1-14
# 42-55: Black 1-14
# 56-60: Escape x5
# 61-65: Pirate x5
# 66-67: Mermaid x2
# 68:    Skull King
# 69:    Tigress

NUM_CARDS = 70


def create_deck() -> List[Card]:
    """Create a full 70-card Skull King deck."""
    cards: List[Card] = []
    card_id = 0

    # Numbered cards: 4 suits x 14 numbers
    for suit in Suit:
        for num in range(1, 15):
            cards.append(Card(card_id=card_id, suit=suit, number=num))
            card_id += 1

    # 5 Escapes
    for _ in range(5):
        cards.append(Card(card_id=card_id, special=SpecialType.ESCAPE))
        card_id += 1

    # 5 Pirates
    for _ in range(5):
        cards.append(Card(card_id=card_id, special=SpecialType.PIRATE))
        card_id += 1

    # 2 Mermaids
    for _ in range(2):
        cards.append(Card(card_id=card_id, special=SpecialType.MERMAID))
        card_id += 1

    # 1 Skull King
    cards.append(Card(card_id=card_id, special=SpecialType.SKULL_KING))
    card_id += 1

    # 1 Tigress
    cards.append(Card(card_id=card_id, special=SpecialType.TIGRESS))
    card_id += 1

    assert card_id == NUM_CARDS
    assert len(cards) == NUM_CARDS
    return cards


def card_id_to_card(card_id: int) -> Card:
    """Convert a card_id back to a Card object."""
    if card_id < 56:
        suit = Suit(card_id // 14)
        number = (card_id % 14) + 1
        return Card(card_id=card_id, suit=suit, number=number)
    elif card_id < 61:
        return Card(card_id=card_id, special=SpecialType.ESCAPE)
    elif card_id < 66:
        return Card(card_id=card_id, special=SpecialType.PIRATE)
    elif card_id < 68:
        return Card(card_id=card_id, special=SpecialType.MERMAID)
    elif card_id == 68:
        return Card(card_id=card_id, special=SpecialType.SKULL_KING)
    elif card_id == 69:
        return Card(card_id=card_id, special=SpecialType.TIGRESS)
    else:
        raise ValueError(f"Invalid card_id: {card_id}")
