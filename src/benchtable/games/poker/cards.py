"""Card representation and deck management for Texas Hold'em."""

from __future__ import annotations

import random
from enum import IntEnum


class Rank(IntEnum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

    def short(self) -> str:
        return {
            Rank.TWO: "2",
            Rank.THREE: "3",
            Rank.FOUR: "4",
            Rank.FIVE: "5",
            Rank.SIX: "6",
            Rank.SEVEN: "7",
            Rank.EIGHT: "8",
            Rank.NINE: "9",
            Rank.TEN: "T",
            Rank.JACK: "J",
            Rank.QUEEN: "Q",
            Rank.KING: "K",
            Rank.ACE: "A",
        }[self]


class Suit(IntEnum):
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3

    def short(self) -> str:
        return "cdhs"[self.value]


class Card:
    """An immutable playing card."""

    __slots__ = ("rank", "suit")

    rank: Rank
    suit: Suit

    def __init__(self, rank: Rank, suit: Suit) -> None:
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "suit", suit)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"Card is immutable; cannot assign {name}")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"Card is immutable; cannot delete {name}")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return self.rank == other.rank and self.suit == other.suit

    def __hash__(self) -> int:
        return hash((self.rank, self.suit))

    def __lt__(self, other: Card) -> bool:
        return (self.rank, self.suit) < (other.rank, other.suit)

    def __repr__(self) -> str:
        return f"Card({self.rank.name}, {self.suit.name})"

    def __str__(self) -> str:
        return f"{self.rank.short()}{self.suit.short()}"

    def to_dict(self) -> dict[str, str]:
        return {"rank": self.rank.short(), "suit": self.suit.short()}


STANDARD_DECK: tuple[Card, ...] = tuple(
    Card(rank, suit) for suit in Suit for rank in Rank
)


class Deck:
    """A seeded, finite deck that draws without replacement."""

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)
        self._cards = list(STANDARD_DECK)
        self._rng.shuffle(self._cards)
        self._drawn = 0

    def draw(self, count: int = 1) -> list[Card]:
        """Draw *count* cards from the top. Raises if not enough remain."""
        if count < 0:
            raise ValueError("count must be non-negative")
        remaining = len(self._cards) - self._drawn
        if count > remaining:
            raise ValueError(
                f"Not enough cards: requested {count}, {remaining} remaining"
            )
        start = self._drawn
        self._drawn += count
        return self._cards[start : self._drawn]

    @property
    def remaining(self) -> int:
        return len(self._cards) - self._drawn
