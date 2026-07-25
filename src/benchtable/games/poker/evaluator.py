"""Texas Hold'em hand evaluation."""

from __future__ import annotations

from itertools import combinations

from benchtable.games.poker.cards import Card


class HandRank:
    """Comparable hand rank: (category, tiebreaker tuple).

    Categories (higher is better):
        1 = High Card
        2 = One Pair
        3 = Two Pair
        4 = Three of a Kind
        5 = Straight
        6 = Flush
        7 = Full House
        8 = Four of a Kind
        9 = Straight Flush
    """

    __slots__ = ("_value",)

    _value: tuple[int, tuple[int, ...]]

    def __init__(self, category: int, tiebreakers: tuple[int, ...]) -> None:
        object.__setattr__(self, "_value", (category, tiebreakers))

    @property
    def category(self) -> int:
        return self._value[0]

    @property
    def tiebreakers(self) -> tuple[int, ...]:
        return self._value[1]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HandRank):
            return NotImplemented
        return self._value == other._value

    def __lt__(self, other: HandRank) -> bool:
        return self._value < other._value

    def __le__(self, other: HandRank) -> bool:
        return self._value <= other._value

    def __repr__(self) -> str:
        return f"HandRank({CATEGORY_NAMES.get(self.category, '?')}, {self.tiebreakers})"


CATEGORY_NAMES = {
    1: "High Card",
    2: "One Pair",
    3: "Two Pair",
    4: "Three of a Kind",
    5: "Straight",
    6: "Flush",
    7: "Full House",
    8: "Four of a Kind",
    9: "Straight Flush",
}


def _rank_values(cards: list[Card]) -> list[int]:
    return sorted([c.rank.value for c in cards], reverse=True)


def _evaluate_five(cards: list[Card]) -> HandRank:
    """Evaluate exactly 5 cards."""
    values = _rank_values(cards)
    suits = [c.suit for c in cards]

    is_flush = len(set(suits)) == 1

    # Check straight
    unique = sorted(set(values), reverse=True)
    is_straight = False
    straight_high = 0
    if len(unique) == 5:
        if unique[0] - unique[4] == 4:
            is_straight = True
            straight_high = unique[0]
        # Ace-low straight (A-2-3-4-5)
        elif unique == [14, 5, 4, 3, 2]:
            is_straight = True
            straight_high = 5

    if is_straight and is_flush:
        return HandRank(9, (straight_high,))

    # Count ranks
    from collections import Counter

    counts = Counter(values)
    most_common = counts.most_common()
    # Sort by count desc, then by rank desc
    most_common.sort(key=lambda x: (x[1], x[0]), reverse=True)

    group_ranks = [rank for rank, _ in most_common]
    group_counts = [count for _, count in most_common]

    if group_counts == [4, 1]:
        return HandRank(8, tuple(group_ranks))

    if group_counts == [3, 2]:
        return HandRank(7, tuple(group_ranks))

    if is_flush:
        return HandRank(6, tuple(values))

    if is_straight:
        return HandRank(5, (straight_high,))

    if group_counts == [3, 1, 1]:
        return HandRank(4, tuple(group_ranks))

    if group_counts == [2, 2, 1]:
        return HandRank(3, tuple(group_ranks))

    if group_counts == [2, 1, 1, 1]:
        return HandRank(2, tuple(group_ranks))

    return HandRank(1, tuple(values))


def evaluate_hand(cards: list[Card]) -> HandRank:
    """Evaluate a 5-card hand."""
    if len(cards) != 5:
        raise ValueError("evaluate_hand requires exactly 5 cards")
    return _evaluate_five(cards)


def best_hand(cards: list[Card]) -> HandRank:
    """Evaluate the best 5-card hand from up to 7 cards."""
    if len(cards) < 5:
        raise ValueError("best_hand requires at least 5 cards")
    if len(cards) == 5:
        return _evaluate_five(cards)
    best: HandRank | None = None
    for combo in combinations(cards, 5):
        rank = _evaluate_five(list(combo))
        if best is None or rank > best:
            best = rank
    assert best is not None
    return best


def describe_rank(rank: HandRank) -> str:
    """Return a human-readable description of a hand rank."""
    name = CATEGORY_NAMES.get(rank.category, "Unknown")
    return f"{name} ({', '.join(str(v) for v in rank.tiebreakers)})"
