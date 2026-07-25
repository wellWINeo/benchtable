"""Tests for poker hand evaluation."""

from __future__ import annotations

import pytest

from benchtable.games.poker.cards import Card, Rank, Suit
from benchtable.games.poker.evaluator import (
    best_hand,
    describe_rank,
    evaluate_hand,
)


def _card(rank: str, suit: str) -> Card:
    rank_map = {
        "2": Rank.TWO,
        "3": Rank.THREE,
        "4": Rank.FOUR,
        "5": Rank.FIVE,
        "6": Rank.SIX,
        "7": Rank.SEVEN,
        "8": Rank.EIGHT,
        "9": Rank.NINE,
        "T": Rank.TEN,
        "J": Rank.JACK,
        "Q": Rank.QUEEN,
        "K": Rank.KING,
        "A": Rank.ACE,
    }
    suit_map = {"c": Suit.CLUBS, "d": Suit.DIAMONDS, "h": Suit.HEARTS, "s": Suit.SPADES}
    return Card(rank_map[rank.upper()], suit_map[suit.lower()])


def _hand(spec: str) -> list[Card]:
    cards = []
    for i in range(0, len(spec), 2):
        cards.append(_card(spec[i], spec[i + 1]))
    return cards


class TestEvaluateHand:
    def test_high_card(self) -> None:
        rank = evaluate_hand(_hand("AhKdQcJh9s"))
        assert rank.category == 1

    def test_one_pair(self) -> None:
        rank = evaluate_hand(_hand("AhAcKdQcJh"))
        assert rank.category == 2

    def test_two_pair(self) -> None:
        rank = evaluate_hand(_hand("AhAcKdKcJh"))
        assert rank.category == 3

    def test_three_of_a_kind(self) -> None:
        rank = evaluate_hand(_hand("AhAcAdKcJh"))
        assert rank.category == 4

    def test_straight(self) -> None:
        rank = evaluate_hand(_hand("AhKdQcJhTs"))
        assert rank.category == 5

    def test_flush(self) -> None:
        rank = evaluate_hand(_hand("AhKhQhJh9h"))
        assert rank.category == 6

    def test_full_house(self) -> None:
        rank = evaluate_hand(_hand("AhAcAdKcKs"))
        assert rank.category == 7

    def test_four_of_a_kind(self) -> None:
        rank = evaluate_hand(_hand("AhAcAdAsKc"))
        assert rank.category == 8

    def test_straight_flush(self) -> None:
        rank = evaluate_hand(_hand("AhKhQhJhTh"))
        assert rank.category == 9

    def test_ace_low_straight(self) -> None:
        rank = evaluate_hand(_hand("Ah2d3c4h5s"))
        assert rank.category == 5
        assert rank.tiebreakers == (5,)

    def test_pair_beats_high_card(self) -> None:
        pair = evaluate_hand(_hand("AhAcKdQcJh"))
        high = evaluate_hand(_hand("KhQdJcTs8s"))
        assert pair > high

    def test_full_house_beats_flush(self) -> None:
        fh = evaluate_hand(_hand("AhAcAdKcKs"))
        flush = evaluate_hand(_hand("KhQhJhTh8h"))
        assert fh > flush

    def test_equal_hands_compare_by_kickers(self) -> None:
        h1 = evaluate_hand(_hand("AhKdQcJh9s"))
        h2 = evaluate_hand(_hand("AhKdQcJh8s"))
        assert h1 > h2

    def test_wrong_card_count_raises(self) -> None:
        with pytest.raises(ValueError):
            evaluate_hand(_hand("AhKd"))


class TestBestHand:
    def test_best_from_seven(self) -> None:
        cards = _hand("AhKhQhJhTh2d3c")
        rank = best_hand(cards)
        assert rank.category == 9

    def test_best_from_six(self) -> None:
        cards = _hand("AhKhQhJhTh9s")
        rank = best_hand(cards)
        assert rank.category == 9

    def test_best_from_five(self) -> None:
        cards = _hand("AhKhQhJhTh")
        rank = best_hand(cards)
        assert rank.category == 9

    def test_less_than_five_raises(self) -> None:
        with pytest.raises(ValueError):
            best_hand(_hand("AhKdQc"))


class TestDescribeRank:
    def test_describe_high_card(self) -> None:
        rank = evaluate_hand(_hand("AhKdQcJh9s"))
        desc = describe_rank(rank)
        assert "High Card" in desc

    def test_describe_pair(self) -> None:
        rank = evaluate_hand(_hand("AhAcKdQcJh"))
        desc = describe_rank(rank)
        assert "One Pair" in desc
