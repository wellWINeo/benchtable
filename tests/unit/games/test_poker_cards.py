"""Tests for poker card and deck primitives."""

from __future__ import annotations

import pytest

from benchtable.games.poker.cards import STANDARD_DECK, Card, Deck, Rank, Suit


class TestCard:
    def test_standard_deck_has_52_cards(self) -> None:
        assert len(STANDARD_DECK) == 52

    def test_all_cards_unique(self) -> None:
        assert len(set(STANDARD_DECK)) == 52

    def test_card_equality(self) -> None:
        assert Card(Rank.ACE, Suit.SPADES) == Card(Rank.ACE, Suit.SPADES)
        assert Card(Rank.ACE, Suit.SPADES) != Card(Rank.ACE, Suit.HEARTS)

    def test_card_ordering(self) -> None:
        assert Card(Rank.TWO, Suit.CLUBS) < Card(Rank.THREE, Suit.CLUBS)
        assert Card(Rank.ACE, Suit.CLUBS) > Card(Rank.KING, Suit.SPADES)

    def test_card_string(self) -> None:
        assert str(Card(Rank.ACE, Suit.SPADES)) == "As"
        assert str(Card(Rank.TEN, Suit.HEARTS)) == "Th"
        assert str(Card(Rank.TWO, Suit.HEARTS)) == "2h"

    def test_card_to_dict(self) -> None:
        d = Card(Rank.QUEEN, Suit.DIAMONDS).to_dict()
        assert d == {"rank": "Q", "suit": "d"}

    def test_card_hashable(self) -> None:
        cards = {Card(Rank.ACE, Suit.SPADES), Card(Rank.ACE, Suit.SPADES)}
        assert len(cards) == 1

    def test_card_is_immutable(self) -> None:
        card = Card(Rank.ACE, Suit.SPADES)

        with pytest.raises(AttributeError):
            card.rank = Rank.KING  # type: ignore[misc]
        with pytest.raises(AttributeError):
            card._rank = Rank.KING  # type: ignore[misc]

    def test_rank_symbols_are_unambiguous(self) -> None:
        symbols = [rank.short() for rank in Rank]

        assert symbols == [
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "T",
            "J",
            "Q",
            "K",
            "A",
        ]


class TestDeck:
    def test_seeded_shuffle_deterministic(self) -> None:
        d1 = Deck(seed=42)
        d2 = Deck(seed=42)
        cards1 = d1.draw(52)
        cards2 = d2.draw(52)
        assert cards1 == cards2

    def test_different_seeds_different_order(self) -> None:
        d1 = Deck(seed=1)
        d2 = Deck(seed=2)
        cards1 = d1.draw(52)
        cards2 = d2.draw(52)
        assert cards1 != cards2

    def test_draw_without_replacement(self) -> None:
        deck = Deck(seed=42)
        drawn = deck.draw(52)
        assert len(set(drawn)) == 52
        assert deck.remaining == 0

    def test_draw_exhaustion(self) -> None:
        deck = Deck(seed=42)
        deck.draw(52)
        with pytest.raises(ValueError, match="Not enough cards"):
            deck.draw(1)

    def test_draw_partial(self) -> None:
        deck = Deck(seed=42)
        first = deck.draw(5)
        second = deck.draw(5)
        assert len(first) == 5
        assert len(second) == 5
        assert set(first).isdisjoint(set(second))
        assert deck.remaining == 42

    def test_draw_negative_raises(self) -> None:
        deck = Deck(seed=42)
        with pytest.raises(ValueError, match="non-negative"):
            deck.draw(-1)

    def test_separate_decks_independent(self) -> None:
        d1 = Deck(seed=42)
        d2 = Deck(seed=42)
        d1.draw(10)
        assert d2.remaining == 52

    def test_standard_deck_surface_is_immutable(self) -> None:
        with pytest.raises(AttributeError):
            STANDARD_DECK.append(Card(Rank.ACE, Suit.CLUBS))  # type: ignore[attr-defined]
