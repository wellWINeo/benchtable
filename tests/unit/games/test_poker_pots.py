"""Tests for poker pot construction and settlement."""

from __future__ import annotations

import pytest

from benchtable.games.poker.cards import Card, Rank, Suit
from benchtable.games.poker.pots import (
    Pot,
    PotContribution,
    build_pots,
    settle_pots,
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


class TestBuildPots:
    def test_simple_two_player_pot(self) -> None:
        contributions = {
            "a": PotContribution(player="a", amount=100),
            "b": PotContribution(player="b", amount=100),
        }
        pots = build_pots(contributions)
        assert len(pots) == 1
        assert pots[0].amount == 200
        assert set(pots[0].eligible) == {"a", "b"}

    def test_unequal_contributions(self) -> None:
        contributions = {
            "a": PotContribution(player="a", amount=50),
            "b": PotContribution(player="b", amount=100),
        }
        pots = build_pots(contributions)
        assert len(pots) >= 1
        total = sum(p.amount for p in pots)
        assert total == 100
        assert pots[0].returned == {"b": 50}

    def test_folded_player_excluded(self) -> None:
        contributions = {
            "a": PotContribution(player="a", amount=100),
            "b": PotContribution(player="b", amount=100, folded=True),
        }
        pots = build_pots(contributions)
        assert len(pots) == 1
        assert pots[0].amount == 200
        assert pots[0].eligible == ["a"]

    def test_folded_contribution_remains_in_pot_amount(self) -> None:
        contributions = {
            "a": PotContribution(player="a", amount=100),
            "b": PotContribution(player="b", amount=100, folded=True),
            "c": PotContribution(player="c", amount=100),
        }

        pots = build_pots(contributions)

        assert sum(p.amount for p in pots) == 300
        assert pots[0].eligible == ["a", "c"]

    def test_uncalled_excess_is_returned_before_pot_settlement(self) -> None:
        contributions = {
            "a": PotContribution(player="a", amount=100),
            "b": PotContribution(player="b", amount=50),
        }

        pots = build_pots(contributions)

        assert sum(p.amount for p in pots) == 100
        assert pots[0].returned == {"a": 50}

    def test_folded_dead_money_prevents_return_and_forms_side_pot(self) -> None:
        contributions = {
            "a": PotContribution(player="a", amount=100),
            "b": PotContribution(player="b", amount=50),
            "c": PotContribution(player="c", amount=100, folded=True),
        }

        pots = build_pots(contributions)

        assert [(pot.amount, pot.eligible) for pot in pots] == [
            (150, ["a", "b"]),
            (100, ["a"]),
        ]
        assert all(not pot.returned for pot in pots)

    def test_all_in_creates_side_pot(self) -> None:
        contributions = {
            "a": PotContribution(player="a", amount=50, all_in_at=50),
            "b": PotContribution(player="b", amount=100),
            "c": PotContribution(player="c", amount=100),
        }
        pots = build_pots(contributions)
        assert len(pots) >= 2
        total = sum(p.amount for p in pots)
        assert total == 250
        assert [(pot.amount, pot.eligible) for pot in pots] == [
            (150, ["a", "b", "c"]),
            (100, ["b", "c"]),
        ]

    def test_odd_chips_distributed(self) -> None:
        contributions = {
            "a": PotContribution(player="a", amount=100),
            "b": PotContribution(player="b", amount=100),
            "c": PotContribution(player="c", amount=100),
        }
        pots = build_pots(contributions)
        total = sum(p.amount for p in pots)
        assert total == 300

    def test_empty_contributions(self) -> None:
        pots = build_pots({})
        assert pots == []

    def test_all_folded_contributions_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="eligible"):
            build_pots(
                {
                    "a": PotContribution(player="a", amount=100, folded=True),
                    "b": PotContribution(player="b", amount=50, folded=True),
                }
            )


class TestSettlePots:
    def test_empty_eligibility_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="eligible"):
            settle_pots(
                [Pot(amount=100, eligible=[])],
                {},
                {"a": PotContribution(player="a", amount=100, folded=True)},
            )

    def test_contribution_total_must_equal_pot_total(self) -> None:
        with pytest.raises(ValueError, match="conservation"):
            settle_pots(
                [Pot(amount=100, eligible=["a", "b"])],
                {},
                {
                    "a": PotContribution(player="a", amount=75),
                    "b": PotContribution(player="b", amount=75),
                },
            )

    def test_winner_takes_all(self) -> None:
        pot = Pot(amount=200, eligible=["a", "b"])
        hands = {
            "a": [
                _card("A", "s"),
                _card("K", "s"),
                _card("Q", "s"),
                _card("J", "s"),
                _card("T", "s"),
                _card("2", "d"),
                _card("3", "c"),
            ],
            "b": [
                _card("2", "d"),
                _card("3", "c"),
                _card("4", "d"),
                _card("5", "d"),
                _card("7", "d"),
                _card("8", "s"),
                _card("9", "c"),
            ],
        }
        contributions = {
            "a": PotContribution(player="a", amount=100),
            "b": PotContribution(player="b", amount=100),
        }
        results = settle_pots([pot], hands, contributions)
        assert any(r.amount == 200 for r in results)
        assert results[0].payouts == {"a": 200}

    def test_split_pot(self) -> None:
        # Both players have the same best hand
        cards = [
            _card("A", "s"),
            _card("K", "s"),
            _card("Q", "s"),
            _card("J", "s"),
            _card("T", "s"),
        ]
        pot = Pot(amount=200, eligible=["a", "b"])
        hands = {
            "a": cards + [_card("2", "d"), _card("3", "c")],
            "b": cards + [_card("4", "d"), _card("5", "c")],
        }
        contributions = {
            "a": PotContribution(player="a", amount=100),
            "b": PotContribution(player="b", amount=100),
        }
        results = settle_pots([pot], hands, contributions)
        split_results = [r for r in results if len(r.winners) > 1]
        assert len(split_results) >= 1
        assert split_results[0].share == 100

    def test_odd_chip_to_first_eligible(self) -> None:
        pot = Pot(amount=301, eligible=["a", "b"])
        cards_a = [
            _card("A", "s"),
            _card("K", "s"),
            _card("Q", "s"),
            _card("J", "s"),
            _card("T", "s"),
            _card("2", "d"),
            _card("3", "c"),
        ]
        cards_b = [
            _card("9", "s"),
            _card("K", "s"),
            _card("Q", "s"),
            _card("J", "s"),
            _card("T", "s"),
            _card("2", "d"),
            _card("3", "c"),
        ]
        hands = {"a": cards_a, "b": cards_b}
        contributions = {
            "a": PotContribution(player="a", amount=151),
            "b": PotContribution(player="b", amount=150),
        }
        results = settle_pots([pot], hands, contributions)
        total_distributed = sum(r.amount for r in results)
        assert total_distributed == 301

    def test_odd_chip_recipient_follows_explicit_seat_order(self) -> None:
        cards = [
            _card("A", "s"),
            _card("K", "s"),
            _card("Q", "s"),
            _card("J", "s"),
            _card("T", "s"),
        ]
        pot = Pot(amount=301, eligible=["a", "b"])
        hands = {
            "a": cards + [_card("2", "d"), _card("3", "c")],
            "b": cards + [_card("4", "d"), _card("5", "c")],
        }
        contributions = {
            "a": PotContribution(player="a", amount=151),
            "b": PotContribution(player="b", amount=150),
        }

        result = settle_pots([pot], hands, contributions, seat_order=["b", "a"])[0]

        assert result.payouts == {"b": 151, "a": 150}

    def test_settlement_conserves_every_pot(self) -> None:
        pots = [
            Pot(amount=150, eligible=["a", "b", "c"]),
            Pot(amount=100, eligible=["b", "c"]),
        ]
        cards = [
            _card("A", "s"),
            _card("K", "s"),
            _card("Q", "s"),
            _card("J", "s"),
            _card("T", "s"),
        ]
        hands = {
            "a": cards + [_card("2", "d"), _card("3", "c")],
            "b": [_card("2", "d")] * 7,
            "c": [_card("3", "c")] * 7,
        }
        contributions = {
            "a": PotContribution(player="a", amount=50, all_in_at=50),
            "b": PotContribution(player="b", amount=100),
            "c": PotContribution(player="c", amount=100),
        }

        results = settle_pots(pots, hands, contributions, seat_order=["a", "b", "c"])

        assert len(results) == len(pots)
        assert all(sum(result.payouts.values()) == result.amount for result in results)
        assert sum(result.amount for result in results) == sum(
            sum(result.payouts.values()) for result in results
        )

    def test_settlement_does_not_mutate_inputs(self) -> None:
        pot = Pot(amount=200, eligible=["a", "b"])
        hands = {
            "a": [_card("A", "s")] * 7,
            "b": [_card("K", "s")] * 7,
        }
        contributions = {
            "a": PotContribution(player="a", amount=100),
            "b": PotContribution(player="b", amount=100),
        }
        original_eligible = list(pot.eligible)
        original_contributions = {
            key: (value.amount, value.folded) for key, value in contributions.items()
        }

        settle_pots([pot], hands, contributions, seat_order=["a", "b"])

        assert pot.eligible == original_eligible
        assert {
            key: (value.amount, value.folded) for key, value in contributions.items()
        } == original_contributions
