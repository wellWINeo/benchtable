"""Pot construction and settlement for Texas Hold'em."""

from __future__ import annotations

from dataclasses import dataclass, field

from benchtable.games.poker.cards import Card
from benchtable.games.poker.evaluator import HandRank, best_hand


@dataclass
class PotContribution:
    """A player's total contribution to the pot across all rounds."""

    player: str
    amount: int = 0
    all_in_at: int | None = None
    folded: bool = False


@dataclass(frozen=True)
class PotResult:
    """Settlement result for a single pot."""

    winners: list[str]
    amount: int
    share: int
    payouts: dict[str, int]


@dataclass
class Pot:
    """A single pot with eligible players."""

    amount: int = 0
    eligible: list[str] = field(default_factory=list[str])
    returned: dict[str, int] = field(default_factory=dict[str, int])


def build_pots(contributions: dict[str, PotContribution]) -> list[Pot]:
    """Build main and side pots from player contributions.

    Players who are all-in can only win chips up to their contribution level.
    Side pots are created for the excess.
    """
    if not contributions:
        return []
    if any(contribution.amount < 0 for contribution in contributions.values()):
        raise ValueError("Contributions must not be negative")

    effective = {
        player: contribution.amount for player, contribution in contributions.items()
    }
    returned: dict[str, int] = {}
    active = [
        player
        for player, contribution in contributions.items()
        if not contribution.folded and contribution.amount > 0
    ]
    if (
        any(contribution.amount > 0 for contribution in contributions.values())
        and not active
    ):
        raise ValueError("Contributions have no eligible players")
    if len(active) >= 2:
        highest = max(effective[player] for player in active)
        leaders = [player for player in active if effective[player] == highest]
        if len(leaders) == 1:
            leader = leaders[0]
            second = max(
                (effective[player] for player in active if player != leader),
                default=0,
            )
            excess = highest - second
            if excess > 0:
                effective[leader] -= excess
                returned[leader] = excess

    levels = sorted({amount for amount in effective.values() if amount > 0})

    pots: list[Pot] = []
    processed_amount = 0

    for level in levels:
        pot_amount = 0
        eligible: list[str] = []
        for p, c in contributions.items():
            contribution_at_level = min(effective[p], level) - processed_amount
            if contribution_at_level > 0:
                pot_amount += contribution_at_level
                if not c.folded and effective[p] >= level:
                    eligible.append(p)

        if pot_amount > 0:
            if eligible:
                pots.append(Pot(amount=pot_amount, eligible=eligible))
            elif pots:
                # Folded dead money above the last contested level remains in
                # that pot rather than becoming an unclaimable side pot.
                pots[-1].amount += pot_amount
        processed_amount = level

    if pots and returned:
        pots[0].returned.update(returned)

    contribution_total = sum(
        contribution.amount for contribution in contributions.values()
    )
    pot_total = sum(pot.amount for pot in pots)
    returned_total = sum(amount for pot in pots for amount in pot.returned.values())
    if contribution_total != pot_total + returned_total:
        raise ValueError("Pot construction conservation violation")

    return pots


def settle_pots(
    pots: list[Pot],
    hands: dict[str, list[Card]],
    contributions: dict[str, PotContribution],
    seat_order: list[str] | None = None,
) -> list[PotResult]:
    """Settle each pot using hand evaluation.

    Ties split the pot. Odd chips go to the earliest eligible player by seat order.
    """
    if any(contribution.amount < 0 for contribution in contributions.values()):
        raise ValueError("Contributions must not be negative")
    contribution_total = sum(
        contribution.amount for contribution in contributions.values()
    )
    pot_total = sum(pot.amount for pot in pots)
    returned_total = sum(amount for pot in pots for amount in pot.returned.values())
    if contribution_total != pot_total + returned_total:
        raise ValueError("Settlement conservation violation")

    results: list[PotResult] = []

    for pot in pots:
        if not pot.eligible:
            raise ValueError("Pot has no eligible players")
        if pot.amount <= 0:
            raise ValueError("Pot amount must be positive")

        ranked: list[tuple[str, HandRank]] = []
        for player in pot.eligible:
            if player in hands and len(hands[player]) >= 5:
                rank = best_hand(hands[player])
                ranked.append((player, rank))

        if not ranked:
            raise ValueError("Pot has no eligible hand to settle")

        best_rank = max(r[1] for r in ranked)
        winners = [p for p, r in ranked if r == best_rank]

        order = seat_order if seat_order is not None else pot.eligible
        if any(winner not in order for winner in winners):
            raise ValueError("Seat order must contain every pot winner")
        winners = sorted(winners, key=order.index)

        share = pot.amount // len(winners)
        payouts = {
            winner: share + (1 if index < pot.amount % len(winners) else 0)
            for index, winner in enumerate(winners)
        }
        if sum(payouts.values()) != pot.amount:
            raise ValueError("Pot payout does not conserve chips")
        results.append(
            PotResult(
                winners=winners,
                amount=pot.amount,
                share=share,
                payouts=payouts,
            )
        )

    if sum(result.amount for result in results) != pot_total:
        raise ValueError("Settlement payout conservation violation")
    return results
