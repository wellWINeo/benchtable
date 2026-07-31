"""Poker session state machine and plugin factory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

from benchtable.contracts import (
    GameMetrics,
    GameResult,
    JsonObject,
    MatchMemorySummary,
    Observation,
    PluginEvent,
    ToolSpec,
    Transition,
)
from benchtable.errors import InvalidActionError
from benchtable.games.poker.cards import Card, Deck
from benchtable.games.poker.config import PokerConfig
from benchtable.games.poker.pots import (
    PotContribution,
    build_pots,
    settle_pots,
)


class PokerAction(str):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALL_IN = "all_in"

    ALL_VALUES = (FOLD, CHECK, CALL, BET, RAISE, ALL_IN)


class Street(str):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"

    ALL = (PREFLOP, FLOP, TURN, RIVER)


def _empty_card_list() -> list[Card]:
    return []


def _empty_str_list() -> list[str]:
    return []


def _empty_str_int_dict() -> dict[str, int]:
    return {}


def _empty_str_set() -> set[str]:
    return set()


@dataclass
class _PlayerState:
    id: str
    stack: int
    active: bool = True
    hole_cards: list[Card] = field(default_factory=_empty_card_list)
    folded: bool = False
    all_in: bool = False


@dataclass
class _HandState:
    hand_index: int
    seed: int
    deck: Deck
    dealer_id: str
    small_blind_id: str
    big_blind_id: str
    community: list[Card] = field(default_factory=_empty_card_list)
    street: str = Street.PREFLOP
    current_bet: int = 0
    min_raise: int = 0
    bet_open: bool = False
    pot: int = 0
    hand_contributions: dict[str, int] = field(default_factory=_empty_str_int_dict)
    street_contributions: dict[str, int] = field(default_factory=_empty_str_int_dict)
    acted_this_round: set[str] = field(default_factory=_empty_str_set)
    last_raiser: str | None = None
    burn_cards: list[Card] = field(default_factory=_empty_card_list)
    action_history: list[str] = field(default_factory=_empty_str_list)
    pending_actors: list[str] = field(default_factory=_empty_str_list)
    start_stacks: dict[str, int] = field(default_factory=_empty_str_int_dict)


@dataclass(frozen=True)
class _ActionState:
    """Street-local values needed to validate one actor's action."""

    total_commitment: int
    street_commitment: int
    to_call: int
    available_stack: int
    current_bet: int
    minimum_raise: int
    bet_open: bool


class PokerSession:
    """Private session state machine for a poker match."""

    def __init__(
        self,
        *,
        players: list[str],
        initial_stack: int,
        small_blind: int,
        big_blind: int,
        hands_per_match: int,
        seed: int,
        invalid_turn_policy: str = "forced_fold",
        memory_max_entries: int = 100,
        memory_max_chars: int = 20000,
    ) -> None:
        if len(players) < 2:
            raise ValueError("At least 2 players required")
        if small_blind <= 0:
            raise ValueError("Small blind must be positive")
        if big_blind <= small_blind:
            raise ValueError("Big blind must be greater than small blind")
        if initial_stack < big_blind:
            raise ValueError("Initial stack must cover big blind")
        if hands_per_match <= 0:
            raise ValueError("hands_per_match must be positive")

        self._player_ids = list(players)
        self._initial_stack = initial_stack
        self._small_blind = small_blind
        self._big_blind = big_blind
        self._hands_per_match = hands_per_match
        self._seed = seed
        self._invalid_turn_policy = invalid_turn_policy
        self._memory_max_entries = memory_max_entries
        self._memory_max_chars = memory_max_chars

        self._players = {pid: _PlayerState(pid, initial_stack) for pid in players}
        self._hand_index = 0
        self._hands_played = 0
        self._match_over = False
        self._public_hand_summaries: list[JsonObject] = []
        self._failure_calls: list[str] = []
        self._recovery_count = 0
        self._failure_reason: str | None = None
        self._finish_reason: str | None = None
        self._wins: dict[str, int] = {pid: 0 for pid in players}
        self._hand_events: list[JsonObject] = []
        self._pending_memory_summaries: list[MatchMemorySummary] = []
        self._current_actor_id = ""
        self._current_actor_index = 0
        self._active_order: list[str] = []
        self._hand: _HandState | None = None
        self._dealer_id = self._player_ids[0]
        self._in_hand_turn = 0
        self._start_new_hand()

    def _active_players(self) -> list[str]:
        return [p for p in self._player_ids if self._players[p].active]

    def _start_new_hand(self) -> None:
        active = self._active_players()
        if len(active) < 2:
            self._match_over = True
            return
        if self._hands_played >= self._hands_per_match:
            self._match_over = True
            return

        self._in_hand_turn = 0
        hand_seed = self._seed + self._hand_index
        deck = Deck(hand_seed)

        if self._dealer_id not in active:
            self._dealer_id = self._next_active_seat(self._dealer_id, active)
        dealer_id = self._dealer_id
        dealer_index = active.index(dealer_id)
        sb_id, bb_id = self._blinds_positions(active, dealer_index)

        self._hand = _HandState(
            hand_index=self._hand_index,
            seed=hand_seed,
            deck=deck,
            dealer_id=dealer_id,
            small_blind_id=sb_id,
            big_blind_id=bb_id,
        )

        for pid in active:
            self._players[pid].hole_cards = []
            self._players[pid].folded = False
            self._players[pid].all_in = False
            self._hand.hand_contributions[pid] = 0
            self._hand.street_contributions[pid] = 0
            self._hand.start_stacks[pid] = self._players[pid].stack

        for _ in range(2):
            for pid in active:
                self._players[pid].hole_cards.extend(deck.draw(1))

        sb_amount = min(self._small_blind, self._players[sb_id].stack)
        bb_amount = min(self._big_blind, self._players[bb_id].stack)
        self._post_blind(sb_id, sb_amount)
        self._post_blind(bb_id, bb_amount)
        self._hand.current_bet = bb_amount
        self._hand.min_raise = self._big_blind

        self._active_order = list(active)
        self._set_preflop_actor(active, sb_id, bb_id)

        self._hand_events.append(
            cast(
                JsonObject,
                {
                    "event_type": "hand_start",
                    "hand_index": self._hand_index,
                    "seed": hand_seed,
                    "dealer": dealer_id,
                    "small_blind": sb_id,
                    "big_blind": bb_id,
                    "small_blind_amount": sb_amount,
                    "big_blind_amount": bb_amount,
                    "players": list(active),
                },
            )
        )

        if not any(self._can_act(pid) for pid in active):
            self._runout_remaining()

    def _next_active_seat(self, player_id: str, active: list[str]) -> str:
        start = self._player_ids.index(player_id)
        for offset in range(1, len(self._player_ids) + 1):
            candidate = self._player_ids[(start + offset) % len(self._player_ids)]
            if candidate in active:
                return candidate
        raise ValueError("No active seat available")

    def _blinds_positions(
        self, active: list[str], dealer_index: int
    ) -> tuple[str, str]:
        if len(active) == 2:
            sb_id = active[dealer_index]
            bb_id = active[(dealer_index + 1) % 2]
        else:
            sb_id = active[(dealer_index + 1) % len(active)]
            bb_id = active[(dealer_index + 2) % len(active)]
        return sb_id, bb_id

    def _set_preflop_actor(self, active: list[str], sb_id: str, bb_id: str) -> None:
        if len(active) == 2:
            start = sb_id
        else:
            bb_idx = active.index(bb_id)
            start = active[(bb_idx + 1) % len(active)]
        self._set_next_actionable(start, include_start=True)

    def _set_next_actionable(self, start: str, *, include_start: bool = False) -> bool:
        if not self._active_order:
            return False
        start_idx = self._active_order.index(start)
        offsets = (
            range(len(self._active_order))
            if include_start
            else range(1, len(self._active_order) + 1)
        )
        for offset in offsets:
            candidate_idx = (start_idx + offset) % len(self._active_order)
            candidate = self._active_order[candidate_idx]
            if self._can_act(candidate):
                self._current_actor_id = candidate
                self._current_actor_index = candidate_idx
                return True
        return False

    def _post_blind(self, pid: str, amount: int) -> None:
        player = self._players[pid]
        actual = min(amount, player.stack)
        player.stack -= actual
        player.all_in = player.stack == 0
        assert self._hand is not None
        self._hand.pot += actual
        self._hand.hand_contributions[pid] = (
            self._hand.hand_contributions.get(pid, 0) + actual
        )
        self._hand.street_contributions[pid] = (
            self._hand.street_contributions.get(pid, 0) + actual
        )

    @property
    def current_actor_id(self) -> str:
        return self._current_actor_id

    @property
    def requires_exact_agent_ids(self) -> bool:
        return True

    @property
    def conversation_scope_id(self) -> str:
        return f"hand-{self._hand_index}"

    @property
    def memory_max_entries(self) -> int:
        return self._memory_max_entries

    @property
    def memory_max_chars(self) -> int:
        return self._memory_max_chars

    def get_turn_context(self) -> JsonObject:
        return cast(
            JsonObject,
            {"hand_index": self._hand_index, "in_hand_turn": self._in_hand_turn},
        )

    def get_observation(self, actor_id: str) -> Observation:
        player = self._players[actor_id]
        hand = self._hand
        assert hand is not None

        community_str = (
            " ".join(str(c) for c in hand.community)
            if hand.community
            else "(no community cards)"
        )
        hole_str = " ".join(str(c) for c in player.hole_cards)

        active = self._active_players()
        stacks_info = ", ".join(f"{p}: {self._players[p].stack}" for p in active)

        legal = self._legal_actions(actor_id)
        legal_str = ", ".join(legal)

        parts = [
            f"You are {actor_id}.",
            f"Hand {hand.hand_index + 1} of {self._hands_per_match}.",
            f"Street: {hand.street}.",
            f"Your hole cards: {hole_str}.",
            f"Community: {community_str}.",
            f"Pot: {hand.pot}.",
            f"Stacks: {stacks_info}.",
            f"Current bet: {hand.current_bet}.",
            f"Your committed: {self._hand_contributions(actor_id)}.",
            f"Legal actions: {legal_str}.",
        ]

        if hand.action_history:
            parts.append("Action history:")
            for entry in hand.action_history:
                parts.append(f"  {entry}")

        return Observation(
            actor_id=actor_id,
            text=" ".join(parts),
            metadata={
                "hand_index": hand.hand_index,
                "street": hand.street,
                "pot": hand.pot,
                "current_bet": hand.current_bet,
            },
        )

    def _hand_contributions(self, actor_id: str) -> int:
        assert self._hand is not None
        return self._hand.hand_contributions.get(actor_id, 0)

    def _street_contribution(self, actor_id: str) -> int:
        assert self._hand is not None
        return self._hand.street_contributions.get(actor_id, 0)

    def _action_state(self, actor_id: str) -> _ActionState:
        player = self._players[actor_id]
        hand = self._hand
        assert hand is not None
        street_commitment = self._street_contribution(actor_id)
        return _ActionState(
            total_commitment=self._hand_contributions(actor_id),
            street_commitment=street_commitment,
            to_call=max(hand.current_bet - street_commitment, 0),
            available_stack=player.stack,
            current_bet=hand.current_bet,
            minimum_raise=hand.min_raise,
            bet_open=hand.bet_open,
        )

    def get_tools(self, actor_id: str) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="poker_action",
                description="Take a poker action.",
                parameters={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": list(PokerAction.ALL_VALUES),
                            "description": "The poker action to take.",
                        },
                        "amount": {
                            "type": "integer",
                            "description": (
                                "Amount for bet/raise (target total commitment)."
                            ),
                        },
                    },
                    "required": ["action"],
                },
            )
        ]

    def _legal_actions(self, actor_id: str) -> list[str]:
        player = self._players[actor_id]
        hand = self._hand
        assert hand is not None

        if player.folded or not player.active or player.all_in:
            return []

        actions: list[str] = []
        state = self._action_state(actor_id)

        actions.append(PokerAction.FOLD)

        if state.to_call == 0:
            actions.append(PokerAction.CHECK)

        if state.to_call > 0 and player.stack > 0:
            actions.append(PokerAction.CALL)

        if player.stack > 0:
            if state.current_bet == 0 and self._bet_is_available(state):
                actions.append(PokerAction.BET)
            elif (
                (state.bet_open or hand.street == Street.PREFLOP)
                and actor_id not in hand.acted_this_round
                and self._raise_is_available(state)
            ):
                actions.append(PokerAction.RAISE)
            all_in_street_bet = state.street_commitment + state.available_stack
            all_in_is_raise = all_in_street_bet > state.current_bet
            if not (all_in_is_raise and actor_id in hand.acted_this_round):
                actions.append(PokerAction.ALL_IN)

        return actions

    @staticmethod
    def _bet_is_available(state: _ActionState) -> bool:
        max_street_bet = state.street_commitment + state.available_stack - 1
        return max_street_bet >= state.minimum_raise

    @staticmethod
    def _raise_is_available(state: _ActionState) -> bool:
        if state.available_stack <= 1:
            return False
        max_street_bet = state.street_commitment + state.available_stack - 1
        return max_street_bet - state.current_bet >= state.minimum_raise

    def _can_act(self, actor_id: str) -> bool:
        player = self._players[actor_id]
        return player.active and not player.folded and not player.all_in

    def _non_folded_active(self) -> list[str]:
        return [p for p in self._active_players() if not self._players[p].folded]

    def _non_folded_active_all_in(self) -> bool:
        active = self._non_folded_active()
        return all(self._players[p].all_in for p in active)

    def _all_in_runout_ready(self, players: list[str]) -> bool:
        """Return true once every live opponent has answered an all-in."""
        hand = self._hand
        assert hand is not None
        if not any(self._players[player].all_in for player in players):
            return False
        return all(
            self._players[player].all_in
            or (
                player in hand.acted_this_round
                and self._street_contribution(player) >= hand.current_bet
            )
            for player in players
        )

    def apply_action(
        self, actor_id: str, tool_name: str, arguments: JsonObject
    ) -> Transition:
        if self._match_over:
            raise InvalidActionError("Match is already over")

        hand = self._hand
        assert hand is not None

        if actor_id != self._current_actor_id:
            raise InvalidActionError(
                f"Not {actor_id}'s turn; current actor is {self._current_actor_id}"
            )

        if tool_name != "poker_action":
            raise InvalidActionError(f"Unknown tool: {tool_name}")

        action_str = arguments.get("action")
        if not isinstance(action_str, str):
            raise InvalidActionError("Missing or invalid 'action' field")

        if action_str not in PokerAction.ALL_VALUES:
            raise InvalidActionError(f"Invalid action: {action_str}")

        legal = self._legal_actions(actor_id)
        if action_str not in legal:
            raise InvalidActionError(
                f"Action '{action_str}' is not legal. Legal: {legal}"
            )

        player = self._players[actor_id]
        amount = arguments.get("amount")
        state = self._action_state(actor_id)
        summary = ""

        if (
            action_str
            in {
                PokerAction.FOLD,
                PokerAction.CHECK,
                PokerAction.CALL,
                PokerAction.ALL_IN,
            }
            and "amount" in arguments
        ):
            raise InvalidActionError(f"Action '{action_str}' does not accept amount")

        if action_str == PokerAction.FOLD:
            player.folded = True
            summary = f"{actor_id} folds"

        elif action_str == PokerAction.CHECK:
            summary = f"{actor_id} checks"

        elif action_str == PokerAction.CALL:
            call_amount = min(state.to_call, player.stack)
            self._commit(actor_id, call_amount)
            summary = f"{actor_id} calls {call_amount}"

        elif action_str == PokerAction.BET:
            target = self._strict_target(amount, PokerAction.BET)
            if state.current_bet != 0:
                raise InvalidActionError("Bet is only legal when no bet is open")
            delta = self._target_delta(target, state, PokerAction.BET)
            new_street_bet = state.street_commitment + delta
            if new_street_bet < state.minimum_raise:
                raise InvalidActionError(
                    "Bet must be at least target "
                    f"{state.total_commitment + state.minimum_raise}"
                )
            self._commit(actor_id, delta)
            hand.current_bet = new_street_bet
            hand.min_raise = new_street_bet
            hand.bet_open = True
            hand.last_raiser = actor_id
            summary = f"{actor_id} bets to {target}"
            hand.acted_this_round = {actor_id}

        elif action_str == PokerAction.RAISE:
            target = self._strict_target(amount, PokerAction.RAISE)
            if not state.bet_open and hand.street != Street.PREFLOP:
                raise InvalidActionError("Raise is only legal when a bet is open")
            new_street_bet = target - state.total_commitment + state.street_commitment
            raise_by = new_street_bet - state.current_bet
            if raise_by < state.minimum_raise:
                raise InvalidActionError(
                    f"Raise must increase the bet by at least {state.minimum_raise}"
                )
            delta = self._target_delta(target, state, PokerAction.RAISE)
            self._commit(actor_id, delta)
            hand.current_bet = new_street_bet
            hand.min_raise = raise_by
            hand.bet_open = True
            hand.last_raiser = actor_id
            summary = f"{actor_id} raises to {target}"
            hand.acted_this_round = {actor_id}

        elif action_str == PokerAction.ALL_IN:
            actual = player.stack
            self._commit(actor_id, actual)
            new_street_bet = self._street_contribution(actor_id)
            increase = new_street_bet - state.current_bet
            if increase > 0:
                hand.current_bet = new_street_bet
                hand.bet_open = True
                if increase >= state.minimum_raise:
                    hand.min_raise = increase
                    hand.acted_this_round = {actor_id}
                hand.last_raiser = actor_id
            summary = f"{actor_id} goes all-in to {self._hand_contributions(actor_id)}"
            hand.acted_this_round.add(actor_id)

        hand.action_history.append(summary)
        self._in_hand_turn += 1
        self._advance_actor()
        if not self._check_street_end() and not any(
            self._can_act(pid) for pid in self._non_folded_active()
        ):
            self._runout_remaining()

        return Transition(
            summary=summary,
            metrics={"pot": hand.pot, "street": hand.street},
        )

    @staticmethod
    def _strict_target(amount: object, action: str) -> int:
        if type(amount) is not int or amount <= 0:
            raise InvalidActionError(
                f"{action} requires a strict positive integer target amount"
            )
        return amount

    @staticmethod
    def _target_delta(target: int, state: _ActionState, action: str) -> int:
        delta = target - state.total_commitment
        if delta <= 0:
            raise InvalidActionError(f"{action} target must exceed current commitment")
        if delta >= state.available_stack:
            raise InvalidActionError(
                f"{action} target would consume the available stack; use all_in"
            )
        return delta

    def _commit(self, actor_id: str, amount: int) -> None:
        if amount < 0:
            raise InvalidActionError("Commitment amount must not be negative")
        player = self._players[actor_id]
        hand = self._hand
        assert hand is not None
        player.stack -= amount
        hand.hand_contributions[actor_id] = (
            hand.hand_contributions.get(actor_id, 0) + amount
        )
        hand.street_contributions[actor_id] = (
            hand.street_contributions.get(actor_id, 0) + amount
        )
        hand.pot += amount
        player.all_in = player.stack == 0

    def _advance_actor(self) -> None:
        hand = self._hand
        assert hand is not None

        non_folded = self._non_folded_active()
        if len(non_folded) <= 1:
            return

        hand.acted_this_round.add(self._current_actor_id)
        current_idx = self._active_order.index(self._current_actor_id)
        for _ in range(len(self._active_order)):
            current_idx = (current_idx + 1) % len(self._active_order)
            next_pid = self._active_order[current_idx]
            if self._can_act(next_pid):
                self._current_actor_id = next_pid
                self._current_actor_index = current_idx
                return

    def _check_street_end(self) -> bool:
        hand = self._hand
        assert hand is not None

        non_folded = self._non_folded_active()
        if len(non_folded) <= 1:
            self._end_hand(non_folded)
            return True

        if self._all_in_runout_ready(non_folded):
            self._runout_remaining()
            return True

        if not any(self._can_act(pid) for pid in non_folded):
            self._runout_remaining()
            return True

        all_acted = all(
            p in hand.acted_this_round or self._players[p].all_in for p in non_folded
        )
        if all_acted:
            all_matched = all(
                (
                    self._street_contribution(p) == hand.current_bet
                    or self._players[p].all_in
                )
                for p in non_folded
            )
            if all_matched:
                self._advance_street(non_folded)
                return True

        return False

    def _advance_street(self, non_folded: list[str]) -> None:
        hand = self._hand
        assert hand is not None
        assert hand.deck is not None

        if hand.street == Street.RIVER:
            self._end_hand(non_folded)
            return

        for pid in self._active_order:
            hand.street_contributions[pid] = 0

        hand.current_bet = 0
        hand.min_raise = self._big_blind
        hand.bet_open = False
        hand.acted_this_round = set()
        hand.last_raiser = None

        if hand.street == Street.PREFLOP:
            hand.street = Street.FLOP
            hand.burn_cards.append(hand.deck.draw(1)[0])
            hand.community.extend(hand.deck.draw(3))
        elif hand.street == Street.FLOP:
            hand.street = Street.TURN
            hand.burn_cards.append(hand.deck.draw(1)[0])
            hand.community.extend(hand.deck.draw(1))
        elif hand.street == Street.TURN:
            hand.street = Street.RIVER
            hand.burn_cards.append(hand.deck.draw(1)[0])
            hand.community.extend(hand.deck.draw(1))

        if not self._set_next_actionable(
            hand.dealer_id, include_start=len(non_folded) == 2
        ):
            self._runout_remaining()

    def _runout_remaining(self) -> None:
        hand = self._hand
        assert hand is not None
        assert hand.deck is not None

        while hand.street != Street.RIVER:
            if hand.street == Street.PREFLOP:
                hand.street = Street.FLOP
                hand.burn_cards.append(hand.deck.draw(1)[0])
                hand.community.extend(hand.deck.draw(3))
            elif hand.street == Street.FLOP:
                hand.street = Street.TURN
                hand.burn_cards.append(hand.deck.draw(1)[0])
                hand.community.extend(hand.deck.draw(1))
            elif hand.street == Street.TURN:
                hand.street = Street.RIVER
                hand.burn_cards.append(hand.deck.draw(1)[0])
                hand.community.extend(hand.deck.draw(1))

        non_folded = self._non_folded_active()
        self._end_hand(non_folded)

    def _end_hand(self, non_folded: list[str]) -> None:
        hand = self._hand
        assert hand is not None

        payouts: dict[str, int] = {}
        uncalled_returns: dict[str, int] = {}
        settlement: list[JsonObject] = []
        finish_reason = "showdown"
        if len(non_folded) == 1:
            winner = non_folded[0]
            self._players[winner].stack += hand.pot
            payouts[winner] = hand.pot
            self._wins[winner] += 1
            summary = f"{winner} wins {hand.pot} (others folded)"
            finish_reason = "fold"
        elif len(non_folded) >= 2:
            contributions: dict[str, PotContribution] = {}
            for pid in self._active_order:
                total = hand.hand_contributions.get(pid, 0)
                contributions[pid] = PotContribution(
                    player=pid,
                    amount=total,
                    folded=self._players[pid].folded,
                )
            hands: dict[str, list[Card]] = {
                pid: self._players[pid].hole_cards + hand.community
                for pid in non_folded
            }
            pots = build_pots(contributions)
            seat_order = self._active_order
            results = settle_pots(pots, hands, contributions, seat_order=seat_order)

            uncalled_returns = {
                pid: amount for pot in pots for pid, amount in pot.returned.items()
            }
            returned = sum(uncalled_returns.values())
            for pid, amount in uncalled_returns.items():
                self._players[pid].stack += amount

            for r in results:
                for winner, payout in r.payouts.items():
                    payouts[winner] = payouts.get(winner, 0) + payout
                settlement.append(
                    cast(
                        JsonObject,
                        {
                            "amount": r.amount,
                            "winners": list(r.winners),
                            "payouts": dict(r.payouts),
                        },
                    )
                )

            total_payouts = sum(payouts.values())
            assert total_payouts + returned == hand.pot, (
                f"Conservation violation: payouts {total_payouts} + returned "
                f"{returned} != pot {hand.pot}"
            )

            for pid, payout in payouts.items():
                self._players[pid].stack += payout

            winners = [w for w in seat_order if w in payouts]
            for winner in winners:
                self._wins[winner] += 1
            summary = f"Showdown: {', '.join(winners)} win(s) {hand.pot}"
        else:
            summary = "Hand ended with no active players"

        seat_deltas = {
            pid: self._players[pid].stack
            - hand.start_stacks.get(pid, self._players[pid].stack)
            for pid in self._active_order
            if pid in hand.start_stacks
        }
        seat_delta_text = ", ".join(
            f"{pid} {delta:+d}" for pid, delta in seat_deltas.items()
        )
        compact_summary = (
            f"Hand {hand.hand_index} finished by {finish_reason}. "
            f"Seat deltas: {seat_delta_text}"
        )
        public_summary = cast(
            JsonObject,
            {
                "hand_index": hand.hand_index,
                "finish_reason": finish_reason,
                "summary": compact_summary,
                "seat_deltas": seat_deltas,
                "payouts": dict(payouts),
            },
        )
        self._public_hand_summaries.append(public_summary)

        for pid in self._active_order:
            self._pending_memory_summaries.append(
                MatchMemorySummary(
                    actor_id=pid,
                    text=compact_summary,
                    hand=hand.hand_index,
                    turn=self._in_hand_turn,
                )
            )

        self._hand_events.append(
            cast(
                JsonObject,
                {
                    "event_type": "hand_end",
                    "hand_index": hand.hand_index,
                    "seed": hand.seed,
                    "dealer": hand.dealer_id,
                    "board": [str(c) for c in hand.community],
                    "small_blind": hand.small_blind_id,
                    "big_blind": hand.big_blind_id,
                    "summary": summary,
                    "finish_reason": finish_reason,
                    "payouts": dict(payouts),
                    "uncalled_returns": dict(uncalled_returns),
                    "settlement": settlement,
                },
            )
        )

        for pid in self._active_order:
            if self._players[pid].stack <= 0:
                self._players[pid].active = False
                self._players[pid].stack = 0

        self._hands_played += 1
        self._hand_index += 1

        active = self._active_players()
        if len(active) < 2 or self._hands_played >= self._hands_per_match:
            self._match_over = True
            self._finish_reason = "elimination" if len(active) < 2 else "hand_limit"
        else:
            self._dealer_id = self._next_active_seat(hand.dealer_id, active)
            self._start_new_hand()

    @property
    def is_terminal(self) -> bool:
        return self._match_over

    def drain_hand_events(self) -> list[PluginEvent]:
        events = list(self._hand_events)
        self._hand_events.clear()
        return [
            PluginEvent(
                event_type=cast(str, event["event_type"]),
                payload={
                    key: value for key, value in event.items() if key != "event_type"
                },
            )
            for event in events
        ]

    def drain_match_memory_summaries(self) -> list[MatchMemorySummary]:
        summaries = list(self._pending_memory_summaries)
        self._pending_memory_summaries.clear()
        return summaries

    def get_result(self) -> GameResult:
        stacks: dict[str, int] = {
            pid: self._players[pid].stack for pid in self._player_ids
        }
        deltas: dict[str, int] = {
            pid: stacks[pid] - self._initial_stack for pid in self._player_ids
        }
        completed = self._match_over and self._failure_reason is None
        outcome: dict[str, object] = {
            "stacks": stacks,
            "deltas": deltas,
            "hands_played": self._hands_played,
            "wins": dict(self._wins),
            "finish_reason": self._finish_reason
            or ("in_progress" if not self._match_over else "failed"),
            "completed": completed,
        }
        if completed:
            highest_stack = max(stacks.values())
            winners = [pid for pid in self._player_ids if stacks[pid] == highest_stack]
            if len(winners) == 1:
                outcome["winner"] = winners[0]
            else:
                outcome["winners"] = winners

        return GameResult(
            completed=completed,
            outcome=cast(JsonObject, outcome),
            metrics=cast(
                GameMetrics,
                {
                    "hands_played": self._hands_played,
                    "hand_summaries": self._public_hand_summaries,
                    "failure_count": len(self._failure_calls),
                    "recovery_count": self._recovery_count,
                    "wins": dict(self._wins),
                    "final_stacks": stacks,
                    "deltas": deltas,
                    "finish_reason": outcome["finish_reason"],
                    "completed": completed,
                },
            ),
        )

    def handle_failed_turn(self, actor_id: str, reason: str) -> Transition | None:
        self._failure_calls.append(reason)
        if self._invalid_turn_policy == "forced_fold":
            player = self._players.get(actor_id)
            if player and not player.folded:
                player.folded = True
                self._recovery_count += 1
                self._in_hand_turn += 1
                if self._hand is not None:
                    self._hand.action_history.append(
                        f"{actor_id} forced fold (failed turn)"
                    )
                self._advance_actor()
                self._check_street_end()
                return Transition(
                    summary=f"{actor_id} forced fold (failed turn)",
                    metrics={"reason": reason},
                )
        elif self._invalid_turn_policy == "fail_match":
            self._match_over = True
            self._failure_reason = reason
            self._finish_reason = "failed_turn"
        return None


class PokerGame:
    """First-party poker game plugin factory."""

    PLUGIN_VERSION = "0.2.0"

    @property
    def name(self) -> str:
        return "poker"

    @property
    def version(self) -> str:
        return self.PLUGIN_VERSION

    @property
    def player_ids(self) -> list[str]:
        return ["player-1", "player-2"]

    @property
    def requires_exact_agent_ids(self) -> bool:
        return True

    def player_ids_from_config(self, game_config: JsonObject) -> list[str]:
        return list(PokerConfig.model_validate(game_config).players)

    def validate_config(self, game_config: JsonObject) -> None:
        PokerConfig.model_validate(game_config)

    def system_prompt(self, actor_id: str) -> str:
        return (
            f"You are {actor_id} in a No-Limit Texas Hold'em poker game. "
            "On your turn, call the 'poker_action' tool with your chosen action "
            "(fold, check, call, bet, raise, or all_in) and an optional amount."
        )

    def create_session(
        self, *, seed: int, game_config: JsonObject | None = None
    ) -> PokerSession:
        config = game_config or {}
        validated = PokerConfig.model_validate(config)
        return PokerSession(
            players=validated.players,
            initial_stack=validated.initial_stack,
            small_blind=validated.small_blind,
            big_blind=validated.big_blind,
            hands_per_match=validated.hands_per_match,
            seed=seed,
            invalid_turn_policy=validated.invalid_turn_policy,
            memory_max_entries=validated.memory_max_entries,
            memory_max_chars=validated.memory_max_chars,
        )
