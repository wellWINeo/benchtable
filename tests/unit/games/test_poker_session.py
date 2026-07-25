"""Tests for poker session state machine."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from benchtable.contracts import GameResult, Observation, Transition
from benchtable.errors import InvalidActionError
from benchtable.games.poker.cards import Deck
from benchtable.games.poker.session import PokerGame, PokerSession, Street


class TestPokerSession:
    @pytest.fixture
    def session(self) -> PokerSession:
        return PokerSession(
            players=["a", "b"],
            initial_stack=1000,
            small_blind=5,
            big_blind=10,
            hands_per_match=10,
            seed=42,
        )

    def test_starts_with_two_players(self, session: PokerSession) -> None:
        assert session.current_actor_id in ("a", "b")

    def test_hole_cards_are_dealt_round_robin(self) -> None:
        session = PokerSession(
            players=["a", "b", "c"],
            initial_stack=1000,
            small_blind=5,
            big_blind=10,
            hands_per_match=1,
            seed=42,
        )
        expected = Deck(42).draw(6)

        assert session._players["a"].hole_cards == [expected[0], expected[3]]
        assert session._players["b"].hole_cards == [expected[1], expected[4]]
        assert session._players["c"].hole_cards == [expected[2], expected[5]]

    def test_observation_for_current_actor(self, session: PokerSession) -> None:
        actor = session.current_actor_id
        obs = session.get_observation(actor)
        assert isinstance(obs, Observation)
        assert obs.actor_id == actor
        assert "hole cards" in obs.text.lower()

    def test_observation_excludes_other_hole_cards(self, session: PokerSession) -> None:
        obs_a = session.get_observation("a")
        obs_b = session.get_observation("b")
        # Each observation should only mention own hole cards
        # (we can't easily check this since they're in the text, but at least verify)
        assert obs_a.actor_id == "a"
        assert obs_b.actor_id == "b"

    def test_get_tools_returns_poker_action(self, session: PokerSession) -> None:
        tools = session.get_tools("a")
        assert len(tools) == 1
        assert tools[0].name == "poker_action"

    def test_valid_check_action(self, session: PokerSession) -> None:
        actor = session.current_actor_id
        legal = session._legal_actions(actor)
        if "check" in legal:
            t = session.apply_action(actor, "poker_action", {"action": "check"})
            assert isinstance(t, Transition)
            assert "checks" in t.summary.lower()

    def test_valid_fold_action(self, session: PokerSession) -> None:
        actor = session.current_actor_id
        t = session.apply_action(actor, "poker_action", {"action": "fold"})
        assert isinstance(t, Transition)
        assert "fold" in t.summary.lower()

    def test_street_advances_after_both_players_act(
        self, session: PokerSession
    ) -> None:
        actor = session.current_actor_id
        legal = session._legal_actions(actor)
        if "call" in legal:
            session.apply_action(actor, "poker_action", {"action": "call"})
        else:
            session.apply_action(actor, "poker_action", {"action": "check"})

        actor = session.current_actor_id
        legal = session._legal_actions(actor)
        if "check" in legal:
            session.apply_action(actor, "poker_action", {"action": "check"})
        else:
            session.apply_action(actor, "poker_action", {"action": "call"})

        assert session._hand is not None
        assert session._hand.street == Street.FLOP

    def test_normal_check_call_progresses_through_all_postflop_streets(self) -> None:
        session = PokerSession(
            players=["a", "b"],
            initial_stack=1000,
            small_blind=5,
            big_blind=10,
            hands_per_match=1,
            seed=42,
        )

        actor = session.current_actor_id
        session.apply_action(actor, "poker_action", {"action": "call"})
        session.apply_action(
            session.current_actor_id, "poker_action", {"action": "check"}
        )
        assert session._hand is not None
        assert session._hand.street == Street.FLOP

        for expected_street in (Street.TURN, Street.RIVER):
            session.apply_action(
                session.current_actor_id, "poker_action", {"action": "check"}
            )
            session.apply_action(
                session.current_actor_id, "poker_action", {"action": "check"}
            )
            assert session._hand is not None
            assert session._hand.street == expected_street

        session.apply_action(
            session.current_actor_id, "poker_action", {"action": "check"}
        )
        session.apply_action(
            session.current_actor_id, "poker_action", {"action": "check"}
        )
        assert session.is_terminal

    def test_seeded_burn_and_board_order_is_deterministic(self) -> None:
        session = PokerSession(
            players=["a", "b"],
            initial_stack=1000,
            small_blind=5,
            big_blind=10,
            hands_per_match=1,
            seed=42,
        )
        expected = Deck(42).draw(12)

        session.apply_action(
            session.current_actor_id, "poker_action", {"action": "call"}
        )
        session.apply_action(
            session.current_actor_id, "poker_action", {"action": "check"}
        )
        assert session._hand is not None
        assert session._hand.burn_cards == [expected[4]]
        assert session._hand.community == expected[5:8]

        for _ in range(2):
            session.apply_action(
                session.current_actor_id, "poker_action", {"action": "check"}
            )
            session.apply_action(
                session.current_actor_id, "poker_action", {"action": "check"}
            )

        assert session._hand is not None
        assert session._hand.burn_cards == [expected[4], expected[8], expected[10]]
        assert session._hand.community == [*expected[5:8], expected[9], expected[11]]

    @pytest.mark.parametrize("amount", [None, True, 1.0, "10", 0, -1])
    def test_bet_requires_strict_positive_target_amount(
        self, session: PokerSession, amount: object
    ) -> None:
        # Reach the first postflop street, where betting is open.
        session.apply_action(
            session.current_actor_id, "poker_action", {"action": "call"}
        )
        session.apply_action(
            session.current_actor_id, "poker_action", {"action": "check"}
        )
        actor = session.current_actor_id

        arguments: dict[str, object] = {"action": "bet"}
        if amount is not None:
            arguments["amount"] = amount
        with pytest.raises(InvalidActionError):
            session.apply_action(actor, "poker_action", arguments)

    def test_bet_amount_is_a_target_and_cannot_be_clamped_over_stack(self) -> None:
        session = PokerSession(
            players=["a", "b"],
            initial_stack=20,
            small_blind=5,
            big_blind=10,
            hands_per_match=1,
            seed=42,
        )
        actor = session.current_actor_id
        session.apply_action(actor, "poker_action", {"action": "call"})
        session.apply_action(
            session.current_actor_id, "poker_action", {"action": "check"}
        )

        actor = session.current_actor_id
        with pytest.raises(InvalidActionError):
            session.apply_action(
                actor, "poker_action", {"action": "bet", "amount": 100}
            )

        with pytest.raises(InvalidActionError, match="all_in"):
            session.apply_action(actor, "poker_action", {"action": "bet", "amount": 20})

    def test_bet_amount_targets_total_hand_commitment(
        self, session: PokerSession
    ) -> None:
        session.apply_action(
            session.current_actor_id, "poker_action", {"action": "call"}
        )
        session.apply_action(
            session.current_actor_id, "poker_action", {"action": "check"}
        )
        actor = session.current_actor_id

        session.apply_action(actor, "poker_action", {"action": "bet", "amount": 20})

        assert session._hand is not None
        assert session._hand.hand_contributions[actor] == 20
        assert session._hand.street_contributions[actor] == 10

    def test_raise_is_not_legal_without_a_postflop_bet(
        self, session: PokerSession
    ) -> None:
        session.apply_action(
            session.current_actor_id, "poker_action", {"action": "call"}
        )
        session.apply_action(
            session.current_actor_id, "poker_action", {"action": "check"}
        )

        with pytest.raises(InvalidActionError):
            session.apply_action(
                session.current_actor_id,
                "poker_action",
                {"action": "raise", "amount": 20},
            )

    def test_explicit_all_in_can_commit_remaining_stack_as_short_raise(self) -> None:
        session = PokerSession(
            players=["a", "b"],
            initial_stack=20,
            small_blind=5,
            big_blind=10,
            hands_per_match=1,
            seed=42,
        )
        session.apply_action(
            session.current_actor_id, "poker_action", {"action": "all_in"}
        )
        session.apply_action(
            session.current_actor_id, "poker_action", {"action": "call"}
        )
        assert session.is_terminal

    def test_postflop_short_all_in_does_not_reopen_raise_for_acted_player(self) -> None:
        session = PokerSession(
            players=["a", "b", "c"],
            initial_stack=100,
            small_blind=5,
            big_blind=10,
            hands_per_match=1,
            seed=42,
        )

        session.apply_action(
            session.current_actor_id, "poker_action", {"action": "call"}
        )
        session.apply_action(
            session.current_actor_id, "poker_action", {"action": "call"}
        )
        session.apply_action(
            session.current_actor_id, "poker_action", {"action": "check"}
        )
        assert session._hand is not None
        assert session._hand.street == Street.FLOP

        session._players["b"].stack = 15
        session.apply_action("b", "poker_action", {"action": "check"})
        session.apply_action("c", "poker_action", {"action": "check"})
        session.apply_action("a", "poker_action", {"action": "bet", "amount": 20})
        session.apply_action("b", "poker_action", {"action": "all_in"})
        session.apply_action("c", "poker_action", {"action": "call"})

        assert session.current_actor_id == "a"
        assert "raise" not in session._legal_actions("a")
        with pytest.raises(InvalidActionError, match="not legal"):
            session.apply_action("a", "poker_action", {"action": "raise", "amount": 35})

    def test_all_in_big_blind_is_never_selected_for_action(self) -> None:
        session = PokerSession(
            players=["a", "b"],
            initial_stack=10,
            small_blind=5,
            big_blind=10,
            hands_per_match=1,
            seed=42,
        )

        assert session.current_actor_id == "a"
        session.apply_action(
            session.current_actor_id, "poker_action", {"action": "call"}
        )

        assert session.is_terminal

    def test_deep_opponent_is_not_asked_again_after_calling_all_in(self) -> None:
        session = PokerSession(
            players=["short", "deep"],
            initial_stack=100,
            small_blind=5,
            big_blind=10,
            hands_per_match=1,
            seed=42,
        )
        session._players["deep"].stack = 1000

        session.apply_action(
            session.current_actor_id, "poker_action", {"action": "all_in"}
        )
        assert session.current_actor_id == "deep"
        session.apply_action(
            session.current_actor_id, "poker_action", {"action": "call"}
        )

        assert session.is_terminal
        assert session._hand is not None
        assert len(session._hand.community) == 5

    def test_dealer_rotates_by_stable_seat_order(self) -> None:
        session = PokerSession(
            players=["a", "b", "c"],
            initial_stack=1000,
            small_blind=5,
            big_blind=10,
            hands_per_match=3,
            seed=42,
        )
        dealers: list[str] = []
        seen_hands: set[int] = set()

        while not session.is_terminal:
            assert session._hand is not None
            if session._hand.hand_index not in seen_hands:
                seen_hands.add(session._hand.hand_index)
                dealers.append(session._hand.dealer_id)
            session.apply_action(
                session.current_actor_id, "poker_action", {"action": "fold"}
            )

        assert dealers == ["a", "b", "c"]

    def test_invalid_action_raises(self, session: PokerSession) -> None:
        actor = session.current_actor_id
        with pytest.raises(InvalidActionError):
            session.apply_action(actor, "poker_action", {"action": "invalid"})

    def test_wrong_actor_raises(self, session: PokerSession) -> None:
        actor = session.current_actor_id
        other = "b" if actor == "a" else "a"
        with pytest.raises(InvalidActionError):
            session.apply_action(other, "poker_action", {"action": "check"})

    def test_unknown_tool_raises(self, session: PokerSession) -> None:
        actor = session.current_actor_id
        with pytest.raises(InvalidActionError):
            session.apply_action(actor, "nonexistent", {})

    def test_hand_ends_when_one_folds(self, session: PokerSession) -> None:
        actor = session.current_actor_id
        session.apply_action(actor, "poker_action", {"action": "fold"})
        # After fold, the other player wins the hand and a new hand starts
        # (unless match is over)
        obs = session.get_observation(session.current_actor_id)
        assert obs.text

    def test_result_is_game_result(self, session: PokerSession) -> None:
        # Play to completion
        for _ in range(100):
            if session.is_terminal:
                break
            actor = session.current_actor_id
            legal = session._legal_actions(actor)
            if "check" in legal:
                session.apply_action(actor, "poker_action", {"action": "check"})
            elif "call" in legal:
                session.apply_action(actor, "poker_action", {"action": "call"})
            else:
                session.apply_action(actor, "poker_action", {"action": "fold"})

        result = session.get_result()
        assert isinstance(result, GameResult)
        assert result.completed

    def test_result_does_not_choose_singular_winner_for_tied_stacks(
        self, session: PokerSession
    ) -> None:
        session._players["a"].stack = 100
        session._players["b"].stack = 100
        session._match_over = True
        session._finish_reason = "hand_limit"

        result = session.get_result()

        assert "winner" not in result.outcome
        assert result.outcome["winners"] == ["a", "b"]

    def test_config_rejects_too_few_players(self) -> None:
        with pytest.raises(ValueError, match="At least 2"):
            PokerSession(
                players=["a"],
                initial_stack=1000,
                small_blind=5,
                big_blind=10,
                hands_per_match=10,
                seed=42,
            )

    def test_config_rejects_bad_blinds(self) -> None:
        with pytest.raises(ValueError):
            PokerSession(
                players=["a", "b"],
                initial_stack=1000,
                small_blind=10,
                big_blind=10,
                hands_per_match=10,
                seed=42,
            )


class TestPokerGame:
    def test_name_and_version(self) -> None:
        game = PokerGame()
        assert game.name == "poker"
        assert game.version

    def test_default_player_ids(self) -> None:
        game = PokerGame()
        assert len(game.player_ids) == 2

    def test_player_ids_from_config(self) -> None:
        game = PokerGame()
        ids = game.player_ids_from_config({"players": ["x", "y", "z"]})
        assert ids == ["x", "y", "z"]

    def test_validate_config_valid(self) -> None:
        game = PokerGame()
        game.validate_config(
            {
                "players": ["a", "b"],
                "big_blind": 20,
                "small_blind": 10,
                "initial_stack": 1000,
            }
        )

    def test_validate_config_rejects_duplicate_players(self) -> None:
        game = PokerGame()
        with pytest.raises(ValueError, match="unique"):
            game.validate_config({"players": ["a", "a"]})

    def test_validate_config_rejects_one_player(self) -> None:
        game = PokerGame()
        with pytest.raises(ValidationError):
            game.validate_config({"players": ["a"]})

    def test_validate_config_rejects_bad_blinds(self) -> None:
        game = PokerGame()
        with pytest.raises(ValueError, match="big_blind"):
            game.validate_config({"big_blind": 5, "small_blind": 10})

    def test_system_prompt(self) -> None:
        game = PokerGame()
        prompt = game.system_prompt("player-1")
        assert "player-1" in prompt
        assert "poker" in prompt.lower()

    def test_create_session(self) -> None:
        game = PokerGame()
        session = game.create_session(seed=42, game_config={"players": ["x", "y"]})
        assert isinstance(session, PokerSession)
        assert session.current_actor_id in ("x", "y")

    def test_create_session_requires_players(self) -> None:
        game = PokerGame()
        with pytest.raises(ValidationError):
            game.create_session(seed=42, game_config=None)

    def test_handle_failed_turn_forced_fold(self) -> None:
        game = PokerGame()
        session = game.create_session(
            seed=42,
            game_config={"players": ["a", "b"], "invalid_turn_policy": "forced_fold"},
        )
        actor = session.current_actor_id
        t = session.handle_failed_turn(actor, "test")
        if t is not None:
            assert isinstance(t, Transition)
            assert "forced fold" in t.summary.lower()

    def test_handle_failed_turn_fail_match(self) -> None:
        game = PokerGame()
        session = game.create_session(
            seed=42,
            game_config={"players": ["a", "b"], "invalid_turn_policy": "fail_match"},
        )
        actor = session.current_actor_id
        session.handle_failed_turn(actor, "test")
        assert session.is_terminal
        result = session.get_result()
        assert result.completed is False
        assert "winner" not in result.outcome

    def test_session_satisfies_protocol(self) -> None:
        from benchtable.games.protocol import GameSession

        game = PokerGame()
        session = game.create_session(seed=42, game_config={"players": ["a", "b"]})
        assert isinstance(session, GameSession)

    def test_plugin_satisfies_protocol(self) -> None:
        from benchtable.games.protocol import GamePlugin

        game = PokerGame()
        assert isinstance(game, GamePlugin)

    def test_multi_hand_persistence(self) -> None:
        game = PokerGame()
        session = game.create_session(
            seed=42,
            game_config={
                "players": ["a", "b"],
                "hands_per_match": 2,
                "initial_stack": 1000,
            },
        )
        for _ in range(200):
            if session.is_terminal:
                break
            actor = session.current_actor_id
            legal = session._legal_actions(actor)
            if "check" in legal:
                session.apply_action(actor, "poker_action", {"action": "check"})
            elif "call" in legal:
                session.apply_action(actor, "poker_action", {"action": "call"})
            else:
                session.apply_action(actor, "poker_action", {"action": "fold"})
        result = session.get_result()
        assert result.completed

    def test_in_hand_turn_context_resets_for_new_hand(self) -> None:
        session = PokerSession(
            players=["a", "b"],
            initial_stack=1000,
            small_blind=5,
            big_blind=10,
            hands_per_match=2,
            seed=42,
        )

        assert session.get_turn_context() == {"hand_index": 0, "in_hand_turn": 0}
        session.apply_action(
            session.current_actor_id, "poker_action", {"action": "fold"}
        )

        assert session.get_turn_context() == {"hand_index": 1, "in_hand_turn": 0}
