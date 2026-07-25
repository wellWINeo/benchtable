"""Contract tests for the poker game plugin."""

from __future__ import annotations

import pytest

from benchtable.contracts import GameResult, Observation, ToolSpec, Transition
from benchtable.errors import InvalidActionError
from benchtable.games.poker.session import PokerGame, PokerSession
from benchtable.games.protocol import GamePlugin, GameSession


class TestPokerGameContract:
    @pytest.fixture
    def game(self) -> PokerGame:
        return PokerGame()

    @pytest.fixture
    def session(self, game: PokerGame) -> PokerSession:
        return game.create_session(
            seed=42,
            game_config={"players": ["a", "b"], "hands_per_match": 2},
        )

    def test_satisfies_game_plugin_protocol(self, game: PokerGame) -> None:
        assert isinstance(game, GamePlugin)

    def test_has_name_and_version(self, game: PokerGame) -> None:
        assert game.name == "poker"
        assert isinstance(game.version, str)

    def test_player_ids_from_config(self, game: PokerGame) -> None:
        ids = game.player_ids_from_config({"players": ["x", "y"]})
        assert ids == ["x", "y"]

    def test_validate_config_accepts_valid(self, game: PokerGame) -> None:
        game.validate_config(
            {
                "players": ["a", "b"],
                "initial_stack": 1000,
                "small_blind": 5,
                "big_blind": 10,
            }
        )

    def test_session_satisfies_game_session_protocol(
        self, session: PokerSession
    ) -> None:
        assert isinstance(session, GameSession)

    def test_current_actor_is_a_or_b(self, session: PokerSession) -> None:
        assert session.current_actor_id in ("a", "b")

    def test_observation_for_current_actor(self, session: PokerSession) -> None:
        actor = session.current_actor_id
        obs = session.get_observation(actor)
        assert isinstance(obs, Observation)
        assert obs.actor_id == actor
        assert obs.text

    def test_observation_for_other_actor(self, session: PokerSession) -> None:
        other = "b" if session.current_actor_id == "a" else "a"
        obs = session.get_observation(other)
        assert isinstance(obs, Observation)
        assert obs.actor_id == other

    def test_tools_are_spec_list(self, session: PokerSession) -> None:
        actor = session.current_actor_id
        tools = session.get_tools(actor)
        assert len(tools) >= 1
        assert all(isinstance(t, ToolSpec) for t in tools)

    def test_valid_action_produces_transition(self, session: PokerSession) -> None:
        actor = session.current_actor_id
        legal = session._legal_actions(actor)
        action = legal[0]
        t = session.apply_action(actor, "poker_action", {"action": action})
        assert isinstance(t, Transition)
        assert t.summary

    def test_invalid_tool_name_raises(self, session: PokerSession) -> None:
        actor = session.current_actor_id
        with pytest.raises(InvalidActionError):
            session.apply_action(actor, "nonexistent", {})

    def test_session_terminates(self, session: PokerSession) -> None:
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
        assert session.is_terminal

    def test_terminal_result_is_game_result(self, session: PokerSession) -> None:
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
        assert isinstance(result, GameResult)
        assert result.completed

    def test_handle_failed_turn_returns_transition_or_none(
        self, session: PokerSession
    ) -> None:
        actor = session.current_actor_id
        result = session.handle_failed_turn(actor, "test failure")
        assert result is None or isinstance(result, Transition)

    def test_observations_isolate_each_player_hole_cards(
        self, session: PokerSession
    ) -> None:
        obs_a = session.get_observation("a")
        obs_b = session.get_observation("b")

        a_cards = " ".join(str(card) for card in session._players["a"].hole_cards)
        b_cards = " ".join(str(card) for card in session._players["b"].hole_cards)
        assert obs_a.actor_id == "a"
        assert obs_b.actor_id == "b"
        assert a_cards in obs_a.text
        assert b_cards not in obs_a.text
        assert b_cards in obs_b.text
        assert a_cards not in obs_b.text

    def test_transition_does_not_expose_raw_state(self, session: PokerSession) -> None:
        actor = session.current_actor_id
        legal = session._legal_actions(actor)
        if legal:
            t = session.apply_action(actor, "poker_action", {"action": legal[0]})
            assert isinstance(t.summary, str)
            assert len(t.summary) > 0
