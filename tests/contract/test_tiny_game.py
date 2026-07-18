"""Contract tests for the tiny test game fixture."""

from __future__ import annotations

import pytest
from tests.fixtures.tiny_game import TinyGame

from benchtable.contracts import Observation, ToolSpec
from benchtable.errors import InvalidActionError
from benchtable.games.protocol import GamePlugin, GameSession


class TestTinyGameContract:
    @pytest.fixture
    def game(self) -> TinyGame:
        return TinyGame()

    @pytest.fixture
    def session(self, game: TinyGame) -> GameSession:
        return game.create_session(seed=42, game_config={})

    def test_satisfies_game_plugin_protocol(self, game: TinyGame) -> None:
        assert isinstance(game, GamePlugin)

    def test_has_two_actors(self, game: TinyGame) -> None:
        assert game.player_ids == ["a", "b"]

    def test_produces_system_prompt_per_actor(self, game: TinyGame) -> None:
        prompt_a = game.system_prompt("a")
        prompt_b = game.system_prompt("b")
        assert "a" in prompt_a.lower()
        assert "b" in prompt_b.lower()
        assert prompt_a != prompt_b

    def test_session_satisfies_game_session_protocol(
        self, session: GameSession
    ) -> None:
        assert isinstance(session, GameSession)

    def test_current_actor_is_a_or_b(self, session: GameSession) -> None:
        assert session.current_actor_id in ("a", "b")

    def test_observation_for_current_actor(self, session: GameSession) -> None:
        actor = session.current_actor_id
        obs = session.get_observation(actor)
        assert isinstance(obs, Observation)
        assert obs.actor_id == actor
        assert obs.text

    def test_observation_for_other_actor(self, session: GameSession) -> None:
        other = "b" if session.current_actor_id == "a" else "a"
        obs = session.get_observation(other)
        assert isinstance(obs, Observation)
        assert obs.actor_id == other

    def test_observations_are_actor_isolated(self, session: GameSession) -> None:
        obs_a = session.get_observation("a")
        obs_b = session.get_observation("b")
        assert "secret_b" not in obs_a.text.lower()
        assert "secret_a" not in obs_b.text.lower()

    def test_tools_are_spec_list(self, session: GameSession) -> None:
        actor = session.current_actor_id
        tools = session.get_tools(actor)
        assert len(tools) >= 1
        assert all(isinstance(t, ToolSpec) for t in tools)

    def test_single_legal_tool(self, session: GameSession) -> None:
        actor = session.current_actor_id
        tools = session.get_tools(actor)
        names = [t.name for t in tools]
        assert "act" in names

    def test_valid_action_produces_transition(self, session: GameSession) -> None:
        actor = session.current_actor_id
        transition = session.apply_action(actor, "act", {})
        assert transition.summary
        assert "act" in transition.summary.lower() or actor in transition.summary

    def test_invalid_tool_name_raises(self, session: GameSession) -> None:
        actor = session.current_actor_id
        with pytest.raises(InvalidActionError):
            session.apply_action(actor, "nonexistent", {})

    def test_handle_failed_turn_returns_none(self, session: GameSession) -> None:
        actor = session.current_actor_id
        result = session.handle_failed_turn(actor, "test failure")
        assert result is None

    def test_session_terminates_after_max_actions(self) -> None:
        game = TinyGame()
        session = game.create_session(seed=42, game_config={})

        for _ in range(10):
            if session.is_terminal:
                break
            actor = session.current_actor_id
            session.apply_action(actor, "act", {})

        assert session.is_terminal

    def test_terminal_result_is_game_result(self, session: GameSession) -> None:
        from benchtable.contracts import GameResult

        # Play to completion
        for _ in range(10):
            if session.is_terminal:
                break
            actor = session.current_actor_id
            session.apply_action(actor, "act", {})

        result = session.get_result()
        assert isinstance(result, GameResult)
        assert result.completed
        assert "total_actions" in result.outcome
