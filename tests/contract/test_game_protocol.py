from __future__ import annotations

import pytest

from benchtable.contracts import (
    GameResult,
    JsonObject,
    Observation,
    ToolSpec,
    Transition,
)
from benchtable.errors import InvalidActionError
from benchtable.games.protocol import GameSession


class TestGameProtocol:
    def test_failed_turn_handler_is_optional(self) -> None:
        from tests.fixtures.tiny_game import TinyGame

        class SessionWithoutFailedTurn:
            current_actor_id = "a"
            is_terminal = False

            def get_observation(self, actor_id: str) -> Observation:
                return (
                    TinyGame()
                    .create_session(seed=1, game_config={})
                    .get_observation(actor_id)
                )

            def get_tools(self, actor_id: str) -> list[ToolSpec]:
                return (
                    TinyGame()
                    .create_session(seed=1, game_config={})
                    .get_tools(actor_id)
                )

            def apply_action(
                self, actor_id: str, tool_name: str, arguments: JsonObject
            ) -> Transition:
                return Transition(summary="applied")

            def get_result(self) -> GameResult:
                return GameResult(completed=False)

        assert isinstance(SessionWithoutFailedTurn(), GameSession)

    def test_fake_session_satisfies_protocol(self) -> None:
        from tests.fixtures.tiny_game import TinyGame

        game = TinyGame()
        session = game.create_session(seed=42, game_config={})

        assert session.current_actor_id in ("a", "b")

    def test_returns_observation_for_current_actor(self) -> None:
        from tests.fixtures.tiny_game import TinyGame

        game = TinyGame()
        session = game.create_session(seed=42, game_config={})
        actor = session.current_actor_id
        obs = session.get_observation(actor)

        assert isinstance(obs, Observation)
        assert obs.actor_id == actor
        assert obs.text

    def test_returns_tools(self) -> None:
        from tests.fixtures.tiny_game import TinyGame

        game = TinyGame()
        session = game.create_session(seed=42, game_config={})
        actor = session.current_actor_id
        tools = session.get_tools(actor)

        assert len(tools) >= 1
        assert all(isinstance(t, ToolSpec) for t in tools)

    def test_valid_action_produces_transition(self) -> None:
        from tests.fixtures.tiny_game import TinyGame

        game = TinyGame()
        session = game.create_session(seed=42, game_config={})
        actor = session.current_actor_id
        tools = session.get_tools(actor)

        result = session.apply_action(actor, tools[0].name, {})
        assert isinstance(result, Transition)

    def test_invalid_action_raises_domain_error(self) -> None:
        from tests.fixtures.tiny_game import TinyGame

        game = TinyGame()
        session = game.create_session(seed=42, game_config={})
        actor = session.current_actor_id

        with pytest.raises(InvalidActionError):
            session.apply_action(actor, "nonexistent_tool", {})

    def test_session_terminates(self) -> None:
        from tests.fixtures.tiny_game import TinyGame

        game = TinyGame()
        session = game.create_session(seed=42, game_config={})

        for _ in range(10):
            if session.is_terminal:
                break
            actor = session.current_actor_id
            tools = session.get_tools(actor)
            session.apply_action(actor, tools[0].name, {})

        assert session.is_terminal
        result = session.get_result()
        assert isinstance(result, GameResult)
        assert result.completed

    def test_observations_never_leak_other_actor_data(self) -> None:
        from tests.fixtures.tiny_game import TinyGame

        game = TinyGame()
        session = game.create_session(seed=42, game_config={})

        obs_a = session.get_observation("a")
        obs_b = session.get_observation("b")

        assert "private_b" not in obs_a.text.lower()
        assert "private_a" not in obs_b.text.lower()

    def test_transition_does_not_expose_raw_state(self) -> None:
        from tests.fixtures.tiny_game import TinyGame

        game = TinyGame()
        session = game.create_session(seed=42, game_config={})
        actor = session.current_actor_id
        tools = session.get_tools(actor)

        transition = session.apply_action(actor, tools[0].name, {})
        assert isinstance(transition.summary, str)
        assert len(transition.summary) > 0
