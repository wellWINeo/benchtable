from __future__ import annotations

import pytest

from benchtable.contracts import (
    GameResult,
    JsonObject,
    MatchMemorySummary,
    Observation,
    ToolSpec,
    Transition,
)
from benchtable.errors import InvalidActionError
from benchtable.games.protocol import (
    ConversationScopedSession,
    GameSession,
    MatchMemorySummarySession,
)


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

    def test_optional_conversation_scope_capability_is_discoverable(self) -> None:
        class ScopedSession:
            current_actor_id = "a"

            @property
            def conversation_scope_id(self) -> str:
                return "hand-1"

        assert isinstance(ScopedSession(), ConversationScopedSession)

    def test_optional_memory_summary_capability_is_discoverable(self) -> None:
        class SummarySession:
            current_actor_id = "a"

            def drain_match_memory_summaries(self) -> list[MatchMemorySummary]:
                return []

        assert isinstance(SummarySession(), MatchMemorySummarySession)

    def test_legacy_session_without_optional_caps_remains_valid(self) -> None:
        class LegacySession:
            current_actor_id = "a"
            is_terminal = False

            def get_observation(self, actor_id: str) -> Observation:
                return Observation(actor_id=actor_id, text="Legacy.")

            def get_tools(self, actor_id: str) -> list[ToolSpec]:
                return []

            def apply_action(
                self, actor_id: str, tool_name: str, arguments: JsonObject
            ) -> Transition:
                return Transition(summary="legacy")

            def get_result(self) -> GameResult:
                return GameResult(completed=False)

        assert isinstance(LegacySession(), GameSession)

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

    def test_player_ids_from_config_returns_static_when_not_overridden(self) -> None:
        from tests.fixtures.tiny_game import TinyGame

        game = TinyGame()
        assert game.player_ids_from_config({}) == ["a", "b"]

    def test_validate_config_returns_none_for_valid_config(self) -> None:
        from tests.fixtures.tiny_game import TinyGame

        game = TinyGame()
        assert game.validate_config({}) is None
        assert game.validate_config({"max_actions": 5}) is None

    def test_validate_config_raises_for_invalid_config(self) -> None:
        from tests.fixtures.tiny_game import TinyGame

        game = TinyGame()
        with pytest.raises(ValueError):
            game.validate_config({"max_actions": "not_an_int"})
