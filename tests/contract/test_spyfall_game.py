"""Contract tests for the Spyfall game plugin."""

from __future__ import annotations

import pytest

from benchtable.contracts import GameResult, Observation, ToolSpec
from benchtable.errors import InvalidActionError
from benchtable.games.protocol import (
    ConversationScopedSession,
    GamePlugin,
    GameSession,
    JudgedActionSession,
    JudgeDependencySession,
    PluginEventSession,
)
from benchtable.games.registry import GameRegistry
from benchtable.games.spyfall.session import SpyfallGame, SpyfallSession


def _valid_config() -> dict[str, object]:
    return {
        "players": ["p1", "p2", "p3"],
        "rounds_per_match": 2,
        "question_rounds": 1,
        "accusation_vote_threshold": 0.75,
        "judge_id": "spyfall-leak-judge",
        "judge_leak_threshold": 0.85,
        "locations": [{"name": "airport"}],
    }


class TestSpyfallGameContract:
    @pytest.fixture
    def game(self) -> SpyfallGame:
        return SpyfallGame()

    @pytest.fixture
    def session(self, game: SpyfallGame) -> SpyfallSession:
        return game.create_session(seed=42, game_config=_valid_config())

    def test_registry_discovers_spyfall(self) -> None:
        registry = GameRegistry()
        registry.discover()
        plugin = registry.load("spyfall")
        assert plugin.name == "spyfall"

    def test_satisfies_game_plugin_protocol(self, game: SpyfallGame) -> None:
        assert isinstance(game, GamePlugin)

    def test_has_name_and_version(self, game: SpyfallGame) -> None:
        assert game.name == "spyfall"
        assert isinstance(game.version, str)

    def test_requires_exact_agent_ids(self, game: SpyfallGame) -> None:
        assert game.requires_exact_agent_ids is True

    def test_player_ids_from_config(self, game: SpyfallGame) -> None:
        assert game.player_ids_from_config(_valid_config()) == ["p1", "p2", "p3"]

    def test_validate_config_accepts_valid(self, game: SpyfallGame) -> None:
        game.validate_config(_valid_config())

    def test_validate_config_rejects_too_few_players(self, game: SpyfallGame) -> None:
        config = _valid_config()
        config["players"] = ["p1", "p2"]
        with pytest.raises(Exception):
            game.validate_config(config)

    def test_system_prompt_marks_dialogue_untrusted(self, game: SpyfallGame) -> None:
        prompt = game.system_prompt("p1")
        assert "untrusted" in prompt

    def test_min_max_turns_matches_turn_budget(self, game: SpyfallGame) -> None:
        assert game.min_max_turns(_valid_config()) == 2 * 3 * 3 * 1

    def test_session_satisfies_optional_session_protocols(
        self, session: SpyfallSession
    ) -> None:
        assert isinstance(session, GameSession)
        assert isinstance(session, ConversationScopedSession)
        assert isinstance(session, PluginEventSession)
        assert isinstance(session, JudgedActionSession)
        assert isinstance(session, JudgeDependencySession)

    def test_observation_for_current_actor(self, session: SpyfallSession) -> None:
        actor = session.current_actor_id
        obs = session.get_observation(actor)
        assert isinstance(obs, Observation)
        assert obs.actor_id == actor
        assert obs.text

    def test_tools_are_spec_list(self, session: SpyfallSession) -> None:
        tools = session.get_tools(session.current_actor_id)
        assert tools
        assert all(isinstance(tool, ToolSpec) for tool in tools)

    def test_required_judge_ids_come_from_config(self, session: SpyfallSession) -> None:
        assert session.required_judge_ids() == ["spyfall-leak-judge"]

    def test_judgment_request_rejects_illegal_action(
        self, session: SpyfallSession
    ) -> None:
        with pytest.raises(InvalidActionError):
            session.judgment_request(
                session.current_actor_id, "spyfall_answer", {"text": "hi"}
            )

    def test_invalid_tool_name_raises(self, session: SpyfallSession) -> None:
        actor = session.current_actor_id
        with pytest.raises(InvalidActionError):
            session.apply_judged_action(actor, "nonexistent", {}, None)

    def test_conversation_scope_is_round_scoped(self, session: SpyfallSession) -> None:
        assert session.conversation_scope_id == "round-0"

    def test_completed_match_returns_game_result(self, session: SpyfallSession) -> None:
        players = ("p1", "p2", "p3")
        for _ in range(2):
            for _ in range(3):
                questioner = session.current_actor_id
                target = next(p for p in players if p != questioner)
                session.apply_judged_action(
                    questioner,
                    "spyfall_question",
                    {"target_id": target, "text": "Is it crowded?"},
                    None,
                )
                answerer = session.current_actor_id
                session.apply_judged_action(
                    answerer, "spyfall_answer", {"text": "Sometimes."}, None
                )
        result = session.get_result()
        assert isinstance(result, GameResult)
        assert result.completed
        assert len(result.outcome["rounds"]) == 2
