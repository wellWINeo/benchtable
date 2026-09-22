"""Contract tests for the bunker game plugin."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from benchtable.contracts import GameResult, Observation, ToolSpec, Transition
from benchtable.errors import InvalidActionError
from benchtable.games.bunker.session import BunkerGame, BunkerSession
from benchtable.games.protocol import GamePlugin, GameSession, PluginEventSession

CONFIG = {
    "players": ["player-1", "player-2", "player-3", "player-4"],
    "scenario": "sealed shelter after a solar storm",
    "shelter_capacity": 2,
}


class TestBunkerGameContract:
    @pytest.fixture
    def game(self) -> BunkerGame:
        return BunkerGame()

    @pytest.fixture
    def session(self, game: BunkerGame) -> BunkerSession:
        return game.create_session(seed=42, game_config=dict(CONFIG))

    def test_satisfies_game_plugin_protocol(self, game: BunkerGame) -> None:
        assert isinstance(game, GamePlugin)

    def test_has_name_and_version(self, game: BunkerGame) -> None:
        assert game.name == "bunker"
        assert isinstance(game.version, str)

    def test_registry_discovers_bunker(self) -> None:
        from benchtable.games.registry import GameRegistry

        registry = GameRegistry()
        registry.discover()
        assert "bunker" in [plugin.name for plugin in registry.list()]

    def test_player_ids_from_config(self, game: BunkerGame) -> None:
        assert game.player_ids_from_config(dict(CONFIG)) == list(CONFIG["players"])

    def test_validate_config_accepts_valid(self, game: BunkerGame) -> None:
        game.validate_config(dict(CONFIG))

    def test_validate_config_rejects_invalid(self, game: BunkerGame) -> None:
        with pytest.raises(ValueError):
            game.validate_config({**CONFIG, "shelter_capacity": 4})
        with pytest.raises(ValueError):
            game.validate_config({**CONFIG, "unexpected": True})

    def test_requires_exact_agent_ids(self, game: BunkerGame) -> None:
        assert game.requires_exact_agent_ids is True

    def test_min_max_turns_closed_form_bound(self, game: BunkerGame) -> None:
        assert game.min_max_turns(dict(CONFIG)) == 14
        eight_players = {
            "players": [f"player-{index}" for index in range(1, 9)],
            "scenario": "sealed shelter after a solar storm",
            "shelter_capacity": 1,
        }
        assert game.min_max_turns(eight_players) == 70

    def test_min_max_turns_validates_game_config(self, game: BunkerGame) -> None:
        invalid = dict(CONFIG)
        invalid["shelter_capacity"] = len(CONFIG["players"])
        with pytest.raises(ValidationError):
            game.min_max_turns(invalid)

    def test_system_prompt_marks_discussion_untrusted(self, game: BunkerGame) -> None:
        prompt = game.system_prompt("player-1")
        assert "untrusted" in prompt
        assert "bunker_speak" in prompt
        assert "bunker_vote_eliminate" in prompt

    def test_session_satisfies_session_protocols(self, session: BunkerSession) -> None:
        assert isinstance(session, GameSession)
        assert isinstance(session, PluginEventSession)

    def test_session_has_no_judged_action_capability(
        self, session: BunkerSession
    ) -> None:
        assert getattr(session, "judgment_request", None) is None
        assert getattr(session, "apply_judged_action", None) is None

    def test_observation_for_current_actor(self, session: BunkerSession) -> None:
        actor = session.current_actor_id
        obs = session.get_observation(actor)
        assert isinstance(obs, Observation)
        assert obs.actor_id == actor
        assert obs.text
        assert "sealed shelter" in obs.text

    def test_tools_expose_only_phase_legal_tool(self, session: BunkerSession) -> None:
        actor = session.current_actor_id
        tools = session.get_tools(actor)
        assert [isinstance(tool, ToolSpec) for tool in tools]
        assert [tool.name for tool in tools] == ["bunker_speak"]

    def test_invalid_tool_name_raises(self, session: BunkerSession) -> None:
        actor = session.current_actor_id
        with pytest.raises(InvalidActionError):
            session.apply_action(actor, "nonexistent", {})

    def test_play_through_terminates_with_completed_result(
        self, session: BunkerSession
    ) -> None:
        while not session.is_terminal:
            actor = session.current_actor_id
            if session.get_tools(actor)[0].name == "bunker_speak":
                transition = session.apply_action(
                    actor, "bunker_speak", {"text": f"{actor} addresses the group"}
                )
            else:
                survivors = session.get_observation(actor).metadata["survivors"]
                target = next(pid for pid in survivors if pid != actor)
                transition = session.apply_action(
                    actor,
                    "bunker_vote_eliminate",
                    {"target_id": target},
                )
            assert isinstance(transition, Transition)

        result = session.get_result()
        assert isinstance(result, GameResult)
        assert result.completed is True
        assert result.outcome["completion_reason"] == "capacity_reached"
