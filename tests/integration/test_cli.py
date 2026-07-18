"""Integration tests for the CLI."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from tests.fixtures.fake_agent import FakeAgent
from tests.fixtures.tiny_game import TinyGame
from typer.testing import CliRunner

from benchtable import cli
from benchtable.cli import app
from benchtable.errors import PluginError, ProviderError

runner = CliRunner()


@pytest.fixture(autouse=True)
def _reset_cli_overrides() -> Any:
    """Reset injectable CLI dependencies after each test."""
    yield
    cli.set_registry(None)
    cli.set_agent_factory(None)


class TestListGames:
    def test_lists_discovered_plugins(self) -> None:
        from benchtable.games.registry import GameRegistry

        registry = GameRegistry()
        registry.register(TinyGame())
        cli.set_registry(registry)
        result = runner.invoke(app, ["list-games"])
        assert result.exit_code == 0

    def test_lists_no_production_games_by_default(self) -> None:
        result = runner.invoke(app, ["list-games"])
        assert result.exit_code == 0

    def test_registry_is_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from benchtable.games.registry import GameRegistry

        discoveries: list[GameRegistry] = []

        def discover(registry: GameRegistry) -> None:
            discoveries.append(registry)

        monkeypatch.setattr(GameRegistry, "discover", discover)
        cli.set_registry(None)

        first = cli._get_registry()
        second = cli._get_registry()

        assert second is first
        assert discoveries == [first]

    def test_discovery_errors_are_rendered_as_cli_errors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from benchtable.games.registry import GameRegistry

        def fail_discovery(registry: GameRegistry) -> None:
            raise PluginError("broken entry point")

        monkeypatch.setattr(GameRegistry, "discover", fail_discovery)
        cli.set_registry(None)

        result = runner.invoke(app, ["list-games"])

        assert result.exit_code != 0
        assert "Plugin discovery error" in result.output
        assert "broken entry point" in result.output

    def test_malformed_plugin_metadata_is_not_an_attribute_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        malformed_plugin = SimpleNamespace(
            name="broken",
            player_ids=["p1"],
            system_prompt=lambda actor_id: "stub",
            create_session=lambda **kwargs: object(),
        )
        entry_points = [
            SimpleNamespace(name="broken-entry-point", load=lambda: malformed_plugin)
        ]
        monkeypatch.setattr(
            "importlib.metadata.entry_points", lambda *, group: entry_points
        )
        cli.set_registry(None)

        result = runner.invoke(app, ["list-games"])

        assert result.exit_code != 0
        assert not isinstance(result.exception, AttributeError)
        assert "Plugin discovery error" in result.output


class TestRunCommand:
    def test_run_forwards_completion_token_limit_to_agent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from benchtable import engine
        from benchtable.games.registry import GameRegistry

        captured_agent_configs: list[dict[str, Any]] = []

        def fake_agent_factory(agent_config: Any) -> FakeAgent:
            captured_agent_configs.append(agent_config.model_dump(mode="json"))
            return FakeAgent(tool_name="act")

        class _FakeEngine:
            def __init__(self, **kwargs: Any) -> None:
                pass

            async def run(self) -> Any:
                return type("Result", (), {"success": True})()

        cli.set_agent_factory(fake_agent_factory)
        monkeypatch.setattr(engine, "RunEngine", _FakeEngine)

        registry = GameRegistry()
        registry.register(TinyGame())
        cli.set_registry(registry)

        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            textwrap.dedent("""\
            [run]
            game = "tiny"
            matches = 1

            [[agents]]
            id = "a"
            role = "player"
            model = "gpt-4o"
            api_key_env = "TEST_API_KEY"
            max_completion_tokens = 128
        """)
        )

        result = runner.invoke(
            app, ["run", "--config", str(cfg_path), "--output", str(tmp_path / "out")]
        )

        assert result.exit_code == 0
        assert captured_agent_configs == [
            {
                "id": "a",
                "role": "player",
                "model": "gpt-4o",
                "base_url": None,
                "api_key_env": "TEST_API_KEY",
                "timeout": None,
                "max_completion_tokens": 128,
            }
        ]

    def test_run_uses_injected_agents_and_records_sanitized_configuration(
        self, tmp_path: Path
    ) -> None:
        from benchtable.games.registry import GameRegistry

        created_agents: dict[str, FakeAgent] = {}

        def fake_agent_factory(agent_config: Any) -> FakeAgent:
            agent = FakeAgent(tool_name="act")
            created_agents[agent_config.id] = agent
            return agent

        cli.set_agent_factory(fake_agent_factory)

        registry = GameRegistry()
        game = TinyGame()
        registry.register(game)
        cli.set_registry(registry)

        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            """\
            [run]
            game = "tiny"
            matches = 1
            seed = 42

            [run.game_config]
            max_actions = 2
            label = "offline"

            [[agents]]
            id = "a"
            role = "first"
            model = "fake-a"
            api_key_env = "FAKE_A_KEY"
            timeout = 1.5
            max_completion_tokens = 64

            [[agents]]
            id = "b"
            role = "second"
            model = "fake-b"
            base_url = "https://fake.example/v1"
            api_key_env = "FAKE_B_KEY"
            timeout = 2
            max_completion_tokens = 32
            """
        )

        output_dir = tmp_path / "out"
        result = runner.invoke(
            app,
            ["run", "--config", str(cfg_path), "--output", str(output_dir)],
        )

        assert result.exit_code == 0
        assert set(created_agents) == {"a", "b"}
        assert {request.actor_id for request in created_agents["a"].requests} == {"a"}
        assert {request.actor_id for request in created_agents["b"].requests} == {"b"}
        assert game.created_game_configs == [{"max_actions": 2, "label": "offline"}]

        events = [
            json.loads(line)
            for line in (output_dir / "events.jsonl").read_text().splitlines()
            if line
        ]
        run_config = events[0]["payload"]
        assert run_config["game_config"] == {"max_actions": 2, "label": "offline"}
        assert run_config["agents"] == [
            {
                "id": "a",
                "role": "first",
                "model": "fake-a",
                "base_url": None,
                "api_key_env": "FAKE_A_KEY",
                "timeout": 1.5,
                "max_completion_tokens": 64,
            },
            {
                "id": "b",
                "role": "second",
                "model": "fake-b",
                "base_url": "https://fake.example/v1",
                "api_key_env": "FAKE_B_KEY",
                "timeout": 2.0,
                "max_completion_tokens": 32,
            },
        ]
        assert "FAKE_A_KEY" in json.dumps(run_config)
        assert "FAKE_B_KEY" in json.dumps(run_config)
        assert "sk-" not in json.dumps(run_config)

    def test_failed_run_returns_nonzero_and_writes_failure_trace(
        self, tmp_path: Path
    ) -> None:
        from benchtable.games.registry import GameRegistry

        cli.set_agent_factory(
            lambda agent_config: FakeAgent(
                raise_on_respond=ProviderError(
                    "offline failure", provider="fake", model=agent_config.model
                )
            )
        )

        registry = GameRegistry()
        registry.register(TinyGame())
        cli.set_registry(registry)

        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            """\
            [run]
            game = "tiny"
            matches = 1
            max_provider_retries = 0

            [[agents]]
            id = "a"
            model = "fake"
            """
        )
        output_dir = tmp_path / "out"

        result = runner.invoke(
            app,
            ["run", "--config", str(cfg_path), "--output", str(output_dir)],
        )

        assert result.exit_code != 0
        events = [
            json.loads(line)
            for line in (output_dir / "events.jsonl").read_text().splitlines()
            if line
        ]
        assert events[-1]["event_type"] == "run_end"
        assert events[-1]["payload"]["status"] == "failed"

    def test_reusing_output_directory_gets_unique_run_ids(self, tmp_path: Path) -> None:
        from benchtable.games.registry import GameRegistry

        cli.set_agent_factory(lambda agent_config: FakeAgent(tool_name="act"))
        registry = GameRegistry()
        registry.register(TinyGame())
        cli.set_registry(registry)

        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            """\
            [run]
            game = "tiny"
            matches = 1
            seed = 42

            [run.game_config]
            max_actions = 1

            [[agents]]
            id = "a"
            model = "fake"
            """
        )
        output_dir = tmp_path / "out"
        args = ["run", "--config", str(cfg_path), "--output", str(output_dir)]

        assert runner.invoke(app, args).exit_code == 0
        assert runner.invoke(app, args).exit_code == 0

        events = [
            json.loads(line)
            for line in (output_dir / "events.jsonl").read_text().splitlines()
            if line
        ]
        run_configs = [event for event in events if event["event_type"] == "run_config"]
        match_starts = [
            event for event in events if event["event_type"] == "match_start"
        ]
        assert len({event["run_id"] for event in run_configs}) == 2
        assert len(match_starts) == 2
        assert match_starts[0]["payload"]["seed"] == match_starts[1]["payload"]["seed"]

    def test_rejects_unknown_game(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            textwrap.dedent("""\
            [run]
            game = "nonexistent"
            matches = 1

            [[agents]]
            id = "p1"
            role = "player"
            model = "gpt-4o"
        """)
        )
        result = runner.invoke(
            app, ["run", "--config", str(cfg_path), "--output", str(tmp_path / "out")]
        )
        assert result.exit_code != 0

    def test_multi_agent_configuration_rejects_unknown_ids_before_construction(
        self, tmp_path: Path
    ) -> None:
        from benchtable.games.registry import GameRegistry

        constructed: list[str] = []

        def fake_agent_factory(agent_config: Any) -> FakeAgent:
            constructed.append(agent_config.id)
            return FakeAgent(tool_name="act")

        cli.set_agent_factory(fake_agent_factory)
        registry = GameRegistry()
        registry.register(TinyGame())
        cli.set_registry(registry)

        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            textwrap.dedent("""\
            [run]
            game = "tiny"
            matches = 1

            [[agents]]
            id = "a"
            model = "fake-a"

            [[agents]]
            id = "b"
            model = "fake-b"

            [[agents]]
            id = "extra"
            model = "fake-extra"
        """)
        )

        result = runner.invoke(
            app, ["run", "--config", str(cfg_path), "--output", str(tmp_path / "out")]
        )

        assert result.exit_code != 0
        assert "unknown" in result.output.lower()
        assert constructed == []

    def test_multi_agent_configuration_rejects_missing_actor_ids_before_construction(
        self, tmp_path: Path
    ) -> None:
        from benchtable.games.registry import GameRegistry

        class ThreeActorGame(TinyGame):
            @property
            def player_ids(self) -> list[str]:
                return ["a", "b", "c"]

        constructed: list[str] = []

        def fake_agent_factory(agent_config: Any) -> FakeAgent:
            constructed.append(agent_config.id)
            return FakeAgent(tool_name="act")

        cli.set_agent_factory(fake_agent_factory)
        registry = GameRegistry()
        registry.register(ThreeActorGame())
        cli.set_registry(registry)

        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            textwrap.dedent("""\
            [run]
            game = "tiny"
            matches = 1

            [[agents]]
            id = "a"
            model = "fake-a"

            [[agents]]
            id = "b"
            model = "fake-b"
        """)
        )

        result = runner.invoke(
            app, ["run", "--config", str(cfg_path), "--output", str(tmp_path / "out")]
        )

        assert result.exit_code != 0
        assert "missing" in result.output.lower()
        assert constructed == []

    def test_run_discovery_errors_are_rendered_before_agent_construction(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from benchtable.games.registry import GameRegistry

        constructed: list[str] = []

        def fail_discovery(registry: GameRegistry) -> None:
            raise PluginError("broken entry point")

        def fake_agent_factory(agent_config: Any) -> FakeAgent:
            constructed.append(agent_config.id)
            return FakeAgent(tool_name="act")

        monkeypatch.setattr(GameRegistry, "discover", fail_discovery)
        cli.set_agent_factory(fake_agent_factory)
        cli.set_registry(None)

        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            textwrap.dedent("""\
            [run]
            game = "tiny"
            matches = 1

            [[agents]]
            id = "a"
            model = "fake-a"
        """)
        )

        result = runner.invoke(
            app, ["run", "--config", str(cfg_path), "--output", str(tmp_path / "out")]
        )

        assert result.exit_code != 0
        assert "Plugin discovery error" in result.output
        assert "broken entry point" in result.output
        assert constructed == []

    def test_run_with_tiny_game_writes_trace(self, tmp_path: Path) -> None:
        from benchtable.games.registry import GameRegistry

        cli.set_agent_factory(lambda agent_config: FakeAgent(tool_name="act"))

        registry = GameRegistry()
        registry.register(TinyGame())
        cli.set_registry(registry)

        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            textwrap.dedent("""\
            [run]
            game = "tiny"
            matches = 1
            seed = 42
            max_turns = 20

            [[agents]]
            id = "a"
            role = "player"
            model = "gpt-4o"
        """)
        )

        output_dir = tmp_path / "out"
        result = runner.invoke(
            app,
            ["run", "--config", str(cfg_path), "--output", str(output_dir)],
        )

        assert result.exit_code == 0
        # Should produce an events.jsonl
        trace_path = output_dir / "events.jsonl"
        assert trace_path.exists()

        raw = trace_path.read_text()
        events = [json.loads(line) for line in raw.splitlines() if line]
        assert len(events) > 0
        assert events[0]["event_type"] == "run_config"

    def test_cli_errors_have_nonzero_exit_code(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "bad.toml"
        cfg_path.write_text("not valid toml [[[")
        result = runner.invoke(
            app, ["run", "--config", str(cfg_path), "--output", str(tmp_path / "out")]
        )
        assert result.exit_code != 0
