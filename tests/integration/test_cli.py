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
from benchtable.config import AgentConfig
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

    def test_unexpected_discovery_errors_are_redacted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from benchtable.games.registry import GameRegistry

        def fail_discovery(registry: GameRegistry) -> None:
            raise RuntimeError("api_key: discovery-secret")

        monkeypatch.setattr(GameRegistry, "discover", fail_discovery)
        cli.set_registry(None)

        result = runner.invoke(app, ["list-games"])

        assert result.exit_code != 0
        assert "Plugin discovery error" in result.output
        assert "[REDACTED]" in result.output
        assert "discovery-secret" not in result.output

    def test_unexpected_configuration_errors_are_redacted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def fail_load_config(path: Path) -> Any:
            raise RuntimeError("api_key: config-load-secret")

        monkeypatch.setattr(cli, "load_config", fail_load_config)

        result = runner.invoke(
            app,
            [
                "run",
                "--config",
                str(tmp_path / "config.toml"),
                "--output",
                str(tmp_path / "out"),
            ],
        )

        assert result.exit_code != 0
        assert "Configuration error" in result.output
        assert "[REDACTED]" in result.output
        assert "config-load-secret" not in result.output

    def test_listing_exceptions_are_redacted(self) -> None:
        from benchtable.games.registry import GameRegistry

        class BrokenListRegistry(GameRegistry):
            def list(self) -> list[Any]:
                raise RuntimeError("api_key: list-secret")

        cli.set_registry(BrokenListRegistry())

        result = runner.invoke(app, ["list-games"])

        assert result.exit_code != 0
        assert "Plugin listing error" in result.output
        assert "[REDACTED]" in result.output
        assert "list-secret" not in result.output

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
    def test_default_agent_factory_dispatches_gigachat_adapter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from benchtable.agents import gigachat

        captured_kwargs: dict[str, Any] = {}

        class _GigaChatAgent:
            def __init__(self, **kwargs: Any) -> None:
                captured_kwargs.update(kwargs)

        monkeypatch.setattr(gigachat, "GigaChatAgent", _GigaChatAgent)
        config = AgentConfig(
            id="a",
            provider="gigachat",
            model="GigaChat-3-Ultra",
            credential_env="GIGA_ENV",
            timeout=30.0,
            max_completion_tokens=128,
        )

        agent = cli._default_agent_factory(config)

        assert isinstance(agent, _GigaChatAgent)
        assert captured_kwargs == {
            "model": "GigaChat-3-Ultra",
            "credential_env": "GIGA_ENV",
            "scope": "GIGACHAT_API_PERS",
            "timeout": 30.0,
            "max_completion_tokens": 128,
        }

    def test_default_agent_factory_dispatches_yandex_adapter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from benchtable.agents import yandex_ai_studio

        captured_kwargs: dict[str, Any] = {}

        class _YandexAIStudioAgent:
            def __init__(self, **kwargs: Any) -> None:
                captured_kwargs.update(kwargs)

        monkeypatch.setattr(
            yandex_ai_studio, "YandexAIStudioAgent", _YandexAIStudioAgent
        )
        config = AgentConfig(
            id="a",
            provider="yandex_ai_studio",
            model="aliceai-llm",
            credential_env="YC_ENV",
            folder_id="folder-123",
            credential_kind="api_key",
            timeout=30.0,
            max_completion_tokens=128,
        )

        agent = cli._default_agent_factory(config)

        assert isinstance(agent, _YandexAIStudioAgent)
        assert captured_kwargs == {
            "model": "aliceai-llm",
            "folder_id": "folder-123",
            "credential_kind": "api_key",
            "credential_env": "YC_ENV",
            "timeout": 30.0,
            "max_completion_tokens": 128,
        }

    def test_single_agent_passes_id_for_exact_mapping_game(
        self, tmp_path: Path
    ) -> None:
        from benchtable.games.registry import GameRegistry

        class SingleActorExactMappingGame(TinyGame):
            @property
            def requires_exact_agent_ids(self) -> bool:
                return True

        created_agents: dict[str, FakeAgent] = {}

        def fake_agent_factory(agent_config: Any) -> FakeAgent:
            agent = FakeAgent(tool_name="act")
            created_agents[agent_config.id] = agent
            return agent

        cli.set_agent_factory(fake_agent_factory)
        registry = GameRegistry()
        registry.register(SingleActorExactMappingGame())
        cli.set_registry(registry)

        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            textwrap.dedent("""\
            [run]
            game = "tiny"
            matches = 1

            [run.game_config]
            players = ["a"]
            max_actions = 1

            [[agents]]
            id = "a"
            model = "fake"
        """)
        )

        result = runner.invoke(
            app, ["run", "--config", str(cfg_path), "--output", str(tmp_path / "out")]
        )

        assert result.exit_code == 0
        assert set(created_agents) == {"a"}
        assert {request.actor_id for request in created_agents["a"].requests} == {"a"}

    def test_exact_mapping_capability_errors_are_rendered_as_cli_errors(
        self, tmp_path: Path
    ) -> None:
        from benchtable.games.registry import GameRegistry

        class BrokenCapabilityGame(TinyGame):
            @property
            def requires_exact_agent_ids(self) -> bool:
                raise RuntimeError("Authorization: top-secret")

        registry = GameRegistry()
        registry.register(BrokenCapabilityGame())
        cli.set_registry(registry)

        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            textwrap.dedent("""\
            [run]
            game = "tiny"
            matches = 1

            [run.game_config]
            players = ["a", "b"]
            max_actions = 1

            [[agents]]
            id = "a"
            model = "fake"

            [[agents]]
            id = "b"
            model = "fake"
        """)
        )

        result = runner.invoke(
            app, ["run", "--config", str(cfg_path), "--output", str(tmp_path / "out")]
        )

        assert result.exit_code != 0
        assert "Plugin error" in result.output
        assert "[REDACTED]" in result.output
        assert "top-secret" not in result.output

    def test_plugin_validation_exceptions_are_redacted(self, tmp_path: Path) -> None:
        from benchtable.games.registry import GameRegistry

        class BrokenValidationGame(TinyGame):
            def validate_config(self, game_config: dict[str, Any]) -> None:
                raise RuntimeError("Authorization: config-secret")

        registry = GameRegistry()
        registry.register(BrokenValidationGame())
        cli.set_registry(registry)

        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            textwrap.dedent("""\
            [run]
            game = "tiny"
            matches = 1

            [[agents]]
            id = "a"
            model = "fake"
        """)
        )

        result = runner.invoke(
            app, ["run", "--config", str(cfg_path), "--output", str(tmp_path / "out")]
        )

        assert result.exit_code != 0
        assert "Game configuration error" in result.output
        assert "[REDACTED]" in result.output
        assert "config-secret" not in result.output

    def test_agent_factory_exceptions_are_redacted(self, tmp_path: Path) -> None:
        from benchtable.games.registry import GameRegistry

        def broken_agent_factory(agent_config: Any) -> FakeAgent:
            raise RuntimeError("api_key: agent-secret")

        cli.set_agent_factory(broken_agent_factory)
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
            model = "fake"
        """)
        )

        result = runner.invoke(
            app, ["run", "--config", str(cfg_path), "--output", str(tmp_path / "out")]
        )

        assert result.exit_code != 0
        assert "Agent error" in result.output
        assert "[REDACTED]" in result.output
        assert "agent-secret" not in result.output

    def test_unexpected_run_exceptions_are_redacted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from benchtable import engine
        from benchtable.games.registry import GameRegistry

        class BrokenEngine:
            def __init__(self, **kwargs: Any) -> None:
                pass

            async def run(self) -> Any:
                raise RuntimeError("api_key: run-secret")

        cli.set_agent_factory(lambda agent_config: FakeAgent(tool_name="act"))
        registry = GameRegistry()
        registry.register(TinyGame())
        cli.set_registry(registry)
        monkeypatch.setattr(engine, "RunEngine", BrokenEngine)

        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            textwrap.dedent("""\
            [run]
            game = "tiny"
            matches = 1

            [[agents]]
            id = "a"
            model = "fake"
        """)
        )

        result = runner.invoke(
            app, ["run", "--config", str(cfg_path), "--output", str(tmp_path / "out")]
        )

        assert result.exit_code != 0
        assert "Run error" in result.output
        assert "[REDACTED]" in result.output
        assert "run-secret" not in result.output

    def test_unexpected_plugin_load_errors_are_redacted(self, tmp_path: Path) -> None:
        from benchtable.games.registry import GameRegistry

        class BrokenLoadRegistry(GameRegistry):
            def load(self, name: str) -> Any:
                raise RuntimeError("api_key: load-secret")

        cli.set_registry(BrokenLoadRegistry())

        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            textwrap.dedent("""\
            [run]
            game = "tiny"
            matches = 1

            [[agents]]
            id = "a"
            model = "fake"
        """)
        )

        result = runner.invoke(
            app, ["run", "--config", str(cfg_path), "--output", str(tmp_path / "out")]
        )

        assert result.exit_code != 0
        assert "Plugin error" in result.output
        assert "[REDACTED]" in result.output
        assert "load-secret" not in result.output

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
                "provider": "openai_compatible",
                "base_url": None,
                "api_key_env": "TEST_API_KEY",
                "credential_env": None,
                "scope": None,
                "folder_id": None,
                "credential_kind": None,
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
                "provider": "openai_compatible",
                "base_url": None,
                "api_key_env": "FAKE_A_KEY",
                "credential_env": None,
                "scope": None,
                "folder_id": None,
                "credential_kind": None,
                "timeout": 1.5,
                "max_completion_tokens": 64,
            },
            {
                "id": "b",
                "role": "second",
                "model": "fake-b",
                "provider": "openai_compatible",
                "base_url": "https://fake.example/v1",
                "api_key_env": "FAKE_B_KEY",
                "credential_env": None,
                "scope": None,
                "folder_id": None,
                "credential_kind": None,
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

    def test_multi_agent_configuration_uses_configured_actor_ids(
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
        registry.register(TinyGame())
        cli.set_registry(registry)

        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            textwrap.dedent("""\
            [run]
            game = "tiny"
            matches = 1

            [run.game_config]
            players = ["north", "south"]
            max_actions = 2

            [[agents]]
            id = "north"
            model = "fake-north"

            [[agents]]
            id = "south"
            model = "fake-south"
        """)
        )

        result = runner.invoke(
            app,
            ["run", "--config", str(cfg_path), "--output", str(tmp_path / "out")],
        )

        assert result.exit_code == 0
        assert {request.actor_id for request in created_agents["north"].requests} == {
            "north"
        }
        assert {request.actor_id for request in created_agents["south"].requests} == {
            "south"
        }

    def test_actor_resolution_errors_are_reported_before_agent_construction(
        self, tmp_path: Path
    ) -> None:
        from benchtable.games.registry import GameRegistry

        class _BrokenActorConfigGame(TinyGame):
            def player_ids_from_config(self, game_config):
                raise ValueError("players are malformed")

        constructed: list[str] = []

        def fake_agent_factory(agent_config: Any) -> FakeAgent:
            constructed.append(agent_config.id)
            return FakeAgent(tool_name="act")

        cli.set_agent_factory(fake_agent_factory)
        registry = GameRegistry()
        registry.register(_BrokenActorConfigGame())
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
            app,
            ["run", "--config", str(cfg_path), "--output", str(tmp_path / "out")],
        )

        assert result.exit_code != 0
        assert "game configuration error" in result.output.lower()
        assert "players are malformed" in result.output
        assert constructed == []

    def test_single_agent_still_validates_actor_configuration_before_construction(
        self, tmp_path: Path
    ) -> None:
        from benchtable.games.registry import GameRegistry

        class _BrokenActorConfigGame(TinyGame):
            def player_ids_from_config(self, game_config):
                raise ValueError("players are malformed")

        constructed: list[str] = []
        cli.set_agent_factory(
            lambda agent_config: (
                constructed.append(agent_config.id) or FakeAgent(tool_name="act")
            )
        )
        registry = GameRegistry()
        registry.register(_BrokenActorConfigGame())
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
        """)
        )

        result = runner.invoke(
            app,
            ["run", "--config", str(cfg_path), "--output", str(tmp_path / "out")],
        )

        assert result.exit_code != 0
        assert "game configuration error" in result.output.lower()
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

    def test_run_validates_game_config_before_agent_construction(
        self, tmp_path: Path
    ) -> None:
        from benchtable.games.registry import GameRegistry

        class _ConfigValidatingGame(TinyGame):
            def validate_config(self, game_config):
                if game_config.get("fail_validation"):
                    raise ValueError("intentional validation failure")

        constructed: list[str] = []

        def fake_agent_factory(agent_config: Any) -> FakeAgent:
            constructed.append(agent_config.id)
            return FakeAgent(tool_name="act")

        cli.set_agent_factory(fake_agent_factory)
        registry = GameRegistry()
        registry.register(_ConfigValidatingGame())
        cli.set_registry(registry)

        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            textwrap.dedent("""\
            [run]
            game = "tiny"
            matches = 1

            [run.game_config]
            fail_validation = true

            [[agents]]
            id = "a"
            model = "fake-a"
        """)
        )

        result = runner.invoke(
            app,
            ["run", "--config", str(cfg_path), "--output", str(tmp_path / "out")],
        )

        assert result.exit_code != 0
        assert (
            "validation failure" in result.output.lower()
            or "error" in result.output.lower()
        )
        assert constructed == []

    def test_run_passes_game_config_to_validate(self, tmp_path: Path) -> None:
        from benchtable.games.registry import GameRegistry

        received_configs: list[dict] = []

        class _RecordingGame(TinyGame):
            def validate_config(self, game_config):
                received_configs.append(game_config)

        cli.set_agent_factory(lambda agent_config: FakeAgent(tool_name="act"))
        registry = GameRegistry()
        registry.register(_RecordingGame())
        cli.set_registry(registry)

        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            textwrap.dedent("""\
            [run]
            game = "tiny"
            matches = 1

            [run.game_config]
            key = "value"

            [[agents]]
            id = "a"
            model = "fake-a"
        """)
        )

        output_dir = tmp_path / "out"
        result = runner.invoke(
            app,
            ["run", "--config", str(cfg_path), "--output", str(output_dir)],
        )

        assert result.exit_code == 0
        assert received_configs == [{"key": "value"}]

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

    def test_run_forwards_max_memory_operations_per_turn_to_engine(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from benchtable import engine
        from benchtable.games.registry import GameRegistry

        captured_kwargs: dict[str, Any] = {}

        class _CaptureEngine:
            def __init__(self, **kwargs: Any) -> None:
                captured_kwargs.update(kwargs)

            async def run(self) -> Any:
                return type("Result", (), {"success": True})()

        cli.set_agent_factory(lambda agent_config: FakeAgent(tool_name="act"))
        monkeypatch.setattr(engine, "RunEngine", _CaptureEngine)

        registry = GameRegistry()
        registry.register(TinyGame())
        cli.set_registry(registry)

        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            textwrap.dedent("""\
            [run]
            game = "tiny"
            matches = 1
            max_memory_operations_per_turn = 10

            [[agents]]
            id = "a"
            model = "fake-a"
        """)
        )

        result = runner.invoke(
            app,
            ["run", "--config", str(cfg_path), "--output", str(tmp_path / "out")],
        )

        assert result.exit_code == 0
        assert captured_kwargs["max_memory_operations_per_turn"] == 10

    def test_run_records_max_memory_operations_per_turn_in_trace(
        self, tmp_path: Path
    ) -> None:
        from benchtable.games.registry import GameRegistry

        cli.set_agent_factory(lambda agent_config: FakeAgent(tool_name="act"))
        registry = GameRegistry()
        registry.register(TinyGame(max_actions=1))
        cli.set_registry(registry)

        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            textwrap.dedent("""\
            [run]
            game = "tiny"
            matches = 1
            max_memory_operations_per_turn = 10

            [[agents]]
            id = "a"
            model = "fake-a"
        """)
        )
        output_dir = tmp_path / "out"

        result = runner.invoke(
            app,
            ["run", "--config", str(cfg_path), "--output", str(output_dir)],
        )

        assert result.exit_code == 0
        events = [
            json.loads(line)
            for line in (output_dir / "events.jsonl").read_text().splitlines()
            if line
        ]
        assert events[0]["payload"]["max_memory_operations_per_turn"] == 10

    def test_poker_config_validates_before_agent_construction(
        self, tmp_path: Path
    ) -> None:
        from benchtable.games.poker.session import PokerGame
        from benchtable.games.registry import GameRegistry

        constructed: list[str] = []

        def fake_agent_factory(agent_config: Any) -> FakeAgent:
            constructed.append(agent_config.id)
            return FakeAgent(tool_name="act")

        cli.set_agent_factory(fake_agent_factory)
        registry = GameRegistry()
        registry.register(PokerGame())
        cli.set_registry(registry)

        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            textwrap.dedent("""\
            [run]
            game = "poker"
            matches = 1
            seed = 42
            max_turns = 10

            [run.game_config]
            players = ["a", "b"]
            hands_per_match = 1
            initial_stack = 1000
            small_blind = 5
            big_blind = 10

            [[agents]]
            id = "a"
            model = "fake-a"

            [[agents]]
            id = "b"
            model = "fake-b"
        """)
        )

        result = runner.invoke(
            app,
            ["run", "--config", str(cfg_path), "--output", str(tmp_path / "out")],
        )

        assert result.exit_code == 0
        assert set(constructed) == {"a", "b"}

    def test_poker_list_games_includes_poker(self) -> None:
        from benchtable.games.poker.session import PokerGame
        from benchtable.games.registry import GameRegistry

        registry = GameRegistry()
        registry.register(PokerGame())
        cli.set_registry(registry)

        result = runner.invoke(app, ["list-games"])
        assert result.exit_code == 0
        assert "poker" in result.output.lower()

    def test_poker_rejects_missing_player_ids(self, tmp_path: Path) -> None:
        from benchtable.games.poker.session import PokerGame
        from benchtable.games.registry import GameRegistry

        constructed: list[str] = []

        def fake_agent_factory(agent_config: Any) -> FakeAgent:
            constructed.append(agent_config.id)
            return FakeAgent(tool_name="act")

        cli.set_agent_factory(fake_agent_factory)
        registry = GameRegistry()
        registry.register(PokerGame())
        cli.set_registry(registry)

        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(
            textwrap.dedent("""\
            [run]
            game = "poker"
            matches = 1

            [run.game_config]
            players = ["a", "b"]

            [[agents]]
            id = "a"
            model = "fake-a"

            [[agents]]
            id = "c"
            model = "fake-c"
        """)
        )

        result = runner.invoke(
            app,
            ["run", "--config", str(cfg_path), "--output", str(tmp_path / "out")],
        )

        assert result.exit_code != 0
        assert "missing" in result.output.lower() or "unknown" in result.output.lower()
        assert constructed == []
