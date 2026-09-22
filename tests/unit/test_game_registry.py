"""Tests for the game plugin registry."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchtable.errors import PluginError
from benchtable.games.registry import GameRegistry


class _StubPlugin:
    """Minimal plugin for registry tests."""

    def __init__(self, name: str = "stub", version: str = "1.0.0") -> None:
        self._name = name
        self._version = version

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    @property
    def player_ids(self) -> list[str]:
        return ["p1"]

    def player_ids_from_config(self, game_config: dict) -> list[str]:
        return self.player_ids

    def validate_config(self, game_config: dict) -> None:
        pass

    def system_prompt(self, actor_id: str) -> str:
        return "stub"

    def create_session(self, *, seed: int) -> object:
        return object()


class TestGameRegistry:
    def test_register_and_list_plugins(self) -> None:
        registry = GameRegistry()
        registry.register(_StubPlugin("alpha", "1.0"))
        registry.register(_StubPlugin("beta", "2.0"))

        plugins = registry.list()
        assert len(plugins) == 2
        names = [p.name for p in plugins]
        assert names == sorted(names)  # sorted by name

    def test_register_duplicate_name_raises(self) -> None:
        registry = GameRegistry()
        registry.register(_StubPlugin("alpha"))
        with pytest.raises(PluginError, match="already registered"):
            registry.register(_StubPlugin("alpha"))

    def test_register_rejects_plugin_with_missing_version(self) -> None:
        plugin = SimpleNamespace(
            name="broken",
            player_ids=["p1"],
            system_prompt=lambda actor_id: "stub",
            create_session=lambda **kwargs: object(),
        )

        with pytest.raises(PluginError, match="version"):
            GameRegistry().register(plugin)

    def test_register_rejects_plugin_with_missing_required_callable(self) -> None:
        plugin = SimpleNamespace(
            name="broken",
            version="1.0",
            player_ids=["p1"],
            create_session=lambda **kwargs: object(),
        )

        with pytest.raises(PluginError, match="system_prompt"):
            GameRegistry().register(plugin)

    def test_register_rejects_broken_optional_capability(self) -> None:
        class _BrokenOptionalPlugin(_StubPlugin):
            @property
            def player_ids_from_config(self):
                raise RuntimeError("capability unavailable")

        with pytest.raises(PluginError, match="optional attribute"):
            GameRegistry().register(_BrokenOptionalPlugin())

    def test_load_returns_cached_plugin(self) -> None:
        registry = GameRegistry()
        plugin = _StubPlugin("alpha")
        registry.register(plugin)

        loaded = registry.load("alpha")
        assert loaded is plugin

    def test_load_unknown_name_raises(self) -> None:
        registry = GameRegistry()
        with pytest.raises(PluginError, match="not found"):
            registry.load("nonexistent")

    def test_list_returns_stable_metadata(self) -> None:
        registry = GameRegistry()
        registry.register(_StubPlugin("b", "2.0"))
        registry.register(_StubPlugin("a", "1.0"))

        plugins = registry.list()
        assert [(p.name, p.version) for p in plugins] == [
            ("a", "1.0"),
            ("b", "2.0"),
        ]

    def test_discover_duplicate_entry_point_names_raise(self, monkeypatch) -> None:
        plugins = [
            SimpleNamespace(name="alpha", load=lambda: _StubPlugin("alpha")),
            SimpleNamespace(name="alpha", load=lambda: _StubPlugin("alpha")),
        ]

        def fake_entry_points(*, group: str):
            assert group == "benchtable.games"
            return plugins

        monkeypatch.setattr("importlib.metadata.entry_points", fake_entry_points)

        with pytest.raises(PluginError, match="duplicate"):
            GameRegistry().discover()

    def test_discover_keys_plugins_by_metadata_name(self, monkeypatch) -> None:
        plugins = [
            SimpleNamespace(
                name="entry-point-alias", load=lambda: _StubPlugin("canonical")
            )
        ]

        monkeypatch.setattr("importlib.metadata.entry_points", lambda *, group: plugins)

        registry = GameRegistry()
        registry.discover()

        assert [plugin.name for plugin in registry.list()] == ["canonical"]
        assert registry.load("canonical").name == "canonical"
        with pytest.raises(PluginError, match="not found"):
            registry.load("entry-point-alias")

    def test_discover_rejects_duplicate_plugin_metadata_names(
        self, monkeypatch
    ) -> None:
        plugins = [
            SimpleNamespace(name="alpha", load=lambda: _StubPlugin("canonical")),
            SimpleNamespace(name="beta", load=lambda: _StubPlugin("canonical")),
        ]

        monkeypatch.setattr("importlib.metadata.entry_points", lambda *, group: plugins)

        with pytest.raises(PluginError, match="metadata name"):
            GameRegistry().discover()

    def test_discover_rejects_invalid_plugin_metadata_name(self, monkeypatch) -> None:
        plugins = [SimpleNamespace(name="entry-point", load=lambda: _StubPlugin("   "))]

        monkeypatch.setattr("importlib.metadata.entry_points", lambda *, group: plugins)

        with pytest.raises(PluginError, match="name"):
            GameRegistry().discover()

    def test_validate_plugin_config_calls_plugin_validate_config(self) -> None:
        registry = GameRegistry()
        plugin = _StubPlugin("test", "1.0")
        registry.register(plugin)
        # Should not raise
        registry.validate_plugin_config("test", {})

    def test_validate_plugin_config_rejects_unknown_plugin(self) -> None:
        registry = GameRegistry()
        with pytest.raises(PluginError, match="not found"):
            registry.validate_plugin_config("nonexistent", {})

    def test_validate_plugin_config_propagates_plugin_validation_error(self) -> None:
        class _InvalidPlugin(_StubPlugin):
            def validate_config(self, game_config):
                raise ValueError("bad config")

        registry = GameRegistry()
        registry.register(_InvalidPlugin("bad", "1.0"))
        with pytest.raises(ValueError, match="bad config"):
            registry.validate_plugin_config("bad", {"key": "value"})

    def test_validate_plugin_config_passes_game_config(self) -> None:
        class _CheckingPlugin(_StubPlugin):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.received_config = None

            def validate_config(self, game_config):
                self.received_config = game_config

        registry = GameRegistry()
        plugin = _CheckingPlugin("checker", "1.0")
        registry.register(plugin)
        config = {"players": ["a", "b"]}
        registry.validate_plugin_config("checker", config)
        assert plugin.received_config == config

    def test_legacy_plugin_uses_static_actor_ids_and_noop_validation(self) -> None:
        plugin = SimpleNamespace(
            name="legacy",
            version="1.0",
            player_ids=["p1"],
            system_prompt=lambda actor_id: "stub",
            create_session=lambda **kwargs: object(),
        )
        registry = GameRegistry()
        registry.register(plugin)

        registry.validate_plugin_config("legacy", {})
        assert registry.resolve_player_ids("legacy", {}) == ["p1"]

    def test_configured_actor_plugin_does_not_require_static_actor_ids(self) -> None:
        plugin = SimpleNamespace(
            name="configured",
            version="1.0",
            player_ids_from_config=lambda config: ["a", "b"],
            system_prompt=lambda actor_id: "stub",
            create_session=lambda **kwargs: object(),
        )

        registry = GameRegistry()
        registry.register(plugin)

        assert registry.resolve_player_ids("configured", {}) == ["a", "b"]

    @pytest.mark.parametrize("actor_ids", [["a", "a"], [""], ["a", 1], "a"])
    def test_resolve_player_ids_rejects_invalid_results(self, actor_ids) -> None:
        plugin = SimpleNamespace(
            name="invalid",
            version="1.0",
            player_ids_from_config=lambda config: actor_ids,
            system_prompt=lambda actor_id: "stub",
            create_session=lambda **kwargs: object(),
        )
        registry = GameRegistry()
        registry.register(plugin)

        with pytest.raises(PluginError, match="actor IDs"):
            registry.resolve_player_ids("invalid", {})


class TestPokerPlugin:
    def test_poker_entry_point_resolves(self, monkeypatch) -> None:
        from benchtable.games.poker.session import PokerGame

        plugins = [
            SimpleNamespace(
                name="poker",
                load=lambda: PokerGame,
            )
        ]
        monkeypatch.setattr("importlib.metadata.entry_points", lambda *, group: plugins)

        registry = GameRegistry()
        registry.discover()

        poker = registry.load("poker")
        assert poker.name == "poker"
        assert poker.version

    def test_poker_validate_config_valid(self) -> None:
        from benchtable.games.poker.session import PokerGame

        game = PokerGame()
        game.validate_config(
            {
                "players": ["a", "b"],
                "initial_stack": 1000,
                "small_blind": 5,
                "big_blind": 10,
            }
        )

    def test_poker_validate_config_rejects_bad_players(self) -> None:
        from benchtable.games.poker.session import PokerGame

        game = PokerGame()
        with pytest.raises(ValueError, match="unique"):
            game.validate_config({"players": ["a", "a"]})

    def test_poker_player_ids_from_config(self) -> None:
        from benchtable.games.poker.session import PokerGame

        game = PokerGame()
        ids = game.player_ids_from_config({"players": ["x", "y", "z"]})
        assert ids == ["x", "y", "z"]


class TestMinMaxTurns:
    """The optional ``min_max_turns`` plugin hook and registry accessor."""

    def test_returns_plugin_value(self) -> None:
        plugin = _StubPlugin("min-turns", "1.0")
        plugin.min_max_turns = lambda game_config: 135  # type: ignore[method-assign]
        registry = GameRegistry()
        registry.register(plugin)

        assert registry.min_max_turns("min-turns", {}) == 135

    def test_returns_none_when_hook_is_absent(self) -> None:
        registry = GameRegistry()
        registry.register(_StubPlugin("plain", "1.0"))

        assert registry.min_max_turns("plain", {}) is None

    def test_rejects_non_callable_hook(self) -> None:
        plugin = _StubPlugin("min-turns", "1.0")
        plugin.min_max_turns = "135"  # type: ignore[method-assign]
        registry = GameRegistry()
        registry.register(plugin)

        with pytest.raises(PluginError, match="min_max_turns"):
            registry.min_max_turns("min-turns", {})

    @pytest.mark.parametrize("value", ["135", 13.5, True, 0, -3])
    def test_rejects_invalid_hook_results(self, value: object) -> None:
        plugin = _StubPlugin("min-turns", "1.0")
        plugin.min_max_turns = lambda game_config: value  # type: ignore[method-assign]
        registry = GameRegistry()
        registry.register(plugin)

        with pytest.raises(PluginError, match="integer >= 1"):
            registry.min_max_turns("min-turns", {})

    def test_wraps_hook_failures_in_plugin_error(self) -> None:
        def _broken(game_config: object) -> int:
            raise ValueError("boom")

        plugin = _StubPlugin("min-turns", "1.0")
        plugin.min_max_turns = _broken  # type: ignore[method-assign]
        registry = GameRegistry()
        registry.register(plugin)

        with pytest.raises(PluginError, match="could not be read"):
            registry.min_max_turns("min-turns", {})
