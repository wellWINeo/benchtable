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
