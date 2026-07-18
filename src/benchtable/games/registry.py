"""Game plugin registry with entry-point discovery and direct registration."""

from __future__ import annotations

from typing import cast

from benchtable.errors import PluginError
from benchtable.games.protocol import GamePlugin


class GameRegistry:
    """Registry for game plugins supporting both entry-point discovery
    and direct injection (for tests)."""

    def __init__(self) -> None:
        self._plugins: dict[str, GamePlugin] = {}

    def register(self, plugin: GamePlugin) -> None:
        """Register a plugin directly.  Raises PluginError on duplicate names."""
        name = _validated_plugin_name(plugin)
        if name in self._plugins:
            raise PluginError(f"Plugin '{name}' is already registered")
        self._plugins[name] = plugin

    def list(self) -> list[GamePlugin]:
        """Return registered plugins sorted by name."""
        return sorted(
            self._plugins.values(),
            key=lambda p: p.name,
        )

    def load(self, name: str) -> GamePlugin:
        """Load a plugin by name.  Raises PluginError if not found."""
        plugin = self._plugins.get(name)
        if plugin is None:
            available = ", ".join(sorted(self._plugins.keys())) or "(none)"
            raise PluginError(f"Game plugin '{name}' not found. Available: {available}")
        return plugin

    def discover(self) -> None:
        """Discover plugins from the benchtable.games entry-point group."""
        try:
            from importlib.metadata import entry_points
        except ImportError:
            return

        eps = entry_points(group="benchtable.games")
        entry_point_names: set[str] = set()
        for ep in sorted(eps, key=lambda e: e.name):
            if ep.name in entry_point_names:
                raise PluginError(f"duplicate game plugin entry point name '{ep.name}'")
            entry_point_names.add(ep.name)
            try:
                plugin = ep.load()
                instance = cast(GamePlugin, plugin() if callable(plugin) else plugin)
                metadata_name = _validated_plugin_name(instance)
                if metadata_name in self._plugins:
                    raise PluginError(
                        f"duplicate plugin metadata name '{metadata_name}'"
                    )
                self._plugins[metadata_name] = instance
            except Exception as exc:
                raise PluginError(f"Failed to load plugin '{ep.name}': {exc}") from exc


def _validated_plugin_name(plugin: GamePlugin) -> str:
    """Validate plugin metadata and return its name for registry keys."""
    try:
        name = cast(object, plugin.name)
    except Exception as exc:
        raise PluginError("Plugin metadata name could not be read") from exc
    if not isinstance(name, str) or not name.strip():
        raise PluginError("Plugin metadata name must be a non-empty string")

    try:
        version = cast(object, plugin.version)
    except Exception as exc:
        raise PluginError("Plugin metadata version could not be read") from exc
    if not isinstance(version, str) or not version.strip():
        raise PluginError("Plugin metadata version must be a non-empty string")

    try:
        player_ids = cast(object, plugin.player_ids)
    except Exception as exc:
        raise PluginError("Plugin player_ids could not be read") from exc
    if not isinstance(player_ids, list) or not all(
        isinstance(player_id, str) for player_id in cast(list[object], player_ids)
    ):
        raise PluginError("Plugin player_ids must be a list of strings")

    for attribute in ("system_prompt", "create_session"):
        try:
            value = getattr(plugin, attribute)
        except Exception as exc:
            raise PluginError(
                f"Plugin required attribute '{attribute}' could not be read"
            ) from exc
        if not callable(value):
            raise PluginError(f"Plugin attribute '{attribute}' must be callable")
    return name
