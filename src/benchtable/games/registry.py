"""Game plugin registry with entry-point discovery and direct registration."""

from __future__ import annotations

from typing import Callable, cast

from benchtable.contracts import JsonObject
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

    def validate_plugin_config(self, name: str, game_config: JsonObject) -> None:
        """Validate game configuration through the plugin.

        Raises PluginError if the plugin is not found, and re-raises any
        ValueError raised by the plugin's ``validate_config`` method.
        """
        plugin = self.load(name)
        validator = getattr(plugin, "validate_config", None)
        if validator is not None:
            if not callable(validator):
                raise PluginError(
                    f"Plugin '{name}' configuration validator must be callable"
                )
            cast(Callable[[JsonObject], None], validator)(game_config)

    def resolve_player_ids(self, name: str, game_config: JsonObject) -> list[str]:
        """Resolve actor IDs, supporting plugins from before config-aware actors."""
        plugin = self.load(name)
        resolver = getattr(plugin, "player_ids_from_config", None)
        if resolver is None:
            try:
                player_ids = getattr(plugin, "player_ids")
            except AttributeError as exc:
                raise PluginError(
                    f"Plugin '{name}' must define player_ids or player_ids_from_config"
                ) from exc
            return _validated_player_ids(player_ids, name)
        if not callable(resolver):
            raise PluginError(f"Plugin '{name}' actor resolver must be callable")
        return _validated_player_ids(
            cast(Callable[[JsonObject], list[str]], resolver)(game_config), name
        )

    def requires_exact_agent_ids(self, name: str) -> bool:
        """Check if a plugin requires exact agent-to-player mapping."""
        plugin = self.load(name)
        try:
            return bool(getattr(plugin, "requires_exact_agent_ids"))
        except AttributeError:
            return False

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
        player_ids = getattr(plugin, "player_ids", None)
    except Exception as exc:
        raise PluginError("Plugin player_ids could not be read") from exc
    if player_ids is None:
        resolver = getattr(plugin, "player_ids_from_config", None)
        if not callable(resolver):
            raise PluginError("Plugin must define player_ids or player_ids_from_config")
    else:
        _validated_player_ids(player_ids, name)

    for attribute in ("system_prompt", "create_session"):
        try:
            value = getattr(plugin, attribute)
        except Exception as exc:
            raise PluginError(
                f"Plugin required attribute '{attribute}' could not be read"
            ) from exc
        if not callable(value):
            raise PluginError(f"Plugin attribute '{attribute}' must be callable")

    # Validate optional configuration-aware methods if present
    for attribute in ("player_ids_from_config", "validate_config"):
        try:
            value = getattr(plugin, attribute)
        except Exception:
            continue
        if not callable(value):
            raise PluginError(f"Plugin attribute '{attribute}' must be callable")

    return name


def _validated_player_ids(value: object, plugin_name: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or not all(
            type(player_id) is str and bool(player_id.strip())
            for player_id in cast(list[object], value)
        )
    ):
        raise PluginError(
            f"Plugin '{plugin_name}' actor IDs must be a list of non-empty strings"
        )
    player_ids = cast(list[str], value)
    if len(set(player_ids)) != len(player_ids):
        raise PluginError(f"Plugin '{plugin_name}' actor IDs must be unique")
    return player_ids
