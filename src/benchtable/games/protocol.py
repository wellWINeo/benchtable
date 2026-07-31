"""Game plugin and session protocols."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from benchtable.contracts import (
    GameResult,
    JsonObject,
    MatchMemorySummary,
    Observation,
    PluginEvent,
    ToolSpec,
    Transition,
)


@runtime_checkable
class GameSession(Protocol):
    """A running game instance.  The engine never accesses raw state."""

    @property
    def current_actor_id(self) -> str: ...

    def get_observation(self, actor_id: str) -> Observation: ...
    def get_tools(self, actor_id: str) -> list[ToolSpec]: ...
    def apply_action(
        self, actor_id: str, tool_name: str, arguments: JsonObject
    ) -> Transition: ...

    @property
    def is_terminal(self) -> bool: ...

    def get_result(self) -> GameResult: ...


@runtime_checkable
class ConversationScopedSession(Protocol):
    """Optional capability exposing the current conversation scope."""

    @property
    def conversation_scope_id(self) -> str: ...


@runtime_checkable
class MatchMemorySummarySession(Protocol):
    """Optional capability exposing drained public memory summaries."""

    def drain_match_memory_summaries(self) -> list[MatchMemorySummary]: ...


@runtime_checkable
class PluginEventSession(Protocol):
    """Optional capability exposing typed plugin lifecycle events."""

    def drain_hand_events(self) -> list[PluginEvent]: ...


@runtime_checkable
class GamePlugin(Protocol):
    """Factory for game sessions with metadata."""

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    def system_prompt(self, actor_id: str) -> str: ...

    def create_session(
        self, *, seed: int, game_config: JsonObject | None = None
    ) -> GameSession: ...


class StaticActorGamePlugin(GamePlugin, Protocol):
    """Optional compatibility contract for plugins with fixed actor IDs."""

    @property
    def player_ids(self) -> list[str]: ...


class ConfiguredActorGamePlugin(GamePlugin, Protocol):
    """Optional contract for plugins whose actors come from configuration."""

    def player_ids_from_config(self, game_config: JsonObject) -> list[str]: ...


class ExactAgentMappingGamePlugin(GamePlugin, Protocol):
    """Optional contract for plugins requiring one agent per player."""

    @property
    def requires_exact_agent_ids(self) -> bool: ...
