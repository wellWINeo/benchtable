"""Game plugin and session protocols."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from benchtable.contracts import (
    GameResult,
    JsonObject,
    Observation,
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
class GamePlugin(Protocol):
    """Factory for game sessions with metadata."""

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def player_ids(self) -> list[str]: ...

    def system_prompt(self, actor_id: str) -> str: ...

    def create_session(
        self, *, seed: int, game_config: JsonObject | None = None
    ) -> GameSession: ...
