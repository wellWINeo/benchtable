"""A tiny deterministic test game for infrastructure testing.

Two players alternate choosing "act". The game ends after a configurable
number of total actions.  Each player has a private marker string that must
never appear in the other player's observation.
"""

from __future__ import annotations

from benchtable.contracts import (
    GameResult,
    JsonObject,
    Observation,
    ToolSpec,
    Transition,
)


class _TinySession:
    """Session for the tiny alternating-choice game."""

    def __init__(
        self,
        *,
        seed: int,
        max_actions: int = 4,
        reject_actions: bool = False,
        failed_turn_recovery: bool = False,
        recovery_terminates: bool = False,
        failed_turn_calls: list[str] | None = None,
    ) -> None:
        self._seed = seed
        self._actions_taken = 0
        self._max_actions = max_actions
        self._reject_actions = reject_actions
        self._failed_turn_recovery = failed_turn_recovery
        self._recovery_terminates = recovery_terminates
        self._failed_turn_calls = failed_turn_calls
        self._actors = ("a", "b")
        self._private = {"a": "secret_a", "b": "secret_b"}

    @property
    def current_actor_id(self) -> str:
        return self._actors[self._actions_taken % 2]

    def get_observation(self, actor_id: str) -> Observation:
        return Observation(
            actor_id=actor_id,
            text=(
                f"You are player {actor_id}. "
                f"Actions taken so far: {self._actions_taken}. "
                f"It is your turn."
            ),
            metadata={"actions_taken": self._actions_taken, "seed": self._seed},
        )

    def get_tools(self, actor_id: str) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="act",
                description="Take an action in the game.",
                parameters={"type": "object", "properties": {}},
            )
        ]

    def apply_action(
        self, actor_id: str, tool_name: str, arguments: JsonObject
    ) -> Transition:
        if self._reject_actions:
            from benchtable.errors import InvalidActionError

            raise InvalidActionError("This fixture rejects the action")
        if tool_name != "act":
            from benchtable.errors import InvalidActionError

            raise InvalidActionError(
                f"Unknown tool: {tool_name}",
                details={"actor_id": actor_id, "tool_name": tool_name},
            )
        self._actions_taken += 1
        return Transition(
            summary=f"player {actor_id} acted",
            metrics={"actions_taken": self._actions_taken},
        )

    def handle_failed_turn(self, actor_id: str, reason: str) -> Transition | None:
        if self._failed_turn_calls is not None:
            self._failed_turn_calls.append(reason)
        if self._failed_turn_recovery:
            if self._recovery_terminates:
                self._actions_taken = self._max_actions
            return Transition(
                summary=f"recovered failed turn for {actor_id}",
                metrics={"reason": reason},
            )
        return None

    @property
    def is_terminal(self) -> bool:
        return self._actions_taken >= self._max_actions

    def get_result(self) -> GameResult:
        return GameResult(
            completed=True,
            outcome={"total_actions": self._actions_taken},
            metrics={"seed": self._seed},
        )


class TinyGame:
    """A minimal test game plugin.  Not registered as an entry point."""

    def __init__(
        self,
        *,
        max_actions: int = 4,
        reject_actions: bool = False,
        failed_turn_recovery: bool = False,
        recovery_terminates: bool = False,
    ) -> None:
        self._max_actions = max_actions
        self._reject_actions = reject_actions
        self._failed_turn_recovery = failed_turn_recovery
        self._recovery_terminates = recovery_terminates
        self.failed_turn_calls: list[str] = []
        self.created_game_configs: list[JsonObject] = []

    @property
    def name(self) -> str:
        return "tiny"

    @property
    def version(self) -> str:
        return "0.0.1"

    @property
    def player_ids(self) -> list[str]:
        return ["a", "b"]

    def system_prompt(self, actor_id: str) -> str:
        return (
            f"You are player {actor_id} in a simple alternating-choice game. "
            "When it is your turn, call the 'act' tool exactly once."
        )

    def create_session(
        self, *, seed: int, game_config: JsonObject | None = None
    ) -> _TinySession:
        config = game_config or {}
        self.created_game_configs.append(config)
        configured_max_actions = config.get("max_actions")
        max_actions = (
            configured_max_actions
            if isinstance(configured_max_actions, int)
            else self._max_actions
        )
        return _TinySession(
            seed=seed,
            max_actions=max_actions,
            reject_actions=self._reject_actions,
            failed_turn_recovery=self._failed_turn_recovery,
            recovery_terminates=self._recovery_terminates,
            failed_turn_calls=self.failed_turn_calls,
        )
