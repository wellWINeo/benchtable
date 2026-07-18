# Game Plugin Authoring Guide

Benchtable runs language models as agents in interchangeable games. Games are
plugins discovered through the `benchtable.games` Python entry-point group.

## Quick Start

A game plugin implements two protocols: `GamePlugin` (the factory) and
`GameSession` (a running game instance).

```python
from benchtable.contracts import (
    GameResult,
    JsonObject,
    Observation,
    ToolSpec,
    Transition,
)


class MyGame:
    @property
    def name(self) -> str:
        return "my-game"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def player_ids(self) -> list[str]:
        return ["player-1", "player-2"]

    def system_prompt(self, actor_id: str) -> str:
        return f"You are {actor_id}. Call 'act' when it is your turn."

    def create_session(
        self, *, seed: int, game_config: JsonObject | None = None
    ) -> "MySession":
        return MySession(seed=seed, game_config=game_config or {})


class MySession:
    def __init__(self, *, seed: int, game_config: JsonObject) -> None:
        self._seed = seed
        self._game_config = game_config
        self._turns = 0

    @property
    def current_actor_id(self) -> str:
        return "player-1" if self._turns % 2 == 0 else "player-2"

    def get_observation(self, actor_id: str) -> Observation:
        return Observation(
            actor_id=actor_id,
            text=f"Turn {self._turns}. You are {actor_id}.",
            metadata={"turn": self._turns},
        )

    def get_tools(self, actor_id: str) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="act",
                description="Take your turn.",
                parameters={"type": "object", "properties": {}},
            )
        ]

    def apply_action(
        self, actor_id: str, tool_name: str, arguments: JsonObject
    ) -> Transition:
        if tool_name != "act":
            from benchtable.errors import InvalidActionError
            raise InvalidActionError(f"Unknown tool: {tool_name}")
        self._turns += 1
        return Transition(
            summary=f"{actor_id} acted",
            metrics={"turns": self._turns},
        )

    def handle_failed_turn(
        self, actor_id: str, reason: str
    ) -> Transition | None:
        return None

    @property
    def is_terminal(self) -> bool:
        return self._turns >= 10

    def get_result(self) -> GameResult:
        return GameResult(
            completed=True,
            outcome={"total_turns": self._turns},
            metrics={"seed": self._seed},
        )
```

## Key Rules

### Session Configuration and Agent Mapping

Values under `[run.game_config]` are validated as JSON-compatible data and
passed unchanged to `create_session(seed=..., game_config=...)` for every
match. Keep game-specific setup in this mapping and keep the authoritative
state inside the session.

For multiple configured agents, each agent `id` is an actor ID from the
plugin's `current_actor_id` values. The engine constructs every configured
agent and dispatches each turn by that ID. A single configured agent remains a
convenience fallback for games with one shared agent; a missing mapping in a
multi-agent run is an explicit failed match.

The run trace records each configured agent's `id`, `role`, `model`,
`base_url`, `api_key_env`, `timeout`, and `max_completion_tokens`. It records
the environment variable name, never the literal API key.

### Strict Observations

The engine calls `get_observation(actor_id)` for the **current actor only**.
Never include another player's private information in an observation. The
engine passes only your observation, the system prompt, and the tools to the
model.

### Tool Definitions

Return OpenAI-compatible function tools via `get_tools(actor_id)`. Use
`ToolSpec` which provides `to_openai_tool()` for serialization.

### Action Validation

`apply_action` receives the actor ID, tool name, and parsed arguments. If the
action is invalid, raise `InvalidActionError`. The engine will retry within the
configured budget.

### Failed Turns

`handle_failed_turn` is called when the model exhausts its invalid-action
budget. Return `None` to let the engine end the match as failed, or return a
`Transition` to apply a game-specific recovery.

### No Raw State Exposure

The engine never sees your internal state. It only interacts through
observations, tools, and transitions. Keep your state private.

## Registering a Plugin

In your package's `pyproject.toml`:

```toml
[project.entry-points."benchtable.games"]
my-game = "my_package.games:MyGame"
```

Install the package and `benchtable list-games` will discover it.

## Trace Events

The engine records these events for each turn:

- `observation` - the exact text and metadata sent to the actor
- `tool_schemas` - the OpenAI tool definitions sent to the actor
- `model_request` - the request metadata
- `model_response` - the raw and normalized model output
- `validation` - whether the action was valid
- `transition` - your plugin-supplied summary and metrics

Credential-shaped fields are redacted before writing.
