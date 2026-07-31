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

    def player_ids_from_config(self, game_config: JsonObject) -> list[str]:
        return self.player_ids

    def validate_config(self, game_config: JsonObject) -> None:
        # Reject invalid game-specific values before providers are constructed.
        return None

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

Plugins that support configurable actors should implement
`player_ids_from_config(game_config)` and return the actor IDs for that run.
Implement `validate_config(game_config)` to reject malformed game settings
before the CLI constructs any agents. The registry falls back to `player_ids`
and skips validation for older plugins that do not provide these hooks.

For multiple configured agents, each agent `id` is an actor ID from the
plugin's `current_actor_id` values. The engine constructs every configured
agent and dispatches each turn by that ID. A single configured agent remains a
convenience fallback for games with one shared agent; a missing mapping in a
multi-agent run is an explicit failed match.
Plugins that expose `requires_exact_agent_ids = True` must receive exactly one
agent for every configured actor, including when the engine is used directly.

The run trace records each configured agent's `id`, `role`, `model`,
`base_url`, `api_key_env`, `timeout`, and `max_completion_tokens`. It records
the environment variable name, never the literal API key.

The run configuration also records `max_memory_operations_per_turn`, the
finite budget reserved for engine-owned memory operations.

### Conversations and Optional Session Capabilities

The engine owns the transcript. Each actor has an isolated transcript for the
current conversation scope. On every turn, the engine appends a fresh
actor-specific observation as a user message. Assistant tool-call messages and
matching tool-result messages remain in the transcript as valid Chat
Completions history; the system prompt is kept separate.

Sessions may optionally expose:

```python
from benchtable.contracts import MatchMemorySummary


class MySession:
    @property
    def conversation_scope_id(self) -> str:
        """Stable identifier for the current conversation scope."""

    def drain_match_memory_summaries(self) -> list[MatchMemorySummary]:
        """Return pending public summaries addressed to specific actors."""
```

When `conversation_scope_id` changes, the engine clears all actor transcripts
before the next request. A session without this capability uses a stable
match-scoped conversation for legacy compatibility. The summary drain is also
optional. When present, the engine drains and stores summaries before the next
actor request and before terminal match completion. A summary must identify
its target actor and contain only compact public information, not raw state or
private observations.

### Reserved Memory Tools

The engine provides two reserved tools to every agent during a match:

- `read_memory`: Returns that actor's private notes and actor-targeted system
  summaries in chronological order. Each entry has a `source` marker of
  `agent` or `system`.
- `write_memory(text)`: Appends a free-form private note.

System summaries are stored separately from agent notes and do not consume the
configured agent-note entry or character limits.

Memory-only calls do not advance the game turn. The engine processes them and
requests another model response. Memory writes are committed when their
memory-only calls succeed and cannot be paired with a game action. A response
that mixes memory and game calls is invalid and retried in the pre-action loop.
The action-phase response must contain exactly one recognized game action.
`read_memory` cannot be combined with that action.

Memory state is per-match and per-actor. Notes are never shared between agents
or across matches.

When a response contains several memory-only calls, the engine processes them
in response order and counts each call against the memory budget. Each
assistant tool-call message and matching tool result is retained in the actor's
transcript before the next request.

### Strict Observations

The engine calls `get_observation(actor_id)` for the **current actor only**.
Never include another player's private information in an observation. The
engine appends only that observation to the current actor's transcript and
passes that transcript, the system prompt, and the applicable tools to the
model. The current actor's accumulated transcript and memory results addressed
to that actor are permitted. Never expose raw game state, another actor's
observation, transcript, or memory.

### Tool Definitions

Return OpenAI-compatible function tools via `get_tools(actor_id)`. Use
`ToolSpec` which provides `to_openai_tool()` for serialization.

### Action Validation

`apply_action` receives the actor ID, tool name, and parsed arguments. If the
action is invalid, raise `InvalidActionError`. The engine will retry within the
configured budget.

After applying an accepted action, the engine appends the assistant tool-call
message and a tool result containing only the public transition summary,
metrics, and terminal status. It then makes exactly one tool-free finalization
call. Finalization must contain assistant text or a finish reason. A
finalization response containing any tool call fails immediately and is not
retried.

### Failed Turns

`handle_failed_turn` is called when the model exhausts its invalid-action
budget. Return `None` to let the engine end the match as failed, or return a
`Transition` to apply a game-specific recovery. A nonterminal recovery lets
the engine continue the match; a terminal recovery is completed only when the
session returns a completed result.

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

Sessions may optionally expose `drain_hand_events()` to return public plugin
events. The engine emits each event with match, turn, and actor context before
the match result. Poker uses this for public `hand_start` and `hand_end`
events; private hole cards must not appear in those payloads.
