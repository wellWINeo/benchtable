# Benchtable Infrastructure Design

## Goal

Benchtable runs language models as agents in games and records enough information
to measure outcomes and investigate behavior. Games are interchangeable plugins.
The first implementation provides the benchmark infrastructure only; poker will
be implemented as a separate game plugin.

The first usable version is local and CLI-driven. It supports OpenAI-compatible
providers through the official OpenAI Python SDK, including configurable
`base_url` values for compatible services.

## Decisions

- Use Python 3.12 with `uv` for project and dependency management.
- Use Pydantic for typed configuration and runtime contracts.
- Use the official `openai` Python SDK with its asynchronous client.
- Use Chat Completions tool calls as the first model action protocol.
- Use Typer for the CLI.
- Use `pytest`, Ruff, and Pyright for tests, linting, and type checking.
- Persist runs as append-only JSONL files; do not introduce a database yet.
- Run matches sequentially initially, while keeping provider calls asynchronous.
- Enforce strict per-agent observations at the engine/plugin boundary.
- Use a deterministic fake agent and a small test-only game for infrastructure tests.

## Architecture

The repository is a Python package with a thin CLI over an in-process benchmark
kernel.

### Game plugins

A game plugin defines:

- Game metadata and plugin version.
- Player or role definitions.
- The game system prompt.
- Initial session/state creation.
- The next actor to receive a turn.
- An actor-specific observation.
- OpenAI-compatible tool definitions.
- Action validation and application.
- Terminal results and game-specific metrics.

The plugin owns authoritative game state and all game rules. The engine never
receives or forwards raw state. It asks the plugin for the current actor's
observation and tools.

Plugins are discoverable through a `benchtable.games` Python entry-point group.
The library API also accepts plugin objects directly so tests do not need package
installation or subprocesses.

### Agent and provider layer

The first concrete adapter is `OpenAICompatibleAgent`. Its configuration
includes the model, base URL, API-key environment variable, request timeout,
retry policy, and relevant generation limits.

The engine depends on an `Agent` protocol rather than directly on the SDK. This
allows deterministic fake agents in tests and keeps provider normalization in one
place. API credentials are read from the environment and are never included in
run configuration or trace payloads.

### Run engine

For a sequential turn, the engine:

1. Selects the current actor from the game session.
2. Requests that actor's strict observation and available tools.
3. Builds the system and turn messages.
4. Requests a model response from the configured agent.
5. Parses assistant text and tool calls.
6. Validates exactly one game action.
7. Retries invalid output within the configured turn budget, if allowed.
8. Applies the accepted action through the game session.
9. Records the transition and continues until the game returns a result.

The engine owns turn sequencing, request attempts, budgets, event sequencing,
and common metrics. The game owns rule decisions and state transitions.

```text
GamePlugin -> GameSession -> observation/tools/action/result
                    ^
                    |
RunEngine -> Agent -> OpenAI-compatible provider
                    |
                EventWriter -> JSONL trace
```

The first runner processes matches sequentially for stable event ordering and
local reproducibility. Provider calls are asynchronous internally so the
provider boundary does not need to change when controlled concurrency is added.

## Trace and data model

Each run creates a directory containing an append-only `events.jsonl`. Every line
uses a versioned event envelope:

```json
{
  "schema_version": 1,
  "sequence": 42,
  "event_type": "model_response",
  "run_id": "...",
  "match_id": "...",
  "turn_index": 7,
  "actor_id": "player-1",
  "occurred_at": "...",
  "payload": {}
}
```

The first event records the experiment configuration, game/plugin version,
random seed, agent configurations, and run metadata. Provider credentials are
represented by environment-variable names only.

The engine emits events for:

- Run and match start/end.
- Turn start/end.
- Exact per-agent observations and tool schemas issued.
- Provider requests and raw provider responses.
- Normalized assistant text, tool calls, usage, latency, and finish reason.
- Action validation, including invalid-action errors.
- Applied state-transition summaries.
- Provider, timeout, budget, cancellation, and engine failures.
- Game results and generic metrics.

Provider payloads are retained for forensic analysis, while normalized fields
make traces provider-independent. Known credential fields must be redacted
before serialization. The writer flushes each event so an interrupted run
retains the largest possible valid prefix.

The normalized turn protocol accepts assistant text plus tool calls. The first
valid game tool call is the action. Missing, malformed, or multiple-action
responses are invalid. The engine may return a tool-level error and retry within
the configured budget. Exhausted budgets produce explicit failed-match events;
they are not silently converted to wins or losses. A game may provide
game-specific handling for failed turns when its rules require it.

Assistant text is recorded as provider output. The system does not request or
depend on hidden chain-of-thought capture.

Game results provide a winner/result shape and game-specific metrics. The engine
adds common metrics such as completion status, turn count, invalid actions,
provider failures, token usage, and latency. Win rate and behavioral
classifications are downstream analyses over traces, not hard-coded engine
concepts.

## Configuration and CLI

The CLI accepts a TOML experiment configuration and an output directory. A
configuration has run settings and one or more agent entries:

```toml
[run]
game = "poker"
matches = 10
seed = 20260717
max_turns = 200
max_invalid_attempts = 2

[[agents]]
id = "player-1"
role = "player"
model = "..."
base_url = "https://..."
api_key_env = "OPENAI_API_KEY"
```

Configuration is validated before any provider request. The initial CLI surface
is intentionally small:

- `list-games`: list installed game plugins and versions.
- `run`: validate configuration, execute matches, and write the trace.

Analysis and report commands are deferred until real game traces exist.

## Failure and reproducibility policy

- Transient provider failures are retried according to configuration, and every
  attempt is logged.
- Invalid model output receives a tool-level error and may be retried within the
  turn budget.
- Exhausted invalid-action or provider budgets produce explicit failed-match
  results.
- Cancellation and process errors write terminal error events whenever possible.
- The random seed is recorded and supplied to the game plugin.
- Prompts, observations, tool schemas, normalized responses, and provider
  payloads are recorded so a run can be reconstructed for analysis.
- Provider calls are not assumed deterministic; replay initially means replaying
  or analyzing recorded events, not rerunning a provider.

## Testing strategy

Testing uses a deterministic fake agent and a tiny test-only game:

- Unit tests for typed contract validation and configuration parsing.
- Unit tests for turn sequencing and strict observation isolation.
- Unit tests for tool parsing, invalid-action retries, budgets, and metrics.
- Unit tests for JSONL ordering, flushing, schema versioning, and redaction.
- Provider adapter tests at the fake agent/transport boundary without API keys.
- Game plugin contract tests.
- CLI smoke tests that create and inspect a complete trace.
- Golden normalized traces with timestamps and other volatile fields excluded.

## Repository shape

```text
pyproject.toml
src/benchtable/
  cli.py
  config.py
  contracts.py
  engine.py
  errors.py
  events.py
  plugins.py
  agents/
    protocol.py
    openai_compatible.py
  games/
    protocol.py
    registry.py
tests/
  fixtures/
    tiny_game.py
    fake_agent.py
  unit/
  contract/
  integration/
docs/
  plugin-authoring.md
```

## Scaffold phases

1. Bootstrap the Python package, `uv` project, linting, type checking, and test
   tooling.
2. Define typed contracts for games, sessions, observations, tools, actions,
   agents, results, and events.
3. Implement the JSONL event writer with sequence numbers, flush behavior,
   schema versioning, and credential redaction.
4. Implement the OpenAI-compatible asynchronous agent adapter and normalized
   response parsing.
5. Implement the engine loop, retry and budget policies, metrics aggregation,
   and plugin registry.
6. Add TOML configuration validation and `list-games` / `run` CLI commands.
7. Add the deterministic test game, fake agent, contract tests, integration
   tests, and plugin authoring documentation.

## Explicitly out of scope

- Poker implementation.
- Web UI or dashboard.
- Database, queue, distributed workers, or hosted API.
- Provider adapters beyond OpenAI-compatible endpoints.
- Statistical analysis and report generation.
- Hidden chain-of-thought capture.
