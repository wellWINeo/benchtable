# Benchtable Agent Instructions

## Read First

Before changing code, read these documents in order:

1. `docs/plans/2026-07-17-benchmark-infrastructure-design.md`
2. `docs/plans/2026-07-17-benchmark-infrastructure.md`
3. The tests nearest to the code being changed

The design document records approved architectural decisions. The implementation
plan is the execution order for the initial infrastructure scaffold. If an
implementation needs to diverge materially from either document, update the
design or ask for approval before proceeding.

## Project Mission

Benchtable runs language models as agents in interchangeable games and records
traces for outcome and behavior analysis. The initial infrastructure must support
future poker experiments without embedding poker rules in the benchmark engine.

The first release is local and CLI-driven. It supports OpenAI-compatible model
providers only.

## Technology

- Python 3.12 or newer.
- `uv` for project, environment, and lockfile management.
- Pydantic 2 for validated configuration and serialized contracts.
- The official `openai` Python SDK with its asynchronous client.
- Chat Completions function tool calls as the initial model action protocol.
- Typer for the CLI.
- `pytest` and `pytest-asyncio` for tests.
- Ruff for formatting and linting.
- Pyright for type checking.
- `tomllib` from the standard library for TOML configuration.
- Append-only JSONL files for run persistence; do not add a database for the
  initial scaffold.

Use the `src/` layout. Runtime code belongs under `src/benchtable/`; tests belong
under `tests/`.

## Architecture Invariants

### Game plugins

Games are plugins discovered through the `benchtable.games` Python entry-point
group. The library must also allow direct plugin injection for tests.

A game plugin owns:

- Game metadata and plugin version.
- Player and role definitions.
- The game system prompt.
- Initial state and session creation.
- Turn selection.
- Actor-specific observations.
- OpenAI-compatible tool definitions.
- Action validation and state transitions.
- Terminal results and game-specific metrics.

The game plugin owns authoritative state and all game rules. Do not move game
rules into the generic engine.

### Strict observations

The engine must request an observation for the current actor and pass only that
observation, the applicable system prompt, and the applicable tools to the
agent. Never pass raw game state to a model. Test private-information isolation
explicitly for every game fixture.

### Agent and provider boundary

The engine depends on an `Agent` protocol, not directly on the OpenAI SDK. The
initial concrete adapter is `OpenAICompatibleAgent`.

Provider configuration may include a model, configurable `base_url`, timeout,
retry settings, generation limits, and the name of an environment variable
containing the API key. Read credentials from the environment. Never accept,
persist, print, or log literal API keys.

Keep provider normalization inside the adapter. Do not add provider-specific
branches or additional provider abstractions until a concrete requirement exists.

### Run engine

For each sequential turn, the engine:

1. Selects the current actor.
2. Requests that actor's observation and tools.
3. Builds the model messages.
4. Requests a model response.
5. Parses assistant text and tool calls.
6. Requires exactly one recognized game action.
7. Retries invalid output within the configured budget when allowed.
8. Applies the accepted action through the game session.
9. Records the transition and continues until the game returns a result.

The engine owns turn sequencing, request attempts, budgets, event sequencing,
and common metrics. The game owns rule decisions and state transitions.

Matches run sequentially initially for stable event ordering and local
reproducibility. Provider calls should remain asynchronous internally.

Invalid model output, provider failures, timeouts, cancellation, and exhausted
budgets must become explicit events and outcomes. Never silently convert a failed
match into a win or loss. Game-specific failed-turn semantics may be supplied by
the game plugin.

## Trace Rules

Each run writes an append-only `events.jsonl` with a versioned envelope containing
at least:

- `schema_version`
- Monotonic `sequence`
- `event_type`
- `run_id`
- Match and turn identifiers when applicable
- Actor identifier when applicable
- Timestamp
- JSON-compatible payload

The trace must preserve, after redaction:

- Run configuration, game/plugin version, and random seeds
- Exact observations and tool schemas issued to agents
- Provider requests and raw responses
- Normalized assistant text, tool calls, usage, latency, and finish reason
- Validation results and invalid-action errors
- Plugin-provided transition summaries
- Provider, timeout, budget, cancellation, and engine failures
- Game results and generic metrics

The event writer is the credential-redaction boundary. Redact credential-shaped
fields recursively, including API keys, authorization values, access tokens,
passwords, client secrets, and nested headers. Preserve legitimate usage fields
such as `prompt_tokens` and `completion_tokens`.

Flush each event after writing so interrupted runs retain a valid prefix. Do not
capture or request hidden chain-of-thought. Assistant text may be logged as
provider output, but it is not a required reasoning field.

## Configuration And CLI

Experiment configuration is TOML and must be validated before any provider
request. It contains run settings, agent entries, and optional game-specific
configuration.

The initial CLI surface is intentionally small:

- `benchtable list-games` lists installed game plugins and versions.
- `benchtable run --config PATH --output PATH` validates configuration, executes
  matches, and writes the trace.

Keep registry and agent construction injectable so CLI tests remain offline.
Use non-zero exit codes and actionable messages for invalid configuration,
unknown games, missing credentials, and failed runs.

## Repository Shape

```text
pyproject.toml
src/benchtable/
  cli.py
  config.py
  contracts.py
  engine.py
  errors.py
  events.py
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
  plans/
  plugin-authoring.md
```

## Testing Requirements

Use test-first development for behavior changes. Write a failing test, run it to
confirm the failure, implement the smallest correct change, then run focused and
full verification.

Use a deterministic fake agent and a small non-poker test game. Tests must not
require API keys or make live provider requests.

Cover:

- Typed contract and TOML validation
- Turn sequencing and strict observation isolation
- Tool parsing and exactly-one-action enforcement
- Invalid-action and provider retry budgets
- Event ordering, flush behavior, schema versioning, and redaction
- Provider response normalization
- Plugin discovery and contract compliance
- CLI errors and complete offline traces
- Golden normalized traces with timestamps and other volatile fields excluded

Run the complete verification suite before declaring infrastructure work complete:

```bash
uv sync
uv run pytest -q
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv lock --check
```

## Scope Guardrails

The initial scaffold explicitly excludes:

- Poker rules or poker-specific metrics
- Web UI or dashboard
- Database, queue, distributed workers, or hosted API
- Provider adapters beyond OpenAI-compatible endpoints
- Statistical analysis and report generation
- Hidden chain-of-thought capture

Do not add configuration knobs, abstractions, dependencies, concurrency, or
backward-compatibility layers without a concrete requirement.

## Change Workflow

- Preserve user changes in the worktree; do not reset or overwrite unrelated
  files.
- Keep edits focused and match the surrounding style.
- Prefer standard-library solutions for small utilities.
- Stage only intended files.
- Do not commit or push unless explicitly requested.
- When implementation is complete, inspect `git status`, the diff, and the full
  verification output before reporting the result.
