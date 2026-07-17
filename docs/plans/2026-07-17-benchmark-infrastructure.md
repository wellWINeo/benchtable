# Benchmark Infrastructure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the local, plugin-based benchmark infrastructure for running agents in games and recording complete OpenAI-compatible traces, without implementing poker.

**Architecture:** A Python package exposes typed game and agent protocols, an in-process sequential run engine, an OpenAI-compatible async adapter, and an append-only JSONL event writer. Games own state and rules; the engine owns orchestration, budgets, retries, and common metrics; the CLI loads TOML configurations and resolves installed game plugins.

**Tech Stack:** Python 3.12, `uv`, Pydantic 2, official `openai` Python SDK, Typer, `pytest`, `pytest-asyncio`, Ruff, and Pyright. Configuration uses the Python standard library `tomllib`; persistence uses standard-library JSONL writing.

---

## Task 1: Bootstrap the Python project

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md`
- Create: `src/benchtable/__init__.py`
- Create: `tests/unit/test_package.py`

**Step 1: Write the failing package smoke test**

Create a test that imports `benchtable` and asserts it exposes a non-empty version string:

```python
def test_package_exposes_version():
    import benchtable

    assert benchtable.__version__
```

**Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/test_package.py -q`

Expected: FAIL because the project and package metadata do not exist yet.

**Step 3: Add the minimal project configuration**

Create `pyproject.toml` with:

- Project name `benchtable`, Python requirement `>=3.12`.
- Runtime dependencies `pydantic`, `openai`, and `typer`.
- Development dependencies `pytest`, `pytest-asyncio`, `ruff`, and `pyright`.
- A `benchtable` console script pointing to `benchtable.cli:app`.
- Ruff configuration for the project source and tests.
- Pyright configuration with the source directory included.

Create `src/benchtable/__init__.py` with a package version constant. Add `.gitignore` entries for virtual environments, caches, generated `runs/`, and Python bytecode. Add a short README containing setup, test, and future plugin commands.

**Step 4: Run the test and tooling checks**

Run: `uv sync`

Expected: `uv` creates the environment and lockfile successfully.

Run: `uv run pytest tests/unit/test_package.py -q`

Expected: PASS.

Run: `uv run ruff check .`

Expected: PASS.

**Step 5: Commit**

```bash
git add pyproject.toml uv.lock .gitignore README.md src/benchtable/__init__.py tests/unit/test_package.py
git commit -m "feat: bootstrap benchtable Python project"
```

## Task 2: Define core typed contracts

**Files:**
- Create: `src/benchtable/contracts.py`
- Create: `src/benchtable/errors.py`
- Create: `src/benchtable/games/__init__.py`
- Create: `src/benchtable/games/protocol.py`
- Create: `src/benchtable/agents/__init__.py`
- Create: `src/benchtable/agents/protocol.py`
- Create: `tests/unit/test_contracts.py`
- Create: `tests/contract/test_game_protocol.py`

**Step 1: Write failing contract tests**

Cover these behaviors:

- `ToolSpec` rejects an empty name and serializes to the OpenAI function-tool shape.
- `Observation` contains only text, public metadata, and actor identity; it must not expose an arbitrary raw state field.
- `ToolCall` retains the provider call ID, tool name, parsed arguments, and raw argument text when parsing fails.
- `ModelResponse` represents assistant text, zero or more normalized tool calls, finish reason, optional usage, and the raw provider payload.
- `GameResult` represents completion, outcome data, and game-specific metrics.
- A minimal fake session satisfies the `GameSession` protocol and can return a transition and terminal result.
- A missing actor or invalid transition is represented by a typed domain error rather than `None` with no context.

Use Pydantic models with `extra="forbid"` for serialized contracts and `Protocol` definitions for game/session and agent behavior. Keep provider SDK types out of these public contracts.

**Step 2: Run the contract tests to verify they fail**

Run: `uv run pytest tests/unit/test_contracts.py tests/contract/test_game_protocol.py -q`

Expected: FAIL because the contract modules do not exist.

**Step 3: Implement the minimal contracts**

Implement:

- JSON-compatible type aliases for event payloads and game metrics.
- `ToolSpec`, `Observation`, `ToolCall`, `Usage`, `ModelResponse`, `Transition`, and `GameResult` Pydantic models.
- `GameMetadata`, `GamePlugin`, and `GameSession` protocols in `games/protocol.py`.
- `Agent` protocol and request input model in `agents/protocol.py`.
- Explicit errors for configuration, plugin lookup, malformed model output, invalid action, provider failure, and run failure.

The game session protocol must expose the current actor, a system prompt or system-prompt builder, actor-specific observation, actor-specific tool list, action application, optional failed-turn handling, and terminal result. It must not expose raw state through the protocol.

**Step 4: Run the tests and type checks**

Run: `uv run pytest tests/unit/test_contracts.py tests/contract/test_game_protocol.py -q`

Expected: PASS.

Run: `uv run pyright src/benchtable`

Expected: PASS.

**Step 5: Commit**

```bash
git add src/benchtable/contracts.py src/benchtable/errors.py src/benchtable/games src/benchtable/agents tests/unit/test_contracts.py tests/contract/test_game_protocol.py
git commit -m "feat: define game and agent contracts"
```

## Task 3: Implement the JSONL event writer

**Files:**
- Create: `src/benchtable/events.py`
- Create: `tests/unit/test_events.py`

**Step 1: Write failing event tests**

Test that:

- Each emitted event receives a monotonically increasing sequence number starting at one.
- The event envelope contains `schema_version`, event type, run identifiers, actor/turn context when supplied, timestamp, and payload.
- Each event is one valid JSON line and is immediately visible after `emit` because the writer flushes.
- The first event can contain the run configuration and seed.
- Pydantic provider responses are converted to JSON-compatible values.
- Credential-shaped keys such as `api_key`, `authorization`, `access_token`, `password`, and `client_secret` are redacted recursively, including nested headers.
- Unsupported payload values raise a serialization error instead of being silently stringified.
- Closing the writer prevents further writes with a clear error.

Use `tmp_path` and read the file after each write; do not compare volatile timestamps in exact golden assertions.

**Step 2: Run the event tests to verify they fail**

Run: `uv run pytest tests/unit/test_events.py -q`

Expected: FAIL because `EventWriter` and the event envelope are not implemented.

**Step 3: Implement `EventWriter`**

Implement an append-only writer that:

- Creates the run directory and `events.jsonl`.
- Assigns sequence numbers internally rather than trusting callers.
- Normalizes Pydantic models, mappings, lists, strings, numbers, booleans, and null values.
- Redacts only credential fields, preserving legitimate usage fields such as `prompt_tokens`.
- Serializes with deterministic key ordering and one line per event.
- Flushes every line and exposes a context manager for closing the file.

Keep event payload redaction at the writer boundary so provider and game code cannot accidentally bypass it.

**Step 4: Run the tests and lint**

Run: `uv run pytest tests/unit/test_events.py -q`

Expected: PASS.

Run: `uv run ruff check src/benchtable/events.py tests/unit/test_events.py`

Expected: PASS.

**Step 5: Commit**

```bash
git add src/benchtable/events.py tests/unit/test_events.py
git commit -m "feat: add versioned JSONL event writer"
```

## Task 4: Add the OpenAI-compatible agent adapter

**Files:**
- Modify: `src/benchtable/agents/protocol.py`
- Create: `src/benchtable/agents/openai_compatible.py`
- Create: `tests/unit/test_openai_compatible_agent.py`

**Step 1: Write failing adapter tests**

Use an injected fake async client so tests never require a network or API key. Test that:

- The adapter passes the configured model, messages, and OpenAI function-tool schemas to `chat.completions.create`.
- A configured `base_url` and environment-derived API key are used to construct the SDK client.
- Missing or empty API-key environment variables raise a configuration error before a request.
- A response with assistant text and one function tool call normalizes into `ModelResponse` and parses JSON arguments into a mapping.
- A response with no tool calls still preserves assistant text and finish reason.
- Malformed tool arguments remain available as raw text and produce a normalized parse error for the engine to handle.
- Provider exceptions are wrapped with provider/model context without including the secret key.

The fake client should record request kwargs and return a small object exposing `model_dump(mode="json")` plus the SDK response attributes needed by the adapter.

**Step 2: Run the adapter tests to verify they fail**

Run: `uv run pytest tests/unit/test_openai_compatible_agent.py -q`

Expected: FAIL because the concrete adapter is not implemented.

**Step 3: Implement `OpenAICompatibleAgent`**

Implement an async adapter around `AsyncOpenAI`:

- Read the API key from the configured environment variable.
- Pass `base_url` when configured.
- Use Chat Completions with `n=1` and the supplied messages and tools.
- Keep generation settings optional so compatible providers that reject unsupported fields can be used with defaults.
- Convert the first choice into the provider-independent `ModelResponse`.
- Preserve the raw JSON-compatible response for event logging.
- Leave retry decisions to the engine, or disable SDK-internal retries when configured so each attempt is visible to the event log.

Do not add provider-specific branches. OpenAI-compatible behavior belongs in configuration or the common request shape.

**Step 4: Run tests and type checks**

Run: `uv run pytest tests/unit/test_openai_compatible_agent.py -q`

Expected: PASS.

Run: `uv run pyright src/benchtable/agents`

Expected: PASS.

**Step 5: Commit**

```bash
git add src/benchtable/agents/protocol.py src/benchtable/agents/openai_compatible.py tests/unit/test_openai_compatible_agent.py
git commit -m "feat: add OpenAI-compatible agent adapter"
```

## Task 5: Implement game plugin discovery and the test fixture

**Files:**
- Modify: `src/benchtable/games/protocol.py`
- Create: `src/benchtable/games/registry.py`
- Create: `tests/fixtures/tiny_game.py`
- Create: `tests/fixtures/__init__.py`
- Create: `tests/unit/test_game_registry.py`
- Create: `tests/contract/test_tiny_game.py`

**Step 1: Write failing registry and fixture tests**

Test that:

- The registry discovers `benchtable.games` entry points and returns stable metadata sorted by name.
- Duplicate names and missing entry points produce explicit plugin errors.
- A named plugin can be loaded once and cached for the run.
- The tiny test game exposes two actors, a single legal tool, a finite session, and a terminal `GameResult`.
- The tiny test game returns different observations for each actor and never includes the other actor's private marker.
- The tiny test game returns an action transition without exposing its raw internal state.

The fixture should be intentionally non-poker: use a tiny alternating-choice game that terminates after a small number of valid actions.

**Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_game_registry.py tests/contract/test_tiny_game.py -q`

Expected: FAIL because registry and fixture modules are incomplete.

**Step 3: Implement registry and fixture**

Implement `GameRegistry` with:

- `discover()` using `importlib.metadata.entry_points(group="benchtable.games")`.
- `list()` returning metadata without instantiating every session.
- `load(name)` returning the plugin factory/object or a clear lookup error.
- Dependency injection support so tests can construct a registry from explicit plugins.

Implement the test-only game as an object satisfying `GamePlugin` and `GameSession`. Keep the fixture out of runtime entry points; the production package must remain poker-free.

**Step 4: Run tests and type checks**

Run: `uv run pytest tests/unit/test_game_registry.py tests/contract/test_tiny_game.py -q`

Expected: PASS.

Run: `uv run pyright src/benchtable/games tests/fixtures/tiny_game.py`

Expected: PASS.

**Step 5: Commit**

```bash
git add src/benchtable/games tests/fixtures tests/unit/test_game_registry.py tests/contract/test_tiny_game.py
git commit -m "feat: add game plugin registry"
```

## Task 6: Implement the run engine

**Files:**
- Create: `src/benchtable/engine.py`
- Create: `tests/unit/test_engine.py`
- Create: `tests/integration/test_engine_trace.py`
- Modify: `tests/fixtures/fake_agent.py`

**Step 1: Write failing engine tests**

Use the tiny game and a deterministic fake agent. Cover:

- The engine emits run, match, turn, observation, model request/response, validation, transition, and terminal result events in order.
- The agent receives only the current actor's observation, current system prompt, and current tools.
- A valid single tool call is applied and advances the game.
- Assistant text is retained in the model response event even when the action is a tool call.
- Missing tool calls, unknown tools, malformed arguments, and multiple tool calls are rejected and retried up to `max_invalid_attempts`.
- A provider exception is retried up to `max_provider_retries`, and every attempt is logged.
- Exhausted invalid-action/provider budgets produce an explicit failed match and do not fabricate a game winner.
- `max_turns` terminates a non-finishing game with a clear failure reason.
- Match seeds are derived deterministically from the run seed and match index and are recorded.
- Common metrics include completion, turn count, invalid actions, provider failures, and usage totals.
- A game-provided failed-turn handler can produce a valid terminal transition.

The fake agent should return a sequence of responses or raise configured exceptions, and should retain each request for observation-isolation assertions.

**Step 2: Run the engine tests to verify they fail**

Run: `uv run pytest tests/unit/test_engine.py tests/integration/test_engine_trace.py -q`

Expected: FAIL because the engine and fake agent are not implemented.

**Step 3: Implement the engine loop**

Implement an async `RunEngine` that:

1. Creates run and match identifiers and derives match seeds.
2. Emits the initial run configuration without secrets.
3. Creates each session through the selected plugin.
4. Requests an observation and tools for the current actor only.
5. Builds Chat Completions messages without exposing raw game state.
6. Logs the exact request, measures latency, and calls the actor's agent.
7. Logs the raw and normalized response.
8. Requires exactly one recognized tool call for an action.
9. Delegates argument and rule validation to the game session.
10. Sends a structured validation error on retry without mutating game state.
11. Applies valid actions and logs only the plugin-supplied transition summary.
12. Stops on a game result, configured failure, cancellation, or engine error.

Keep retry and budget policy in the engine. Do not catch exceptions without emitting a contextual error event and returning an explicit failed result.

**Step 4: Run the focused tests and full test suite**

Run: `uv run pytest tests/unit/test_engine.py tests/integration/test_engine_trace.py -q`

Expected: PASS.

Run: `uv run pytest -q`

Expected: PASS for all tests implemented so far.

**Step 5: Commit**

```bash
git add src/benchtable/engine.py tests/fixtures/fake_agent.py tests/unit/test_engine.py tests/integration/test_engine_trace.py
git commit -m "feat: implement benchmark run engine"
```

## Task 7: Add TOML configuration and CLI commands

**Files:**
- Create: `src/benchtable/config.py`
- Create: `src/benchtable/cli.py`
- Create: `tests/unit/test_config.py`
- Create: `tests/integration/test_cli.py`

**Step 1: Write failing configuration and CLI tests**

Test that:

- Valid TOML parses into typed run and agent configuration.
- Required fields, positive match counts, non-negative budgets, and unknown fields are validated before provider construction.
- API keys are represented only by environment-variable names.
- `list-games` returns discovered plugin names and versions in stable order.
- `run` rejects an unknown game before making an agent request.
- `run` with the tiny test game and injected fake agent writes a complete `events.jsonl` and reports its location.
- CLI errors have non-zero exit codes and actionable messages.

The CLI tests should inject the registry and agent factory rather than making network requests.

**Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_config.py tests/integration/test_cli.py -q`

Expected: FAIL because configuration and CLI modules are not implemented.

**Step 3: Implement configuration loading**

Use `tomllib` to load a file and Pydantic to validate:

- Run-level game name, match count, seed, maximum turns, invalid-action budget, provider retry budget, and output settings.
- Agent ID, role, model, optional base URL, API-key environment variable, timeout, and optional generation settings.
- Game-specific configuration as a JSON-compatible mapping passed to the plugin.

Reject literal API-key fields and unknown fields. Use a default API-key environment variable only when the field is omitted.

**Step 4: Implement the CLI**

Create a Typer app with:

- `list-games`, which discovers and prints plugin name/version metadata.
- `run --config PATH --output PATH`, which validates TOML, loads the plugin, constructs OpenAI-compatible agents, runs matches through `asyncio.run`, and prints the resulting trace path.

Keep registry and agent construction behind injectable functions or parameters so CLI tests remain offline. Return non-zero exit codes for invalid configuration, unknown games, missing credentials, and failed runs.

**Step 5: Run tests and command checks**

Run: `uv run pytest tests/unit/test_config.py tests/integration/test_cli.py -q`

Expected: PASS.

Run: `uv run benchtable --help`

Expected: Help output lists `list-games` and `run`.

Run: `uv run benchtable list-games`

Expected: The command succeeds and reports no production games until a plugin is installed.

**Step 6: Commit**

```bash
git add src/benchtable/config.py src/benchtable/cli.py tests/unit/test_config.py tests/integration/test_cli.py
git commit -m "feat: add experiment configuration and CLI"
```

## Task 8: Document plugin authoring and complete verification

**Files:**
- Create: `docs/plugin-authoring.md`
- Modify: `README.md`
- Create: `tests/integration/test_plugin_documentation.py` only if documentation examples are executable

**Step 1: Write documentation checks**

If examples are made executable, add a test that imports the documented test plugin shape and verifies it satisfies the contract. Otherwise, review the document manually against the protocol tests.

**Step 2: Write plugin authoring documentation**

Document:

- Required `GamePlugin` and `GameSession` methods.
- How to produce strict actor-specific observations.
- How to define OpenAI function tools and validate one action.
- How to return transition summaries and game-specific metrics without exposing raw state.
- How to handle failed turns.
- How to register a plugin with the `benchtable.games` entry-point group in a separate package.
- The trace event expectations and secret-redaction boundary.
- A minimal non-poker example based on the test fixture.

Do not document poker rules in this scaffold.

**Step 3: Run the complete verification suite**

Run: `uv run pytest -q`

Expected: PASS.

Run: `uv run ruff format --check .`

Expected: PASS.

Run: `uv run ruff check .`

Expected: PASS.

Run: `uv run pyright`

Expected: PASS.

Run: `uv lock --check`

Expected: PASS.

Inspect the generated trace from the CLI integration test and verify that it contains no API key or authorization value.

**Step 4: Commit**

```bash
git add README.md docs/plugin-authoring.md tests/integration/test_plugin_documentation.py
git commit -m "docs: document game plugin authoring"
```

## Final acceptance criteria

- A clean `uv sync` installs the project and development tools.
- `uv run pytest -q`, Ruff, and Pyright pass.
- A deterministic test game can run end-to-end through the engine and CLI using a fake agent.
- The trace contains ordered observations, tool schemas, requests, responses, validation outcomes, transitions, metrics, and terminal status.
- Raw provider payloads are retained after recursive credential redaction.
- The engine never passes raw game state to an agent.
- A future poker package can register as a `benchtable.games` plugin without modifying engine code.
- No poker implementation, UI, database, distributed worker, or analytics/reporting subsystem is included.
