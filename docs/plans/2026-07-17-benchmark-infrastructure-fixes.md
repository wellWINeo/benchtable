# Benchmark Infrastructure Review Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix every issue found in the review of the benchmark infrastructure implementation while preserving the approved plugin-based architecture.

**Architecture:** Keep game rules in plugins and orchestration in `RunEngine`. Make the engine asynchronous, route configured agents by actor ID, and keep the CLI as the synchronous boundary. Make `EventWriter` the append-only, redacting trace boundary and make failed runs explicit in both traces and CLI status.

**Tech Stack:** Python 3.12, Pydantic 2, OpenAI async SDK, Typer, pytest/pytest-asyncio, Ruff, Pyright, uv.

---

### Task 1: Add regression coverage for trace and contract behavior

**Files:**
- Modify: `tests/unit/test_events.py`
- Modify: `tests/unit/test_contracts.py`
- Modify: `tests/unit/test_config.py`
- Modify: `tests/unit/test_game_registry.py`

**Step 1: Write failing tests**

Cover append/sequence recovery, invalid existing JSONL handling, `X-API-Key`
redaction and repeated credential strings, non-object tool arguments, generation
limit validation, and duplicate discovered entry points.

**Step 2: Run focused tests to verify failures**

Run: `uv run pytest tests/unit/test_events.py tests/unit/test_contracts.py tests/unit/test_config.py tests/unit/test_game_registry.py -q`

Expected: the new behavior tests fail against the current implementation.

**Step 3: Implement the smallest supporting changes**

Update the event writer, contracts, configuration models, and registry only as
needed by these tests. Preserve `prompt_tokens` and other legitimate usage
fields during redaction.

**Step 4: Run focused tests**

Run: `uv run pytest tests/unit/test_events.py tests/unit/test_contracts.py tests/unit/test_config.py tests/unit/test_game_registry.py -q`

Expected: PASS.

### Task 2: Fix provider normalization and adapter controls

**Files:**
- Modify: `src/benchtable/agents/openai_compatible.py`
- Modify: `src/benchtable/config.py`
- Modify: `tests/unit/test_openai_compatible_agent.py`

**Step 1: Write failing tests**

Assert that SDK construction disables internal retries, optional generation
limits are forwarded, non-object JSON arguments preserve raw text and produce
a parse error, and raw provider responses remain available.

**Step 2: Run focused tests to verify failures**

Run: `uv run pytest tests/unit/test_openai_compatible_agent.py tests/unit/test_config.py -q`

Expected: the new adapter/configuration tests fail.

**Step 3: Implement provider fixes**

Pass `max_retries=0`, add validated optional generation settings, and normalize
only JSON objects as parsed tool arguments while retaining the original text
for malformed values.

**Step 4: Run tests and type checks**

Run: `uv run pytest tests/unit/test_openai_compatible_agent.py tests/unit/test_config.py -q`

Run: `uv run pyright src/benchtable/agents src/benchtable/config.py`

Expected: PASS and zero type errors.

### Task 3: Refactor engine execution and failure semantics

**Files:**
- Modify: `src/benchtable/engine.py`
- Modify: `src/benchtable/games/protocol.py`
- Modify: `tests/fixtures/tiny_game.py`
- Modify: `tests/fixtures/fake_agent.py`
- Modify: `tests/unit/test_engine.py`
- Modify: `tests/integration/test_engine_trace.py`

**Step 1: Write failing tests**

Add async tests for active-loop execution, terminal completion on the final
allowed turn, valid retry message history, failed-turn recovery, provider and
plugin failure events, cancellation, actor-to-agent dispatch, game config
forwarding, raw request/response events, and complete common metrics.

**Step 2: Run focused tests to verify failures**

Run: `uv run pytest tests/unit/test_engine.py tests/integration/test_engine_trace.py -q`

Expected: the new tests fail, including current synchronous engine calls and
missing trace/failure behavior.

**Step 3: Implement the async engine**

Make `RunEngine.run()` asynchronous and await `Agent.respond`. Keep a single
injected agent as a fallback for existing direct tests, but select configured
agents by actor ID. Pass game configuration to session creation. Return or
record explicit failed-match status, invoke `handle_failed_turn`, and emit
engine/provider/cancellation failure events without fabricating winners.

**Step 4: Implement valid retry history and trace completeness**

Include the original assistant tool call and ID before tool error messages;
use user-level validation messages when no valid tool call exists. Emit exact
provider messages/tools for every attempt and include normalized plus raw
responses. Fix the max-turn boundary and aggregate total tokens and latency.

**Step 5: Run focused tests and type checks**

Run: `uv run pytest tests/unit/test_engine.py tests/integration/test_engine_trace.py -q`

Run: `uv run pyright src/benchtable/engine.py src/benchtable/games`

Expected: PASS and zero type errors.

### Task 4: Fix CLI dispatch, injection, and failure status

**Files:**
- Modify: `src/benchtable/cli.py`
- Modify: `tests/integration/test_cli.py`
- Modify: `README.md`
- Modify: `docs/plugin-authoring.md`

**Step 1: Write failing tests**

Inject a fake agent factory, configure two actors with distinct agents, pass
game-specific configuration, assert failed runs exit non-zero, and verify the
CLI test makes no provider request. Assert run configuration contains sanitized
agent metadata.

**Step 2: Run focused tests to verify failures**

Run: `uv run pytest tests/integration/test_cli.py -q`

Expected: the new injection, dispatch, and failure-status tests fail.

**Step 3: Implement CLI fixes**

Add injectable agent construction, pass the configured actor mapping and game
configuration into `RunEngine`, call the async engine through `asyncio.run`,
return a non-zero exit code for failed runs, and generate a unique run ID for
reused output directories.

**Step 4: Update documentation**

Document actor-ID agent mapping, game configuration passed to sessions, and
the async/provider trace and failure behavior without adding poker content.

**Step 5: Run focused tests**

Run: `uv run pytest tests/integration/test_cli.py -q`

Expected: PASS with no live provider calls.

### Task 5: Complete verification and review

**Files:**
- Review all changed files and the implementation/fix design plans.

**Step 1: Run the complete suite**

Run: `uv sync`

Run: `uv run pytest -q`

Run: `uv run ruff format --check .`

Run: `uv run ruff check .`

Run: `uv run pyright`

Run: `uv lock --check`

Expected: all commands pass.

**Step 2: Inspect the final diff**

Run: `git status --short`, `git diff --check`, and `git diff`.

Verify no secrets, generated artifacts, unintended files, or attribution are
present. Do not commit unless explicitly requested.
