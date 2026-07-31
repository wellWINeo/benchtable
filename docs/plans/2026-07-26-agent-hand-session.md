# Hand-Scoped Agent Sessions and Poker Reliability Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the first-party poker plugin and the generic hand-scoped agent session behavior described in `2026-07-26-agent-hand-session-design.md`, including strict privacy, memory, failure, and trace contracts.

**Architecture:** Keep poker state and rules in `PokerSession`. Keep actor dispatch, conversation scopes, memory calls, retries, finalization, failure outcomes, and event sequencing in `RunEngine`. Keep the provider adapter stateless and use `EventWriter` as the final normalization and credential-redaction boundary.

**Tech Stack:** Python 3.12, `uv`, Pydantic 2, standard-library poker logic, the asynchronous OpenAI-compatible SDK, Chat Completions tool messages, Typer, TOML, `pytest`/`pytest-asyncio`, Ruff, Pyright, and append-only JSONL traces.

---

## Implementation Rules

- Read the canonical design before changing code. Older 2026-07-18, 2026-07-19,
  and 2026-07-24 plans are historical context, not additional requirements.
- Work test-first. For each behavior, write a focused failing test, run it to
  confirm the expected failure, implement the smallest change, and rerun it.
- Do not move poker rules into `RunEngine` or add provider-specific branches.
- Keep observations actor-specific and never add raw game state to generic
  contracts, traces sent to another actor, or tool results.
- Memory calls are separate responses before the game-action response. Memory
  writes commit immediately and cannot be paired with an action.
- Preserve valid Chat Completions history, including tool-call IDs, on retries
  and invalid actions. A failed action gets exactly one matching validation
  tool result.
- Preserve unrelated worktree changes until the final explicit all-changes
  commit requested by the user.
- Do not add dependencies, a database, concurrency, UI, hosted API, or hidden
  chain-of-thought capture.

## Task 1: Establish Strict Contracts and Plugin Capabilities

**Files:**
- Modify: `src/benchtable/contracts.py`
- Modify: `src/benchtable/games/protocol.py`
- Modify: `src/benchtable/games/registry.py`
- Modify: `src/benchtable/errors.py`
- Test: `tests/unit/test_contracts.py`
- Test: `tests/contract/test_game_protocol.py`
- Test: `tests/unit/test_game_registry.py`

### Step 1: Write failing tests

Cover strict JSON-compatible models with rejected unknown fields for:

- Conversation scope identifiers.
- Actor-targeted plugin memory summaries with strict hand and turn context.
- Plugin event type/payload contracts.
- Configuration-aware actor IDs and optional exact-agent requirements.
- Legacy sessions and plugins that omit optional capabilities.
- Duplicate discovered plugins, missing entry points, and explicit lookup
  failures.

Run:

```bash
uv run pytest tests/unit/test_contracts.py tests/contract/test_game_protocol.py tests/unit/test_game_registry.py -q
```

Expected: FAIL because the optional capabilities and strict contracts are not
fully represented.

### Step 2: Implement the smallest contract changes

Add typed optional contracts and protocol definitions that remain discoverable
through `getattr`. Add explicit errors for malformed configuration, plugin
lookup, malformed model output, invalid actions, provider failures, and failed
runs. Keep provider SDK types and raw state out of public contracts.

### Step 3: Verify

```bash
uv run pytest tests/unit/test_contracts.py tests/contract/test_game_protocol.py tests/unit/test_game_registry.py -q
uv run pyright src/benchtable/contracts.py src/benchtable/games src/benchtable/errors.py
```

Expected: PASS with zero type errors.

## Task 2: Harden Configuration, Provider Normalization, and CLI Preflight

**Files:**
- Modify: `src/benchtable/config.py`
- Modify: `src/benchtable/agents/openai_compatible.py`
- Modify: `src/benchtable/cli.py`
- Test: `tests/unit/test_config.py`
- Test: `tests/unit/test_openai_compatible_agent.py`
- Test: `tests/integration/test_cli.py`

### Step 1: Write failing tests

Cover strict run and poker configuration values, generation limits, unknown
fields, environment-only API keys, disabled SDK-internal retries, optional
generation settings, malformed/non-object tool arguments, raw provider response
retention, exact actor mapping, game configuration forwarding, offline agent
factory injection, unique output run IDs, and non-zero failed-run exit status.

Run:

```bash
uv run pytest tests/unit/test_config.py tests/unit/test_openai_compatible_agent.py tests/integration/test_cli.py -q
```

Expected: FAIL for missing strict checks, adapter controls, and CLI preflight.

### Step 2: Implement boundary behavior

Use `tomllib` and strict Pydantic models. Validate game configuration and actor
IDs before constructing agents or making provider requests. Pass validated
`game_config`, actor-to-agent mapping, memory budget, and optional generation
limits into the engine. Construct the asynchronous OpenAI-compatible client
with configured base URL and `max_retries=0`; read the key only from the named
environment variable and never include it in errors or traces.

Normalize only JSON objects as parsed tool arguments. Preserve malformed raw
text and normalized parse errors. Keep registry and agent construction
injectable for offline CLI tests.

### Step 3: Verify

```bash
uv run pytest tests/unit/test_config.py tests/unit/test_openai_compatible_agent.py tests/integration/test_cli.py -q
uv run pyright src/benchtable/config.py src/benchtable/agents/openai_compatible.py src/benchtable/cli.py
```

Expected: PASS with zero type errors and no live provider request.

## Task 3: Implement Correct Poker State and Settlement

**Files:**
- Create or modify: `src/benchtable/games/poker/config.py`
- Create or modify: `src/benchtable/games/poker/cards.py`
- Create or modify: `src/benchtable/games/poker/evaluator.py`
- Create or modify: `src/benchtable/games/poker/pots.py`
- Modify: `src/benchtable/games/poker/session.py`
- Test: `tests/unit/games/test_poker_config.py`
- Test: `tests/unit/games/test_poker_cards.py`
- Test: `tests/unit/games/test_poker_evaluator.py`
- Test: `tests/unit/games/test_poker_pots.py`
- Test: `tests/unit/games/test_poker_session.py`
- Test: `tests/contract/test_poker_game.py`

### Step 1: Write failing tests

Cover:

- Required non-empty unique players, strict integers, blind/stack/deck limits,
  supported invalid-turn policy, and no coercion.
- Immutable cards, unique standard deck, unambiguous serialization, seeded
  shuffle, draw exhaustion, and round-robin hole-card dealing.
- All Hold'em hand categories, ace-low straights, kickers, seven-card selection,
  and exact ties.
- Heads-up and multi-player dealer/blind rotation, active-seat elimination,
  all-in runouts, street progression, fresh per-hand state, and termination.
- Fold, check, call, bet, raise, all-in, target commitments, minimum raises,
  short all-ins, and illegal action/amount rejection.
- Folded contributions, uncalled excess, main/side pots, eligibility, ties,
  odd chips, input immutability, and payout conservation.

Run the smallest relevant group first, for example:

```bash
uv run pytest tests/unit/games/test_poker_config.py tests/unit/games/test_poker_session.py -q
```

Expected: FAIL against incomplete or incorrect state behavior.

### Step 2: Implement poker-owned behavior

Create a strict `PokerConfig` consumed unchanged by `PokerGame.validate_config`
and `create_session`. Use immutable card values and a fresh seeded deck for each
hand. Track persistent match state separately from hand-local contributions,
commitments, pending actors, board, and settlement. Skip all-in seats and run
out the board when no actionable players remain.

Centralize legal-action calculation and use it for observations and validation.
Build pots from all contribution levels, return unmatched excess, determine
eligibility by level, distribute tied pots and odd chips in configured seat
order, assert conservation, and apply payouts exactly once.

### Step 3: Verify

```bash
uv run pytest tests/unit/games tests/contract/test_poker_game.py -q
uv run pyright src/benchtable/games/poker
```

Expected: PASS with zero type errors and no stuck all-in session.

## Task 4: Add Isolated Match Memory and System Summaries

**Files:**
- Modify: `src/benchtable/memory.py`
- Modify: `src/benchtable/games/protocol.py`
- Modify: `tests/unit/test_memory.py`
- Modify: `tests/unit/test_engine_memory.py`

### Step 1: Write failing tests

Cover actor-private notes, chronological reads, empty/whitespace rejection,
entry/character limits, new-match reset, multiple memory calls, accurate hand
and in-hand turn context, and a separate system-summary channel. Assert that
system summaries are actor-targeted, source-marked, chronologically merged,
and do not consume agent-note limits.

Run:

```bash
uv run pytest tests/unit/test_memory.py tests/unit/test_engine_memory.py -q
```

Expected: FAIL for summary channels, limits, context, or isolation gaps.

### Step 2: Implement memory behavior

Keep memory keyed by match and actor. Commit successful memory-only writes
immediately. Add copy-safe system-summary storage and combined reads with
`agent`/`system` source markers. Validate all public input at the boundary and
never expose mutable internal collections.

### Step 3: Verify

```bash
uv run pytest tests/unit/test_memory.py tests/unit/test_engine_memory.py -q
uv run ruff check src/benchtable/memory.py tests/unit/test_memory.py tests/unit/test_engine_memory.py
```

Expected: PASS.

## Task 5: Implement Actor-Scoped Transcripts and the Logical Turn Loop

**Files:**
- Modify: `src/benchtable/engine.py`
- Modify: `tests/fixtures/fake_agent.py`
- Test: `tests/unit/test_engine.py`
- Test: `tests/unit/test_engine_memory.py`
- Test: `tests/integration/test_engine_trace.py`

### Step 1: Write failing tests

Use deterministic fake agents to assert:

- One actor receives its prior hand messages on later turns.
- Another actor never receives that transcript or memory.
- A scope change clears all transcripts; a legacy session retains a match
  transcript.
- Memory-only calls are processed in order and do not advance game turns.
- Mixed memory/action responses are rejected before action validation.
- Exactly one game action is accepted per logical turn.
- Every accepted action receives one public tool result and exactly one
  tool-free finalization response.
- Empty finalization requires a finish reason; finalization tool calls fail
  immediately without retry.
- Invalid action history contains exactly one matching validation tool result
  before retry, recovery, next observation, or next request.
- The exact accumulated messages appear in each model-request trace event.

Run:

```bash
uv run pytest tests/unit/test_engine.py tests/unit/test_engine_memory.py tests/integration/test_engine_trace.py -q
```

Expected: FAIL until transcript persistence and explicit protocol phases exist.

### Step 2: Implement the engine phases

Maintain per-match actor transcript lists and a current generic scope ID. Before
each turn, resolve the optional session scope, clear all transcripts on change,
and append only the current actor's observation. Persist assistant tool calls,
tool results, and final text after every provider response.

Process memory-only calls in response order within their separate budget. Reject
mixed responses. Require one recognized game action, append its tool call before
application, append a public transition result after success, and make one
tool-free finalization request. Validate observation actor IDs before the first
provider request. Preserve tool IDs and valid history for invalid-action retries.

Keep provider retries, invalid-output retries, memory limits, accepted-turn
limits, cancellation, and engine failures as separate outcomes and metrics.

### Step 3: Verify

```bash
uv run pytest tests/unit/test_engine.py tests/unit/test_engine_memory.py tests/integration/test_engine_trace.py -q
uv run pyright src/benchtable/engine.py
```

Expected: PASS with zero type errors.

## Task 6: Integrate Poker Conversation Scope, Summaries, and Events

**Files:**
- Modify: `src/benchtable/games/poker/session.py`
- Modify: `src/benchtable/engine.py`
- Modify: `src/benchtable/contracts.py`
- Test: `tests/unit/games/test_poker_session.py`
- Test: `tests/contract/test_poker_game.py`
- Test: `tests/integration/test_poker_trace.py`

### Step 1: Write failing integration tests

Assert that the poker scope is stable within a hand and changes for the next
hand, observations exclude previous-hand summaries and all private data, and
one compact public summary is queued per actor at hand completion. Assert that
forced-fold and terminal recovery paths drain summaries before the next request
or match completion.

Assert `hand_start` and `hand_end` events contain public hand index, derived
seed, dealer/blinds, board, settlement, finish information, and correct
per-hand payouts. Assert terminal results contain final stacks, deltas, hand
count, winners, finish reason, completion status, and recovery/failure metrics.

Run:

```bash
uv run pytest tests/unit/games/test_poker_session.py tests/contract/test_poker_game.py tests/integration/test_poker_trace.py -q
```

Expected: FAIL until scope, summary, and lifecycle events are integrated.

### Step 2: Implement plugin capabilities and event draining

Expose the current hand as the generic conversation scope. Remove previous-hand
summary text from poker observations. Queue compact public summaries containing
only allowed data and drain them from the engine after successful or recovery
transitions, before the next actor/provider request, and before terminal match
completion. Emit plugin events with generic match/turn/actor context without
inspecting poker state in the engine.

### Step 3: Verify

```bash
uv run pytest tests/unit/games/test_poker_session.py tests/contract/test_poker_game.py tests/integration/test_poker_trace.py -q
uv run pyright src/benchtable/games/poker/session.py src/benchtable/engine.py
```

Expected: PASS with zero type errors.

## Task 7: Complete Event Redaction, Documentation, and Offline Trace Coverage

**Files:**
- Modify: `src/benchtable/events.py`
- Modify: `README.md`
- Modify: `docs/plugin-authoring.md`
- Modify: `tests/unit/test_events.py`
- Modify: `tests/integration/test_cli.py`
- Modify: `tests/integration/test_poker_trace.py`

### Step 1: Write failing redaction and trace tests

Cover append/sequence recovery, malformed existing JSONL, immediate flush,
schema versioning, recursive credential keys, nested headers, free-form
`Authorization: value`, `password=value`, `api_key` text, escaped keys, raw
provider response retention, and preservation of usage/correlation fields.

Add offline multi-player, multi-hand, multi-match coverage for actor mapping,
memory reset and persistence, hand events, system summaries, finalization,
forced folds, failed matches, deterministic seeds, seat-ordered payouts, and
normalized event comparisons with volatile fields removed.

Run:

```bash
uv run pytest tests/unit/test_events.py tests/integration/test_cli.py tests/integration/test_poker_trace.py -q
```

Expected: FAIL for any incomplete redaction or lifecycle trace behavior.

### Step 2: Implement boundary and documentation updates

Make `EventWriter` append only valid prefixes, assign sequences, normalize
Pydantic/provider values, redact recursively and in credential-shaped strings,
flush every event, and reject unsupported payloads. Update README and plugin
authoring docs with strict configuration, exact agent mapping, hand-scoped
transcripts, separate memory calls, source-marked summaries, finalization,
observation isolation, failure behavior, and trace guarantees. Do not document
provider-owned sessions or shared transcripts.

### Step 3: Verify

```bash
uv run pytest tests/unit/test_events.py tests/integration/test_cli.py tests/integration/test_poker_trace.py -q
uv run ruff format --check src tests
uv run ruff check src tests
```

Expected: PASS with no formatting or lint errors.

## Task 8: Run Release Verification and Inspect the Feature Commit

**Files:**
- Review: all files changed by Tasks 1-7
- Review: `docs/plans/2026-07-26-agent-hand-session-design.md`
- Review: `docs/plans/2026-07-26-agent-hand-session.md`

### Step 1: Run the complete verification suite

```bash
uv sync
uv run pytest -q
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv lock --check
git diff --check
```

Expected: all commands pass.

### Step 2: Inspect scope and secrets

Review:

```bash
git status --short
git diff --stat
git diff --name-only
```

Confirm that all current implementation, test, README, plugin documentation,
and the two canonical plans are intentional; the redundant
`2026-07-26-agent-hand-session-fixes.md` plan is gone; the earlier infrastructure
fix plans remain as historical context; and no generated traces, credentials,
or attribution markers are present.

### Step 3: Commit the completed feature

Stage explicit intended paths only, including all current feature changes and
the two canonical plans. Do not stage unrelated secrets or generated artifacts.

```bash
git add README.md docs/plugin-authoring.md docs/plans/2026-07-18-poker-game-design.md docs/plans/2026-07-18-poker-game.md docs/plans/2026-07-19-poker-game-repair-design.md docs/plans/2026-07-19-poker-game-repair.md docs/plans/2026-07-24-poker-game-fix-design.md docs/plans/2026-07-24-poker-game-fix.md docs/plans/2026-07-26-agent-hand-session-design.md docs/plans/2026-07-26-agent-hand-session.md pyproject.toml src tests
git diff --cached --check
git commit -m "feat: add hand-scoped poker agent sessions"
```

Expected: one feature commit containing the complete implementation and the
consolidated design/implementation plans.

## Final Acceptance Criteria

- Strict poker configuration and exact agent mapping fail before provider use.
- Poker dealing, betting, all-ins, rotation, elimination, settlement, results,
  and hand events are correct and deterministic.
- Actor transcripts are private and reset on hand scope changes; legacy plugins
  retain match scope.
- Memory is private, bounded, source-marked, immediately committed for
  memory-only calls, and available across hands without leaking private data.
- Exactly one game action and one tool-free finalization follow each logical
  accepted turn.
- Invalid actions leave one matching validation result in history, and failed
  runs cannot become successful outcomes.
- Traces are append-safe, flushed, ordered, complete, and credential-redacted.
- The complete verification suite passes before merge.
