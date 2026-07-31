# Poker Game Implementation Plan

> **Historical / superseded by the 2026-07-26 hand-session design. Do not execute
> this historical plan.** It is retained as historical task context only. Current
> semantics are that memory calls occur in separate model responses before the
> game-action response, mixed memory/action responses are rejected, and memory
> writes are memory-only and committed immediately. The paired, batched, and
> accompanying-write instructions below are obsolete.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a first-party configurable no-limit Texas Hold'em plugin with repeated hands, private per-agent memory, and complete offline traces.

**Architecture:** Keep poker rules inside a `benchtable.games.poker` plugin. Extend the generic engine with configuration-aware actor resolution and an engine-owned per-match memory interaction loop. A logical turn may contain memory-only tool calls, but must end with exactly one poker action; memory writes may accompany that action and are committed atomically with it.

**Tech Stack:** Python 3.12, Pydantic 2, standard-library card/deck logic, existing async run engine, OpenAI Chat Completions tool protocol, TOML configuration, pytest/pytest-asyncio, Ruff, Pyright, and append-only JSONL traces.

---

## Implementation Notes

- Work test-first. Each task starts with a failing test, then the smallest implementation, then focused verification.
- Do not add poker rules to `RunEngine`; it should only classify generic memory operations and delegate poker actions to `PokerSession`.
- Keep all private card, memory, and authoritative state behind plugin/session or memory-store APIs. Never add a raw-state field to `Observation`, `Transition`, or `GameResult`.
- Do not add a dependency. Card representation, shuffling, hand evaluation, and pot settlement use the standard library.
- Do not commit during implementation unless the user explicitly requests commits. Inspect status and diffs at every verification checkpoint.

### Task 1: Make Plugin Actors Configuration-Aware

**Files:**
- Modify: `src/benchtable/games/protocol.py`
- Modify: `src/benchtable/games/registry.py`
- Modify: `src/benchtable/cli.py`
- Modify: `tests/fixtures/tiny_game.py`
- Modify: `tests/contract/test_game_protocol.py`
- Modify: `tests/contract/test_tiny_game.py`
- Modify: `tests/unit/test_game_registry.py`
- Modify: `tests/integration/test_cli.py`

**Step 1: Write the failing tests**

Add coverage for a plugin that resolves actor IDs from `game_config`, and for
the CLI calling plugin configuration validation before constructing any agent.
Update the tiny fixture to expose the new configuration-aware contract while
keeping its existing static actors. Assert that malformed configuration and
unknown actor IDs fail before the fake agent factory is called.

**Step 2: Run the focused tests**

Run: `uv run pytest tests/contract/test_game_protocol.py tests/contract/test_tiny_game.py tests/unit/test_game_registry.py tests/integration/test_cli.py -q`

Expected: FAIL because the protocol and CLI only support the static
`player_ids` property and have no plugin configuration validation hook.

**Step 3: Implement the protocol change**

Replace static actor validation with a configuration-aware plugin API, for
example `player_ids(game_config)` and `validate_config(game_config)`. Make the
registry validate the default metadata path without requiring a game session.
Make the CLI validate the game configuration and resolve actor IDs before
constructing agents. Preserve direct plugin injection for tests.

**Step 4: Run the focused tests and type check**

Run: `uv run pytest tests/contract/test_game_protocol.py tests/contract/test_tiny_game.py tests/unit/test_game_registry.py tests/integration/test_cli.py -q`

Expected: PASS.

Run: `uv run pyright src/benchtable/games src/benchtable/cli.py tests/fixtures/tiny_game.py`

Expected: PASS with zero errors.

### Task 2: Add Run-Level Memory Configuration

**Files:**
- Modify: `src/benchtable/config.py`
- Modify: `src/benchtable/engine.py`
- Modify: `src/benchtable/cli.py`
- Modify: `tests/unit/test_config.py`
- Modify: `tests/unit/test_engine.py`
- Modify: `tests/integration/test_cli.py`

**Step 1: Write the failing tests**

Test that `max_memory_operations_per_turn` accepts a non-negative strict
integer, rejects booleans, strings, floats, and negative values, is recorded in
the run configuration, and is forwarded from the CLI to the engine. Assert the
default is finite and safe for a model that repeatedly calls memory tools.

**Step 2: Run the focused tests**

Run: `uv run pytest tests/unit/test_config.py tests/unit/test_engine.py tests/integration/test_cli.py -q`

Expected: FAIL because the run configuration and engine constructor do not
have a memory-operation budget.

**Step 3: Implement the smallest configuration change**

Add the validated run field, pass it through the CLI, include it in the
`run_config` event, and store it on `RunEngine`. Do not add poker-specific
fields to the generic run model; poker settings remain in `game_config`.

**Step 4: Run the focused tests**

Run: `uv run pytest tests/unit/test_config.py tests/unit/test_engine.py tests/integration/test_cli.py -q`

Expected: PASS.

### Task 3: Implement the Per-Match Memory Store

**Files:**
- Create: `src/benchtable/memory.py`
- Create: `tests/unit/test_memory.py`

**Step 1: Write the failing tests**

Cover the following public behavior:

- A note is stored under the current actor only.
- `read` returns notes in insertion order with hand and turn context.
- One actor cannot read another actor's notes.
- A new match store is empty even when another match used the same actor ID.
- Entry and character limits reject writes explicitly instead of silently
  truncating notes.
- Empty or whitespace-only notes are rejected.
- Reserved `read_memory` and `write_memory` tool specifications have strict
  JSON object schemas.

Use a small JSON-compatible entry model or dataclass; do not expose mutable
internal lists to callers.

**Step 2: Run the focused test**

Run: `uv run pytest tests/unit/test_memory.py -q`

Expected: FAIL because the memory module does not exist.

**Step 3: Implement the store and tool definitions**

Implement an engine-owned store keyed by actor ID. Store note text and stable
sequence/context fields, expose copy-safe reads, validate limits at the write
boundary, and provide the two reserved `ToolSpec` values. Keep memory state
separate from `GameSession` and avoid timestamps in normalized note ordering.

**Step 4: Run the focused test and lint**

Run: `uv run pytest tests/unit/test_memory.py -q`

Expected: PASS.

Run: `uv run ruff check src/benchtable/memory.py tests/unit/test_memory.py`

Expected: PASS.

### Task 4: Add Generic Memory Tool Interaction to the Engine

**Files:**
- Modify: `src/benchtable/engine.py`
- Modify: `src/benchtable/contracts.py`
- Modify: `src/benchtable/events.py` only if a serialization helper is needed
- Modify: `tests/fixtures/fake_agent.py`
- Modify: `tests/unit/test_engine.py`
- Modify: `tests/integration/test_engine_trace.py`

**Step 1: Write the failing tests**

Add deterministic fake-agent responses for:

- A `read_memory` call followed by one valid game action.
- A memory-only `write_memory` call followed by another model response and a
  valid game action.
- A valid game action with an accompanying `write_memory` call.
- A `read_memory` call incorrectly combined with a game action.
- Multiple game actions in one response.
- Memory-operation budget exhaustion.
- A memory write paired with an invalid poker action, asserting that no note is
  committed.

Assert that memory tool result messages use valid Chat Completions history,
that `max_turns` counts only accepted game actions, and that the agent receives
only its own memory results.

**Step 2: Run the focused tests**

Run: `uv run pytest tests/unit/test_engine.py tests/integration/test_engine_trace.py -q`

Expected: FAIL because the engine currently treats every tool call as a game
action and performs only one provider response per turn.

**Step 3: Implement the interaction loop**

Create a `MatchMemory` instance for each match. Add reserved memory tool specs
to the tools sent to the agent, partition normalized calls into memory calls and
game calls, and keep a logical turn open while memory-only calls are processed.
For each memory call, emit a `memory_operation` event and append the assistant
tool call plus a valid tool result to the next request.

Require exactly one game action for the final response. Permit writes with that
action, stage them, validate/apply the game action, then commit the staged notes
only after successful application. Reject a read combined with the final action
with a normal invalid-output retry. Keep provider retries and invalid-action
budgets separate from the memory-operation budget.

Preserve the existing normalized provider response, exact request messages and
tool schemas, latency, usage, and failure events. Use a reserved-name check so a
game cannot accidentally define a conflicting memory tool.

**Step 4: Run the focused tests and type check**

Run: `uv run pytest tests/unit/test_engine.py tests/integration/test_engine_trace.py -q`

Expected: PASS.

Run: `uv run pyright src/benchtable/engine.py src/benchtable/memory.py`

Expected: PASS with zero errors.

### Task 5: Add Card and Deck Primitives

**Files:**
- Create: `src/benchtable/games/poker/__init__.py`
- Create: `src/benchtable/games/poker/cards.py`
- Create: `tests/unit/games/test_poker_cards.py`

**Step 1: Write the failing tests**

Test a standard 52-card deck, unique cards, card equality/serialization,
deterministic seeded shuffles, drawing without replacement, and exhaustion
errors. Test that separate hand seeds produce different deterministic deals
without modifying a previous deck.

**Step 2: Run the focused test**

Run: `uv run pytest tests/unit/games/test_poker_cards.py -q`

Expected: FAIL because the poker package and card primitives do not exist.

**Step 3: Implement the card primitives**

Use standard-library enums or immutable values for suits and ranks. Implement a
fresh deck per hand, seeded shuffling, draw operations, and JSON-safe display
strings. Keep deck internals private to the poker package.

**Step 4: Run the focused test and type check**

Run: `uv run pytest tests/unit/games/test_poker_cards.py -q`

Expected: PASS.

Run: `uv run pyright src/benchtable/games/poker/cards.py`

Expected: PASS with zero errors.

### Task 6: Implement Hold'em Hand Evaluation

**Files:**
- Create: `src/benchtable/games/poker/evaluator.py`
- Create: `tests/unit/games/test_poker_evaluator.py`

**Step 1: Write the failing tests**

Cover all standard categories from high card through straight flush, ace-low
straights, duplicate-rank and duplicate-suit edge cases, comparison of equal
categories by kickers, seven-card best-five selection, and exact ties.

Use known fixed card sets rather than random tests so failures identify the
ranking rule that is wrong.

**Step 2: Run the focused test**

Run: `uv run pytest tests/unit/games/test_poker_evaluator.py -q`

Expected: FAIL because the evaluator does not exist.

**Step 3: Implement the evaluator**

Represent a hand rank as a comparable tuple containing category and tie-breaker
values. Evaluate every five-card subset of a seven-card holding and return the
best rank plus a display category. Do not include private cards in any trace
payload from this module.

**Step 4: Run the focused test**

Run: `uv run pytest tests/unit/games/test_poker_evaluator.py -q`

Expected: PASS.

### Task 7: Implement Pot and Settlement Logic

**Files:**
- Create: `src/benchtable/games/poker/pots.py`
- Create: `tests/unit/games/test_poker_pots.py`

**Step 1: Write the failing tests**

Test contributions from unequal stacks, folded players remaining in pot
contributions, main and side-pot construction, all-in eligibility, tied pot
splits, and deterministic odd-chip distribution by seat order.

**Step 2: Run the focused test**

Run: `uv run pytest tests/unit/games/test_poker_pots.py -q`

Expected: FAIL because pot construction and settlement do not exist.

**Step 3: Implement pot construction and settlement**

Build pots from contribution levels, attach eligible non-folded players, and
settle each pot using evaluator ranks. Return explicit public settlement data
and avoid mutating caller-owned contribution structures.

**Step 4: Run the focused test and lint**

Run: `uv run pytest tests/unit/games/test_poker_pots.py -q`

Expected: PASS.

Run: `uv run ruff check src/benchtable/games/poker/pots.py`

Expected: PASS.

### Task 8: Implement the Poker Session and Plugin

**Files:**
- Create: `src/benchtable/games/poker/session.py`
- Modify: `src/benchtable/games/poker/__init__.py`
- Create: `tests/contract/test_poker_game.py`
- Create: `tests/unit/games/test_poker_session.py`

**Step 1: Write the failing contract and session tests**

Cover:

- `PokerGame` metadata and configuration-aware player IDs.
- Validation of player count, duplicate IDs, stacks, blind relationships,
  hand count, memory limits, invalid-turn policy, and deck capacity.
- Fresh deterministic deals for successive hands.
- Dealer/blind rotation, including heads-up behavior.
- Public observation containing only the current actor's hole cards and public
  table information.
- No other player's hole cards, folded cards, raw state, or memory in an
  observation.
- Legal poker action tool schema and valid action transitions.
- Turn and street progression, fold termination, showdown settlement, and
  repeated-hand stack persistence.
- `handle_failed_turn` forced fold behavior and failure-policy selection.
- Terminal `GameResult` metrics and public hand summaries.

**Step 2: Run the focused tests**

Run: `uv run pytest tests/contract/test_poker_game.py tests/unit/games/test_poker_session.py -q`

Expected: FAIL because the plugin and session are not implemented.

**Step 3: Implement the session state machine**

Add a typed poker configuration validator and a private session state machine.
Initialize a fresh deck and deal each hand, determine the next actor, generate
actor-specific observations and legal tools, validate the single poker action,
advance betting streets, reveal community cards, settle folds/showdowns, and
start the next hand until the match ends.

Keep public summaries separate from hidden state. Render observations as text
plus JSON-compatible metadata; do not add raw state to the generic contracts.
Use the existing `Transition`, `GameResult`, and `InvalidActionError` types.

Implement `PokerGame` as the plugin factory. It must expose the configured
player list through the new actor-resolution method, validate configuration
without a provider call, build actor-specific system prompts, and create a
session from a match seed and copied game configuration.

**Step 4: Run focused tests and type checks**

Run: `uv run pytest tests/contract/test_poker_game.py tests/unit/games/test_poker_session.py -q`

Expected: PASS.

Run: `uv run pyright src/benchtable/games/poker`

Expected: PASS with zero errors.

### Task 9: Register Poker and Wire CLI Configuration

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/benchtable/cli.py`
- Modify: `tests/unit/test_game_registry.py`
- Modify: `tests/integration/test_cli.py`
- Modify: `README.md`
- Modify: `docs/plugin-authoring.md`

**Step 1: Write the failing tests**

Add tests that:

- The installed entry point resolves `poker` and reports its version.
- A poker TOML configuration with arbitrary configured player IDs validates.
- Missing, extra, and duplicate player IDs are rejected before fake-agent
  construction.
- The CLI forwards poker game configuration and memory budget to the engine.
- `list-games` includes poker while direct registry injection continues to work.
- README configuration examples use the new repeated-hand and memory fields.

**Step 2: Run the focused tests**

Run: `uv run pytest tests/unit/test_game_registry.py tests/integration/test_cli.py -q`

Expected: FAIL because the entry point and poker-specific CLI path are absent.

**Step 3: Register and document the plugin**

Add the `poker` entry point in `pyproject.toml`. Keep provider construction
unchanged and pass validated generic memory settings plus `game_config` to the
engine. Update plugin authoring documentation with configuration-aware actor
IDs, reserved memory tools, and strict observation requirements. Update the
README with a short poker run example and explain run/match/hand terminology.

Do not add provider keys or live-provider test paths.

**Step 4: Run the focused tests and command checks**

Run: `uv run pytest tests/unit/test_game_registry.py tests/integration/test_cli.py -q`

Expected: PASS.

Run: `uv run benchtable list-games`

Expected: output contains `poker` and its plugin version.

### Task 10: Add Offline Multi-Hand Poker Trace Coverage

**Files:**
- Modify: `tests/fixtures/fake_agent.py`
- Create: `tests/integration/test_poker_trace.py`
- Modify: `tests/integration/test_engine_trace.py`
- Modify: `tests/unit/test_events.py` only if memory-specific redaction coverage
  is missing

**Step 1: Write the failing integration tests**

Use fake agents with deterministic sequences that write a note about a rival,
read it on a later hand, and take valid poker actions. Run at least three
players over multiple hands and multiple independent matches. Assert:

- New cards are dealt per hand while notes reset between matches.
- Each agent receives only its own memory and private cards.
- The trace contains run, match, hand, memory, request/response, validation,
  transition, hand-end, match-end, and run-end events in sequence.
- Forced folds and fail-match policies produce explicit outcomes.
- Memory contents are redacted by the existing event writer if they contain
  credential-shaped values.
- Normalized golden event payloads are stable after removing volatile fields.

**Step 2: Run the focused integration tests**

Run: `uv run pytest tests/integration/test_poker_trace.py tests/integration/test_engine_trace.py -q`

Expected: FAIL until the complete plugin, engine loop, and trace integration
are wired together.

**Step 3: Fix only integration defects**

Adjust event context, fake-agent response sequences, observation rendering, or
tool-message history only where the focused tests identify a contract mismatch.
Do not weaken isolation assertions or remove private data checks to make a
trace pass.

**Step 4: Run the focused tests**

Run: `uv run pytest tests/integration/test_poker_trace.py tests/integration/test_engine_trace.py -q`

Expected: PASS.

### Task 11: Complete Documentation and Verification

**Files:**
- Review: `docs/plans/2026-07-18-poker-game-design.md`
- Review: all files changed by Tasks 1-10

**Step 1: Run the complete test suite**

Run: `uv run pytest -q`

Expected: all tests pass.

**Step 2: Run formatting and lint checks**

Run: `uv run ruff format --check .`

Expected: no formatting changes required.

Run: `uv run ruff check .`

Expected: no lint errors.

**Step 3: Run type and lockfile checks**

Run: `uv run pyright`

Expected: zero type errors.

Run: `uv lock --check`

Expected: lockfile is current. No dependency or unrelated lockfile changes
should be introduced.

**Step 4: Inspect the final worktree**

Run: `git status --short`, `git diff --check`, and `git diff -- docs/plans/2026-07-18-poker-game-design.md docs/plans/2026-07-18-poker-game.md`

Expected: only intended source, test, and documentation changes are present;
no secrets, generated run artifacts, or attribution markers are present.

## Final Acceptance Criteria

- `poker` is discoverable through the existing plugin registry.
- Configured agents can play a configurable number of hands at one table.
- Hands use fresh deterministic deals and standard Hold'em settlement.
- Stacks persist within a match; notes persist within a match and reset between
  matches.
- Agents explicitly write and read private free-form notes.
- Memory operations do not advance poker turns and cannot leak across actors.
- The engine requires exactly one poker action per accepted logical turn.
- Invalid output, provider failures, memory-loop exhaustion, cancellation, and
  configuration errors have explicit outcomes and trace events.
- Full pytest, Ruff, Pyright, and lockfile verification passes.
