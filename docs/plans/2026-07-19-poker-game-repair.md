# Poker Game Repair Implementation Plan

> **Historical / superseded by the 2026-07-26 hand-session design. Do not execute
> this historical plan.** It is retained as historical task context only. Current
> semantics are that memory calls occur in separate model responses before the
> game-action response, mixed memory/action responses are rejected, and memory
> writes are memory-only and committed immediately. The paired, batched, and
> accompanying-write instructions below are obsolete.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Correct the first-party no-limit Texas Hold'em plugin, generic memory loop, failure semantics, and trace contracts identified in the full implementation review.

**Architecture:** Rebuild the poker state machine around explicit active seats, per-hand contributions, legal action state, and conservation-checked settlement while preserving existing plugin and engine contracts. Add small generic optional capabilities for exact actor mapping, memory limits, turn context, and plugin trace events; keep poker rules inside `PokerSession`.

**Tech Stack:** Python 3.12, Pydantic 2, standard-library card/deck logic, existing async engine, OpenAI Chat Completions tool protocol, pytest/pytest-asyncio, Ruff, Pyright, and append-only JSONL traces.

---

## Task 1: Add Strict Poker Configuration and Exact Agent Mapping

**Files:**
- Create: `src/benchtable/games/poker/config.py`
- Modify: `src/benchtable/games/poker/session.py`
- Modify: `src/benchtable/games/protocol.py`
- Modify: `src/benchtable/games/registry.py`
- Modify: `src/benchtable/cli.py`
- Test: `tests/unit/games/test_poker_config.py`
- Modify: `tests/unit/test_game_registry.py`
- Modify: `tests/integration/test_cli.py`

**Step 1: Write failing configuration tests**

Cover required `players`, non-empty unique IDs, at least two players, strict
integer fields, positive stacks/blinds/hand counts, blind relationships,
stack coverage, non-negative memory limits, supported invalid-turn policies,
and the standard-deck capacity boundary `2 * player_count + 8 <= 52`.

Assert that missing, boolean, string, float, zero, negative, duplicate, and
over-capacity values fail before any agent factory call. Assert that a valid
configuration produces a typed value object without coercion.

**Step 2: Run the focused tests and confirm failure**

Run: `uv run pytest tests/unit/games/test_poker_config.py tests/integration/test_cli.py -q`

Expected: FAIL because poker-specific strict configuration and exact poker
agent mapping are incomplete.

**Step 3: Implement the typed configuration boundary**

Create a Pydantic `PokerConfig` with `extra="forbid"`, strict player strings,
strict integer fields, a literal invalid-turn policy, and model validation for
blind, stack, and deck constraints. Make `PokerGame.validate_config()` call the
model and make `PokerGame.create_session()` consume the validated model rather
than `int()` coercions or fallback defaults.

Add an optional plugin capability for `requires_exact_agent_ids`. The registry
must default this capability to false for legacy plugins; `PokerGame` returns
true. The CLI must compare the configured agent ID set with the poker player
set whenever the capability is true, including the single-agent case.

**Step 4: Run focused tests and type checks**

Run: `uv run pytest tests/unit/games/test_poker_config.py tests/unit/test_game_registry.py tests/integration/test_cli.py -q`

Expected: PASS, with malformed poker configuration and mismatched agent IDs
rejected before agent construction.

Run: `uv run pyright src/benchtable/games/poker/config.py src/benchtable/games/poker/session.py src/benchtable/games/registry.py src/benchtable/cli.py`

Expected: zero errors.

## Task 2: Make Cards and Decks Immutable and Fix Dealing

**Files:**
- Modify: `src/benchtable/games/poker/cards.py`
- Modify: `src/benchtable/games/poker/session.py`
- Modify: `tests/unit/games/test_poker_cards.py`
- Modify: `tests/unit/games/test_poker_session.py`

**Step 1: Write failing card and dealing tests**

Assert that card rank and suit cannot be reassigned, the exported standard deck
cannot mutate shared card state, and each new `Deck` still contains exactly 52
unique cards. For a fixed seed, assert that hole cards are dealt one card per
active seat for two rounds, not as two consecutive cards per player. Assert
burn-card and board order after flop, turn, and river.

**Step 2: Run the focused tests and confirm failure**

Run: `uv run pytest tests/unit/games/test_poker_cards.py tests/unit/games/test_poker_session.py -q`

Expected: FAIL on mutable cards and non-round-robin dealing.

**Step 3: Implement immutable card/deck values and round-robin dealing**

Use a frozen value representation or guarded attribute assignment for `Card`.
Keep the standard deck private or immutable and ensure every `Deck` copies a
stable card sequence. Deal one card to each active seat per round, then burn
and deal the flop, turn, and river through the existing seeded deck.

**Step 4: Run focused tests**

Run: `uv run pytest tests/unit/games/test_poker_cards.py tests/unit/games/test_poker_session.py -q`

Expected: PASS.

## Task 3: Rebuild Active Seats, Blinds, and Hand Lifecycle

**Files:**
- Modify: `src/benchtable/games/poker/session.py`
- Modify: `tests/unit/games/test_poker_session.py`
- Modify: `tests/contract/test_poker_game.py`

**Step 1: Write failing lifecycle tests**

Cover heads-up dealer/small-blind alternation, multi-player dealer/blind
rotation over several hands, correct preflop and postflop first actors,
zero-stack elimination, hand-count termination, elimination termination, and
fresh per-hand deals. Add a fixed all-in test that reaches showdown without
requesting an action from an all-in player.

**Step 2: Run focused tests and confirm failure**

Run: `uv run pytest tests/unit/games/test_poker_session.py tests/contract/test_poker_game.py -q`

Expected: FAIL on heads-up rotation, busted-player lifecycle, or all-in runout.

**Step 3: Implement explicit active-seat lifecycle**

Track the dealer by player ID or active-seat position and derive blinds from the
current active order. Reset hand-local state at each hand. Mark a player
inactive after settlement when their stack is zero, exclude inactive players
from dealing/blinds/actions, and terminate when fewer than two remain.

After every action, detect whether no non-folded player can act. Deal remaining
streets and settle immediately when all remaining players are all-in. Select
the first actor according to Hold'em preflop and postflop rules.

**Step 4: Run focused tests**

Run: `uv run pytest tests/unit/games/test_poker_session.py tests/contract/test_poker_game.py -q`

Expected: PASS with no stuck all-in sessions and correct seat rotation.

## Task 4: Correct Betting Actions and Street State

**Files:**
- Modify: `src/benchtable/games/poker/session.py`
- Modify: `tests/unit/games/test_poker_session.py`

**Step 1: Write failing action-matrix tests**

Cover fold, check, call, bet, raise, and all-in while facing and not facing a
bet. Amounts must represent target total commitment. Assert rejection of
missing/non-integer/negative amounts, target commitments above stack unless the
action is explicit all-in, betting while facing a bet, below-minimum raises,
and illegal raises after a short all-in. Cover short all-in calls and street
reset behavior.

**Step 2: Run focused tests and confirm failure**

Run: `uv run pytest tests/unit/games/test_poker_session.py -q`

Expected: FAIL on at least the current amount semantics, legal action list, or
minimum-raise assertions.

**Step 3: Implement centralized legal-action and commitment validation**

Compute `to_call`, available stack, current street bet, minimum raise size, and
whether a bet/raise is open from one state helper. Apply target commitments as
`target - current_commitment`, never as an unbounded increment. Reject or
explicitly classify short all-ins according to no-limit rules. Reset
street-local commitments, current bet, acted set, and minimum raise at every
street while preserving total hand contributions.

**Step 4: Run focused tests**

Run: `uv run pytest tests/unit/games/test_poker_session.py -q`

Expected: PASS for every action/street case.

## Task 5: Rebuild Pot Construction and Conservation-Checked Settlement

**Files:**
- Modify: `src/benchtable/games/poker/pots.py`
- Modify: `src/benchtable/games/poker/session.py`
- Modify: `tests/unit/games/test_poker_pots.py`
- Modify: `tests/unit/games/test_poker_session.py`

**Step 1: Write failing pot-conservation tests**

Cover current-hand contributions independent of prior hands, folded dead money,
uncalled excess returns, main and side pots, all-in eligibility, tied main and
side pots, explicit seat-order odd-chip distribution, and caller-owned input
immutability. Assert per-player payouts and `sum(payouts) == sum(pot amounts)`
for every case.

**Step 2: Run focused tests and confirm failure**

Run: `uv run pytest tests/unit/games/test_poker_pots.py tests/unit/games/test_poker_session.py -q`

Expected: FAIL on tie overpayment, side-pot edge cases, or multi-hand
contribution accounting.

**Step 3: Implement explicit payout settlement**

Track total contribution per player for the current hand. Build pots from
contribution levels and folded status, return uncalled excess where required,
and pass configured seat order explicitly to settlement. Return one payout
mapping per pot, distribute remainder chips once, reject impossible payout
totals, and apply payouts to stacks exactly once.

**Step 4: Run focused tests**

Run: `uv run pytest tests/unit/games/test_poker_pots.py tests/unit/games/test_poker_session.py -q`

Expected: PASS with chip conservation in every settlement.

## Task 6: Complete Poker Results, Observations, and Plugin Events

**Files:**
- Modify: `src/benchtable/contracts.py`
- Modify: `src/benchtable/games/protocol.py`
- Modify: `src/benchtable/games/poker/session.py`
- Modify: `src/benchtable/engine.py`
- Modify: `tests/contract/test_poker_game.py`
- Modify: `tests/unit/games/test_poker_session.py`
- Modify: `tests/integration/test_poker_trace.py`

**Step 1: Write failing result, observation, and event tests**

Assert that observations include public action history and exclude every other
player's hole/folded cards. Assert that poker emits `hand_start` and
`hand_end` events with hand seed, dealer/blinds, board, settlement, and public
finish data. Assert that all summaries preserve configured seat order.

Assert terminal results include final stacks, deltas, hand count, per-player
wins, finish reason, forced-recovery count, failure count, and completed/failed
status.

**Step 2: Run focused tests and confirm failure**

Run: `uv run pytest tests/contract/test_poker_game.py tests/unit/games/test_poker_session.py tests/integration/test_poker_trace.py -q`

Expected: FAIL because hand events, public history, and complete metrics are
missing.

**Step 3: Implement game-owned public context and event draining**

Add a small typed plugin-event contract with event type and JSON payload. Give
sessions an optional event-drain capability that the engine emits with match,
turn, and actor context. Queue hand-start events on creation and hand-end/start
events around settlement. Add public action history to observations and keep
private state out of event payloads except the existing actor-specific
observation event.

Record derived hand seeds, seat-ordered winner lists, finish reasons, wins, and
recovery/failure counters in `GameResult` and hand summaries.

**Step 4: Run focused tests**

Run: `uv run pytest tests/contract/test_poker_game.py tests/unit/games/test_poker_session.py tests/integration/test_poker_trace.py -q`

Expected: PASS with reconstructable hand lifecycle traces.

## Task 7: Make MatchMemory Transactional and Context-Aware

**Files:**
- Modify: `src/benchtable/memory.py`
- Modify: `src/benchtable/games/protocol.py`
- Modify: `src/benchtable/games/poker/session.py`
- Modify: `src/benchtable/engine.py`
- Modify: `tests/unit/test_memory.py`
- Modify: `tests/unit/test_engine_memory.py`

**Step 1: Write failing memory tests**

Cover staged multi-write preflight, rejected batches with no partial entries,
commit after action success, no commit after invalid action, configured entry
and character limits, multiple memory-only calls, accurate hand/in-hand turn
context, and per-match/per-actor isolation.

**Step 2: Run focused tests and confirm failure**

Run: `uv run pytest tests/unit/test_memory.py tests/unit/test_engine_memory.py -q`

Expected: FAIL on transactional writes, multiple memory calls, limits, or
context values.

**Step 3: Implement memory transactions and plugin memory limits**

Add an immutable prepared-write batch that validates all entries and total
limits before mutation, plus a commit operation that cannot reject a
prevalidated batch. Add an optional plugin memory-limit capability; PokerGame
returns validated `memory_max_entries` and `memory_max_chars`, while legacy
plugins retain current defaults.

Expose a session turn-context capability returning hand index and in-hand turn
index. The engine uses it for memory entries and falls back to its global turn
index for legacy sessions.

Process all memory-only calls in response order, append one assistant tool-call
message and one valid tool result per call, and count every call against the
budget. For a final response, reject reads combined with an action and stage
all writes before applying the single game action. Invalid writes produce a
validation retry instead of being silently ignored.

**Step 4: Run focused tests and type checks**

Run: `uv run pytest tests/unit/test_memory.py tests/unit/test_engine_memory.py tests/unit/test_engine.py -q`

Expected: PASS.

Run: `uv run pyright src/benchtable/memory.py src/benchtable/engine.py src/benchtable/games/protocol.py`

Expected: zero errors.

## Task 8: Correct Engine Failure Semantics and Tool Boundaries

**Files:**
- Modify: `src/benchtable/engine.py`
- Modify: `src/benchtable/games/protocol.py`
- Modify: `src/benchtable/games/poker/session.py`
- Modify: `tests/unit/test_engine.py`
- Modify: `tests/unit/test_engine_memory.py`
- Modify: `tests/integration/test_poker_trace.py`

**Step 1: Write failing engine tests**

Cover reserved game-tool name collisions, forced-fold continuation across
multiple hands, fail-match returning `success=False`, provider and timeout
failures, cancellation, memory budget exhaustion, and exactly one final game
action.

**Step 2: Run focused tests and confirm failure**

Run: `uv run pytest tests/unit/test_engine.py tests/unit/test_engine_memory.py tests/integration/test_poker_trace.py -q`

Expected: FAIL on collision handling and recovery status.

**Step 3: Implement generic boundary and recovery fixes**

Reject any game tool whose name is reserved before emitting schemas. Refactor
failed-turn recovery to return an explicit continue/terminal/failed outcome.
Continue the outer match loop after non-terminal forced folds; complete a
terminal forced-fold session only when the game result is genuinely completed;
never classify a fail-match session as success.

Keep provider retries, invalid-output retries, memory budgets, cancellation,
and engine exceptions as separate traceable failure reasons.

**Step 4: Run focused tests**

Run: `uv run pytest tests/unit/test_engine.py tests/unit/test_engine_memory.py tests/integration/test_poker_trace.py -q`

Expected: PASS.

## Task 9: Strengthen Redaction, Determinism, Documentation, and Integration Coverage

**Files:**
- Modify: `src/benchtable/events.py`
- Modify: `README.md`
- Modify: `docs/plugin-authoring.md`
- Modify: `tests/unit/test_events.py`
- Modify: `tests/integration/test_poker_trace.py`
- Modify: `tests/integration/test_cli.py`

**Step 1: Write failing redaction and lifecycle tests**

Add memory-read integration cases containing `Authorization: value`,
`password=value`, and nested credential-shaped text. Add at least two matches
with multiple hands, writes and reads, and assertions that notes reset between
matches while stacks/deals remain match-local. Normalize traces after removing
timestamps/run IDs and assert stable hand seeds, event order, payouts, and
seat-ordered summaries.

**Step 2: Run focused tests and confirm failure**

Run: `uv run pytest tests/unit/test_events.py tests/integration/test_poker_trace.py tests/integration/test_cli.py -q`

Expected: FAIL on free-form redaction and multi-match lifecycle gaps.

**Step 3: Implement redaction and documentation updates**

Extend string redaction to recognize credential key/value patterns without
altering legitimate usage fields. Update README and plugin documentation with
exact poker agent mapping, strict configuration, hand events, memory limits,
public-history isolation, and failure semantics.

**Step 4: Run focused tests**

Run: `uv run pytest tests/unit/test_events.py tests/integration/test_poker_trace.py tests/integration/test_cli.py -q`

Expected: PASS.

## Task 10: Complete Verification and Review

**Files:**
- Review: all files changed by Tasks 1–9
- Review: `docs/plans/2026-07-19-poker-game-repair-design.md`

**Step 1: Run the complete test suite**

Run: `uv run pytest -q`

Expected: all tests pass, including the new fixed-state poker and multi-match
trace coverage.

**Step 2: Run formatting and lint checks**

Run: `uv run ruff format --check .`

Expected: no formatting changes required.

Run: `uv run ruff check .`

Expected: no lint errors.

**Step 3: Run type and lockfile checks**

Run: `uv run pyright`

Expected: zero type errors.

Run: `uv lock --check`

Expected: lockfile is current and no dependency was added.

**Step 4: Inspect the final worktree**

Run: `git status --short`, `git diff --check`, and `git diff --stat`.

Expected: only intended source, test, and documentation changes are present;
no secrets, generated traces, or attribution markers are present. Do not commit
or push unless explicitly requested.
