# Poker Game Fix Implementation Plan

> **Historical / superseded by the 2026-07-26 hand-session design. Do not execute
> this historical plan.** It is retained as historical task context only. Current
> semantics are that memory calls occur in separate model responses before the
> game-action response, mixed memory/action responses are rejected, and memory
> writes are memory-only and committed immediately. The paired, batched, and
> accompanying-write instructions below are obsolete.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Repair the reviewed poker plugin, generic memory loop, failure semantics, trace redaction, tests, and verification failures without changing the approved architecture.

**Architecture:** Keep poker rules and authoritative state inside `PokerSession` and keep memory, orchestration, retry budgets, failure classification, and event ordering inside `RunEngine`. Add only the optional generic capabilities required by the approved fix design: exact agent mapping, memory limits, turn context, prepared memory writes, and plugin events.

**Tech Stack:** Python 3.12, Pydantic 2, standard-library poker logic, OpenAI Chat Completions tools, TOML, pytest/pytest-asyncio, Ruff, Pyright, and append-only JSONL traces.

---

## Implementation Notes

- Read `docs/plans/2026-07-24-poker-game-fix-design.md` before implementation.
- Work test-first. Every behavior change starts with a failing focused test.
- Preserve the required poker `players` configuration. Update stale tests that
  expect implicit poker players instead of adding compatibility defaults.
- Do not move poker rules into `RunEngine`.
- Do not capture hidden chain-of-thought or expose raw poker state to agents.
- Do not add dependencies, a database, concurrency, provider adapters, or UI.
- Preserve unrelated user changes in the dirty worktree.
- Do not commit or push unless explicitly requested.
- At every checkpoint run the focused tests and inspect `git diff --check`.

## Review Findings Covered

This plan addresses credential-shaped free-form text leakage, postflop state
progression, target commitment amounts, failed-match false success, non-atomic
memory writes, forced-fold continuation, side-pot eligibility, tied settlement
shape, mutable/ambiguous cards, incorrect dealing, all-in blind actors, dealer
rotation, missing hand events, incomplete results, non-strict configuration,
multiple memory calls, invalid paired writes, direct exact-agent enforcement,
incorrect turn context, and all current pytest/Ruff/Pyright failures.

### Task 1: Reconcile Baseline Tests And Strict Poker Configuration

**Files:**
- Create: `tests/unit/games/test_poker_config.py`
- Modify: `src/benchtable/games/poker/config.py`
- Modify: `src/benchtable/games/poker/session.py`
- Modify: `src/benchtable/cli.py`
- Modify: `src/benchtable/games/registry.py` only if the focused tests identify a capability boundary defect
- Modify: `tests/unit/games/test_poker_session.py`
- Modify: `tests/contract/test_poker_game.py`
- Modify: `tests/unit/test_game_registry.py`
- Modify: `tests/integration/test_cli.py`

**Step 1: Add failing configuration tests**

Cover:

- Required `players`.
- At least two players.
- Strict player strings with no empty or whitespace-only IDs.
- Unique IDs.
- Strict positive integer fields for stack, blinds, hand count, and strict
  non-negative integer memory limits.
- `small_blind < big_blind`.
- Initial stack coverage for the configured blinds.
- `2 * player_count + 8 <= 52`.
- Literal invalid-turn policies only.
- Unknown fields rejected.
- All invalid values rejected before an agent factory is called.
- Valid configuration retains exact input values without coercion.

Update existing poker tests that construct sessions without `players` to pass an
explicit player list. Update the one-player assertion to match the approved
typed configuration error rather than restoring an implicit default.

**Step 2: Run the focused tests and confirm the failures**

Run:

```bash
uv run pytest tests/unit/games/test_poker_config.py tests/unit/games/test_poker_session.py tests/contract/test_poker_game.py tests/unit/test_game_registry.py tests/integration/test_cli.py -q
```

Expected: failures for coercible values, blank IDs, missing strict checks, and
any stale test that still expects implicit poker players.

**Step 3: Implement the strict configuration boundary**

Use `StrictStr` for player IDs and `StrictInt` for all poker integer fields.
Validate non-empty IDs, uniqueness, blind relationships, stack coverage, hand
count, memory limits, policy literals, and deck capacity in `PokerConfig`.
Keep `extra="forbid"`.

Make `PokerGame.validate_config()` validate this model before provider
construction. Make `create_session()` consume the validated model only. When a
`players` key is present, `player_ids_from_config()` must not silently fall
back to default actors on malformed values.

Keep the CLI exact-ID set comparison before agent construction, including one
configured agent. Preserve legacy fallback behavior for plugins that do not
advertise exact mapping.

**Step 4: Run focused tests and type checks**

Run:

```bash
uv run pytest tests/unit/games/test_poker_config.py tests/unit/games/test_poker_session.py tests/contract/test_poker_game.py tests/unit/test_game_registry.py tests/integration/test_cli.py -q
uv run pyright src/benchtable/games/poker/config.py src/benchtable/games/poker/session.py src/benchtable/games/registry.py src/benchtable/cli.py
```

Expected: all focused tests pass and no type errors remain in these files.

### Task 2: Make Cards Immutable And Fix Deterministic Dealing

**Files:**
- Modify: `src/benchtable/games/poker/cards.py`
- Modify: `src/benchtable/games/poker/session.py`
- Modify: `tests/unit/games/test_poker_cards.py`
- Modify: `tests/unit/games/test_poker_session.py`

**Step 1: Add failing card and dealing tests**

Assert that:

- Rank and suit cannot be reassigned through public or stored card attributes.
- Rank strings are unique and use `2` through `A`.
- `Card.to_dict()` is unambiguous.
- Mutating an exported deck value cannot alter future decks.
- Every fresh deck has 52 unique cards.
- Fixed-seed dealing gives one card per active seat for each hole-card round.
- Burn, flop, turn, and river order is deterministic.

**Step 2: Run the focused tests and confirm the failures**

Run:

```bash
uv run pytest tests/unit/games/test_poker_cards.py tests/unit/games/test_poker_session.py -q
```

Expected: failures for rank collisions, mutable card/deck state, and
consecutive hole-card dealing.

**Step 3: Implement immutable values and round-robin dealing**

Use a frozen value representation or guarded assignment for `Card`. Replace
the mutable shared `STANDARD_DECK` surface with an immutable sequence or keep
it private, and copy stable card values into every `Deck`.

Give `Rank` explicit display symbols. In `_start_new_hand()`, draw one card for
each active player, then repeat for the second round. Keep burn and board draws
on the same seeded deck.

Add complete annotations for card collections so strict Pyright has no unknown
types.

**Step 4: Run focused tests and type checks**

Run:

```bash
uv run pytest tests/unit/games/test_poker_cards.py tests/unit/games/test_poker_session.py -q
uv run pyright src/benchtable/games/poker/cards.py src/benchtable/games/poker/session.py
```

Expected: PASS with zero type errors.

### Task 3: Rebuild Active Seats And Hand Lifecycle

**Files:**
- Modify: `src/benchtable/games/poker/session.py`
- Modify: `tests/unit/games/test_poker_session.py`
- Modify: `tests/contract/test_poker_game.py`

**Step 1: Add failing lifecycle tests**

Cover:

- Heads-up dealer/small-blind alternation.
- Multi-player rotation over several hands.
- Dealer advancement after the dealer or another seat is eliminated.
- Correct preflop and postflop first actors.
- Zero-stack removal from future hands.
- Short blind all-in players never selected as actors.
- Immediate runout when every non-folded player is all-in.
- Fresh hand state and fresh deals.
- Termination by hand count and by fewer than two active players.

**Step 2: Run the focused tests and confirm the failures**

Run:

```bash
uv run pytest tests/unit/games/test_poker_session.py tests/contract/test_poker_game.py -q
```

Expected: failures for elimination rotation, all-in blind actors, or stuck
all-in sessions.

**Step 3: Implement explicit active-seat lifecycle**

Track the dealer by stable player ID or configured seat position, not by the
current hand count modulo the changing active-player count. Derive blinds from
the active seat order and advance to the next eligible dealer after settlement.

After posting blinds, skip all-in seats when choosing the first actor. If no
non-folded player can act, run out the remaining board and settle immediately.
Reset hand-local state at hand creation while preserving match stacks, active
status, dealer position, hand count, and metrics.

Track an in-hand turn counter that increments for accepted game actions and
game-specific forced-fold recovery actions.

**Step 4: Run focused tests and type checks**

Run:

```bash
uv run pytest tests/unit/games/test_poker_session.py tests/contract/test_poker_game.py -q
uv run pyright src/benchtable/games/poker/session.py
```

Expected: PASS with zero type errors.

### Task 4: Correct Betting Actions And Street State

**Files:**
- Modify: `src/benchtable/games/poker/session.py`
- Modify: `tests/unit/games/test_poker_session.py`

**Step 1: Add failing action-matrix tests**

Cover fold, check, call, bet, raise, and all-in while facing and not facing a
bet. Assert that:

- Amounts are strict integers and booleans are rejected.
- Bet and raise amounts are target total commitments.
- Missing, zero, negative, over-stack, and malformed amounts are rejected.
- Only explicit all-in may commit the remaining stack when the target exceeds
  available chips.
- Betting is not available while facing a bet.
- Raising is not available when no bet is open.
- Minimum raises use the previous raise size.
- Short all-in calls and short all-in raises follow the selected no-limit rule.
- Street-local commitments and minimum raise state reset correctly.
- Normal check/call play advances through flop, turn, and river.

**Step 2: Run the focused tests and confirm the failures**

Run:

```bash
uv run pytest tests/unit/games/test_poker_session.py -q
```

Expected: failures for target amounts, postflop legality, minimum raises, and
short all-ins.

**Step 3: Implement centralized action validation**

Add one internal state calculation for current commitment, street commitment,
to-call amount, available stack, current street bet, minimum raise size, and
whether betting is open. Use it for both legal-action rendering and
`apply_action()` validation.

Apply `target - current_commitment` for bet and raise amounts. Reject invalid
targets instead of clamping them. Reset street-local bet, commitments, acted
set, pending actors, and minimum raise at every street while retaining total
hand contributions.

Ensure every successful action updates actor selection and invokes automatic
runout/street completion exactly once.

**Step 4: Run focused tests and type checks**

Run:

```bash
uv run pytest tests/unit/games/test_poker_session.py -q
uv run pyright src/benchtable/games/poker/session.py
```

Expected: PASS with zero type errors.

### Task 5: Rebuild Pot Construction And Settlement

**Files:**
- Modify: `src/benchtable/games/poker/pots.py`
- Modify: `src/benchtable/games/poker/session.py`
- Modify: `tests/unit/games/test_poker_pots.py`
- Modify: `tests/unit/games/test_poker_session.py`

**Step 1: Add failing conservation tests**

Cover:

- Current-hand contributions reset between hands.
- Folded dead money remains in pots.
- Contribution levels include folded players.
- Uncalled excess is returned only where no eligible opponent matched it.
- Main and side-pot eligibility follows contribution levels.
- All-in players are eligible only for pots they funded.
- Tied main and side pots produce explicit per-player payouts.
- Odd chips use configured seat order.
- Caller-owned contribution and hand structures are not mutated.
- `sum(payouts) == sum(pot amounts)` for every settlement.

**Step 2: Run the focused tests and confirm the failures**

Run:

```bash
uv run pytest tests/unit/games/test_poker_pots.py tests/unit/games/test_poker_session.py -q
```

Expected: failures for folded high contributions, tied result shape, or
conservation.

**Step 3: Implement explicit pot and payout data**

Build tiers from all positive contribution levels. For every tier, include all
contributors in the amount and only non-folded contributors meeting that tier
in eligibility. Handle unmatched excess before settlement instead of adding a
fallback pot with incorrect eligibility.

Change settlement output to represent each pot once with its full amount,
ordered winner list, and a per-player payout mapping. Pass seat order
explicitly, distribute each remainder once, and reject impossible payout
totals.

In `PokerSession._end_hand()`, apply payouts exactly once, retain per-hand
settlement data, and assert total payouts equal the hand pot.

**Step 4: Run focused tests and type checks**

Run:

```bash
uv run pytest tests/unit/games/test_poker_pots.py tests/unit/games/test_poker_session.py -q
uv run pyright src/benchtable/games/poker/pots.py src/benchtable/games/poker/session.py
```

Expected: PASS with zero type errors.

### Task 6: Complete Results And Plugin Lifecycle Events

**Files:**
- Modify: `src/benchtable/contracts.py`
- Modify: `src/benchtable/games/protocol.py`
- Modify: `src/benchtable/games/poker/session.py`
- Modify: `src/benchtable/engine.py`
- Modify: `tests/contract/test_poker_game.py`
- Modify: `tests/unit/games/test_poker_session.py`
- Modify: `tests/integration/test_poker_trace.py`

**Step 1: Add failing result and event tests**

Assert that:

- `hand_start` and `hand_end` events are emitted in sequence.
- Hand events contain public hand index, derived seed, dealer/blind context,
  board, settlement, and finish data.
- Hand-end payouts are per-hand payouts, not final stacks.
- Terminal results contain final stacks, deltas, hand count, seat-ordered wins,
  finish reason, forced-recovery count, failure count, and completion status.
- Failed sessions contain no fabricated winner and report `completed=False`.
- Observations contain public action history and no other hole cards, folded
  cards, raw state, or memory.

**Step 2: Run the focused tests and confirm the failures**

Run:

```bash
uv run pytest tests/contract/test_poker_game.py tests/unit/games/test_poker_session.py tests/integration/test_poker_trace.py -q
```

Expected: failures for missing hand events, incomplete summaries, result
metrics, and public settlement data.

**Step 3: Add the optional plugin-event contract and result data**

Add a small typed plugin event contract containing an event type and JSON
payload. Add an optional session event-drain capability. Queue hand-start and
hand-end events at the correct lifecycle boundaries.

Have `RunEngine` drain events after session creation, after accepted actions,
after recovery, and before match end. Emit them with match, turn, and actor
context without inspecting poker state.

Store structured public hand summaries, seat-ordered winner lists, hand seeds,
finish reasons, wins, and recovery/failure counters. Keep private hole cards
out of plugin event payloads.

**Step 4: Run focused tests and type checks**

Run:

```bash
uv run pytest tests/contract/test_poker_game.py tests/unit/games/test_poker_session.py tests/integration/test_poker_trace.py -q
uv run pyright src/benchtable/contracts.py src/benchtable/games/protocol.py src/benchtable/engine.py src/benchtable/games/poker/session.py
```

Expected: PASS with zero type errors.

### Task 7: Make MatchMemory Transactional And Context-Aware

**Files:**
- Modify: `src/benchtable/memory.py`
- Modify: `src/benchtable/games/protocol.py`
- Modify: `src/benchtable/games/poker/session.py`
- Modify: `src/benchtable/engine.py`
- Modify: `tests/unit/test_memory.py`
- Modify: `tests/unit/test_engine_memory.py`

**Step 1: Add failing memory tests**

Cover:

- Batch preflight validates every write before mutation.
- Rejected batches leave no partial entries.
- Successful game actions commit all staged writes.
- Invalid game actions commit none of the staged writes.
- Entry and character limits apply to the whole batch.
- Multiple memory-only calls process in response order.
- Invalid final writes produce explicit validation results.
- Memory entries contain accurate hand and in-hand turn context.
- Memory stays isolated per actor and per match.

**Step 2: Run the focused tests and confirm the failures**

Run:

```bash
uv run pytest tests/unit/test_memory.py tests/unit/test_engine_memory.py -q
```

Expected: failures for partial writes, multiple calls, invalid paired writes,
and incorrect context.

**Step 3: Implement prepared write batches**

Add an immutable prepared batch containing validated text and context. Validate
all entries and aggregate limits against a snapshot before mutation. Commit the
prepared batch without revalidating or partially applying entries.

Update the engine to process all memory-only calls in response order and count
every memory operation against the budget. For final responses, reject reads
combined with a game action, validate every write before applying the action,
and commit only after successful action application.

Use the optional session turn-context capability and fall back to the global
engine turn index for legacy sessions. Keep memory initialization inside the
guarded match lifecycle so invalid limits produce a `match_end` failure event.

**Step 4: Run focused tests and type checks**

Run:

```bash
uv run pytest tests/unit/test_memory.py tests/unit/test_engine_memory.py tests/unit/test_engine.py -q
uv run pyright src/benchtable/memory.py src/benchtable/engine.py src/benchtable/games/protocol.py
```

Expected: PASS with zero type errors.

### Task 8: Correct Engine Failure Semantics And Tool Boundaries

**Files:**
- Modify: `src/benchtable/engine.py`
- Modify: `src/benchtable/games/protocol.py`
- Modify: `src/benchtable/games/poker/session.py`
- Modify: `tests/unit/test_engine.py`
- Modify: `tests/unit/test_engine_memory.py`
- Modify: `tests/integration/test_poker_trace.py`

**Step 1: Add failing engine tests**

Cover:

- Reserved game-tool name collisions fail before provider requests.
- Direct `RunEngine` use cannot single-agent a plugin requiring exact IDs.
- Nonterminal forced-fold recovery continues through later turns/hands.
- `fail_match` returns `success=False` and `completed=False`.
- Provider, timeout, cancellation, engine, invalid-output, and memory-budget
  failures remain distinct.
- Exactly one final game action is required.
- Memory-only operations never increment accepted turn count.

**Step 2: Run the focused tests and confirm the failures**

Run:

```bash
uv run pytest tests/unit/test_engine.py tests/unit/test_engine_memory.py tests/integration/test_poker_trace.py -q
```

Expected: failures for false success, forced-fold termination, direct mapping,
or missing boundary events.

**Step 3: Implement explicit engine outcomes**

Refactor failed-turn handling to return an explicit continuation, terminal, or
failed outcome. Continue the match loop after a nonterminal forced fold. Treat
`fail_match` as failed even if the session marks itself terminal, and never use
an incomplete result as a successful winner.

Add an engine preflight for plugins with `requires_exact_agent_ids`, resolving
configured IDs from `game_config` and rejecting single-agent fallback before a
provider call. Keep reserved-name validation before emitting tool schemas.

Preserve exact provider request/response history and emit contextual failure
events for every terminal path.

**Step 4: Run focused tests and type checks**

Run:

```bash
uv run pytest tests/unit/test_engine.py tests/unit/test_engine_memory.py tests/integration/test_poker_trace.py -q
uv run pyright src/benchtable/engine.py src/benchtable/games/protocol.py src/benchtable/games/poker/session.py
```

Expected: PASS with zero type errors.

### Task 9: Strengthen Redaction, Integration Coverage, And Documentation

**Files:**
- Modify: `src/benchtable/events.py`
- Modify: `README.md`
- Modify: `docs/plugin-authoring.md`
- Modify: `tests/unit/test_events.py`
- Modify: `tests/integration/test_poker_trace.py`
- Modify: `tests/integration/test_cli.py`

**Step 1: Add failing redaction and lifecycle tests**

Add free-form cases containing:

- `Authorization: value`.
- `password=value`.
- `api_key: value`.
- Nested credential-shaped text in memory read results.

Add integration cases with at least two matches, multiple hands, memory writes
and reads, hand events, forced folds, failed matches, reset notes, deterministic
seeds, seat-ordered payouts, and normalized event comparisons.

**Step 2: Run the focused tests and confirm the failures**

Run:

```bash
uv run pytest tests/unit/test_events.py tests/integration/test_poker_trace.py tests/integration/test_cli.py -q
```

Expected: failures for free-form credential text and incomplete multi-match
trace coverage.

**Step 3: Implement redaction and documentation updates**

Extend string redaction with bounded, case-insensitive credential key/value
patterns. Redact values without changing legitimate usage fields such as
`prompt_tokens`, and preserve the existing recursive structured-key behavior.

Update documentation with strict poker configuration, exact agent mapping,
reserved tools, transaction semantics, hand events, observation isolation,
memory limits, failure outcomes, and run/match/hand terminology.

Fix any remaining Ruff formatting/lint issues in touched files and add explicit
types where Pyright reports unknown or optional values.

**Step 4: Run focused tests and tooling**

Run:

```bash
uv run pytest tests/unit/test_events.py tests/integration/test_poker_trace.py tests/integration/test_cli.py -q
uv run ruff format --check src tests
uv run ruff check src tests
uv run pyright
```

Expected: PASS with no formatting, lint, or type errors.

### Task 10: Complete Verification And Review

**Files:**
- Review: all files changed by Tasks 1-9
- Review: `docs/plans/2026-07-24-poker-game-fix-design.md`
- Review: `docs/plans/2026-07-24-poker-game-fix.md`

**Step 1: Run the complete test suite**

Run:

```bash
uv sync
uv run pytest -q
```

Expected: all tests pass, including poker configuration, state, settlement,
memory, failure, redaction, and normalized trace coverage.

**Step 2: Run formatting, lint, typing, and lock checks**

Run:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv lock --check
```

Expected: all commands exit successfully with zero errors.

**Step 3: Inspect the final worktree**

Run:

```bash
git status --short
git diff --check
git diff --stat
```

Expected: only intended source, test, and documentation changes are present;
no generated traces, secrets, or attribution markers are present. Do not commit
or push unless explicitly requested.

## Final Acceptance Criteria

- Strict poker configuration and exact player mapping fail before provider
  construction.
- Cards and seeded dealing are immutable, unambiguous, and Hold'em-correct.
- Betting, street progression, all-ins, elimination, rotation, and settlement
  are correct and chip-conserving.
- Poker results and hand events are complete, public, deterministic, and
  reconstructable.
- Memory is private, bounded, context-aware, multi-call, and transactional.
- Forced folds continue active matches, while failed policies remain failed.
- No credential-shaped structured or free-form content survives serialization.
- Complete pytest, Ruff, Pyright, and lockfile verification passes.
