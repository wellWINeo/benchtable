# Poker Game Fix Design

> **Historical / superseded by the 2026-07-26 hand-session design. Do not execute
> this historical plan.** It is retained as historical task context only. Current
> semantics are that memory calls occur in separate model responses before the
> game-action response, mixed memory/action responses are rejected, and memory
> writes are memory-only and committed immediately. The paired, batched, and
> accompanying-write instructions below are obsolete.

**Status:** Approved for implementation on 2026-07-24.

## Goal

Correct all defects identified in the review of the poker implementation and
the existing repair plan. The result must provide a correct no-limit Texas
Hold'em session, transactional private memory, honest failure semantics, and
complete redacted traces while preserving the generic game and agent
boundaries.

## Scope

This repair includes:

- Strict poker configuration and exact agent mapping.
- Immutable cards, unambiguous card serialization, and correct dealing.
- Explicit active-seat lifecycle, blinds, betting state, all-in runouts, and
  elimination.
- Target-commitment action validation, minimum raises, short all-ins, and street
  progression.
- Contribution-level pot construction, uncalled excess handling, conservation
  checks, tied settlement, and deterministic odd-chip distribution.
- Complete poker results, public hand summaries, and plugin lifecycle events.
- Transactional memory writes, multiple memory-only calls, valid turn context,
  and memory budget behavior.
- Exact-agent enforcement for direct engine callers and reserved tool checks.
- Forced-fold continuation, explicit failed outcomes, and separate failure
  reasons.
- Recursive structured and free-form credential redaction.
- Documentation, regression tests, formatting, lint, typing, and full suite
  verification.

Poker rules remain inside `PokerSession` and poker support remains a first-party
plugin. No provider API, database, concurrency model, or unrelated dependency
changes are in scope.

## Architecture

The existing boundaries remain authoritative:

- `PokerConfig` owns strict validation of poker-specific configuration.
- `PokerSession` owns match state, hand state, rules, dealing, betting,
  settlement, lifecycle, and poker metrics.
- `MatchMemory` is engine-owned and isolated per actor and match.
- `RunEngine` owns turn orchestration, retries, memory interaction, budgets,
  failure classification, and event sequencing.
- `EventWriter` is the final JSON-compatible normalization and credential
  redaction boundary.

Optional plugin and session capabilities will be used for exact agent mapping,
memory limits, turn context, and plugin events. Legacy plugins retain their
existing fallback behavior where the capability is absent.

## Poker State Model

Match state will include configured seat order, persistent stacks, active seats,
dealer position, match seed, hand count, wins, recovery/failure counters, and
public hand summaries.

Each hand will use a fresh deterministic deck and contain round-robin hole-card
dealing, burn cards, board cards, street-local commitments, total
contributions, current bet, minimum raise, pending actors, action history, and
settlement data. All-in players will never be selected for action. The hand
will run out remaining streets when no actionable players remain.

Action amounts will represent target total commitment. A centralized legal
action calculation will enforce strict argument types, stack limits, calls,
checks, bets, raises, minimum raises, and short all-in rules. Street-local
state resets without losing total hand contributions.

Pot construction will use every contribution level, including folded money,
and will preserve eligibility by contribution level. Settlement will return
explicit per-pot payouts and reject conservation violations. Seat order will be
passed explicitly for ties and odd chips.

## Engine And Memory

Memory-only calls will not advance poker turns. Multiple memory-only calls will
be processed in response order within the configured budget. Writes paired with
one poker action will be prepared and fully validated before action application,
then committed atomically only after the action succeeds. Invalid writes will
produce explicit validation results rather than being ignored.

Failed-turn handling will distinguish continuation, terminal completion, and
failure. Nonterminal forced folds continue the match. `fail_match`, provider
failures, timeouts, cancellation, engine errors, and exhausted budgets produce
failed outcomes and never fabricate a winner.

The engine will reject game tools that collide with reserved memory tools and
will enforce exact agent mapping for plugins that require it, including direct
engine use. Plugin events will be drained with match, turn, and actor context.

## Traces And Isolation

Poker will emit reconstructable `hand_start` and `hand_end` events containing
public hand context, derived seeds, dealer/blind information, board cards,
settlement summaries, and finish data. Terminal results will include final
stacks, deltas, hand count, wins, finish reason, failure/recovery counts, and
completed status.

Observations will contain only the current actor's hole cards and public table
information. They will not contain other hole cards, folded private cards, raw
state, or another actor's memory.

The event writer will redact credential-shaped structured fields recursively and
credential-shaped key/value content inside free-form strings, including
`Authorization: value`, `password=value`, and nested memory read results.
Legitimate usage fields remain intact.

## Testing Strategy

Tests will be added or updated before implementation for:

- Strict poker configuration, player mapping, deck capacity, and memory limits.
- Card immutability, serialization, seeded dealing, and burn/board order.
- Heads-up and multi-player rotation, elimination, all-in runouts, and hand
  termination.
- Every legal action and invalid amount/minimum-raise case.
- Side pots, folded contributions, uncalled excess, ties, odd chips, and chip
  conservation.
- Results, public observations, hand events, and private-information isolation.
- Transactional memory, multiple calls, context, budgets, and match isolation.
- Failure recovery, exact mapping, reserved tools, and failed statuses.
- Structured and free-form redaction.
- Offline multi-player, multi-hand, multi-match normalized traces.

Stale tests that expect implicit poker players will be reconciled with the
approved required `players` contract. The complete verification suite is a
release gate.

## Acceptance Criteria

- All review findings have regression tests and corrected implementations.
- Poker runs valid multi-player, multi-hand, multi-match offline traces.
- Poker configuration rejects coercions, blank IDs, invalid policies, and
  impossible tables before agent construction.
- Normal Hold'em betting, all-ins, settlement, elimination, and rotation are
  correct and chip-conserving.
- Memory writes are private, bounded, context-aware, and transactional.
- Failed runs cannot be reported as successful matches or fabricated wins.
- Hand lifecycle events and complete public results are present in traces.
- No credential-shaped content survives event serialization.
- The following commands pass:

```bash
uv sync
uv run pytest -q
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv lock --check
git diff --check
```
