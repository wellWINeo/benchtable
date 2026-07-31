# Poker Game Repair Design

> **Historical / superseded by the 2026-07-26 hand-session design. Do not execute
> this historical plan.** It is retained as historical task context only. Current
> semantics are that memory calls occur in separate model responses before the
> game-action response, mixed memory/action responses are rejected, and memory
> writes are memory-only and committed immediately. The paired, batched, and
> accompanying-write instructions below are obsolete.

**Status:** Approved for implementation on 2026-07-19.

## Goal

Make the first-party no-limit Texas Hold'em plugin correct and production-ready
without moving poker rules into the generic run engine.

## Decisions

- Poker requires one configured agent for every configured player. The generic
  single-agent fallback remains available only to plugins that do not require
  exact actor mapping.
- Repair the poker core behind the existing `PokerGame`, `PokerSession`,
  `GameSession`, `Transition`, and `GameResult` boundaries instead of adding a
  second poker implementation or changing provider APIs.
- Use a strict typed poker configuration model. Configuration is validated
  before provider construction and the validated values are reused to create
  every match session without coercion or silent defaults.
- Keep match state separate from hand state. Hand contributions are reset for
  every hand; stacks, active seats, dealer position, hand count, and metrics
  persist for the match.
- Use explicit active-seat order for round-robin dealing, blind rotation, actor
  selection, odd-chip distribution, and deterministic summaries.
- Keep memory engine-owned and per-match. Memory-only operations are processed
  within their configured budget. Writes paired with a poker action are
  prevalidated and committed atomically only after the action is accepted.
- Add generic optional plugin/session capabilities for exact actor mapping,
  memory limits, turn context, and plugin trace events. The engine consumes
  these capabilities without implementing poker rules.

## Poker State Model

`PokerSession` owns persistent player stacks, active/busted status, stable seat
order, dealer position, match seed, hand count, wins, recovery/failure counts,
and public hand summaries.

Each hand owns a fresh seeded deck, round-robin hole cards, burn cards, board,
street, dealer/blind positions, per-hand total contributions, street
commitments, current bet, minimum raise, acted/pending actors, public action
history, and settlement data. Each hand deals the remaining board and settles
automatically when no actionable players remain. Zero-stack players are removed
from future hands. The match ends at the configured hand count or below two
active players.

Pot settlement returns explicit per-player payouts. Payout totals must equal
pot totals for every main pot, side pot, tie, and odd-chip distribution.

## Engine and Memory

The engine validates reserved memory-tool names before sending tools to an
agent. It partitions calls into memory and game calls, permits multiple
memory-only calls within the budget, rejects reads combined with a final game
action, and requires exactly one game action.

Memory write batches validate all text, entry limits, and character limits
before the game action is applied. A failed action or failed batch leaves no
note. Memory-only writes commit immediately and return explicit tool results.

Forced-fold recovery continues the match when the session remains active.
`fail_match`, provider failures, cancellation, engine failures, and exhausted
budgets produce incomplete failed outcomes rather than fabricated successes.

## Traces and Observations

Poker emits `hand_start` and `hand_end` plugin events containing hand index,
derived hand seed, dealer/blind context, public board, settlement, and finish
data. The engine attaches match, turn, and actor context to these events.

Observations include public betting history and current legal actions, but never
another player's hole cards, folded private cards, raw state, or another
actor's memory. Memory entries record true hand and in-hand turn context.
Summaries preserve configured seat order and derived hand seeds are recorded so
normalized traces are deterministic.

The event writer redacts credential-shaped key/value content inside both
structured payloads and free-form strings, including memory read results.

## Verification

Tests will cover fixed dealing and rotation, every action and betting edge case,
all-in runouts, elimination, pot conservation, strict configuration, memory
transactions, agent mapping, failure policies, hand events, isolation,
redaction, multi-hand/multi-match traces, and deterministic normalized output.
The complete pytest, Ruff, Pyright, and lockfile checks must pass.
