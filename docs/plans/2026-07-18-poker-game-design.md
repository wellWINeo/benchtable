# Poker Game Design

## Goal

Add the first real game to Benchtable as a first-party `poker` plugin. The game
must support configurable numbers of agents, repeated hands, and private agent
memory so an agent can record and retrieve observations about competitors, such
as "player-1 rarely bluffs".

The core engine remains game-agnostic. Poker owns cards, betting, stacks, pots,
hand evaluation, and all other poker rules.

## Approved Scope

- No-limit Texas Hold'em.
- One configurable table with at least two configured players.
- No arbitrary application-level table-size cap; reject only configurations
  that cannot be dealt from a standard deck.
- A run contains multiple independent matches.
- A match contains a configured number of hands.
- Each hand deals new cards and rotates the dealer and blinds.
- Stacks persist across hands within a match.
- Notes persist across hands within a match and reset at the next match.
- A match ends after the configured hand count or when only one player remains.
- Invalid-turn recovery is configurable between forced fold and match failure.

The first release excludes tournaments, multiple tables, poker variants,
automatic opponent summaries, cross-match memory, and cross-run memory.

## Architecture

### Plugin boundary

Register `poker` through the existing `benchtable.games` entry-point group and
implement it as a first-party game module. `PokerGame` provides metadata,
configuration-aware actor IDs, system prompts, and session creation.

`PokerSession` owns all authoritative state, including:

- Configured player IDs and stable seat order.
- Player stacks, active/busted status, dealer position, and blind positions.
- The seeded random generator and current deck.
- Private hole cards and public community cards.
- Current hand, betting street, actor, commitments, pots, and action history.
- Showdown evaluation and chip settlement.

The existing static actor-ID validation must become configuration-aware so a
plugin can resolve its player IDs from `run.game_config`. Poker will require an
explicit `players` list and the CLI will verify that configured agent IDs match
it before constructing provider agents.

### Generic memory capability

The engine owns a private memory store per actor and per match. The store is
reset when a match starts and is never shared between agents or matches. It is
not part of poker state and is reusable by future games.

The engine exposes two reserved tools:

- `write_memory(text)`: append one free-form private note.
- `read_memory()`: return the actor's notes in chronological order.

Memory-only calls do not advance the game. The engine returns their results as
normal tool messages and requests another model response. A final response must
contain exactly one poker action. It may also contain `write_memory` calls.
`read_memory` cannot be combined with the final action because the read result
cannot influence that already-produced response.

Memory writes and the poker action are committed atomically. If poker action
validation fails, the accompanying writes are discarded. Successful action
application commits them. Memory-only writes commit immediately.

The interaction loop has a bounded memory-operation budget. Accepted poker
actions, not memory calls, count toward `max_turns`.

## Poker Lifecycle

### Match

At match creation, the session validates the players, stack and blind values,
hand count, and deck capacity. It initializes the seat order, stacks, dealer,
seeded random generator, and fresh memory store.

Each hand rotates the dealer and blinds among remaining players. The plugin
uses standard Hold'em dealing: two private cards per player, five community
cards, and burn cards. A new shuffle and deal occur for every hand. The match
seed and hand index determine the deterministic card sequence.

The match stops after the configured hand count or when fewer than two players
remain active. The terminal result includes final stacks, chip deltas, hand
counts, wins, finish status, and failure/recovery metrics.

### Hand

Each hand runs preflop, flop, turn, and river betting streets. A hand ends when
all but one player folds or after the river betting round completes. Standard
Hold'em hand ranking determines showdown winners. All-in play creates main and
side pots. Tied pots are split deterministically, including deterministic
odd-chip distribution by seat order.

The poker tool will represent one action with an action enum such as `fold`,
`check`, `call`, `bet`, `raise`, and `all_in`. Amounts are required only where
applicable and represent the player's target total commitment for the current
hand. The session validates turn order, legal actions, minimum raises, stack
limits, and all pot accounting.

### Observation isolation

An actor's observation contains only:

- That actor's hole cards.
- Public board cards, pot and stack information, current street, and legal
  actions.
- Public betting history for the current hand.
- The previous hand's public settlement summary when available.
- The actor's identity and hand/match context.

It never contains another player's hole cards, folded private cards, another
agent's memory, or raw internal state. The previous hand summary gives an
agent enough public evidence to record a note about competitor behavior before
the next action.

## Configuration

Poker-specific settings remain under `run.game_config`. Generic memory and
engine interaction limits remain run-level settings.

```toml
[run]
game = "poker"
matches = 10
seed = 20260718
max_turns = 500
max_invalid_attempts = 2
max_provider_retries = 2
max_memory_operations_per_turn = 4

[run.game_config]
players = ["player-1", "player-2", "player-3"]
hands_per_match = 50
initial_stack = 1000
small_blind = 5
big_blind = 10
memory_max_entries = 100
memory_max_chars = 20000
invalid_turn_policy = "forced_fold"

[[agents]]
id = "player-1"
role = "player"
model = "gpt-4o"
api_key_env = "OPENAI_API_KEY"
```

Validation happens before any provider request. It rejects duplicate or
missing players, fewer than two players, invalid blind relationships, stacks
that cannot cover the configured blinds, invalid hand counts, impossible deck
capacity, unknown agent IDs, and unsupported invalid-turn policies.

## Failure Semantics

Existing provider retry, invalid-action retry, cancellation, and max-turn
failure behavior remains explicit.

For an invalid model response, the engine retries within the existing invalid
action budget. If that budget is exhausted:

- `forced_fold` invokes poker's failed-turn handler, emits a `forced_fold`
  transition and metric, and continues the hand or match.
- `fail_match` ends the match with an explicit failed result.

Provider failures, timeouts, cancellation, engine errors, and exhausted
memory-operation budgets remain failed outcomes. They never fabricate a poker
winner or loss.

## Trace Design

The existing versioned JSONL envelope remains unchanged. Poker adds:

- `hand_start` and `hand_end` events with hand index, dealer/blind context,
  public board, and settlement summaries.
- Combined poker and memory tool schemas in `tool_schemas` events.
- `memory_operation` events for reads, writes, rejected writes, and budget
  failures.
- Public transition summaries for actions, forced folds, street changes, pot
  settlement, and showdown results.

The run configuration records poker configuration, memory limits, plugin
version, seeds, and sanitized agent metadata. Actor-specific observations,
memory results, normalized responses, and provider payloads remain traceable
with actor, match, hand, and turn context. The event writer remains the
credential-redaction boundary.

The trace may contain private observations for offline forensic analysis, but
those values must never be forwarded to another actor. Volatile timestamps,
run IDs, and provider-specific fields are excluded from normalized golden
trace comparisons.

## Testing Strategy

### Poker unit tests

- Deterministic seeded shuffles and no duplicate cards.
- Standard hand-ranking cases, including ties.
- Dealer/blind rotation and heads-up behavior.
- Legal action validation for every betting street.
- Minimum raises, all-ins, main pots, side pots, and odd-chip distribution.
- Stack settlement and match termination by hand count or elimination.
- Dynamic player counts and deck-capacity validation.
- Forced-fold and fail-match invalid-turn policies.

### Memory and engine tests

- Private memory isolation between actors and matches.
- Append/read ordering, limits, reset behavior, and atomic writes.
- Memory-only calls followed by a poker action.
- Writes combined with a valid action.
- Writes discarded when the accompanying poker action is invalid.
- Rejection of `read_memory` combined with a final poker action.
- Memory-operation budget exhaustion and valid Chat Completions retry history.
- Exactly one poker action enforcement with multiple memory calls allowed by
  the configured budget.

### Contract and integration tests

- Poker plugin and session protocol compliance.
- Observation isolation for hole cards, folded cards, and memory.
- Dynamic actor mapping through the CLI without provider calls.
- Complete offline multi-player, multi-hand, multi-match traces using fake
  agents that write and read notes.
- Event ordering, schema versioning, redaction, failure outcomes, and common
  metrics.
- Golden normalized traces with volatile fields excluded.

## Acceptance Criteria

- `benchtable list-games` discovers and reports `poker` and its version.
- A valid TOML configuration can run multiple configured agents through
  multiple hands and matches using only fake agents in tests.
- Each hand uses a fresh deterministic deal while stacks and private notes
  persist within the match.
- Agents can explicitly write and read private free-form notes about rivals.
- The engine never forwards raw state, hidden cards, or another agent's notes.
- Invalid outputs, provider failures, and memory-loop failures produce explicit
  trace outcomes.
- Complete pytest, Ruff, Pyright, and lockfile verification passes.
