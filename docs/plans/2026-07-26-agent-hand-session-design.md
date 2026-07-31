# Hand-Scoped Agent Sessions and Poker Reliability Design

**Status:** Approved for implementation on 2026-07-26; consolidated on 2026-07-31.

## Goal

Provide a correct first-party no-limit Texas Hold'em plugin whose agents have
isolated conversations for one hand at a time, while preserving private
match-scoped memory across hands. The generic benchmark engine must own
conversation lifetime, memory interaction, retries, failure classification, and
trace sequencing without learning poker rules.

The result must support strict poker configuration, exact actor-to-agent
mapping, deterministic multi-hand matches, complete public hand results,
credential-safe traces, and offline verification with fake agents.

## Scope

Included:

- A first-party `poker` game plugin registered through `benchtable.games`.
- No-limit Texas Hold'em for one configurable table with at least two players.
- Multiple hands per match, multiple matches per run, persistent stacks within a
  match, and fresh cards for every hand.
- Correct heads-up and multi-player dealer/blind rotation, betting streets,
  all-in runouts, elimination, side pots, ties, and odd-chip distribution.
- Strict poker configuration under `run.game_config` and exact poker agent IDs.
- Actor-private notes and actor-targeted public system summaries in match memory.
- One actor-specific model transcript per conversation scope, normally one hand.
- Complete versioned JSONL traces with recursive structured and free-form
  credential redaction.
- Deterministic fake-agent, contract, unit, and integration coverage.

Excluded:

- Tournaments, multiple tables, poker variants, hosted services, databases,
  dashboards, analytics, and statistical reports.
- Provider adapters other than the OpenAI-compatible adapter.
- Provider-owned conversation state or shared transcripts.
- Cross-match or cross-run memory.
- Hidden chain-of-thought capture.

## Architectural Boundaries

### Game plugin

`PokerGame` owns metadata, plugin version, strict configuration validation,
configuration-aware player IDs, system prompts, and session creation.
`PokerSession` owns all authoritative state and poker rules, including:

- Configured seat order, active/busted players, stacks, dealer, blinds, and
  deterministic seeds.
- Fresh per-hand deck, private hole cards, public board, street, commitments,
  current actor, action history, and settlement state.
- Legal actions, target commitment validation, minimum raises, short all-ins,
  all-in runouts, hand completion, and match termination.
- Contribution-level pot construction, uncalled excess, main/side-pot
  eligibility, tie settlement, and deterministic odd-chip distribution.
- Public hand summaries, plugin lifecycle events, terminal results, and
  game-specific metrics.

The engine receives only the session protocol. It never receives raw poker
state and never implements a poker rule.

### Engine and provider

`RunEngine` owns sequential match execution, actor dispatch, request attempts,
retry budgets, memory interaction, conversation scope, finalization, common
metrics, failure outcomes, and event ordering. Provider calls remain
asynchronous internally.

The `Agent` protocol and `OpenAICompatibleAgent` remain stateless. The engine
sends the current actor's system prompt and accumulated hand transcript on each
request. Provider normalization remains inside the adapter; API credentials are
read from the configured environment variable and never enter configuration,
logs, or persisted traces.

### Optional generic capabilities

Legacy plugins remain valid when optional capabilities are absent. The engine
may discover these capabilities with `getattr` without exposing raw state:

- Configuration-aware actor IDs and exact-agent requirements.
- Conversation scope identity, defaulting to a stable match scope.
- Actor turn context for memory entries.
- Actor-targeted plugin memory summaries.
- Plugin event draining.

## Poker Configuration and Lifecycle

Poker configuration is a strict typed model with `extra="forbid"`. It requires
non-empty unique `players`, at least two seats, strict integer values for
`hands_per_match`, `initial_stack`, `small_blind`, `big_blind`, and memory
limits, a valid `small_blind < big_blind` relationship, sufficient initial
stack, supported `invalid_turn_policy`, and standard-deck capacity:
`2 * player_count + 8 <= 52`.

Poker requires one configured agent for every configured player. The CLI and
direct engine entry points enforce the exact actor set before the first
provider request. Plugins without the exact-mapping capability retain the
generic single-agent fallback.

At match creation, the session copies validated configuration, initializes
stable seats and stacks, derives deterministic hand seeds, and creates the
first hand. Each hand resets hand-local state, deals hole cards round-robin,
uses burn cards and board cards in standard order, rotates the dealer among
active seats, and derives blinds and first actors from active-seat order.

All-in or zero-stack seats are never selected for action. If no actionable
players remain, the session runs out the remaining streets and settles. A match
ends after the configured hand count or when fewer than two players remain.
Stacks persist within a match and reset between matches.

Action amounts are target total commitments for the current hand. A centralized
legal-action calculation supplies both the observation and validator with
to-call amount, available stack, street commitment, current bet, minimum raise,
and betting-open state. Invalid amounts, illegal action/street combinations,
short-raise violations, and over-stack commitments are rejected rather than
clamped. Explicit `all_in` handles a remaining-stack commitment.

Settlement uses current-hand contribution levels and preserves folded dead
money, eligibility, uncalled excess, side pots, ties, and seat-ordered odd
chips. Payouts are conservation-checked and applied exactly once. Results
include final stacks, deltas, hand count, seat-ordered winners, finish reason,
recovery/failure counts, completion status, and public hand summaries.

## Conversation Lifecycle

The engine keeps a transcript mapping for each actor in the current match. A
scope change clears every transcript. Poker's scope is its current hand, so a
hand-ending action is finalized against the old transcript and the transcripts
are discarded before the next hand's first request. Legacy sessions without the
capability retain one match-scoped transcript.

Every turn appends a fresh user message containing only the current actor's
observation snapshot. The actor receives its own prior user, assistant, tool,
and finalization messages for the current scope. No actor receives another
actor's transcript, observation, or memory.

The model interaction is ordered as follows:

1. The engine requests one or more optional memory-only responses.
2. Every memory call is appended as an assistant tool call followed by its
   matching tool result in valid Chat Completions history.
3. A later response must contain exactly one recognized game action and no
   memory calls.
4. The engine appends the action tool call, applies the action, and appends a
   public transition tool result containing only plugin summary, metrics, and
   terminal status.
5. The engine makes exactly one tool-free finalization request. It accepts
   assistant text or a finish reason; empty text without a finish reason fails.

Memory writes are memory-only and commit immediately when their calls succeed.
Writes cannot accompany a game action. A response mixing any memory call with a
game action is rejected before action validation. Memory operations do not
advance the accepted game-turn count and use a separate bounded budget.

If an action raises `InvalidActionError`, the failed assistant tool call gets
exactly one matching validation-error tool result before retry, recovery, the
next observation, or the next provider request. Retry and exhaustion paths reuse
that history and never duplicate the assistant tool call. A finalization tool
call fails immediately and is not retried.

## Match Memory

`MatchMemory` is engine-owned and isolated by match and actor. It has two
channels:

- Agent notes are free-form, actor-private, chronological, and bounded by the
  configured entry and character limits.
- System summaries are compact, actor-targeted, chronological, and excluded
  from agent-note limits. They contain only public hand information.

Reads merge both channels in chronological order and mark each entry as
`agent` or `system`. A plugin may provide one compact summary per completed hand
for each actor, containing hand index, finish reason, and seat-ordered chip
deltas or payouts. It must omit board cards, action history, hole cards, and
raw state. The engine drains summaries after successful or recovery transitions,
before the next actor/provider request, and before terminal match completion.

Poker observations do not include previous-hand summaries. Agents retrieve
cross-hand context explicitly through `read_memory`; memory resets at match
boundaries.

The earlier paired-write and transactional-batch proposals are superseded by
this design. No memory write may be paired with an action, so immediate
memory-only commit is the authoritative behavior.

## Traces and Privacy

`EventWriter` is the final JSON-compatible and credential-redaction boundary.
Each append-only `events.jsonl` line contains a schema version, monotonic
sequence, event type, run/match/turn/actor context where applicable, timestamp,
and JSON-compatible payload. Existing valid prefixes may be appended safely;
malformed existing content is rejected.

Traces preserve sanitized run configuration, plugin/version and seeds, exact
actor-specific observations, tool schemas, provider requests and raw responses,
normalized text/tool calls/usage/latency/finish reasons, validation results,
transition summaries, plugin hand events, memory operations, failures, results,
and common metrics. Every event is flushed immediately.

Structured credential fields are redacted recursively, including API keys,
authorization values, access tokens, passwords, client secrets, API-key headers,
and nested headers. Free-form credential-shaped key/value strings are also
redacted, including case variants, `Authorization: value`, `password=value`,
and escaped/nested forms. Correlation IDs and legitimate usage fields such as
`prompt_tokens` and `completion_tokens` remain intact.

Private observations may be retained in traces for offline analysis, but are
never sent to a different actor. Hidden reasoning is neither requested nor
captured.

## Failure Semantics

Configuration, unknown plugins, missing credentials, malformed provider output,
invalid actions, provider failures, timeouts, cancellation, memory-budget
exhaustion, plugin failures, engine failures, finalization failures, and
exhausted retry budgets are explicit traceable failures. They cannot be
converted into fabricated wins or successful matches.

`forced_fold` is the only recovery path that may continue a poker match, and it
does so only when the session reports a nonterminal continuation. `fail_match`
and all generic failure paths produce incomplete failed results. The CLI exits
non-zero for invalid configuration and failed runs while preserving the trace
prefix and terminal failure events.

## Testing Strategy

Use deterministic fake agents and no live provider calls. Test contracts and
boundaries before implementations:

- Strict Pydantic contracts, optional capabilities, configuration, exact agent
  mapping, registry discovery, duplicate detection, and CLI validation.
- Immutable cards, seeded dealing, rotation, legal action matrices, all-in
  runouts, elimination, side pots, ties, odd chips, and chip conservation.
- Private observation isolation, actor ID validation, hand events, complete
  results, summaries, memory channels, limits, context, and reset behavior.
- Transcript accumulation, actor isolation, scope reset, valid tool history,
  separate memory/action responses, action results, finalization, and failure
  retries.
- Event ordering, append/flush behavior, schema versioning, structured and
  free-form redaction, raw-provider preservation, and common metrics.
- Offline multi-player, multi-hand, multi-match traces and normalized golden
  comparisons with volatile fields removed.

## Acceptance Criteria

- Poker is discoverable and configurable through the existing plugin registry.
- Invalid poker configuration and exact-agent mismatches fail before provider
  construction.
- Hold'em dealing, rotation, betting, all-ins, elimination, settlement, and
  results are correct and chip-conserving.
- Each actor has one isolated transcript per hand; transcripts reset between
  hands while state and memory follow their defined match scope.
- Memory calls are separate from game actions, private, bounded, source-marked,
  and available across hands only through `read_memory`.
- Every accepted action has one public transition result and one tool-free
  finalization response.
- Failed actions have matching validation results, and failed runs remain
  failed in both trace and CLI status.
- No credential-shaped content survives event serialization.
- The complete verification suite passes:

```bash
uv sync
uv run pytest -q
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv lock --check
git diff --check
```
