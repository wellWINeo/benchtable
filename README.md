# Benchtable

Run language models as agents in interchangeable games and record traces for outcome and behavior analysis.

## Setup

```bash
uv sync
```

## Run tests

```bash
uv run pytest -q
```

## Lint and type check

```bash
uv run ruff format --check .
uv run ruff check .
uv run pyright
```

## CLI

```bash
uv run benchtable --help
uv run benchtable list-games
uv run benchtable run --config config.toml --output ./runs
```

An experiment configuration passes `[run.game_config]` to the game's
`create_session` method. When multiple agents are configured, each `id` must
match the actor ID returned by the game and the engine dispatches turns to that
agent. A single configured agent is used as the fallback for every actor.
This fallback is for legacy or single-actor plugins. Plugins that require exact
actor mapping, including poker, must provide one configured agent for every
actor.

Each actor gets an isolated transcript for the current conversation scope,
normally one hand. Fresh observations are appended on each turn, while
assistant tool-call and tool-result messages remain in valid history. Plugins
may expose the optional `conversation_scope_id` capability to change that
scope; without it, the engine retains stable match-scoped behavior for legacy
sessions. Match memory is separate: private agent notes and actor-targeted
public summaries are returned by `read_memory` in chronological order with
`agent` or `system` source markers. Summaries are drained and stored before the
next actor request and terminal match completion. System summaries are stored
separately and do not consume agent-note entry or character limits. Requests
may include the current actor's accumulated transcript and actor-addressed
memory results, never another actor's transcript or memory. Poker observations
do not include previous-hand summaries.

```toml
[run]
game = "my-game"
matches = 1
seed = 20260718

[run.game_config]
starting_round = 1

[[agents]]
id = "player-1"
role = "player"
model = "gpt-4o"
api_key_env = "OPENAI_API_KEY"
```

Only the API-key environment variable name is configured and recorded. Literal
credentials are not accepted or written to traces.

## Plugins

Game plugins register via the `benchtable.games` entry-point group. See [docs/plugin-authoring.md](docs/plugin-authoring.md) for details.

## Poker

Benchtable includes a first-party No-Limit Texas Hold'em poker plugin. It
supports configurable player counts, repeated hands, and private per-agent
memory. Stacks persist across hands within a match; private notes and compact
actor-targeted public hand summaries persist across hands and reset between
matches.

### Run configuration

```toml
[run]
game = "poker"
matches = 5
seed = 20260718
max_turns = 500
max_memory_operations_per_turn = 4

[run.game_config]
players = ["player-1", "player-2"]
hands_per_match = 50
initial_stack = 1000
small_blind = 5
big_blind = 10
invalid_turn_policy = "forced_fold"

[[agents]]
id = "player-1"
role = "player"
model = "gpt-4o"
api_key_env = "OPENAI_API_KEY"

[[agents]]
id = "player-2"
role = "player"
model = "gpt-4o"
api_key_env = "OPENAI_API_KEY"
```

The `poker_action` tool accepts `fold`, `check`, `call`, `bet`, `raise`, and
`all_in` actions. Bet and raise amounts are target total commitments for the
current hand, not increments. The engine adds `read_memory` and `write_memory`
tools so agents can record and retrieve notes about opponents. Memory-only
calls happen in separate responses before the final action call, and writes are
not paired with actions. Each accepted action receives a public transition
tool result followed by exactly one tool-free finalization call. Finalization
must contain assistant text or a finish reason; a finalization tool call fails
immediately without retry.
