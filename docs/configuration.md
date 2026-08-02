# Experiment Configuration Reference

Run an experiment from the CLI:

```bash
uv run benchtable run --config experiment.toml --output ./runs/my-experiment
```

`--config` names the TOML experiment file. `--output` is a required CLI-only
option, not a TOML key: it selects the directory where Benchtable creates and
writes `events.jsonl`. A successful command prints that trace path. The CLI
creates the output directory when necessary. Reusing an output directory
appends to its existing `events.jsonl` after validating it and resumes event
sequence numbers from its last event, so one trace file can contain multiple
runs.

## Generic TOML Shape

This complete example satisfies the generic configuration schema. Replace the
game name, game configuration, actor IDs, and provider values with values for
an installed game plugin and provider. The empty `[run.game_config]` table is
valid generically, but a selected plugin can require settings in it.

```toml
[run]
game = "your-game"                         # Required installed plugin name.
matches = 1                                 # Required number of sequential matches.
seed = 0                                    # Optional run seed; 0 is the default.
max_turns = 200                             # Optional per-match turn limit.
max_invalid_attempts = 2                    # Optional invalid-output retry budget.
max_provider_retries = 2                    # Optional provider retry budget.
max_memory_operations_per_turn = 4          # Optional memory-tool operation budget.

[run.game_config]
# Plugin-defined JSON-compatible settings go here.

[[agents]]
id = "actor-1"                             # Plugin actor ID for this agent.
role = "player"                            # Recorded metadata only.
model = "your-model"                       # OpenAI-compatible model identifier.
base_url = "https://provider.example/v1"   # Optional compatible endpoint.
api_key_env = "OPENAI_API_KEY"             # Environment variable, never the key.
timeout = 30.0                              # Optional request timeout in seconds.
max_completion_tokens = 512                 # Optional completion-token limit.
```

## Generic `[run]` Keys

All keys in `[run]` other than `game` and `matches` have defaults. Integer
fields below are strict integers: TOML booleans, strings, and floating-point
values are rejected rather than coerced. The top level accepts only `run` and
`agents`; unknown top-level keys and unknown `[run]` keys are rejected.

| Key | Type, default, and constraints | Effect |
| --- | --- | --- |
| `game` | Required string. No generic nonblank constraint. | Names the game plugin to load. It must resolve to an installed plugin. |
| `matches` | Required strict integer, at least `1`. | Runs this many matches sequentially. |
| `seed` | Strict integer; default `0`; negative values are allowed. | The configured seed is recorded in the `run_config` trace event. For zero-based match index `i`, the match seed is `int(sha256(f"{seed}:{i}").hexdigest()[:16], 16)`. That derived seed is recorded in the `match_start` event and passed to the game session. |
| `max_turns` | Strict integer, at least `1`; default `200`. | Limits turns in each match. A nonterminal game after this many turns ends as an explicit failed match. |
| `max_invalid_attempts` | Strict integer, at least `0`; default `2`. | Allows this many invalid action outputs on a turn. The next invalid output exhausts the budget; the game can apply its failed-turn policy, otherwise the match fails explicitly. |
| `max_provider_retries` | Strict integer, at least `0`; default `2`. | Allows this many retries after an initial provider failure. The engine therefore makes at most `max_provider_retries + 1` provider attempts for one response before failing the match explicitly. |
| `max_memory_operations_per_turn` | Strict integer, at least `0`; default `4`. | Caps `read_memory` and `write_memory` calls processed from memory-only responses for one game turn. Each processed memory call counts; a response mixing memory and game calls is invalid, so its memory calls are neither processed nor counted. Exceeding the limit fails the match explicitly. `0` permits no memory operations. |
| `game_config` | JSON object; default `{}`. TOML has no null literal. Values expressible in an experiment TOML file are strings, finite integers or floats, booleans, arrays, and nested tables or inline tables with string keys. | Carries only plugin-defined configuration. Benchtable validates its generic JSON shape, then supplies the mapping to plugin validation, actor resolution, and every match session. |

## `[[agents]]` Entries

Provide at least one `[[agents]]` table. Unknown agent keys are rejected. `id`,
`model`, `base_url` when present, and `api_key_env` must not be empty or
whitespace-only. Every agent ID must be unique.

| Key | Type, default, and constraints | Effect |
| --- | --- | --- |
| `id` | Required strict nonblank string. | Identifies this configured agent. With multiple agents, it maps directly to a game actor ID. |
| `role` | String; default `"player"`; no nonblank constraint. | Metadata recorded in the trace. It does not control turn assignment or game behavior. |
| `model` | Required strict nonblank string. | Model value sent to the OpenAI-compatible Chat Completions endpoint. |
| `base_url` | Optional strict nonblank string; default omitted (`None`). | Overrides the OpenAI-compatible endpoint for this agent. If omitted, the SDK default endpoint is used. |
| `api_key_env` | Strict nonblank string; default `"OPENAI_API_KEY"`. | Names the environment variable from which the adapter reads this agent's credential. The name, not its value, is recorded. |
| `timeout` | Optional finite positive number; default omitted (`None`). Integers and floats are accepted. | Passed as the provider request timeout. |
| `max_completion_tokens` | Optional strict integer, at least `1`; default omitted (`None`). | When supplied, is sent as `max_completion_tokens` on provider requests. |

The adapter reads `api_key_env` when it is constructed. The named variable
must exist and contain a non-whitespace value or `benchtable run` exits before
making a provider request. Configure the environment in the shell that runs
the command, for example:

```bash
export OPENAI_API_KEY='...'
uv run benchtable run --config experiment.toml --output ./runs/my-experiment
```

Do not put a literal API key in TOML. The loader rejects credential-shaped keys
recursively anywhere in the file, including nested game configuration. This
includes normalized variants of names such as `api_key`, `authorization`,
`access_token`, `password`, `client_secret`, and `token`.

Retries belong to the engine, not the SDK client: the OpenAI-compatible client
is configured with SDK retries disabled so each engine attempt can be traced.
There are no other generic generation settings. In particular, `max_tokens`,
temperature, top-p, and provider-specific settings are not accepted generic
agent keys; use only `max_completion_tokens` where supported by this schema.

## Plugin Configuration And Agent Mapping

Generic validation does not define game rules. Before constructing agents, the
CLI calls the plugin's optional `validate_config(game_config)` hook and resolves
actor IDs with `player_ids_from_config(game_config)`, or `player_ids` when that
method is absent. Resolved IDs must be a nonempty list of unique, nonblank
strings. Plugin authors can find the hook protocol in
[the plugin authoring guide](plugin-authoring.md).

When more than one `[[agents]]` entry is configured, or when a plugin declares
`requires_exact_agent_ids = true`, configured agent IDs must exactly equal the
resolved actor IDs, with no missing or extra IDs; each turn is dispatched to
the matching actor ID. Otherwise, one configured agent is the shared fallback
for every actor and its ID need not match an actor ID.

## Poker `[run.game_config]`

The first-party `poker` plugin requires `[run.game_config]` to define
`players`. It rejects unknown keys and values that do not meet the constraints
below. Poker has `requires_exact_agent_ids = true`: configure one `[[agents]]`
entry for every string in `players`, with exactly the same IDs and no extras.

| Key | Type, default, and constraints | Effect |
| --- | --- | --- |
| `players` | Required list of strict nonblank strings. At least `2`, unique, and limited to `22` players (`2 * player_count + 8 <= 52`). | Defines the poker seats and the required agent IDs. Each player starts every match with `initial_stack`. |
| `initial_stack` | Strict integer, at least `1`; default `1000`. It must be at least `big_blind`. | Starting chips for every player at the beginning of a match. Stacks persist across hands within that match. |
| `small_blind` | Strict integer, at least `1`; default `5`. It must be less than `big_blind`. | Small blind amount posted for each hand. |
| `big_blind` | Strict integer, at least `1`; default `10`. It must be greater than `small_blind`, and no greater than `initial_stack`. | Big blind amount posted for each hand and the minimum stack coverage requirement. |
| `hands_per_match` | Strict integer, at least `1`; default `50`. | Maximum hands played in a match. A match can end earlier when fewer than two players remain active. |
| `invalid_turn_policy` | One of `"forced_fold"` or `"fail_match"`; default `"forced_fold"`. | Handles a turn whose invalid-action budget is exhausted. `forced_fold` folds the current player and continues when possible; `fail_match` marks the poker match failed. |
| `memory_max_entries` | Strict integer, at least `0`; default `100`. | Maximum private agent-authored memory notes per actor per match. `0` rejects all note writes. Plugin-provided system summaries do not count toward this limit. |
| `memory_max_chars` | Strict integer, at least `0`; default `20000`. | Maximum total characters in private agent-authored notes per actor per match. `0` rejects all nonempty note writes. System summaries do not count toward this limit. |

For example, these poker player IDs require exactly two agent entries with
`id = "player-1"` and `id = "player-2"`:

```toml
[run.game_config]
players = ["player-1", "player-2"]
```
