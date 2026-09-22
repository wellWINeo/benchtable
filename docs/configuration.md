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
values are rejected rather than coerced. The top level accepts only `run`,
`agents`, and `judges`; unknown top-level keys and unknown `[run]` keys are
rejected.

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
`model`, and provider credential fields must not be empty or whitespace-only.
Every agent ID must be unique.

| Key | Type, default, and constraints | Effect |
| --- | --- | --- |
| `id` | Required strict nonblank string. | Identifies this configured agent. With multiple agents, it maps directly to a game actor ID. |
| `role` | String; default `"player"`; no nonblank constraint. | Metadata recorded in the trace. It does not control turn assignment or game behavior. |
| `model` | Required strict nonblank string. | Model value sent to the OpenAI-compatible Chat Completions endpoint. |
| `provider` | One of `"openai_compatible"`, `"gigachat"`, or `"yandex_ai_studio"`; default `"openai_compatible"`. | Selects the provider adapter. Omitting it preserves existing OpenAI-compatible configurations. |
| `base_url` | OpenAI-compatible only; optional strict nonblank string. | Overrides the OpenAI-compatible endpoint. |
| `api_key_env` | OpenAI-compatible only; default `"OPENAI_API_KEY"`. | Names the environment variable containing the credential. |
| `credential_env` | Required for GigaChat and Yandex; strict nonblank string. | Names the environment variable containing the provider credential. |
| `scope` | GigaChat only; strict nonblank string; default `"GIGACHAT_API_PERS"`. | Selects the GigaChat API scope. |
| `folder_id` | Yandex only; required strict nonblank string. | Selects the Yandex Cloud folder billed for inference. |
| `credential_kind` | Yandex only; `"oauth"` or `"api_key"`, default `"oauth"`. | Documents the credential type while the SDK receives the unmodified value and handles OAuth renewal. |
| `timeout` | Optional finite positive number; default omitted (`None`). Integers and floats are accepted. | Passed as the provider request timeout. |
| `max_completion_tokens` | Optional strict integer, at least `1`; default omitted (`None`). | When supplied, is sent as `max_completion_tokens` on provider requests. |

The selected adapter reads its configured environment variable when it is
constructed. The named variable must exist and contain a non-whitespace value
or `benchtable run` exits before making a provider request. Never place a
credential value in TOML. Configure the environment in the shell that runs the
command, for example:

```bash
export OPENAI_API_KEY='...'
uv run benchtable run --config experiment.toml --output ./runs/my-experiment
```

Do not put a literal API key in TOML. The loader rejects credential-shaped keys
recursively anywhere in the file, including nested game configuration. This
includes normalized variants of names such as `api_key`, `authorization`,
`access_token`, `password`, `client_secret`, and `token`.

Retries belong to the engine, not the SDK clients: OpenAI-compatible and
GigaChat clients disable SDK retries so each engine attempt can be traced.
There are no other generic generation settings. In particular, `max_tokens`,
temperature, top-p, and provider-specific settings are not accepted generic
agent keys; use only `max_completion_tokens` where supported by this schema.

## `[[judges]]` Entries

Judges are optional typed-decision models used by games that request
pre-action judgments. The top level accepts any number of `[[judges]]`
tables; judge IDs must be unique. Unknown judge keys are rejected.

| Key | Type, default, and constraints | Effect |
| --- | --- | --- |
| `id` | Required strict nonblank string. | Names the judge; games reference it from game configuration (for example `judge_id`). |
| `adapter` | One of `"openrouter_decisions"`; default `"openrouter_decisions"`. | Selects the judge adapter. |
| `model` | Required strict nonblank string. | Pinned model ID sent to the adapter. Rolling aliases (any ID starting with `~`) are rejected so a threshold stays calibrated against one model. |
| `base_url` | Optional strict nonblank string. | Overrides the Decisions endpoint; default `https://openrouter.ai/api/alpha/decisions`. |
| `api_key_env` | Strict nonblank string; default `"OPENROUTER_API_KEY"`. | Names the environment variable containing the credential. |
| `timeout` | Optional finite positive number. | Per-request timeout in seconds; the adapter default is `30` when omitted. |
| `max_retries` | Strict integer, at least `0`; default `2`. | Engine-owned retry budget: each judgment is attempted at most `max_retries + 1` times. |

Like agents, judges read their credential from the environment at
construction; `benchtable run` exits before any provider request when the
named variable is missing or blank, and no literal key is ever accepted or
recorded. The first-party Spyfall plugin is the current consumer; see its
section below and docs/spyfall-judge-calibration.md before selecting a
leak threshold.

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

Plugins may also expose an optional `min_max_turns(game_config)` hook. When
present, the CLI rejects any configuration whose `run.max_turns` is below the
returned minimum before constructing agents. A missing or non-integer return
is a plugin error; a plugin without the hook is unconstrained.

## Provider Examples

OpenAI-compatible remains the default when `provider` is omitted:

```toml
[[agents]]
id = "player-1"
model = "gpt-4o"
api_key_env = "OPENAI_API_KEY"
```

GigaChat uses its long-lived authorization credential. The SDK exchanges it
for access tokens and renews them as needed; TLS verification remains enabled
and SDK retries are disabled. Its scope defaults to `GIGACHAT_API_PERS` when
omitted.

```toml
[[agents]]
id = "player-1"
provider = "gigachat"
model = "GigaChat-3-Ultra"
credential_env = "GIGACHAT_AUTHORIZATION_KEY"
scope = "GIGACHAT_API_PERS"
```

Yandex AI Studio requires a folder ID. OAuth credentials can be refreshed by
the SDK; an API key can be selected with `credential_kind = "api_key"`.
Direct IAM-token configuration is rejected by the typed configuration.

```toml
[[agents]]
id = "player-1"
provider = "yandex_ai_studio"
model = "aliceai-llm"
folder_id = "folder-123"
credential_kind = "oauth"
credential_env = "YC_OAUTH_TOKEN"
```

Model availability is controlled by the provider account. No credential value,
OAuth token-exchange payload, or direct IAM token belongs in TOML or traces.

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

## Spyfall `[run.game_config]`

The first-party `spyfall` plugin requires the keys below. It rejects unknown
keys, coerced values, and duplicate player or location names. Spyfall has
`requires_exact_agent_ids = true`: configure one `[[agents]]` entry for every
string in `players`, and one `[[judges]]` entry whose `id` matches
`judge_id`.

```toml
[[judges]]
id = "spyfall-leak-judge"
model = "typesafe/jev-1.13"
api_key_env = "OPENROUTER_API_KEY"

[run]
game = "spyfall"
matches = 1
seed = 20260920
max_turns = 135

[run.game_config]
players = ["player-1", "player-2", "player-3"]
rounds_per_match = 5
question_rounds = 3
accusation_vote_threshold = 0.75
judge_id = "spyfall-leak-judge"
judge_leak_threshold = 0.85

[[run.game_config.locations]]
name = "airport"

[[run.game_config.locations]]
name = "underwater research station"
```

| Key | Type, default, and constraints | Effect |
| --- | --- | --- |
| `players` | Required list of strict nonblank strings; `3` through `8` entries, unique. | The round roster and the required agent IDs. Each round's Spy and location are seeded from the match seed. |
| `rounds_per_match` | Strict integer, at least `1`. | Rounds played per match; every round awards one point to each player on its winning side. |
| `question_rounds` | Strict integer, at least `1`. | Complete questioner rotations per round. With three players and `question_rounds = 3`, a round holds up to nine question/answer exchanges. |
| `accusation_vote_threshold` | Strict number in `(0, 1]`. | Conviction needs `ceil(threshold * eligible_voters)` yes votes; the accused is not eligible. |
| `judge_id` | Required strict nonblank string. | The `[[judges]]` entry consulted for every public question and answer. A missing match is a configuration failure. |
| `judge_leak_threshold` | Strict number in `[0, 1]`. | A normalized leak probability at or above this value ends the round in favor of the Spy. Set it explicitly; see docs/spyfall-judge-calibration.md. |
| `locations` | Required list of tables, each with a unique nonblank `name`. | The location catalog the session selects from each round. The engine never selects a location. |

A normal question/answer slot costs two engine turns and a non-convicting
accusation consumes one whole slot at `N` turns, so a match needs at most
`rounds_per_match * players^2 * question_rounds` turns. `run.max_turns` must
be at least that value and the plugin enforces it: with three players, five
rounds, and three question rotations the minimum is `5 * 3^2 * 3 = 135`, and
`benchtable run` rejects smaller values before any provider request. Judge
calls and finalization calls add provider requests but not engine turns.

## Bunker `[run.game_config]`

The first-party `bunker` plugin is a deterministic, Bunker-inspired survival
selection game. Players hold private seeded dossiers, discuss and optionally
reveal facts about themselves, and secretly vote in elimination rounds until
the survivors fit the public shelter capacity. It rejects unknown keys and
values that do not meet the constraints below. Bunker has
`requires_exact_agent_ids = true`: configure one `[[agents]]` entry for every
string in `players`, with exactly the same IDs and no extras.

| Key | Type, default, and constraints | Effect |
| --- | --- | --- |
| `players` | Required list of strict nonblank strings. At least `4`, at most `8`, unique. | Defines the contestants and the required agent IDs. |
| `scenario` | Required strict nonblank string. | Public crisis text shown to every player from match start. |
| `shelter_capacity` | Required strict integer from `1` through one fewer than the player count. | Number of survivors the shelter admits; the match ends when the survivor count reaches it. |

Dossiers are authored by this project: original, neutral values across the
`profession`, `health`, `skill`, and `trait` categories, never copied from a
commercial game, with sensitive real-world attributes deliberately excluded.
The engine-provided match seed assigns dossiers and resolves vote ties, so an
identical seed and configuration produce identical matches.

Each elimination round runs one discussion phase, where every surviving
player calls `bunker_speak` once (at most 500 characters, optionally revealing
one still-hidden category of their own dossier), followed by a secret ballot
in which every survivor calls `bunker_vote_eliminate` targeting another
surviving player; abstention and self-voting are rejected. Only aggregate
totals, the eliminated player, and whether a seeded tie break occurred become
public. A game-attributable failed turn eliminates only the failed player for
that round. Survivors at capacity win, and the result reports
`completion_reason = "capacity_reached"`.

Normal play needs exactly
`player_count * (player_count + 1) - shelter_capacity * (shelter_capacity + 1)`
engine turns: one discussion and one ballot action per survivor per
elimination round. Eight players with capacity `1` need `70` turns. Configure
`[run].max_turns` at least to that value plus any headroom; the CLI rejects
lower values before constructing any agent by consulting the plugin's
`min_max_turns` hook.

For example, the configuration below needs `4 * 5 - 2 * 3 = 14` turns, so
`max_turns = 20` leaves headroom:

```toml
[run]
game = "bunker"
matches = 1
max_turns = 20

[run.game_config]
players = ["player-1", "player-2", "player-3", "player-4"]
scenario = "sealed shelter after a solar storm"
shelter_capacity = 2
```
