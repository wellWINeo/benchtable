# Native Provider Adapters Design

## Goal

Add first-class GigaChat and Yandex AI Studio agent providers without changing
the engine's canonical `Agent` protocol, OpenAI-style transcript, game rules,
turn budgeting, or trace sequencing.

The design supports GigaChat 3 Ultra and Yandex `aliceai-llm`. It uses each
service's official Python SDK and supports automatic renewal of expiring
credentials.

## Research Findings

### GigaChat

GigaChat documents partial OpenAI compatibility at
`https://developers.sber.ru/docs/ru/gigachat/guides/compatible-openai`. Its
access token expires after 30 minutes and is obtained from an authorization
credential. Its documented custom-function protocol uses legacy `functions`
and `function_call` fields, rather than OpenAI's `tools` and `tool_calls`.

GigaChat 3 Ultra is called through the `/v1/chat/completions` endpoint with
model identifier `GigaChat-3-Ultra`. The model page currently states that it
is available in the personal freemium offering. All GigaChat models support
custom functions according to the function-calling guide.

The current `OpenAICompatibleAgent` cannot renew GigaChat credentials or
translate the legacy function and transcript protocol. GigaChat therefore
needs a dedicated adapter.

### Yandex AI Studio

The official Yandex AI Studio SDK exposes native chat completions for
`aliceai-llm`, resolves model URIs from a folder ID, supports tools,
`tool_choice`, and parallel-tool-call control, and has provider-specific
authentication behavior. OAuth credentials can be exchanged and refreshed by
the SDK; API keys are long-lived and require no refresh.

The Yandex AI Studio documentation site presents SmartCaptcha to automated
requests. These findings use the official SDK documentation, which specifies
the same native chat, tool, model-URI, and authentication behavior. The existing
OpenAI SDK adapter cannot supply Yandex's SDK model URI or credential lifecycle,
so Yandex also needs a dedicated adapter.

## Architecture

The engine, game plugins, `Agent` protocol, and canonical OpenAI-style
transcript remain provider-neutral. Provider dispatch happens only in the CLI
agent factory.

Three adapters implement `Agent`:

- `OpenAICompatibleAgent` remains the default for existing configurations.
- `GigaChatAgent` wraps the official GigaChat SDK.
- `YandexAIStudioAgent` wraps the official Yandex AI Studio SDK.

The native adapter clients are constructed through injectable factories so all
tests run without service credentials or network access. SDK-level inference
retries are disabled where configurable; a single `respond` invocation is one
engine provider attempt. The engine remains the sole owner of retry budgets,
failure outcomes, latency accounting, cancellation, and event sequencing.

### GigaChat Adapter

`GigaChatAgent` receives an authorization-credential environment-variable name
and GigaChat scope. It delegates access-token acquisition and renewal to the
native SDK, keeps TLS verification enabled, disables SDK inference retries, and
sends the model configured by the user through the SDK's native asynchronous
chat method.

The adapter translates canonical `ToolSpec` values into GigaChat legacy
`functions` with `function_call="auto"`. It translates canonical transcript
messages in both directions:

- Canonical assistant `tool_calls` become the corresponding legacy assistant
  `function_call` message.
- Canonical `role="tool"` result messages become legacy
  `role="function"` messages with the correlated function name.
- Legacy `function_call` responses become one canonical `ToolCall`.

GigaChat legacy function responses do not contain a call identifier. The
adapter generates a unique non-secret call ID so the engine can preserve its
canonical transcript and associate later function results correctly. Native
arguments are normalized to the same parsed and raw-argument semantics as the
existing adapter. The adapter stores GigaChat's `functions_state_id` internally
against that generated call ID and restores it when it converts the canonical
assistant/function-result history into the legacy continuation format. That
state is neither part of the engine transcript nor written to traces.

### Yandex Adapter

`YandexAIStudioAgent` receives the configured model name, a Yandex Cloud folder
ID, credential kind, and credential environment-variable name. The initial
credential kinds are:

- `oauth`, which delegates IAM-token renewal to the official SDK.
- `api_key`, which supplies a long-lived API key without a renewal step.

Direct IAM-token configuration is excluded because it cannot meet the
automatic-refresh requirement. The adapter delegates model URI creation to the
official SDK, maps canonical tools and transcript messages into the SDK's
native request format, and normalizes text, tool calls, usage, finish reason,
and raw response back into `ModelResponse`.

It requests `parallel_tool_calls=False`, ensuring provider behavior matches the
engine's exactly-one-action requirement. Any unexpected multiple game calls
remain an engine validation failure, as they do for every other provider.

## Configuration

Agent entries gain a strict `provider` discriminator. When omitted, it defaults
to `openai_compatible` to preserve the repository's documented configuration
shape. Each provider accepts only its own fields; unrelated fields are rejected
before an SDK client is constructed.

```toml
[[agents]]
id = "player-1"
role = "player"
provider = "gigachat"
model = "GigaChat-3-Ultra"
credential_env = "GIGACHAT_AUTHORIZATION_KEY"
scope = "GIGACHAT_API_PERS"

[[agents]]
id = "player-2"
role = "player"
provider = "yandex_ai_studio"
model = "aliceai-llm"
folder_id = "your-folder-id"
credential_kind = "oauth"
credential_env = "YC_OAUTH_TOKEN"
```

`credential_env` records only an environment-variable name. Literal keys,
tokens, authorization credentials, and nested credential-shaped configuration
values remain rejected. The existing OpenAI-compatible fields, including
`base_url` and `api_key_env`, remain exclusive to the default provider.

## Traces And Security

`ModelResponse` gains an optional `raw_provider_request` field alongside its
existing raw response. `ProviderError` gains the same optional diagnostic field
for failed calls. The engine includes those generic fields in response and
provider-error trace payloads without branching on provider type. The event
writer remains the only serialization and recursive credential-redaction
boundary.

Credential acquisition and renewal are private SDK operations; their
credential-bearing requests and responses are not persisted. Renewal failures
are normalized as `ProviderError` values with the provider and model but never
credential values.

GigaChat `reasoning_content`, and comparable native reasoning fields, are
discarded. Benchtable neither requests nor stores hidden reasoning.

## Failure Handling

The configuration loader rejects unsupported providers, blank credential
environment-variable names, provider-field mismatches, missing Yandex folder
IDs, and unsupported Yandex credential kinds before any provider request.

The adapters normalize native SDK errors, malformed responses, invalid legacy
function history, missing correlations, and credential-renewal failures to
`ProviderError` values with their concrete provider and model fields. Errors
retain safe request/response diagnostics when available. Engine retry behavior
and explicit failed-match outcomes are unchanged.

## Testing

All tests are offline.

- Add configuration tests for valid provider examples, default OpenAI
  compatibility, required fields, mutually exclusive or misplaced provider
  fields, credential-kind validation, and literal-secret rejection.
- Add isolated adapter tests using injected native-SDK fakes. Verify client
  setup, OAuth-refresh delegation, request and transcript conversion, tool
  conversion, response normalization, malformed output, timeout/error wrapping,
  and credential-safe messages.
- Add engine and trace tests using the provider fakes. Verify action sequencing,
  translated native request retention, recursive redaction, canonical
  tool-result continuity, exactly-one-action enforcement, and actor isolation.
- Update configuration documentation with GigaChat 3 Ultra and `aliceai-llm`
  examples, credential setup requirements, and the no-live-tests policy.
- Add the two official SDKs as explicit runtime dependencies and run the full
  repository verification suite without making live calls.

## Acceptance Criteria

- Existing OpenAI-compatible configurations continue to construct the existing
  adapter.
- GigaChat runs use automatic access-token renewal and correctly complete
  legacy function-call loops through the canonical engine transcript.
- Yandex `aliceai-llm` runs use the official SDK with a folder ID and either
  refreshed OAuth credentials or a long-lived API key.
- No provider-specific behavior is added to the run engine or game plugins.
- Every provider failure is explicit, credential-safe, and governed by the
  existing engine retry budget.
- Traces retain provider-native inference requests and raw responses after
  redaction, but never retain literal credentials or hidden reasoning.
- The complete offline test, format, lint, type-check, and lockfile-validation
  suite passes.
