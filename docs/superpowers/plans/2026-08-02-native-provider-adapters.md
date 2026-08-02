# Native Provider Adapters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add offline-testable native GigaChat and Yandex AI Studio agents with automatic expiring-credential handling, while preserving the provider-neutral run engine.

**Architecture:** Agent configuration selects `OpenAICompatibleAgent`, `GigaChatAgent`, or `YandexAIStudioAgent` in the CLI factory. Every adapter consumes the engine's canonical `AgentRequest` and produces the existing normalized response shape; native protocol translation and authentication remain inside the adapter. The engine gains generic native-request diagnostics only, never provider-specific branches.

**Tech Stack:** Python 3.12+, Pydantic 2, `openai`, `gigachat>=0.2.3,<1`, `yandex-ai-studio-sdk>=0.22.1,<1`, Typer, pytest, pytest-asyncio, Ruff, and Pyright.

## Global Constraints

- Preserve `Agent.respond(request: AgentRequest) -> ModelResponse` as the engine-facing boundary.
- Keep game state, rules, plugin APIs, turn budgets, retries, cancellation, and event sequencing provider-neutral.
- Keep all provider calls asynchronous; call GigaChat through `await client.achat(payload)` and use `AsyncAIStudio` for Yandex.
- Configure GigaChat with `max_retries=0` and do not add adapter-owned inference retries.
- Never accept, log, persist, print, or add to test fixtures any literal credential, authorization header, access token, OAuth token, API key, or reasoning content.
- The event writer remains the credential-redaction boundary; provider-native inference requests and raw responses must pass through it.
- Preserve existing OpenAI-compatible TOML by treating an omitted `provider` as `openai_compatible`.
- All new provider tests use injected fakes and make no network requests.

---

### Task 1: Add Strict Provider Configuration

**Files:**
- Modify: `src/benchtable/config.py`
- Modify: `tests/unit/test_config.py`

**Interfaces:**
- Consumes: existing `ExperimentConfig`, `AgentConfig`, `load_config`, and credential-shaped-key rejection.
- Produces: `AgentConfig` with `provider`, `credential_env`, `scope`, `folder_id`, and `credential_kind` fields for CLI factory dispatch.

- [ ] **Step 1: Add failing provider-configuration tests**

Add tests that load the following valid configurations and assert their typed fields and serialized trace metadata. Also add parameterized invalid cases for each prohibited field combination.

```toml
[[agents]]
id = "a"
model = "gpt-4o"
# provider intentionally omitted
```

```toml
[[agents]]
id = "a"
provider = "gigachat"
model = "GigaChat-3-Ultra"
credential_env = "GIGACHAT_AUTHORIZATION_KEY"
scope = "GIGACHAT_API_PERS"
```

```toml
[[agents]]
id = "a"
provider = "yandex_ai_studio"
model = "aliceai-llm"
folder_id = "folder-123"
credential_kind = "oauth"
credential_env = "YC_OAUTH_TOKEN"
```

Cover these invalid conditions explicitly:

```python
@pytest.mark.parametrize(
    "agent_fields",
    [
        'provider = "gigachat"\nmodel = "GigaChat-3-Ultra"',
        'provider = "yandex_ai_studio"\nmodel = "aliceai-llm"\ncredential_env = "YC_OAUTH_TOKEN"',
        'provider = "yandex_ai_studio"\nmodel = "aliceai-llm"\nfolder_id = "folder"\ncredential_kind = "iam"\ncredential_env = "TOKEN"',
        'provider = "gigachat"\nmodel = "GigaChat-3-Ultra"\ncredential_env = "KEY"\napi_key_env = "OPENAI_API_KEY"',
        'provider = "openai_compatible"\nmodel = "gpt-4o"\ncredential_env = "KEY"',
    ],
)
def test_rejects_invalid_provider_field_combinations(
    tmp_path: Path, agent_fields: str
) -> None:
    cfg_path = tmp_path / "invalid-provider.toml"
    cfg_path.write_text(
        "[run]\ngame = \"tiny\"\nmatches = 1\n\n[[agents]]\nid = \"a\"\n"
        + agent_fields
        + "\n"
    )

    with pytest.raises(ConfigurationError):
        load_config(cfg_path)
```

Assert that provider-specific error messages do not echo credential environment-variable names that look like credentials or any literal configured secret.

- [ ] **Step 2: Run the focused tests to verify failure**

Run: `uv run pytest tests/unit/test_config.py -q`

Expected: FAIL because `provider`, `credential_env`, `scope`, `folder_id`, and `credential_kind` are unknown configuration fields.

- [ ] **Step 3: Extend `AgentConfig` and validate provider fields**

Add provider literals and optional provider-specific fields to the existing model so direct construction and `load_config()` retain one public type:

```python
ProviderName = Literal["openai_compatible", "gigachat", "yandex_ai_studio"]
YandexCredentialKind = Literal["oauth", "api_key"]


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: StrictStr = Field(min_length=1)
    role: str = "player"
    model: StrictStr = Field(min_length=1)
    provider: ProviderName = "openai_compatible"
    timeout: float | None = Field(default=None, gt=0)
    max_completion_tokens: StrictInt | None = Field(default=None, ge=1)
    base_url: StrictStr | None = Field(default=None, min_length=1)
    api_key_env: StrictStr | None = Field(default=None, min_length=1)
    credential_env: StrictStr | None = Field(default=None, min_length=1)
    scope: StrictStr | None = Field(default=None, min_length=1)
    folder_id: StrictStr | None = Field(default=None, min_length=1)
    credential_kind: YandexCredentialKind | None = None
```

In an `after` model validator, inspect `model_fields_set` so a field explicitly supplied for the wrong provider is rejected even if its value equals a default. Apply exactly these rules:

- `openai_compatible`: permit `base_url` and `api_key_env`; default a missing `api_key_env` to `OPENAI_API_KEY`; reject `credential_env`, `scope`, `folder_id`, and `credential_kind`.
- `gigachat`: require `credential_env`; default a missing `scope` to `GIGACHAT_API_PERS`; reject `base_url`, `api_key_env`, `folder_id`, and `credential_kind`.
- `yandex_ai_studio`: require `credential_env` and `folder_id`; default a missing `credential_kind` to `oauth`; reject `base_url`, `api_key_env`, and `scope`.

Include `provider`, `credential_env`, `scope`, `folder_id`, and `credential_kind` in the existing nonblank-string validation. Keep the recursive credential-key scanner unchanged: it rejects values, while typed `*_env` fields hold environment-variable names only.

- [ ] **Step 4: Run configuration tests and type checking**

Run: `uv run pytest tests/unit/test_config.py -q`

Expected: PASS.

Run: `uv run pyright src/benchtable/config.py`

Expected: PASS.

- [ ] **Step 5: Commit the provider configuration**

```bash
git add src/benchtable/config.py tests/unit/test_config.py
git commit -m "feat: add provider agent configuration"
```

### Task 2: Preserve Native Inference Request Diagnostics

**Files:**
- Modify: `src/benchtable/contracts.py`
- Modify: `src/benchtable/errors.py`
- Modify: `src/benchtable/agents/openai_compatible.py`
- Modify: `src/benchtable/engine.py`
- Modify: `tests/unit/test_contracts.py`
- Modify: `tests/unit/test_openai_compatible_agent.py`
- Modify: `tests/integration/test_engine_trace.py`

**Interfaces:**
- Consumes: normalized `ModelResponse`, `ProviderError`, and existing model-response/provider-error trace events.
- Produces: `raw_provider_request: JsonObject | None` on successful and failed provider interactions.

- [ ] **Step 1: Add failing contract, adapter, and trace tests**

Add a `ModelResponse` test that preserves a JSON-compatible raw request, and a `ProviderError` test that preserves the same field. Extend the OpenAI adapter test to assert its result contains the exact SDK request payload. Add a trace test with a fake agent returning:

```python
ModelResponse(
    assistant_text="",
    tool_calls=[ToolCall(call_id="call-1", name="act", arguments={})],
    raw_provider_request={"provider": "fake", "request": {"model": "test"}},
)
```

Assert that `model_response.payload["raw_provider_request"]` equals the value above. Add a provider-failure fake that raises:

```python
ProviderError(
    "failed",
    provider="fake",
    model="test",
    raw_provider_request={"authorization": "Bearer should-redact"},
)
```

Assert that the `provider_error` event retains the field but its authorization value is redacted.

- [ ] **Step 2: Run focused tests to verify failure**

Run: `uv run pytest tests/unit/test_contracts.py tests/unit/test_openai_compatible_agent.py tests/integration/test_engine_trace.py -q`

Expected: FAIL because the two contracts and event payloads have no raw request field.

- [ ] **Step 3: Add generic raw-request fields and trace emission**

Add the optional field to the normalized contracts:

```python
class ModelResponse(BaseModel):
    assistant_text: str
    tool_calls: list[ToolCall]
    finish_reason: str | None = None
    usage: Usage | None = None
    raw_provider_request: JsonObject | None = None
    raw_provider_response: JsonObject | None = None


class ProviderError(BenchtableError):
    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        raw_provider_request: JsonObject | None = None,
        raw_provider_response: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.raw_provider_request = raw_provider_request
        self.raw_provider_response = raw_provider_response
```

Build the OpenAI request payload once before the SDK call. Use it both as the
`chat.completions.create(**request_kwargs)` argument and as
`ModelResponse.raw_provider_request`. Pass it to `_provider_error()` on SDK
construction, request, and normalization failures where it exists.

In `RunEngine._get_response`, add `raw_provider_request` to `model_response`
payloads and `provider_error` payloads. Do not condition engine event selection
on provider type; the event writer's existing recursive redaction must handle
every value.

- [ ] **Step 4: Run focused tests and formatting**

Run: `uv run pytest tests/unit/test_contracts.py tests/unit/test_openai_compatible_agent.py tests/integration/test_engine_trace.py -q`

Expected: PASS.

Run: `uv run ruff format --check src/benchtable/contracts.py src/benchtable/errors.py src/benchtable/agents/openai_compatible.py src/benchtable/engine.py tests/unit/test_contracts.py tests/unit/test_openai_compatible_agent.py tests/integration/test_engine_trace.py`

Expected: PASS.

- [ ] **Step 5: Commit generic provider diagnostics**

```bash
git add src/benchtable/contracts.py src/benchtable/errors.py src/benchtable/agents/openai_compatible.py src/benchtable/engine.py tests/unit/test_contracts.py tests/unit/test_openai_compatible_agent.py tests/integration/test_engine_trace.py
git commit -m "feat: trace native provider requests"
```

### Task 3: Add The GigaChat Agent

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `src/benchtable/agents/gigachat.py`
- Modify: `src/benchtable/agents/__init__.py`
- Create: `tests/unit/test_gigachat_agent.py`

**Interfaces:**
- Consumes: `AgentRequest`, `ToolSpec`, `ToolCall`, `ModelResponse`, and `ProviderError`.
- Produces: `GigaChatAgent(model, credential_env, scope, timeout, max_completion_tokens, _client_factory)` implementing `Agent`.

- [ ] **Step 1: Write failing GigaChat adapter tests**

Create fakes with `achat` as an async callable and `model_dump(mode="json")` on
responses. Test these concrete behaviors:

- Constructor requires a nonblank model, credential environment-variable name, and valid optional timeout/token limit; missing or whitespace-only credential values raise `ConfigurationError` without revealing the environment-variable name or value.
- The lazy client factory receives `credentials`, `scope`, `model`, `verify_ssl_certs=True`, `max_retries=0`, and configured timeout.
- The initial request contains a system message, canonical user messages, model, legacy `functions`, `function_call="auto"`, and the mapped native `max_tokens` limit.
- A native response with `function_call={"name": "act", "arguments": {"choice": "go"}}` produces one `ToolCall` with parsed arguments, a generated nonblank call ID, and raw arguments equal to compact JSON.
- A response with no function call retains assistant text, finish reason, usage, and raw native response while discarding `reasoning_content`.
- A second request containing the generated canonical assistant tool call and its `role="tool"` result becomes a GigaChat assistant `function_call` message plus `role="function"`, `name="act"`, and the original `functions_state_id`.
- The retained raw provider response omits `functions_state_id` because that value is only an in-memory continuation correlation key.
- Missing tool-result correlation, two historical calls in one assistant message, malformed native function arguments, a malformed response, and an SDK exception raise `ProviderError(provider="gigachat", model="GigaChat-3-Ultra")` with a safe raw request and no credential content.

Use the following response fixture shape for the correlation test:

```python
{
    "choices": [{
        "message": {
            "role": "assistant",
            "content": "",
            "function_call": {"name": "act", "arguments": {"choice": "go"}},
            "functions_state_id": "provider-state-1",
            "reasoning_content": "must not be retained",
        },
        "finish_reason": "function_call",
    }],
    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
}
```

- [ ] **Step 2: Run the GigaChat test to verify failure**

Run: `uv run pytest tests/unit/test_gigachat_agent.py -q`

Expected: FAIL because the adapter module and native SDK dependency do not exist.

- [ ] **Step 3: Add the SDK dependency and lockfile**

Run: `uv add 'gigachat>=0.2.3,<1'`

Expected: `pyproject.toml` and `uv.lock` add the GigaChat SDK without upgrading unrelated direct dependencies.

- [ ] **Step 4: Implement the GigaChat adapter**

Use a lazy injected `GigaChat` client factory. Construct the production client
with the long-lived authorization credential read from `credential_env`, the
configured scope/model/timeout, `verify_ssl_certs=True`, and `max_retries=0`.

Build a JSON-compatible native request dictionary first, then validate/send the
same data through the SDK's legacy chat model:

```python
native_request = {
    "model": self._model,
    "messages": self._to_gigachat_messages(request),
    "functions": [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
        for tool in request.tools
    ],
    "function_call": "auto" if request.tools else "none",
}
if self._max_completion_tokens is not None:
    native_request["max_tokens"] = self._max_completion_tokens

response = await self._get_client().achat(native_request)
```

Implement `_to_gigachat_messages()` as a chronological state machine:

1. Start with the system message from `AgentRequest.system_prompt`.
2. Copy ordinary user and assistant-text messages after validating string
   content.
3. When a canonical assistant message has one `tool_calls` item, record
   `call_id -> function name`, recover a previously stored
   `call_id -> functions_state_id`, and emit the legacy assistant
   `function_call` object.
4. When a canonical `role="tool"` message follows, require a known call ID,
   emit `{"role": "function", "name": function_name, "content": content}`,
   and remove the pending correlation.
5. Reject unfinished, duplicated, unknown, or multiple-call history with a
   contextual `ProviderError` before a native request is sent.

When a native function response supplies `functions_state_id`, store it by the
generated `ToolCall.call_id`. Generate the ID with `uuid4().hex`, serialize an
object argument with `json.dumps(arguments, separators=(",", ":"))`, and use
the existing finite-JSON validation behavior for malformed values. Before
assigning the raw native response, remove `reasoning_content` and
`functions_state_id` recursively. Never copy either value into `ModelResponse`
or `raw_provider_request`.

Wrap all SDK failures with `ProviderError(provider="gigachat", model=self._model,
raw_provider_request=native_request,
raw_provider_response=self._raw_provider_response(response))`. Use
`model_dump(mode="json")` only when it is available and JSON-compatible.

- [ ] **Step 5: Run focused GigaChat verification**

Run: `uv run pytest tests/unit/test_gigachat_agent.py -q`

Expected: PASS.

Run: `uv run pyright src/benchtable/agents/gigachat.py`

Expected: PASS.

- [ ] **Step 6: Commit the GigaChat adapter**

```bash
git add pyproject.toml uv.lock src/benchtable/agents/__init__.py src/benchtable/agents/gigachat.py tests/unit/test_gigachat_agent.py
git commit -m "feat: add GigaChat agent adapter"
```

### Task 4: Add The Yandex AI Studio Agent

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `src/benchtable/agents/yandex_ai_studio.py`
- Modify: `src/benchtable/agents/__init__.py`
- Create: `tests/unit/test_yandex_ai_studio_agent.py`

**Interfaces:**
- Consumes: `AgentRequest`, `ToolSpec`, `ToolCall`, `ModelResponse`, `ProviderError`, and Yandex `AsyncAIStudio`.
- Produces: `YandexAIStudioAgent(model, folder_id, credential_kind, credential_env, timeout, max_completion_tokens, _client_factory)` implementing `Agent`.

- [ ] **Step 1: Write failing Yandex adapter tests**

Use an injected `AsyncAIStudio` fake that records its construction fields and
returns a fake configured chat-completion model. Cover:

- Required nonblank model, folder ID, and credential environment-variable name; only `oauth` and `api_key` are accepted credential kinds.
- The client factory receives `folder_id` and the credential value from the configured environment variable. OAuth and API-key cases pass their unmodified secret to the SDK, so SDK auto-authentication can refresh OAuth-derived IAM credentials.
- The adapter selects `sdk.chat.completions(model)`, configures `tools`, `parallel_tool_calls=False`, and maps `max_completion_tokens` to native `max_tokens`.
- A canonical system prompt and transcript are supplied as OpenAI-compatible messages, including assistant tool calls and matching tool results.
- Native text, one native function call, usage, finish reason, and raw response normalize to `ModelResponse`; malformed arguments retain raw content and a parse error.
- Native `reasoning_content` is omitted from normalized fields and the retained raw provider response before event serialization.
- Client construction, model execution, and normalization failures become safe `ProviderError(provider="yandex_ai_studio", model="aliceai-llm")` values with a raw native request.

Use this expected configured-model call shape:

```python
configured = sdk.chat.completions("aliceai-llm").configure(
    tools=expected_tools,
    parallel_tool_calls=False,
    max_tokens=128,
)
result = await configured.run(expected_messages, timeout=30.0)
```

- [ ] **Step 2: Run the Yandex test to verify failure**

Run: `uv run pytest tests/unit/test_yandex_ai_studio_agent.py -q`

Expected: FAIL because the adapter module and Yandex SDK dependency do not exist.

- [ ] **Step 3: Add the SDK dependency and lockfile**

Run: `uv add 'yandex-ai-studio-sdk>=0.22.1,<1'`

Expected: `pyproject.toml` and `uv.lock` include the official Yandex SDK and its required transport/auth dependencies.

- [ ] **Step 4: Implement the Yandex adapter**

Create the client lazily through an injected factory whose production
implementation constructs `AsyncAIStudio(folder_id=folder_id, auth=credential)`.
Read the credential only from `credential_env`. Do not add direct IAM-token
configuration, token exchange code, provider retries, or a second credential
store.

Build an OpenAI-compatible native request payload for tracing and derive the
SDK call from it:

```python
native_request = {
    "model": self._model,
    "messages": [
        {"role": "system", "content": request.system_prompt},
        *request.messages,
    ],
    "tools": [tool.to_openai_tool() for tool in request.tools],
    "parallel_tool_calls": False,
}
if self._max_completion_tokens is not None:
    native_request["max_tokens"] = self._max_completion_tokens

completion = self._get_client().chat.completions(self._model).configure(
    tools=native_request["tools"],
    parallel_tool_calls=False,
    max_tokens=native_request.get("max_tokens"),
)
response = await completion.run(native_request["messages"], timeout=self._timeout)
```

Omit `max_tokens` from `configure()` when no completion limit is configured.
Normalize one result choice into the existing `ModelResponse` shape. Retain
only JSON-compatible raw request/response data and remove reasoning fields from
the raw response recursively before assigning it. Preserve provider call IDs
when the native result supplies them; generate a UUID call ID only when the SDK
does not supply one. Use the same finite JSON argument parsing rules as the
OpenAI adapter.

- [ ] **Step 5: Run focused Yandex verification**

Run: `uv run pytest tests/unit/test_yandex_ai_studio_agent.py -q`

Expected: PASS.

Run: `uv run pyright src/benchtable/agents/yandex_ai_studio.py`

Expected: PASS.

- [ ] **Step 6: Commit the Yandex adapter**

```bash
git add pyproject.toml uv.lock src/benchtable/agents/__init__.py src/benchtable/agents/yandex_ai_studio.py tests/unit/test_yandex_ai_studio_agent.py
git commit -m "feat: add Yandex AI Studio agent adapter"
```

### Task 5: Dispatch Providers, Document Configuration, And Verify End To End

**Files:**
- Modify: `src/benchtable/cli.py`
- Modify: `tests/integration/test_cli.py`
- Modify: `docs/configuration.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: validated `AgentConfig.provider` and the three adapter constructors.
- Produces: CLI construction of each configured provider and documented no-secret configuration examples.

- [ ] **Step 1: Add failing CLI integration tests**

Inject stub constructors for each concrete adapter and assert that
`_default_agent_factory()` dispatches with exactly the typed provider fields.
Cover an omitted provider selecting `OpenAICompatibleAgent`, a GigaChat
configuration selecting `GigaChatAgent`, and a Yandex OAuth configuration
selecting `YandexAIStudioAgent`.

Add a complete offline CLI test using a GigaChat-configured fake agent factory
and `TinyGame`. Assert its `run_config` event records:

```python
{
    "provider": "gigachat",
    "credential_env": "GIGACHAT_AUTHORIZATION_KEY",
    "scope": "GIGACHAT_API_PERS",
}
```

and does not contain the environment value. Add the same assertion for Yandex
OAuth configuration with `credential_kind`, `credential_env`, and `folder_id`.

- [ ] **Step 2: Run CLI tests to verify failure**

Run: `uv run pytest tests/integration/test_cli.py -q`

Expected: FAIL because the default factory always constructs `OpenAICompatibleAgent`.

- [ ] **Step 3: Dispatch the typed provider configurations**

Replace the single constructor call in `_default_agent_factory()` with a
closed `match` over the validated provider literal:

```python
match agent_config.provider:
    case "openai_compatible":
        api_key_env = agent_config.api_key_env
        if api_key_env is None:
            raise AssertionError("validated OpenAI configuration lacks api_key_env")
        return OpenAICompatibleAgent(
            model=agent_config.model,
            api_key_env=api_key_env,
            base_url=agent_config.base_url,
            timeout=agent_config.timeout,
            max_completion_tokens=agent_config.max_completion_tokens,
        )
    case "gigachat":
        credential_env = agent_config.credential_env
        scope = agent_config.scope
        if credential_env is None or scope is None:
            raise AssertionError("validated GigaChat configuration is incomplete")
        return GigaChatAgent(
            model=agent_config.model,
            credential_env=credential_env,
            scope=scope,
            timeout=agent_config.timeout,
            max_completion_tokens=agent_config.max_completion_tokens,
        )
    case "yandex_ai_studio":
        credential_env = agent_config.credential_env
        folder_id = agent_config.folder_id
        credential_kind = agent_config.credential_kind
        if credential_env is None or folder_id is None or credential_kind is None:
            raise AssertionError("validated Yandex configuration is incomplete")
        return YandexAIStudioAgent(
            model=agent_config.model,
            folder_id=folder_id,
            credential_kind=credential_kind,
            credential_env=credential_env,
            timeout=agent_config.timeout,
            max_completion_tokens=agent_config.max_completion_tokens,
        )
```

Use assertions or narrowed local variables immediately before each native
constructor so Pyright sees the provider-required fields as non-`None`. Do not
add a fallback case: `AgentConfig` validation makes every provider exhaustive.

- [ ] **Step 4: Document each provider precisely**

Extend `docs/configuration.md` with three complete agent examples:

- OpenAI-compatible with omitted `provider` and `api_key_env`.
- GigaChat 3 Ultra with `credential_env` and a valid GigaChat scope, explaining
  that the SDK exchanges and renews access tokens from the authorization
  credential.
- Yandex `aliceai-llm` with `folder_id`, `credential_kind = "oauth"`, and
  `credential_env`, explaining OAuth renewal and the API-key alternative.

State that direct IAM tokens are rejected, model availability is controlled by
the provider account, GigaChat TLS verification remains enabled, and no
credential value belongs in TOML. Add a short README link to this provider
configuration reference; do not add live credential commands or secrets.

- [ ] **Step 5: Run focused integration verification**

Run: `uv run pytest tests/integration/test_cli.py tests/integration/test_engine_trace.py -q`

Expected: PASS.

Run: `uv run ruff check src/benchtable/cli.py tests/integration/test_cli.py`

Expected: PASS.

- [ ] **Step 6: Run complete verification**

Run these commands in order:

```bash
uv sync
uv run pytest -q
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv lock --check
```

Expected: every command exits successfully with no live Sber or Yandex request.

- [ ] **Step 7: Inspect and commit the integration and documentation**

Run:

```bash
git status --short
```

Expected: only the provider-dispatch, tests, and documentation changes for this task are present.

Commit:

```bash
git add src/benchtable/cli.py tests/integration/test_cli.py docs/configuration.md README.md
git commit -m "docs: describe native model providers"
```

## Final Acceptance Check

- [ ] An existing agent TOML with no `provider` still selects the OpenAI-compatible adapter.
- [ ] GigaChat uses the native async SDK with credentials, automatic access-token refresh, TLS verification, and SDK retries disabled.
- [ ] GigaChat legacy function calls and function-result history round-trip through canonical engine transcripts, including internal `functions_state_id` correlation.
- [ ] Yandex uses native `AsyncAIStudio`, supports refreshed OAuth or API keys, requires a folder ID, and sets `parallel_tool_calls=False`.
- [ ] `raw_provider_request` is preserved for success and provider failures, then redacted by `EventWriter` without engine provider branches.
- [ ] Provider reasoning content, credentials, and token-exchange payloads are absent from normalized contracts and traces.
- [ ] The full verification suite passes offline.
