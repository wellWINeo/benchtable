# Benchmark Infrastructure Review Fixes

## Goal

Bring the implementation described by `2026-07-17-benchmark-infrastructure.md`
into compliance with its approved trace, failure, plugin, provider, and CLI
requirements without adding poker-specific behavior or external services.

## Design

The run engine will expose an asynchronous execution path and await agents
directly. The CLI will own the synchronous `asyncio.run` boundary. A run will
return an explicit success status based on all match outcomes, and failed runs
will produce a non-zero CLI exit code while retaining their failure events.

The engine will keep a single injected agent as a test convenience, but
configured runs will use an actor-to-agent mapping. Game configuration will be
passed into session creation. Run configuration events will include sanitized
agent metadata and game configuration.

The event writer will append to an existing JSONL file, recover the next
sequence number from its valid prefix, and reject malformed existing content.
Credential redaction will cover standard API-key headers and repeated
credential-shaped strings. Provider request events will contain the exact
messages and tools sent, and response events will retain normalized fields and
the raw provider payload.

Provider retries will be controlled by the engine by disabling SDK-internal
retries. Invalid-action retry messages will use valid Chat Completions history,
including original tool-call IDs where applicable. Malformed JSON values that
are not objects will remain normalized as malformed tool arguments with their
raw text preserved.

Plugin/session failures, cancellation, exhausted budgets, and failed-turn
recovery will be explicit trace outcomes. Common metrics will include token
totals and latency. Optional generation limits will be validated and forwarded
to the OpenAI-compatible adapter. Duplicate discovered plugins will raise an
explicit error.

## Testing

Regression tests will cover each review finding, including exact retry message
shapes, active event-loop execution, terminal completion at the turn limit,
append and redaction behavior, multi-agent dispatch, game configuration,
offline CLI injection, failure exit codes, plugin failures, and raw provider
trace preservation. The complete pytest, Ruff, Pyright, and lockfile checks
remain required before completion.
