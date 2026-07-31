"""Tests for the JSONL event writer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from benchtable.events import EventWriter, redact_json_text, redact_text, redact_value


class TestEventWriter:
    def test_events_have_monotonic_sequence_numbers(self, tmp_path: Path) -> None:
        with EventWriter(tmp_path / "run") as writer:
            writer.emit("run_start", {"a": 1})
            writer.emit("turn_start", {"b": 2})
            writer.emit("turn_end", {"c": 3})

        lines = _read_events(tmp_path / "run" / "events.jsonl")
        assert [e["sequence"] for e in lines] == [1, 2, 3]

    def test_envelope_contains_required_fields(self, tmp_path: Path) -> None:
        with EventWriter(tmp_path / "run", run_id="r1") as writer:
            writer.emit(
                "turn_start",
                {"x": 1},
                match_id="m1",
                turn_index=5,
                actor_id="player-1",
            )

        lines = _read_events(tmp_path / "run" / "events.jsonl")
        e = lines[0]
        assert e["schema_version"] == 1
        assert e["event_type"] == "turn_start"
        assert e["run_id"] == "r1"
        assert e["match_id"] == "m1"
        assert e["turn_index"] == 5
        assert e["actor_id"] == "player-1"
        assert "occurred_at" in e
        assert e["payload"] == {"x": 1}

    @pytest.mark.parametrize(
        ("event_type", "expected"),
        [
            ("turn_start", "turn_start"),
            ("api_key=event-secret", "api_key=[REDACTED]"),
        ],
    )
    def test_event_type_is_redacted_without_changing_ordinary_names(
        self, tmp_path: Path, event_type: str, expected: str
    ) -> None:
        with EventWriter(tmp_path / "run") as writer:
            writer.emit(event_type, {})

        event = _read_events(tmp_path / "run" / "events.jsonl")[0]

        assert event["event_type"] == expected

    def test_protocol_envelope_identifiers_are_preserved_exactly(
        self, tmp_path: Path
    ) -> None:
        with EventWriter(tmp_path / "run", run_id="api_key=run-secret") as writer:
            writer.emit(
                "turn_start",
                {},
                match_id='Bearer "match-secret"!',
                actor_id="password: actor-secret",
            )

        event = _read_events(tmp_path / "run" / "events.jsonl")[0]

        assert event["run_id"] == "api_key=run-secret"
        assert event["match_id"] == 'Bearer "match-secret"!'
        assert event["actor_id"] == "password: actor-secret"

    def test_event_redaction_narrows_protocol_identifier_preservation(
        self, tmp_path: Path
    ) -> None:
        payload = {
            "id": "api_key=identifier-secret",
            "call_id": "authorization: Basic call-secret",
            "tool_call_id": "password: tool-secret",
            "password": "payload-secret",
            "nested": {"id": "token=nested-identifier", "token": "nested-secret"},
            "tool_call": {
                "id": "client_secret=tool-call-id",
                "type": "function",
                "function": {
                    "name": "act",
                    "arguments": "{}",
                },
            },
        }

        with EventWriter(tmp_path / "run") as writer:
            writer.emit("model_response", payload)

        event = _read_events(tmp_path / "run" / "events.jsonl")[0]
        redacted = event["payload"]

        assert {key: redacted[key] for key in payload if key != "password"} == {
            "id": "api_key=[REDACTED]",
            "call_id": "authorization: Basic call-secret",
            "tool_call_id": "password: tool-secret",
            "nested": {"id": "token=[REDACTED]", "token": "[REDACTED]"},
            "tool_call": {
                "id": "client_secret=tool-call-id",
                "type": "function",
                "function": {"name": "act", "arguments": "{}"},
            },
        }
        assert redacted["password"] == "[REDACTED]"

    def test_non_string_protocol_identifiers_are_recursively_redacted(
        self, tmp_path: Path
    ) -> None:
        payload = {
            "call_id": {"password": "call-secret"},
            "tool_call": {
                "id": {"password": "function-call-secret"},
                "type": "function",
                "function": {"name": "act", "arguments": "{}"},
            },
        }

        with EventWriter(tmp_path / "run") as writer:
            writer.emit("model_response", payload)

        redacted = _read_events(tmp_path / "run" / "events.jsonl")[0]["payload"]

        assert redacted["call_id"] == {"password": "[REDACTED]"}
        assert redacted["tool_call"]["id"] == {"password": "[REDACTED]"}

    def test_each_event_is_one_flushed_json_line(self, tmp_path: Path) -> None:
        with EventWriter(tmp_path / "run") as writer:
            writer.emit("run_start", {"a": 1})
            # File should already be flushed after emit
            raw = (tmp_path / "run" / "events.jsonl").read_text()
            assert raw.count("\n") == 1

            writer.emit("run_start", {"b": 2})
            raw = (tmp_path / "run" / "events.jsonl").read_text()
            assert raw.count("\n") == 2

    def test_first_event_can_contain_run_configuration(self, tmp_path: Path) -> None:
        config = {"game": "tiny", "matches": 5, "seed": 42}
        with EventWriter(tmp_path / "run", run_id="r1") as writer:
            writer.emit("run_config", config)

        lines = _read_events(tmp_path / "run" / "events.jsonl")
        assert lines[0]["payload"]["game"] == "tiny"
        assert lines[0]["payload"]["seed"] == 42

    def test_pydantic_models_are_converted_to_json_compatible(
        self, tmp_path: Path
    ) -> None:
        class Inner(BaseModel):
            value: int

        with EventWriter(tmp_path / "run") as writer:
            writer.emit("test", {"model": Inner(value=42)})

        lines = _read_events(tmp_path / "run" / "events.jsonl")
        assert lines[0]["payload"]["model"] == {"value": 42}

    def test_credential_fields_are_redacted(self, tmp_path: Path) -> None:
        payload = {
            "api_key": "sk-secret123",
            "authorization": "Bearer tok",
            "access_token": "at_secret",
            "password": "hunter2",
            "client_secret": "cs_secret",
            "headers": {"Authorization": "Bearer nested"},
            "prompt_tokens": 100,
            "model": "gpt-4o",
        }
        with EventWriter(tmp_path / "run") as writer:
            writer.emit("test", payload)

        lines = _read_events(tmp_path / "run" / "events.jsonl")
        p = lines[0]["payload"]
        assert p["api_key"] == "[REDACTED]"
        assert p["authorization"] == "[REDACTED]"
        assert p["access_token"] == "[REDACTED]"
        assert p["password"] == "[REDACTED]"
        assert p["client_secret"] == "[REDACTED]"
        assert p["headers"]["Authorization"] == "[REDACTED]"
        # Non-credential fields preserved
        assert p["prompt_tokens"] == 100
        assert p["model"] == "gpt-4o"

    @pytest.mark.parametrize(
        "credential_key",
        ["accessToken", "clientSecret", "X_API_KEY", "api_token", "bearer_token"],
    )
    def test_credential_key_matching_ignores_case_and_separators(
        self, tmp_path: Path, credential_key: str
    ) -> None:
        payload = {
            "nested": [{credential_key: "nested-secret"}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

        with EventWriter(tmp_path / "run") as writer:
            writer.emit("test", payload)

        event = _read_events(tmp_path / "run" / "events.jsonl")[0]
        assert event["payload"]["nested"][0][credential_key] == "[REDACTED]"
        assert event["payload"]["usage"] == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }

    @pytest.mark.parametrize(
        "credential_key",
        [
            "api_secret",
            "API.SECRET",
            "session-token",
            "SESSION_TOKEN",
            "client.token",
            "CLIENT_TOKEN",
        ],
    )
    def test_compound_credential_keys_are_redacted_in_nested_payloads(
        self, tmp_path: Path, credential_key: str
    ) -> None:
        with EventWriter(tmp_path / "run") as writer:
            writer.emit("test", {"nested": [{credential_key: "compound-secret"}]})

        event = _read_events(tmp_path / "run" / "events.jsonl")[0]

        assert event["payload"]["nested"][0][credential_key] == "[REDACTED]"

    @pytest.mark.parametrize(
        "credential_key", ["api_secret", "session-token", "client.token"]
    )
    def test_compound_credential_keys_are_redacted_in_text(
        self, credential_key: str
    ) -> None:
        assert redact_text(f"{credential_key}=compound-secret") == (
            f"{credential_key}=[REDACTED]"
        )

    def test_compound_credential_names_do_not_redact_protocol_identifiers(
        self, tmp_path: Path
    ) -> None:
        payload = {
            "call_id": "api_secret=stable-call-id",
            "tool_call_id": "session_token=stable-tool-id",
            "tool_call": {
                "id": "client_token=stable-function-id",
                "type": "function",
                "function": {"name": "act", "arguments": "{}"},
            },
        }

        with EventWriter(tmp_path / "run") as writer:
            writer.emit("model_response", payload)

        redacted = _read_events(tmp_path / "run" / "events.jsonl")[0]["payload"]

        assert redacted["call_id"] == "api_secret=stable-call-id"
        assert redacted["tool_call_id"] == "session_token=stable-tool-id"
        assert redacted["tool_call"]["id"] == "client_token=stable-function-id"

    def test_unsupported_payload_values_raise_serialization_error(
        self, tmp_path: Path
    ) -> None:
        with EventWriter(tmp_path / "run") as writer:
            with pytest.raises(TypeError, match="not JSON-serializable"):
                writer.emit("test", {"bad": object()})

            assert (tmp_path / "run" / "events.jsonl").read_text() == ""
            writer.emit("valid", {})

        events = _read_events(tmp_path / "run" / "events.jsonl")
        assert [(event["sequence"], event["event_type"]) for event in events] == [
            (1, "valid")
        ]

    def test_non_finite_numbers_are_rejected_without_writing_an_event(
        self, tmp_path: Path
    ) -> None:
        with EventWriter(tmp_path / "run") as writer:
            with pytest.raises(ValueError, match="Out of range float values"):
                writer.emit("invalid", {"value": float("nan")})

            assert (tmp_path / "run" / "events.jsonl").read_text() == ""
            writer.emit("valid", {})

        events = _read_events(tmp_path / "run" / "events.jsonl")
        assert events[0]["sequence"] == 1

    def test_closed_writer_rejects_further_writes(self, tmp_path: Path) -> None:
        writer = EventWriter(tmp_path / "run")
        writer.emit("test", {})
        writer.close()

        with pytest.raises(RuntimeError, match="closed"):
            writer.emit("test", {})

    def test_credential_values_in_strings_are_redacted(self, tmp_path: Path) -> None:
        payload = {
            "error": "Incorrect API key provided: sk-secret123abc.",
            "details": "Bearer mytoken123456 was rejected.",
        }
        with EventWriter(tmp_path / "run") as writer:
            writer.emit("test", payload)

        lines = _read_events(tmp_path / "run" / "events.jsonl")
        p = lines[0]["payload"]
        assert "sk-secret" not in p["error"]
        assert "[REDACTED]" in p["error"]
        assert "mytoken" not in p["details"]

    def test_redacts_complete_dotted_authorization_values(self, tmp_path: Path) -> None:
        with EventWriter(tmp_path / "run") as writer:
            writer.emit("test", {"error": "Authorization: Bearer abc.def"})

        payload = _read_events(tmp_path / "run" / "events.jsonl")[0]["payload"]

        assert payload["error"] == "Authorization: [REDACTED]"
        assert "abc" not in payload["error"]
        assert ".def" not in payload["error"]

    @pytest.mark.parametrize(
        ("key", "scheme_value"),
        [
            ("authorization", "Basic dXNlcjpwYXNz"),
            (
                "authorization",
                'Digest username="Mufasa", realm="test", nonce="nonce-value"',
            ),
            ("proxy_authorization", "Basic dXNlcjpwYXNz"),
            (
                "proxy_authorization",
                'Digest username="Mufasa", realm="test", nonce="nonce-value"',
            ),
        ],
    )
    def test_redacts_complete_basic_and_digest_authorization_values(
        self, key: str, scheme_value: str
    ) -> None:
        value = f"prefix; {key}: {scheme_value}; retry=1"

        redacted = redact_text(value)

        assert redacted == f"prefix; {key}: [REDACTED]; retry=1"
        assert all(secret not in redacted for secret in ("dXNlcjpwYXNz", "Mufasa"))

    def test_redacts_authorization_values_in_retry_history(
        self, tmp_path: Path
    ) -> None:
        payload = {
            "retry_history": [
                {"content": "authorization: Basic dXNlcjpwYXNz"},
                {
                    "content": (
                        'proxy_authorization: Digest username="Mufasa", realm="test"'
                    )
                },
            ]
        }

        with EventWriter(tmp_path / "run") as writer:
            writer.emit("validation_error", payload)

        retry_history = _read_events(tmp_path / "run" / "events.jsonl")[0]["payload"][
            "retry_history"
        ]

        assert retry_history == [
            {"content": "authorization: [REDACTED]"},
            {"content": "proxy_authorization: [REDACTED]"},
        ]

    @pytest.mark.parametrize(
        ("key", "scheme_value", "secrets"),
        [
            (
                "authorization",
                "Negotiate token-one, delegated-secret",
                ("token-one", "delegated-secret"),
            ),
            (
                "proxy_authorization",
                'OAuth oauth_consumer_key="consumer;secret", '
                'oauth_signature="signature-secret"',
                ("consumer;secret", "signature-secret"),
            ),
        ],
    )
    def test_redacts_complete_unsupported_authorization_values_in_event_text(
        self,
        tmp_path: Path,
        key: str,
        scheme_value: str,
        secrets: tuple[str, ...],
    ) -> None:
        with EventWriter(tmp_path / "run") as writer:
            writer.emit("provider_error", {"error": f"{key}: {scheme_value}; retry=1"})

        error = _read_events(tmp_path / "run" / "events.jsonl")[0]["payload"]["error"]

        assert error == f"{key}: [REDACTED]; retry=1"
        assert all(secret not in error for secret in secrets)

    def test_redacts_complete_unsupported_authorization_values_in_retry_history(
        self, tmp_path: Path
    ) -> None:
        payload = {
            "retry_history": [
                {"content": ("authorization: NTLM token-one token-two; attempt=1")},
                {
                    "content": (
                        "proxy_authorization: AWS4-HMAC-SHA256 "
                        "Credential=AKIASECRET, SignedHeaders=host, "
                        "Signature=SIGNATURESECRET; attempt=2"
                    )
                },
            ]
        }

        with EventWriter(tmp_path / "run") as writer:
            writer.emit("validation_error", payload)

        retry_history = _read_events(tmp_path / "run" / "events.jsonl")[0]["payload"][
            "retry_history"
        ]

        assert retry_history == [
            {"content": "authorization: [REDACTED]; attempt=1"},
            {"content": "proxy_authorization: [REDACTED]; attempt=2"},
        ]

    @pytest.mark.parametrize("key", ["authorization", "proxy_authorization"])
    @pytest.mark.parametrize(
        "scheme_value",
        [
            "Bearer bearer-secret, second-bearer-secret",
            "Basic basic-secret, second-basic-secret",
            'Digest username="digest-user", nonce="digest-secret"',
            "Negotiate negotiate-secret, delegated-secret",
            'OAuth oauth_consumer_key="oauth-secret", oauth_signature="sig-secret"',
            (
                "AWS4-HMAC-SHA256 Credential=aws-secret, "
                "SignedHeaders=host, Signature=signature-secret"
            ),
            "Custom custom-secret, second-custom-secret",
        ],
    )
    def test_redacts_full_authorization_field_before_semicolon_context(
        self, key: str, scheme_value: str
    ) -> None:
        value = f"prefix; {key}: {scheme_value}; retry=1"

        redacted = redact_text(value)

        assert redacted == f"prefix; {key}: [REDACTED]; retry=1"
        assert "secret" not in redacted

    @pytest.mark.parametrize("key", ["authorization", "proxy_authorization"])
    def test_digest_commas_are_redacted_as_part_of_the_credential_field(
        self, key: str
    ) -> None:
        value = (
            f'{key}: Digest username="digest-user", realm="digest-realm", '
            'nonce="digest-secret", response="response-secret"; retry=1'
        )

        assert redact_text(value) == f"{key}: [REDACTED]; retry=1"

    def test_generic_redaction_does_not_preserve_arbitrary_id_values(self) -> None:
        assert redact_value({"id": "api_key=identifier-secret"}) == {
            "id": "api_key=[REDACTED]"
        }

    @pytest.mark.parametrize(
        "credential_key",
        [
            "X_API_KEY",
            "x-api-key",
            "xapikey",
            "refresh_token",
            "refresh-token",
            "refreshtoken",
            "proxy_authorization",
            "proxy-authorization",
            "proxyauthorization",
        ],
    )
    def test_redact_text_covers_normalized_credential_key_variants(
        self, credential_key: str
    ) -> None:
        redacted = redact_text(f"{credential_key}=variant-secret")

        assert "variant-secret" not in redacted
        assert redacted == f"{credential_key}=[REDACTED]"

    @pytest.mark.parametrize(
        "credential_key",
        [
            "refresh_token",
            "X_API_KEY",
            "proxy_authorization",
        ],
    )
    def test_redacts_quoted_json_credential_values(self, credential_key: str) -> None:
        value = f'{{"{credential_key}":"secret"}}'

        assert redact_text(value) == f'{{"{credential_key}":"[REDACTED]"}}'

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ('{"api_key":123}', '{"api_key":"[REDACTED]"}'),
            ('{"api_key":["secret"]}', '{"api_key":"[REDACTED]"}'),
            ("api_key=[secret]", "api_key=[[REDACTED]]"),
            ("api_key=secret", "api_key=[REDACTED]"),
        ],
    )
    def test_redacts_json_values_without_breaking_value_syntax(
        self, value: str, expected: str
    ) -> None:
        redacted = redact_text(value)

        assert redacted == expected
        if value.startswith("{"):
            assert json.loads(redacted) == {"api_key": "[REDACTED]"}

    def test_redacts_escaped_quoted_credential_values_without_suffix(
        self, tmp_path: Path
    ) -> None:
        with EventWriter(tmp_path / "run") as writer:
            writer.emit("test", {"error": 'password="abc\\".def"'})

        payload = _read_events(tmp_path / "run" / "events.jsonl")[0]["payload"]

        assert payload["error"] == 'password="[REDACTED]"'
        assert "abc" not in payload["error"]
        assert ".def" not in payload["error"]

    @pytest.mark.parametrize("delimiters", [("[", "]"), ("{", "}"), ("(", ")")])
    def test_redacts_delimited_values_preserving_both_delimiters(
        self, delimiters: tuple[str, str]
    ) -> None:
        opening, closing = delimiters

        assert redact_text(f"api_key={opening}VALUE{closing}") == (
            f"api_key={opening}[REDACTED]{closing}"
        )

    @pytest.mark.parametrize(
        "credential_key", ["private_key", "privatekey", "credentials"]
    )
    def test_redacts_additional_credential_key_variants(
        self, credential_key: str
    ) -> None:
        assert redact_text(f"{credential_key}=VALUE") == (
            f"{credential_key}=[REDACTED]"
        )

    def test_redacts_escaped_json_credential_values(self) -> None:
        value = r"{\"api_key\": \"VALUE\"}"

        assert redact_text(value) == r"{\"api_key\": \"[REDACTED]\"}"

    def test_recursive_redaction_helper_preserves_message_shape(self) -> None:
        message = {
            "role": "assistant",
            "content": "api_key=prior-secret",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "act",
                        "arguments": r"{\"api_key\": \"argument-secret\"}",
                    },
                }
            ],
        }

        redacted = redact_value(message)

        assert redacted == {
            "role": "assistant",
            "content": "api_key=[REDACTED]",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "act",
                        "arguments": r"{\"api_key\": \"[REDACTED]\"}",
                    },
                }
            ],
        }

    def test_appends_to_existing_events_and_recovers_sequence(
        self, tmp_path: Path
    ) -> None:
        run_dir = tmp_path / "run"
        events_path = run_dir / "events.jsonl"
        run_dir.mkdir()
        events_path.write_text(json.dumps({"schema_version": 1, "sequence": 7}) + "\n")

        with EventWriter(run_dir) as writer:
            writer.emit("next", {})

        events = _read_events(events_path)
        assert [event["sequence"] for event in events] == [7, 8]
        assert events[-1]["event_type"] == "next"

    def test_rejects_malformed_existing_event_content(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        events_path = run_dir / "events.jsonl"
        run_dir.mkdir()
        existing = '{"schema_version": 1, "sequence": 1}\nnot-json\n'
        events_path.write_text(existing)

        with pytest.raises(ValueError, match=r"events\.jsonl.*line 2"):
            EventWriter(run_dir)

        assert events_path.read_text() == existing

    def test_rejects_existing_sequences_that_are_not_strictly_ordered(
        self, tmp_path: Path
    ) -> None:
        run_dir = tmp_path / "run"
        events_path = run_dir / "events.jsonl"
        run_dir.mkdir()
        events_path.write_text(
            "\n".join(
                json.dumps({"schema_version": 1, "sequence": sequence})
                for sequence in (1, 3, 2)
            )
            + "\n"
        )

        with pytest.raises(ValueError, match=r"events\.jsonl.*line 3"):
            EventWriter(run_dir)

    @pytest.mark.parametrize("sequence", [0, -1])
    def test_rejects_existing_sequences_below_one(
        self, tmp_path: Path, sequence: int
    ) -> None:
        run_dir = tmp_path / "run"
        events_path = run_dir / "events.jsonl"
        run_dir.mkdir()
        events_path.write_text(
            json.dumps({"schema_version": 1, "sequence": sequence}) + "\n"
        )

        with pytest.raises(ValueError, match=r"events\.jsonl.*line 1"):
            EventWriter(run_dir)

    def test_redacts_api_key_headers_and_all_repeated_credentials(
        self, tmp_path: Path
    ) -> None:
        payload = {
            "headers": {
                "X-API-Key": "header-secret",
                "x-api-key": "lower-header-secret",
            },
            "messages": (
                "sk-alpha12345 and sk-beta67890; "
                "Bearer token-alpha123 and Bearer token-beta456"
            ),
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

        with EventWriter(tmp_path / "run") as writer:
            writer.emit("test", payload)

        event = _read_events(tmp_path / "run" / "events.jsonl")[0]
        redacted = event["payload"]
        assert redacted["headers"] == {
            "X-API-Key": "[REDACTED]",
            "x-api-key": "[REDACTED]",
        }
        assert redacted["messages"] == (
            "sk-[REDACTED] and sk-[REDACTED]; Bearer [REDACTED] and Bearer [REDACTED]"
        )
        assert redacted["usage"] == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }

    def test_redacts_normalized_auth_and_proxy_headers_recursively(
        self, tmp_path: Path
    ) -> None:
        payload = {
            "nested": [
                {
                    "X-Auth-Token": "auth-header",
                    "x_auth_token": "auth-underscore",
                    "XAUTHTOKEN": "auth-compact",
                },
                {
                    "Proxy-Authorization": "proxy-header",
                    "proxy_authorization": "proxy-underscore",
                    "PROXYAUTHORIZATION": "proxy-compact",
                },
            ],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
            },
        }

        with EventWriter(tmp_path / "run") as writer:
            writer.emit("test", payload)

        event = _read_events(tmp_path / "run" / "events.jsonl")[0]
        headers = event["payload"]["nested"]
        assert all(value == "[REDACTED]" for value in headers[0].values())
        assert all(value == "[REDACTED]" for value in headers[1].values())
        assert event["payload"]["usage"] == {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
        }

    def test_redacts_case_insensitive_bearer_and_api_key_strings(
        self, tmp_path: Path
    ) -> None:
        payload = {
            "message": (
                "bearer tok then BEARER abc then BeArEr xyz; "
                "SK-alpha12345 and Sk-beta67890 and sk-gamma12345"
            )
        }

        with EventWriter(tmp_path / "run") as writer:
            writer.emit("test", payload)

        event = _read_events(tmp_path / "run" / "events.jsonl")[0]
        assert event["payload"]["message"] == (
            "bearer [REDACTED] then BEARER [REDACTED] then "
            "BeArEr [REDACTED]; SK-[REDACTED] and Sk-[REDACTED] and "
            "sk-[REDACTED]"
        )

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ('Bearer abc"secret-suffix', "Bearer [REDACTED]"),
            ('sk-abc"secret-suffix', "sk-[REDACTED]"),
            (
                'Bearer abc","safe":"keep"',
                'Bearer [REDACTED]","safe":"keep"',
            ),
            ('sk-abc"} rest', 'sk-[REDACTED]"} rest'),
            ("Bearer abc, next", "Bearer [REDACTED], next"),
            ("sk-abc, next", "sk-[REDACTED], next"),
        ],
    )
    def test_redacts_prefix_tokens_without_leaking_quote_suffix(
        self, value: str, expected: str
    ) -> None:
        assert redact_text(value) == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (
                'Bearer "quoted-secret" trailing text',
                'Bearer "[REDACTED]" trailing text',
            ),
            ('Bearer "unmatched-secret', "Bearer [REDACTED]"),
        ],
    )
    def test_redacts_quoted_bearer_tokens_with_quoted_value_scanner(
        self, value: str, expected: str
    ) -> None:
        assert redact_text(value) == expected

    @pytest.mark.parametrize("punctuation", [".", "!", "?", ":"])
    def test_redacts_quoted_bearer_tokens_before_post_quote_punctuation(
        self, punctuation: str
    ) -> None:
        value = f'Bearer "quoted-secret"{punctuation} trailing text'

        assert redact_text(value) == f'Bearer "[REDACTED]"{punctuation} trailing text'

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ('api_key="quoted"secret', 'api_key="[REDACTED]'),
            (
                'api_key="quoted" trailing',
                'api_key="[REDACTED]" trailing',
            ),
            (
                'authorization=Bearer "quoted"secret',
                "authorization=[REDACTED]",
            ),
            (
                'authorization=Bearer "quoted" trailing',
                'authorization=Bearer "[REDACTED]" trailing',
            ),
            ('api_key=sk-"quoted"secret', "api_key=[REDACTED]"),
            (
                'api_key=sk-"quoted" trailing',
                "api_key=[REDACTED] trailing",
            ),
        ],
    )
    def test_quoted_credentials_preserve_only_valid_trailing_context(
        self, value: str, expected: str
    ) -> None:
        assert redact_text(value) == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (
                r"Bearer \"escaped-secret\" trailing text",
                r"Bearer \"[REDACTED]\" trailing text",
            ),
            (
                r"Bearer \'escaped-secret\' trailing text",
                r"Bearer \'[REDACTED]\' trailing text",
            ),
            (r"Bearer \"escaped-secret trailing text", "Bearer [REDACTED]"),
        ],
    )
    def test_redacts_escaped_quoted_bearer_tokens_with_quoted_value_scanner(
        self, value: str, expected: str
    ) -> None:
        assert redact_text(value) == expected

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("api_key", 'api_key=abc"secret-suffix'),
            ("password", "password=abc'secret-suffix"),
        ],
    )
    def test_redacts_bare_values_through_end_when_quote_is_encountered(
        self, key: str, value: str
    ) -> None:
        redacted = redact_text(value)

        assert redacted == f"{key}=[REDACTED]"
        assert "secret-suffix" not in redacted

    def test_redacts_bounded_free_form_key_value_credentials_recursively(
        self, tmp_path: Path
    ) -> None:
        payload = {
            "message": (
                "Authorization: Bearer secret-value; password=hunter2; "
                "api_key: sk-free-form"
            ),
            "memory": {
                "read": [
                    {"text": "nested PASSWORD=secret-note and Api_Key: nested-key"}
                ]
            },
            "usage": {"prompt_tokens": 4, "completion_tokens": 2},
        }

        with EventWriter(tmp_path / "run") as writer:
            writer.emit("memory_operation", payload)

        event = _read_events(tmp_path / "run" / "events.jsonl")[0]
        redacted = event["payload"]
        assert redacted["message"] == (
            "Authorization: [REDACTED]; password=[REDACTED]; api_key: [REDACTED]"
        )
        assert redacted["memory"]["read"][0]["text"] == (
            "nested PASSWORD=[REDACTED] and Api_Key: [REDACTED]"
        )
        assert redacted["usage"] == {"prompt_tokens": 4, "completion_tokens": 2}

    def test_redacts_spaced_structured_keys_and_quoted_nested_values(
        self, tmp_path: Path
    ) -> None:
        payload = {
            "values": {
                "api key": "structured-api",
                "client secret": "structured-client",
                "secret_key": "structured-secret",
            },
            "memory": {
                "read": [
                    {"text": ('password="nested-password"; API.KEY: [nested-api]')}
                ]
            },
            "message": "password=\"quoted-password\"; client.secret: 'quoted-client'",
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        }

        with EventWriter(tmp_path / "run") as writer:
            writer.emit("memory_operation", payload)

        redacted = _read_events(tmp_path / "run" / "events.jsonl")[0]["payload"]
        assert redacted["values"] == {
            "api key": "[REDACTED]",
            "client secret": "[REDACTED]",
            "secret_key": "[REDACTED]",
        }
        assert redacted["memory"]["read"][0]["text"] == (
            'password="[REDACTED]"; API.KEY: [[REDACTED]]'
        )
        assert redacted["message"] == (
            "password=\"[REDACTED]\"; client.secret: '[REDACTED]'"
        )
        assert redacted["usage"] == {"prompt_tokens": 3, "completion_tokens": 2}

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (
                "api_key=[outer {nested (quoted value)} tail] after",
                "api_key=[[REDACTED]] after",
            ),
            (
                "password={outer [nested (value)] tail}; next=ok",
                "password={[REDACTED]}; next=ok",
            ),
            (
                "token=(outer [nested {value}] tail), next=ok",
                "token=([REDACTED]), next=ok",
            ),
        ],
    )
    def test_redacts_nested_delimited_values_without_trailing_closers(
        self, value: str, expected: str
    ) -> None:
        assert redact_text(value) == expected

    def test_redacts_escaped_quoted_values_as_one_balanced_value(self) -> None:
        value = 'password="outer \\"quoted\\" and {nested}" after'

        assert redact_text(value) == 'password="[REDACTED]" after'

    @pytest.mark.parametrize("delimiters", [("[", "]"), ("{", "}"), ("(", ")")])
    def test_escaped_quotes_inside_delimiters_do_not_hide_secret_suffix(
        self, delimiters: tuple[str, str]
    ) -> None:
        opening, closing = delimiters
        value = f'api_key={opening}prefix \\"quoted\\" secret-suffix{closing} after'

        redacted = redact_text(value)

        assert redacted == f"api_key={opening}[REDACTED]{closing} after"
        assert "secret-suffix" not in redacted

    @pytest.mark.parametrize(
        "value",
        [
            "api_key=[unmatched-bracket-secret",
            "api_key={outer [mismatched-nested-brace-secret}",
            'password="unmatched-double-quote-secret',
            "password='unmatched-single-quote-secret",
        ],
    )
    def test_redacts_unbalanced_free_form_values_through_end_of_string(
        self, value: str
    ) -> None:
        redacted = redact_text(value)

        assert "secret" not in redacted

    @pytest.mark.parametrize(
        "value",
        [
            '{"api_key":{"nested":{"password":123}},"safe":[1,{"token":456}]}',
            '{"api_key":[1,{"access_token":{"nested":true}}],"count":7}',
            '{"password":123,"container":{"secret":[1,2,3]}}',
        ],
    )
    def test_redact_json_text_returns_valid_json_for_nested_values(
        self, value: str
    ) -> None:
        redacted = redact_json_text(value)

        parsed = json.loads(redacted)

        assert isinstance(parsed, dict)
        assert "[REDACTED]" in redacted
        assert all(
            secret not in redacted
            for secret in ("123", "456", "true", "1,2,3")
            if secret in value
        )

    def test_redact_json_text_falls_back_to_balanced_free_form_redaction(self) -> None:
        value = "api_key=[outer {nested [value]} tail]"

        assert redact_json_text(value) == "api_key=[[REDACTED]]"

    @pytest.mark.parametrize("non_finite", ["NaN", "Infinity", "-Infinity"])
    def test_redact_json_text_falls_back_when_serialization_rejects_non_finite(
        self, non_finite: str
    ) -> None:
        value = f'{{"api_key":"non-finite-secret","value":{non_finite}}}'

        redacted = redact_json_text(value)

        assert "non-finite-secret" not in redacted

    def test_redact_json_text_redacts_escaped_keys_before_nonfinite_fallback(
        self,
    ) -> None:
        value = r'{"\u0061pi_key":"escaped-key-secret","value":NaN}'

        redacted = redact_json_text(value)

        assert '"api_key":"[REDACTED]"' in redacted
        assert "escaped-key-secret" not in redacted
        assert "NaN" in redacted

    @pytest.mark.parametrize("escaped_key", [r"\u0061pi_key", r"ref\u0072esh_token"])
    def test_event_redacts_unicode_escaped_keys_in_malformed_json(
        self, tmp_path: Path, escaped_key: str
    ) -> None:
        value = rf'{{"{escaped_key}":"malformed-json-secret",}}'

        with EventWriter(tmp_path / "run") as writer:
            writer.emit("model_response", {"raw_arguments": value})

        payload = _read_events(tmp_path / "run" / "events.jsonl")[0]["payload"]

        assert payload["raw_arguments"] == (rf'{{"{escaped_key}":"[REDACTED]",}}')
        assert "malformed-json-secret" not in json.dumps(payload)

    @pytest.mark.parametrize(
        ("escaped_key", "scheme_value", "secrets"),
        [
            (
                r"\u0061uthorization",
                r'Digest username="Mufasa", realm="test", '
                r'nonce="nonce-value", response="digest-response"',
                ("Mufasa", "test", "nonce-value", "digest-response"),
            ),
            (
                r"\u0070roxy_authorization",
                "Basic dXNlcjpwYXNz",
                ("dXNlcjpwYXNz",),
            ),
        ],
    )
    def test_redacts_complete_unicode_escaped_auth_values_in_malformed_json(
        self,
        tmp_path: Path,
        escaped_key: str,
        scheme_value: str,
        secrets: tuple[str, ...],
    ) -> None:
        value = rf'{{"{escaped_key}":{scheme_value},}}'

        with EventWriter(tmp_path / "run") as writer:
            writer.emit("model_response", {"raw_arguments": value})

        payload = _read_events(tmp_path / "run" / "events.jsonl")[0]["payload"]

        redacted = payload["raw_arguments"]
        assert redacted.startswith(rf'{{"{escaped_key}":')
        assert "[REDACTED]" in redacted
        assert all(secret not in json.dumps(payload) for secret in secrets)

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (
                r"ordinary \"\u0061uthorization\": ordinary-secret",
                r"ordinary \"\u0061uthorization\": ordinary-secret",
            ),
            (
                r"{\"\u0061pi_key\":\"escaped-key-secret\"}",
                r"{\"\u0061pi_key\":\"[REDACTED]\"}",
            ),
            (
                r"prefix,{\"\u0061pi_key\":\"comma-key-secret\"}",
                r"prefix,{\"\u0061pi_key\":\"[REDACTED]\"}",
            ),
            (
                r"{\"\\u0061pi_key\":\"nested-escaped-key-secret\"}",
                r"{\"\\u0061pi_key\":\"[REDACTED]\"}",
            ),
            (
                r"ordinary \\\"\\u0061uthorization\\\": ordinary-secret",
                r"ordinary \\\"\\u0061uthorization\\\": ordinary-secret",
            ),
        ],
    )
    def test_unicode_escaped_keys_require_structural_escaped_opening_quotes(
        self, value: str, expected: str
    ) -> None:
        assert redact_text(value) == expected

    @pytest.mark.parametrize(
        "value",
        [
            r'"message contains \u0061uthorization: ordinary text"',
            r'prefix "message contains \u0070roxy_authorization: ordinary text" suffix',
        ],
    )
    def test_unicode_escaped_credential_tokens_inside_quoted_strings_are_unchanged(
        self, value: str
    ) -> None:
        assert redact_text(value) == value

    def test_redact_json_text_falls_back_for_an_oversized_integer(self) -> None:
        value = '{"api_key":"oversized-secret","value":' + ("9" * 5000) + "}"

        redacted = redact_json_text(value)

        assert "oversized-secret" not in redacted
        assert '"api_key":"[REDACTED]"' in redacted

    def test_redact_json_text_falls_back_for_deeply_nested_input(self) -> None:
        value = (
            '{"api_key":"deep-secret","nested":'
            + ("[" * 1000)
            + "null"
            + ("]" * 1000)
            + "}"
        )

        redacted = redact_json_text(value)

        assert "deep-secret" not in redacted
        assert '"api_key":"[REDACTED]"' in redacted


def _read_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]
