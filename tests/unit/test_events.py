"""Tests for the JSONL event writer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from benchtable.events import EventWriter


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

    def test_redacts_escaped_quoted_credential_values_without_suffix(
        self, tmp_path: Path
    ) -> None:
        with EventWriter(tmp_path / "run") as writer:
            writer.emit("test", {"error": 'password="abc\\".def"'})

        payload = _read_events(tmp_path / "run" / "events.jsonl")[0]["payload"]

        assert payload["error"] == 'password="[REDACTED]"'
        assert "abc" not in payload["error"]
        assert ".def" not in payload["error"]

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
            'password="[REDACTED]"; API.KEY: [REDACTED]'
        )
        assert redacted["message"] == (
            "password=\"[REDACTED]\"; client.secret: '[REDACTED]'"
        )
        assert redacted["usage"] == {"prompt_tokens": 3, "completion_tokens": 2}


def _read_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]
