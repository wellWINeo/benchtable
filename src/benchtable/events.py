"""Append-only JSONL event writer with credential redaction."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from benchtable.contracts import JsonObject

_CREDENTIAL_KEYS = frozenset(
    {
        "apikey",
        "apitoken",
        "xapikey",
        "xauthtoken",
        "proxyauthorization",
        "authorization",
        "accesstoken",
        "password",
        "passwd",
        "clientsecret",
        "secret",
        "token",
        "bearertoken",
        "auth",
        "privatekey",
        "refreshtoken",
        "credential",
        "credentials",
    }
)

# Pattern to redact API key values in error messages
_CREDENTIAL_PATTERNS = [
    "sk-",  # OpenAI-style keys
    "Bearer ",  # Bearer tokens
]


def _normalize_key(value: str) -> str:
    return value.casefold().replace("-", "").replace("_", "")


def _redact(value: Any) -> Any:
    """Recursively redact credential-shaped keys."""
    if isinstance(value, dict):
        d = cast(dict[str, Any], value)
        result: dict[str, Any] = {}
        for k, v in d.items():
            if _normalize_key(k) in _CREDENTIAL_KEYS:
                result[k] = "[REDACTED]"
            else:
                result[k] = _redact(v)
        return result
    if isinstance(value, list):
        lst = cast(list[Any], value)
        return [_redact(item) for item in lst]
    if isinstance(value, str):
        return _redact_string(value)
    return value


def _redact_string(s: str) -> str:
    """Redact credential-shaped substrings in a string."""
    for prefix in _CREDENTIAL_PATTERNS:
        prefix_lower = prefix.lower()
        search_from = 0
        while True:
            idx = s.lower().find(prefix_lower, search_from)
            if idx == -1:
                break

            # Redact from the prefix to the next whitespace/end.
            end = idx + len(prefix)
            while end < len(s) and s[end] not in " \t\n,.;'\"})]":
                end += 1
            if end > idx + len(prefix):
                s = s[: idx + len(prefix)] + "[REDACTED]" + s[end:]
                search_from = idx + len(prefix) + len("[REDACTED]")
            else:
                search_from = end
    return s


def _normalize(value: Any) -> Any:
    """Convert Pydantic models and nested structures to JSON-compatible values."""
    if isinstance(value, dict):
        d = cast(dict[str, Any], value)
        result: dict[str, Any] = {}
        for k, v in d.items():
            result[k] = _normalize(v)
        return result
    if isinstance(value, list):
        lst = cast(list[Any], value)
        return [_normalize(item) for item in lst]
    if hasattr(value, "model_dump"):
        return _normalize(value.model_dump(mode="json"))
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    raise TypeError(f"Object of type {type(value).__name__} is not JSON-serializable")


class EventWriter:
    """Append-only JSONL writer with credential redaction and flush-on-emit."""

    def __init__(self, run_dir: Path | str, *, run_id: str = "") -> None:
        self._run_dir = Path(run_dir)
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._path = self._run_dir / "events.jsonl"
        self._file = self._path.open("a+", encoding="utf-8")
        try:
            self._file.seek(0)
            existing = self._file.read()
            self._sequence = _recover_sequence(existing)
        except Exception:
            self._file.close()
            raise
        self._file.seek(0, 2)
        self._needs_separator = bool(existing) and not existing.endswith(("\n", "\r"))
        self._run_id = run_id
        self._closed = False

    def emit(
        self,
        event_type: str,
        payload: JsonObject,
        *,
        match_id: str | None = None,
        turn_index: int | None = None,
        actor_id: str | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("Cannot emit to a closed EventWriter")

        next_sequence = self._sequence + 1
        envelope: dict[str, Any] = {
            "schema_version": 1,
            "sequence": next_sequence,
            "event_type": event_type,
            "run_id": self._run_id,
            "occurred_at": datetime.now(timezone.utc).isoformat(),
            "payload": _redact(_normalize(payload)),
        }
        if match_id is not None:
            envelope["match_id"] = match_id
        if turn_index is not None:
            envelope["turn_index"] = turn_index
        if actor_id is not None:
            envelope["actor_id"] = actor_id

        line = json.dumps(
            envelope,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        self._sequence = next_sequence
        if self._needs_separator:
            self._file.write("\n")
            self._needs_separator = False
        self._file.write(line + "\n")
        self._file.flush()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._file.close()

    def __enter__(self) -> EventWriter:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _recover_sequence(existing: str) -> int:
    """Validate existing JSONL and return the next sequence's predecessor."""
    sequence = 0
    previous_sequence: int | None = None
    for line_number, line in enumerate(existing.splitlines(), start=1):
        if not line.strip():
            raise ValueError(
                f"Malformed existing events.jsonl at line {line_number}: empty line"
            )
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Malformed existing events.jsonl at line {line_number}: {exc.msg}"
            ) from exc
        if not isinstance(event, dict):
            raise ValueError(
                f"Malformed existing events.jsonl at line {line_number}: "
                "event must be a JSON object"
            )
        event_object = cast(dict[str, Any], event)
        existing_sequence = event_object.get("sequence")
        if isinstance(existing_sequence, bool) or not isinstance(
            existing_sequence, int
        ):
            raise ValueError(
                f"Malformed existing events.jsonl at line {line_number}: "
                "sequence must be an integer"
            )
        if existing_sequence < 1:
            raise ValueError(
                f"Malformed existing events.jsonl at line {line_number}: "
                "sequence must be at least 1"
            )
        if previous_sequence is not None and existing_sequence <= previous_sequence:
            raise ValueError(
                f"Malformed existing events.jsonl at line {line_number}: "
                "sequence must be strictly increasing"
            )
        previous_sequence = existing_sequence
        sequence = existing_sequence
    return sequence
