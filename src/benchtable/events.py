"""Append-only JSONL event writer with credential redaction."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from benchtable.contracts import JsonObject

_CREDENTIAL_KEYS = frozenset(
    {
        "apikey",
        "apitoken",
        "apisecret",
        "xapikey",
        "xauthtoken",
        "proxyauthorization",
        "authorization",
        "accesstoken",
        "sessiontoken",
        "clienttoken",
        "password",
        "passwd",
        "clientsecret",
        "secret",
        "secretkey",
        "token",
        "bearertoken",
        "auth",
        "privatekey",
        "refreshtoken",
        "credential",
        "credentials",
    }
)
_EVENT_PROTOCOL_IDENTIFIER_KEYS = frozenset(
    {"run_id", "match_id", "actor_id", "call_id", "tool_call_id"}
)


def _credential_key_pattern(key: str) -> str:
    """Build a key pattern that permits common separators between characters."""
    return "".join(
        f"{re.escape(character)}[\\W_]*" for character in key[:-1]
    ) + re.escape(key[-1])


_CREDENTIAL_KEY_PATTERN = (
    "(?:"
    + "|".join(
        _credential_key_pattern(key)
        for key in sorted(_CREDENTIAL_KEYS, key=len, reverse=True)
    )
    + ")"
)

# Pattern to redact API key values in error messages
_CREDENTIAL_PATTERNS = [
    "sk-",  # OpenAI-style keys
    "Bearer ",  # Bearer tokens
]

_CREDENTIAL_KEY_VALUE_PATTERN = re.compile(
    rf"(?P<key>\b{_CREDENTIAL_KEY_PATTERN}\b)"
    r"(?P<key_quote>\\?[\"'])?"
    r"(?P<separator>\s*[:=]\s*)",
    re.IGNORECASE,
)
_UNICODE_ESCAPE_PATTERN = r"\\u[0-9a-f]{4}"
_UNICODE_KEY_CHARACTER = rf"(?:{_UNICODE_ESCAPE_PATTERN}|[A-Za-z0-9_.\- ])"
_UNICODE_CREDENTIAL_KEY_VALUE_PATTERN = re.compile(
    rf"(?<![A-Za-z0-9_.-])"
    rf"(?=(?P<key>{_UNICODE_KEY_CHARACTER}*{_UNICODE_ESCAPE_PATTERN}"
    rf"{_UNICODE_KEY_CHARACTER}*)"
    r"(?P<key_quote>\\?[\"'])?"
    r"(?P<separator>\s*[:=]\s*))",
    re.IGNORECASE,
)
_UNICODE_ESCAPE = re.compile(r"\\u([0-9a-f]{4})", re.IGNORECASE)


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def redact_value(value: Any) -> Any:
    """Recursively redact credential-shaped keys."""
    if isinstance(value, dict):
        d = cast(dict[str, Any], value)
        result: dict[str, Any] = {}
        for k, v in d.items():
            if _normalize_key(k) in _CREDENTIAL_KEYS:
                result[k] = "[REDACTED]"
            else:
                result[k] = redact_value(v)
        return result
    if isinstance(value, list):
        lst = cast(list[Any], value)
        return [redact_value(item) for item in lst]
    if isinstance(value, str):
        return _redact_string(value)
    return value


def _redact_event_value(value: Any) -> Any:
    """Redact an event while preserving only protocol correlation identifiers."""
    if isinstance(value, dict):
        d = cast(dict[str, Any], value)
        is_function_tool_call = d.get("type") == "function" and isinstance(
            d.get("function"), dict
        )
        result: dict[str, Any] = {}
        for k, v in d.items():
            if (
                k in _EVENT_PROTOCOL_IDENTIFIER_KEYS
                or (k == "id" and is_function_tool_call)
            ) and isinstance(v, str):
                result[k] = v
            elif _normalize_key(k) in _CREDENTIAL_KEYS:
                result[k] = "[REDACTED]"
            else:
                result[k] = _redact_event_value(v)
        return result
    if isinstance(value, list):
        lst = cast(list[Any], value)
        return [_redact_event_value(item) for item in lst]
    if isinstance(value, str):
        return _redact_string(value)
    return value


def _redact_string(s: str) -> str:
    """Redact credential-shaped substrings in a string."""
    s = _redact_unicode_escaped_keys(s)
    redacted: list[str] = []
    source_index = 0
    search_from = 0
    while match := _CREDENTIAL_KEY_VALUE_PATTERN.search(s, search_from):
        if _is_inside_prefix_value(s, match.start()):
            search_from = match.end()
            continue
        value_start = match.end()
        value_end, replacement = _redacted_value_span(
            s, value_start, key=_normalize_key(match.group("key"))
        )
        if value_end is None:
            search_from = value_start
            continue
        key_quote = match.group("key_quote") or ""
        if key_quote:
            quote = r"\"" if key_quote.startswith("\\") else '"'
            replacement = f"{quote}[REDACTED]{quote}"
        redacted.append(s[source_index:value_start])
        redacted.append(replacement)
        source_index = value_end
        search_from = value_end
    redacted.append(s[source_index:])
    return _redact_prefixes("".join(redacted))


def _redact_unicode_escaped_keys(value: str) -> str:
    """Redact values whose key becomes a credential name after decoding."""
    redacted: list[str] = []
    source_index = 0
    search_from = 0
    while match := _UNICODE_CREDENTIAL_KEY_VALUE_PATTERN.search(value, search_from):
        decoded_key = _UNICODE_ESCAPE.sub(
            lambda escape: chr(int(escape.group(1), 16)), match.group("key")
        )
        if (
            _normalize_key(decoded_key) not in _CREDENTIAL_KEYS
            or not match.group("key_quote")
            or not _has_unicode_key_context(value, match.start())
        ):
            search_from = match.start() + 1
            continue
        if _is_inside_prefix_value(value, match.start()):
            search_from = match.start() + 1
            continue

        value_start = match.end("separator")
        value_end, replacement = _redacted_value_span(
            value, value_start, key=_normalize_key(decoded_key)
        )
        if value_end is None:
            search_from = value_start
            continue
        key_quote = match.group("key_quote") or ""
        if key_quote:
            quote = '\\"' if key_quote.startswith("\\") else '"'
            replacement = f"{quote}[REDACTED]{quote}"
        redacted.append(value[source_index:value_start])
        redacted.append(replacement)
        source_index = value_end
        search_from = value_end

    if not redacted:
        return value
    redacted.append(value[source_index:])
    return "".join(redacted)


def _has_unicode_key_context(value: str, key_start: int) -> bool:
    """Require a Unicode-escaped key to follow an object delimiter."""
    opening_quote = key_start - 1
    while opening_quote >= 0 and value[opening_quote] == "\\":
        opening_quote -= 1
    if opening_quote < 0 or value[opening_quote] not in {'"', "'"}:
        return False

    context = opening_quote - 1
    while context >= 0 and value[context] == "\\":
        context -= 1
    while context >= 0 and value[context].isspace():
        context -= 1
    return context >= 0 and value[context] in "{,"


def _is_inside_prefix_value(value: str, index: int) -> bool:
    """Return whether a key match is part of a prefix-token value."""
    value_lower = value.lower()
    for prefix in _CREDENTIAL_PATTERNS:
        prefix_index = value_lower.rfind(prefix.lower(), 0, index)
        if prefix_index < 0:
            continue
        token_start = prefix_index + len(prefix)
        while token_start < len(value) and value[token_start].isspace():
            token_start += 1
        if token_start > index:
            continue
        if token_start < len(value) and (
            value[token_start] in {'"', "'"}
            or (
                value[token_start] == "\\"
                and token_start + 1 < len(value)
                and value[token_start + 1] in {'"', "'"}
            )
        ):
            token_end = _quoted_value_end(value, token_start)
        else:
            token_end = token_start
            while (
                token_end < len(value)
                and value[token_end] not in _VALUE_BOUNDARIES
                and value[token_end] not in {"'", '"'}
            ):
                token_end += 1
        if token_end is not None and index < token_end:
            return True
    return False


def _redact_prefixes(s: str) -> str:
    """Redact prefix tokens before key-value scanning can misclassify them."""
    for prefix in _CREDENTIAL_PATTERNS:
        prefix_lower = prefix.lower()
        search_from = 0
        while True:
            idx = s.lower().find(prefix_lower, search_from)
            if idx == -1:
                break

            # Redact from the prefix to the next whitespace/end.
            end = idx + len(prefix)
            token_start = end
            while token_start < len(s) and s[token_start].isspace():
                token_start += 1
            if token_start < len(s) and (
                s[token_start] in {'"', "'"}
                or (
                    s[token_start] == "\\"
                    and token_start + 1 < len(s)
                    and s[token_start + 1] in {'"', "'"}
                )
            ):
                quoted_end = _quoted_value_end(s, token_start)
                if quoted_end is None:
                    end = len(s)
                else:
                    opening = (
                        s[token_start : token_start + 2]
                        if s[token_start] == "\\"
                        else s[token_start]
                    )
                    closing = (
                        s[quoted_end - 2 : quoted_end]
                        if opening.startswith("\\")
                        else s[quoted_end - 1]
                    )
                    s = (
                        s[:token_start]
                        + opening
                        + "[REDACTED]"
                        + closing
                        + s[quoted_end:]
                    )
                    search_from = token_start + len(opening + "[REDACTED]" + closing)
                    continue
            else:
                while end < len(s) and s[end] not in " \t\n,;'\"})]":
                    end += 1
                if (
                    end < len(s)
                    and s[end] in {"'", '"'}
                    and not _is_value_boundary(s, end + 1)
                ):
                    end = len(s)
            if end > idx + len(prefix):
                s = s[: idx + len(prefix)] + "[REDACTED]" + s[end:]
                search_from = idx + len(prefix) + len("[REDACTED]")
            else:
                search_from = end
    return s


def redact_text(value: str) -> str:
    """Redact credential-shaped substrings from model-facing text."""
    return _redact_string(value)


def redact_json_text(value: str) -> str:
    """Redact JSON values while preserving valid JSON syntax."""
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, OverflowError, RecursionError):
        return redact_text(value)
    try:
        redacted = redact_value(parsed)
        separators = (",", ":") if _looks_compact_json(value) else (", ", ": ")
        try:
            return json.dumps(
                redacted,
                ensure_ascii=False,
                allow_nan=False,
                separators=separators,
            )
        except (TypeError, ValueError, OverflowError, RecursionError):
            return json.dumps(
                redacted,
                ensure_ascii=False,
                allow_nan=True,
                separators=separators,
            )
    except (TypeError, ValueError, OverflowError, RecursionError):
        return redact_text(value)


_DELIMITER_PAIRS = {"[": "]", "{": "}", "(": ")"}
_DELIMITER_CLOSERS = frozenset(_DELIMITER_PAIRS.values())
_VALUE_BOUNDARIES = frozenset(" \t\r\n,;)}]")
_QUOTED_VALUE_BOUNDARIES = _VALUE_BOUNDARIES | frozenset(".!?:")


def _redacted_value_span(
    value: str, start: int, *, key: str | None = None
) -> tuple[int | None, str]:
    """Return the end and replacement for one credential value."""
    if start >= len(value):
        return None, ""

    if key in {"authorization", "proxyauthorization"}:
        return _authorization_value_span(value, start)

    if value[start] in {'"', "'"} or (
        value[start] == "\\"
        and start + 1 < len(value)
        and value[start + 1] in {'"', "'"}
    ):
        end, replacement = _quoted_value_span(value, start)
        if end is not None:
            return end, replacement
        return len(value), replacement

    if value[start] in _DELIMITER_PAIRS:
        end = _balanced_value_end(value, start)
        if end is not None:
            return end, f"{value[start]}[REDACTED]{value[end - 1]}"
        return len(value), f"{value[start]}[REDACTED]"

    end = start
    if value[start : start + 6].casefold() == "bearer" and (
        start + 6 == len(value) or value[start + 6].isspace()
    ):
        end += 6
        while end < len(value) and value[end].isspace():
            end += 1
        if end < len(value) and (
            value[end] in {'"', "'"}
            or (
                value[end] == "\\"
                and end + 1 < len(value)
                and value[end + 1] in {'"', "'"}
            )
        ):
            quoted_end, quoted_replacement = _quoted_value_span(value, end)
            if quoted_end is None:
                return len(value), "[REDACTED]"
            return quoted_end, value[start:end] + quoted_replacement
    while (
        end < len(value)
        and value[end] not in _VALUE_BOUNDARIES
        and value[end] not in {"'", '"'}
    ):
        end += 1
    if end < len(value) and value[end] in {"'", '"'}:
        quoted_end, _ = _quoted_value_span(value, end)
        if quoted_end is None:
            return len(value), "[REDACTED]"
        return quoted_end, "[REDACTED]"
    if end == start:
        return None, ""
    return end, "[REDACTED]"


def _authorization_value_span(value: str, start: int) -> tuple[int | None, str]:
    """Redact an authorization scheme and its complete credential value."""
    if value[start] in {'"', "'"} or (
        value[start] == "\\"
        and start + 1 < len(value)
        and value[start + 1] in {'"', "'"}
    ):
        return _quoted_value_span(value, start)

    scheme_end = start
    while (
        scheme_end < len(value)
        and value[scheme_end] not in _VALUE_BOUNDARIES
        and value[scheme_end] not in {"'", '"'}
    ):
        scheme_end += 1
    scheme = value[start:scheme_end].casefold()
    if scheme == "digest":
        return _authorization_field_end(value, scheme_end), "[REDACTED]"
    if scheme == "basic":
        token_start = scheme_end
        while token_start < len(value) and value[token_start].isspace():
            token_start += 1
        if token_start >= len(value):
            return scheme_end, "[REDACTED]"
        if value[token_start] in {'"', "'"} or (
            value[token_start] == "\\"
            and token_start + 1 < len(value)
            and value[token_start + 1] in {'"', "'"}
        ):
            token_end = _quoted_value_end(value, token_start)
            return (len(value) if token_end is None else token_end), "[REDACTED]"
        return _authorization_field_end(value, start), "[REDACTED]"
    if scheme == "bearer":
        token_start = scheme_end
        while token_start < len(value) and value[token_start].isspace():
            token_start += 1
        if token_start < len(value) and (
            value[token_start] in {'"', "'"}
            or (
                value[token_start] == "\\"
                and token_start + 1 < len(value)
                and value[token_start + 1] in {'"', "'"}
            )
        ):
            token_end, token_replacement = _quoted_value_span(value, token_start)
            if token_end is None:
                return len(value), "[REDACTED]"
            return token_end, value[start:token_start] + token_replacement
        return _authorization_field_end(value, start), "[REDACTED]"
    return _authorization_field_end(value, start), "[REDACTED]"


def _authorization_field_end(value: str, start: int) -> int:
    """Find a safe end for an authorization field, including quoted parameters."""
    index = start
    while index < len(value):
        if value[index] in {'"', "'"} or (
            value[index] == "\\"
            and index + 1 < len(value)
            and value[index + 1] in {'"', "'"}
        ):
            quoted_end = _quoted_value_end(value, index)
            if quoted_end is None:
                return len(value)
            index = quoted_end
            continue
        if value[index] in {";", "\r", "\n", ")", "}", "]"}:
            return index
        index += 1
    return index


def _quoted_value_end(value: str, start: int) -> int | None:
    """Find the closing quote for a quoted credential value."""
    escaped_delimiter = value[start] == "\\"
    quote_index = start + 1 if escaped_delimiter else start
    quote = value[quote_index]
    index = quote_index + 1
    while index < len(value):
        if value[index] == "\\":
            if index + 1 >= len(value):
                return None
            if (
                escaped_delimiter
                and value[index + 1] == quote
                and _is_value_boundary(value, index + 2)
            ):
                return index + 2
            index += 2
            continue
        if value[index] == quote and _is_value_boundary(value, index + 1):
            return index + 1
        index += 1
    return None


def _quoted_value_span(value: str, start: int) -> tuple[int | None, str]:
    """Return a quoted value's endpoint and redacted replacement."""
    opening = value[start : start + 2] if value[start] == "\\" else value[start]
    end = _quoted_value_end(value, start)
    if end is None:
        return None, f"{opening}[REDACTED]"
    closing = value[end - 2 : end] if opening.startswith("\\") else value[end - 1]
    return end, f"{opening}[REDACTED]{closing}"


def _balanced_value_end(value: str, start: int) -> int | None:
    """Find the end of a nested delimiter value, ignoring quoted content."""
    stack: list[str] = [_DELIMITER_PAIRS[value[start]]]
    index = start + 1
    while index < len(value):
        character = value[index]
        if character in {'"', "'"} and not _is_escaped(value, index):
            quoted_end = _quoted_value_end(value, index)
            if quoted_end is None:
                return None
            index = quoted_end
            continue
        if character in _DELIMITER_PAIRS:
            stack.append(_DELIMITER_PAIRS[character])
        elif character in _DELIMITER_CLOSERS:
            if stack[-1] != character:
                return None
            stack.pop()
            if not stack:
                return index + 1
        index += 1
    return None


def _is_escaped(value: str, index: int) -> bool:
    """Return whether the character has an odd number of preceding backslashes."""
    backslashes = 0
    index -= 1
    while index >= 0 and value[index] == "\\":
        backslashes += 1
        index -= 1
    return backslashes % 2 == 1


def _is_value_boundary(value: str, index: int) -> bool:
    return index >= len(value) or value[index] in _QUOTED_VALUE_BOUNDARIES


def _looks_compact_json(value: str) -> bool:
    """Keep compact provider argument formatting when it was supplied."""
    return ": " not in value and ", " not in value


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
            "event_type": _redact_string(event_type),
            "run_id": self._run_id,
            "occurred_at": datetime.now(timezone.utc).isoformat(),
            "payload": _redact_event_value(_normalize(payload)),
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
