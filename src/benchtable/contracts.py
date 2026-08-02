"""Typed contracts for benchtable engine, games, and agents."""

from __future__ import annotations

import json
import math
from typing import Any, Self, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

# ---------------------------------------------------------------------------
# JSON-compatible type aliases
# ---------------------------------------------------------------------------

type JsonValue = (
    str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]
)
type JsonObject = dict[str, JsonValue]
type GameMetrics = dict[str, JsonValue]


# ---------------------------------------------------------------------------
# Tool contracts
# ---------------------------------------------------------------------------


class ToolSpec(BaseModel):
    """An OpenAI-compatible function tool specification."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    description: str = ""
    parameters: JsonObject = Field(default_factory=lambda: {"type": "object"})

    def to_openai_tool(self) -> JsonObject:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


class Observation(BaseModel):
    """An actor-specific view of the game.  Never contains raw game state."""

    model_config = ConfigDict(extra="forbid")

    actor_id: str
    text: str
    metadata: JsonObject = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Model response contracts
# ---------------------------------------------------------------------------


class ToolCall(BaseModel):
    """A single normalized tool call from the model."""

    model_config = ConfigDict(extra="forbid")

    call_id: StrictStr = Field(..., min_length=1)
    name: str
    arguments: JsonObject | None = None
    raw_arguments: str | None = None
    parse_error: str | None = None

    @field_validator("call_id")
    @classmethod
    def reject_blank_call_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("call_id must not be empty or whitespace-only")
        return value

    @model_validator(mode="before")
    @classmethod
    def normalize_non_object_arguments(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        input_data = cast(dict[str, Any], value)
        arguments = input_data.get("arguments")
        if arguments is None:
            if (
                input_data.get("raw_arguments") is None
                or input_data.get("parse_error") is not None
            ):
                return input_data
        elif isinstance(arguments, dict):
            return input_data

        normalized: dict[str, Any] = dict(input_data)
        normalized["arguments"] = None
        if normalized.get("parse_error") is None:
            normalized["parse_error"] = "Tool arguments must be a JSON object"
        return normalized


class Usage(BaseModel):
    """Token usage from the provider."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ModelResponse(BaseModel):
    """Provider-independent model response."""

    model_config = ConfigDict(extra="forbid")

    assistant_text: str
    tool_calls: list[ToolCall]
    finish_reason: str | None = None
    usage: Usage | None = None
    raw_provider_request: JsonObject | None = None
    raw_provider_response: JsonObject | None = None

    @model_validator(mode="after")
    def reject_duplicate_tool_call_ids(self) -> Self:
        seen: set[str] = set()
        duplicates: list[str] = []
        for tool_call in self.tool_calls:
            if tool_call.call_id in seen and tool_call.call_id not in duplicates:
                duplicates.append(tool_call.call_id)
            seen.add(tool_call.call_id)
        if duplicates:
            raise ValueError(
                "duplicate call_id values are not allowed: "
                + ", ".join(repr(call_id) for call_id in duplicates)
            )
        return self


# ---------------------------------------------------------------------------
# Game result and transition
# ---------------------------------------------------------------------------


class Transition(BaseModel):
    """The result of applying a valid action."""

    model_config = ConfigDict(extra="forbid")

    summary: str
    metrics: GameMetrics = Field(default_factory=dict)


class PluginEvent(BaseModel):
    """A typed event emitted by a game session for the generic trace."""

    model_config = ConfigDict(extra="forbid")

    event_type: StrictStr = Field(..., min_length=1)
    payload: JsonObject = Field(default_factory=dict)

    @field_validator("event_type")
    @classmethod
    def reject_blank_event_type(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("event_type must not be empty or whitespace-only")
        return value

    @field_validator("payload", mode="before")
    @classmethod
    def reject_non_json_types(cls, value: object) -> object:
        try:
            _validate_json_value(value)
        except ValueError as exc:
            raise ValueError("payload must be JSON-compatible") from exc
        return value

    @field_validator("payload")
    @classmethod
    def reject_non_json_values(cls, value: JsonObject) -> JsonObject:
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("payload must be JSON-compatible") from exc
        return value


def _validate_json_value(value: object) -> None:
    """Reject values Pydantic could otherwise coerce into JSON types."""
    if value is None or type(value) in {str, int, bool}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("non-finite numbers are not JSON-compatible")
        return
    if type(value) is list:
        for item in cast(list[object], value):
            _validate_json_value(item)
        return
    if type(value) is dict:
        for key, item in cast(dict[object, object], value).items():
            if type(key) is not str:
                raise ValueError("JSON object keys must be strings")
            _validate_json_value(item)
        return
    raise ValueError("value is not JSON-compatible")


class GameResult(BaseModel):
    """Terminal outcome of a game session."""

    model_config = ConfigDict(extra="forbid")

    completed: bool
    outcome: JsonObject = Field(default_factory=dict)
    metrics: GameMetrics = Field(default_factory=dict)


class MatchMemorySummary(BaseModel):
    """A compact public memory summary for one actor."""

    model_config = ConfigDict(extra="forbid")

    actor_id: StrictStr = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    hand: StrictInt = Field(..., ge=0)
    turn: StrictInt = Field(..., ge=0)

    @field_validator("actor_id")
    @classmethod
    def reject_blank_actor_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("actor_id must not be empty or whitespace-only")
        return value

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be empty or whitespace-only")
        return value
