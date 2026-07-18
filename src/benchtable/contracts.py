"""Typed contracts for benchtable engine, games, and agents."""

from __future__ import annotations

from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

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

    call_id: str
    name: str
    arguments: JsonObject | None = None
    raw_arguments: str | None = None
    parse_error: str | None = None

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
    raw_provider_response: JsonObject | None = None


# ---------------------------------------------------------------------------
# Game result and transition
# ---------------------------------------------------------------------------


class Transition(BaseModel):
    """The result of applying a valid action."""

    model_config = ConfigDict(extra="forbid")

    summary: str
    metrics: GameMetrics = Field(default_factory=dict)


class GameResult(BaseModel):
    """Terminal outcome of a game session."""

    model_config = ConfigDict(extra="forbid")

    completed: bool
    outcome: JsonObject = Field(default_factory=dict)
    metrics: GameMetrics = Field(default_factory=dict)
