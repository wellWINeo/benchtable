"""TOML experiment configuration loading and validation."""

from __future__ import annotations

import json
import math
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
)

from benchtable.contracts import JsonObject
from benchtable.errors import ConfigurationError

_CREDENTIAL_KEY_NAMES = frozenset(
    {
        "apikey",
        "apitoken",
        "authorization",
        "accesstoken",
        "password",
        "clientsecret",
        "secretkey",
        "bearertoken",
        "token",
        "auth",
        "xapikey",
        "secret",
        "privatekey",
        "refreshtoken",
        "credential",
        "credentials",
    }
)


def _normalize_key(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _contains_credential_key(value: object) -> bool:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        for key, nested_value in mapping.items():
            if isinstance(key, str) and _normalize_key(key) in _CREDENTIAL_KEY_NAMES:
                return True
            if _contains_credential_key(nested_value):
                return True
    elif isinstance(value, list):
        for nested_value in cast(list[object], value):
            if _contains_credential_key(nested_value):
                return True
    return False


def _is_strict_json_value(value: object) -> bool:
    if value is None or type(value) in (str, int, bool):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is list:
        return all(_is_strict_json_value(item) for item in cast(list[object], value))
    if type(value) is dict:
        mapping = cast(dict[object, object], value)
        return all(
            type(key) is str and _is_strict_json_value(nested_value)
            for key, nested_value in mapping.items()
        )
    return False


class AgentConfig(BaseModel):
    """Configuration for a single agent."""

    model_config = ConfigDict(extra="forbid")

    id: StrictStr = Field(min_length=1)
    role: str = "player"
    model: StrictStr = Field(min_length=1)
    base_url: StrictStr | None = Field(default=None, min_length=1)
    api_key_env: StrictStr = "OPENAI_API_KEY"
    timeout: float | None = Field(default=None, gt=0)
    max_completion_tokens: StrictInt | None = Field(default=None, ge=1)

    @field_validator("timeout", mode="before")
    @classmethod
    def validate_timeout(cls, value: object) -> float | None:
        if value is None:
            return None
        try:
            if type(value) is int:
                numeric_value = float(value)
            elif type(value) is float:
                numeric_value = value
            else:
                raise ValueError("Timeout must be a finite positive number")
        except OverflowError as exc:
            raise ConfigurationError(
                "Timeout must be a finite positive number"
            ) from exc
        if not math.isfinite(numeric_value) or numeric_value <= 0:
            raise ValueError("Timeout must be a finite positive number")
        return numeric_value

    @field_validator("id", "model", "base_url", "api_key_env")
    @classmethod
    def reject_whitespace_only_strings(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Provider strings must not be whitespace-only")
        return value


class RunConfig(BaseModel):
    """Run-level configuration."""

    model_config = ConfigDict(extra="forbid")

    game: str
    matches: StrictInt = Field(ge=1)
    seed: StrictInt = 0
    max_turns: StrictInt = Field(default=200, ge=1)
    max_invalid_attempts: StrictInt = Field(default=2, ge=0)
    max_provider_retries: StrictInt = Field(default=2, ge=0)
    max_memory_operations_per_turn: StrictInt = Field(default=4, ge=0)
    game_config: JsonObject = Field(default_factory=dict)

    @field_validator("game_config")
    @classmethod
    def reject_non_finite_numbers(cls, value: JsonObject) -> JsonObject:
        try:
            json.dumps(value, allow_nan=False)
        except ValueError as exc:
            raise ValueError(
                "game_config must contain only finite JSON numbers"
            ) from exc
        return value

    @field_validator("game_config", mode="before")
    @classmethod
    def validate_game_config_input(cls, value: object) -> object:
        if _contains_credential_key(value):
            raise ConfigurationError(
                "Configuration contains a credential-shaped field; "
                "use 'api_key_env' instead of literal credentials."
            )
        if type(value) is not dict:
            raise ValueError("game_config must be a strict JSON object")
        mapping = cast(dict[object, object], value)
        if not _is_strict_json_value(mapping):
            raise ValueError("game_config must be a strict JSON object")
        return mapping


class ExperimentConfig(BaseModel):
    """Full experiment configuration."""

    model_config = ConfigDict(extra="forbid")

    run: RunConfig
    agents: list[AgentConfig] = Field(min_length=1)

    @field_validator("agents")
    @classmethod
    def reject_duplicate_agent_ids(cls, value: list[AgentConfig]) -> list[AgentConfig]:
        agent_ids = [agent.id for agent in value]
        if len(agent_ids) != len(set(agent_ids)):
            raise ValueError("Agent IDs must be unique")
        return value


def load_config(path: Path | str) -> ExperimentConfig:
    """Load and validate an experiment configuration from a TOML file.

    Raises ConfigurationError for invalid or missing fields.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigurationError(f"Configuration file not found: {path}")

    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise ConfigurationError("Invalid TOML configuration") from exc
    except OSError as exc:
        raise ConfigurationError(f"Could not open configuration file: {path}") from exc

    if _contains_credential_key(data):
        raise ConfigurationError(
            "Configuration contains a credential-shaped field; "
            "use 'api_key_env' instead of literal credentials."
        )

    try:
        config = ExperimentConfig.model_validate(data)
    except ValidationError as exc:
        fields = sorted(
            {".".join(str(part) for part in error["loc"]) for error in exc.errors()}
        )
        field_summary = ", ".join(fields) or "configuration fields"
        raise ConfigurationError(f"Invalid configuration for {field_summary}") from exc
    except Exception as exc:
        raise ConfigurationError("Invalid configuration") from exc

    return config
