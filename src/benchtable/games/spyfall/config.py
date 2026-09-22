"""Strict typed configuration for the Spyfall game plugin."""

from __future__ import annotations

import math
from typing import cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from benchtable.contracts import JsonObject


class SpyfallLocation(BaseModel):
    """One playable location in the configured catalog."""

    model_config = ConfigDict(extra="forbid")

    name: StrictStr = Field(min_length=1)

    @field_validator("name")
    @classmethod
    def reject_blank_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Location names must not be blank")
        return value


class SpyfallConfig(BaseModel):
    """Validated Spyfall configuration.  Rejects unknown keys and coerced values."""

    model_config = ConfigDict(extra="forbid")

    players: list[StrictStr] = Field(min_length=3, max_length=8)
    rounds_per_match: StrictInt = Field(ge=1)
    question_rounds: StrictInt = Field(ge=1)
    accusation_vote_threshold: StrictFloat = Field(gt=0.0, le=1.0)
    judge_id: StrictStr = Field(min_length=1)
    judge_leak_threshold: StrictFloat = Field(ge=0.0, le=1.0)
    locations: list[SpyfallLocation] = Field(min_length=1)

    @field_validator(
        "accusation_vote_threshold",
        "judge_leak_threshold",
        mode="before",
    )
    @classmethod
    def require_finite_thresholds(cls, value: object) -> object:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Thresholds must be finite numbers")
        return value

    @field_validator("players", mode="before")
    @classmethod
    def require_player_list(cls, value: object) -> object:
        if type(value) is not list:
            raise ValueError("players must be a list of strict strings")
        return cast(list[object], value)

    @field_validator("players")
    @classmethod
    def validate_player_ids(cls, value: list[str]) -> list[str]:
        if any(not player_id.strip() for player_id in value):
            raise ValueError("Player IDs must not be blank")
        return value

    @model_validator(mode="after")
    def validate_spyfall_constraints(self) -> SpyfallConfig:
        if len(self.players) != len(set(self.players)):
            raise ValueError("Player IDs must be unique")
        names = [location.name for location in self.locations]
        if len(names) != len(set(names)):
            raise ValueError("Location names must be unique")
        return self


def min_required_turns(config: SpyfallConfig) -> int:
    """Return the safe engine-turn upper bound ``R * N * N * Q``."""
    return config.rounds_per_match * len(config.players) ** 2 * config.question_rounds


def parse_spyfall_config(config: JsonObject) -> SpyfallConfig:
    """Parse and validate raw config dict into a typed SpyfallConfig."""
    return SpyfallConfig.model_validate(config)
