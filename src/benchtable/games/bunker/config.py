"""Strict typed configuration for the bunker game plugin."""

from __future__ import annotations

from typing import cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from benchtable.contracts import JsonObject


class BunkerConfig(BaseModel):
    """Validated bunker configuration.  Rejects unknown keys and coerced values."""

    model_config = ConfigDict(extra="forbid")

    players: list[StrictStr] = Field(min_length=4, max_length=8)
    scenario: StrictStr = Field(min_length=1)
    shelter_capacity: StrictInt = Field(ge=1)

    @field_validator("scenario")
    @classmethod
    def reject_blank_scenario(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Scenario must not be blank")
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
    def validate_capacity_constraint(self) -> BunkerConfig:
        if len(self.players) != len(set(self.players)):
            raise ValueError("Player IDs must be unique")
        if self.shelter_capacity >= len(self.players):
            raise ValueError("shelter_capacity must be below the player count")
        return self


def parse_bunker_config(config: JsonObject) -> BunkerConfig:
    """Parse and validate raw config dict into a typed BunkerConfig."""
    return BunkerConfig.model_validate(config)
