"""Strict typed configuration for the poker game plugin."""

from __future__ import annotations

from typing import Literal, cast

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


class PokerConfig(BaseModel):
    """Validated poker configuration.  Rejects unknown keys and coerced values."""

    model_config = ConfigDict(extra="forbid")

    players: list[StrictStr] = Field(min_length=2)
    initial_stack: StrictInt = Field(ge=1, default=1000)
    small_blind: StrictInt = Field(ge=1, default=5)
    big_blind: StrictInt = Field(ge=1, default=10)
    hands_per_match: StrictInt = Field(ge=1, default=50)
    invalid_turn_policy: Literal["forced_fold", "fail_match"] = "forced_fold"
    memory_max_entries: StrictInt = Field(default=100, ge=0)
    memory_max_chars: StrictInt = Field(default=20000, ge=0)

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
    def validate_poker_constraints(self) -> PokerConfig:
        if len(self.players) != len(set(self.players)):
            raise ValueError("Player IDs must be unique")
        if self.big_blind <= self.small_blind:
            raise ValueError("big_blind must be greater than small_blind")
        if self.initial_stack < self.big_blind:
            raise ValueError("initial_stack must cover the big blind")
        if 2 * len(self.players) + 8 > 52:
            raise ValueError(
                f"Too many players ({len(self.players)}): "
                f"2 * player_count + 8 = {2 * len(self.players) + 8} "
                "exceeds 52-card deck"
            )
        return self


def parse_poker_config(config: JsonObject) -> PokerConfig:
    """Parse and validate raw config dict into a typed PokerConfig."""
    return PokerConfig.model_validate(config)
