"""Regression tests for strict bunker configuration validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from benchtable.games.bunker.config import BunkerConfig, parse_bunker_config


def _valid_config() -> dict[str, object]:
    return {
        "players": ["player-1", "player-2", "player-3", "player-4"],
        "scenario": "sealed shelter after a solar storm",
        "shelter_capacity": 2,
    }


def test_valid_config_preserves_strict_values() -> None:
    config = parse_bunker_config(_valid_config())

    assert config.players == ["player-1", "player-2", "player-3", "player-4"]
    assert config.scenario == "sealed shelter after a solar storm"
    assert config.shelter_capacity == 2


@pytest.mark.parametrize(
    "capacity",
    [1, 3],
)
def test_capacity_boundaries_are_valid(capacity: int) -> None:
    raw = _valid_config()
    raw["shelter_capacity"] = capacity

    config = BunkerConfig.model_validate(raw)

    assert config.shelter_capacity == capacity


@pytest.mark.parametrize(
    "raw",
    [
        {**_valid_config(), "players": ["player-1", "player-2", "player-3"]},
        {
            **_valid_config(),
            "players": [f"player-{index}" for index in range(1, 10)],
        },
        {
            **_valid_config(),
            "players": ["player-1", "player-2", "player-1", "player-4"],
        },
        {**_valid_config(), "players": ["player-1", "player-2", " ", "player-4"]},
        {
            **_valid_config(),
            "players": ("player-1", "player-2", "player-3", "player-4"),
        },
        {**_valid_config(), "players": ["player-1", "player-2", 3, "player-4"]},
        {**_valid_config(), "scenario": ""},
        {**_valid_config(), "scenario": "   "},
        {**_valid_config(), "scenario": 7},
        {**_valid_config(), "shelter_capacity": 0},
        {**_valid_config(), "shelter_capacity": -1},
        {**_valid_config(), "shelter_capacity": 4},
        {**_valid_config(), "shelter_capacity": 2.0},
        {**_valid_config(), "shelter_capacity": True},
        {**_valid_config(), "unknown": 1},
    ],
)
def test_invalid_bunker_config_is_rejected(raw: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        BunkerConfig.model_validate(raw)
