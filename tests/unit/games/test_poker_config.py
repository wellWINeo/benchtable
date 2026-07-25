"""Regression tests for strict poker configuration validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from benchtable.games.poker.config import PokerConfig
from benchtable.games.poker.session import PokerGame


def _valid_config() -> dict[str, object]:
    return {
        "players": ["alice", "bob"],
        "initial_stack": 1000,
        "small_blind": 5,
        "big_blind": 10,
        "hands_per_match": 2,
        "memory_max_entries": 10,
        "memory_max_chars": 1000,
        "invalid_turn_policy": "forced_fold",
    }


def test_valid_config_preserves_strict_values() -> None:
    config = PokerConfig.model_validate(_valid_config())

    assert config.players == ["alice", "bob"]
    assert config.initial_stack == 1000
    assert config.small_blind == 5
    assert config.big_blind == 10


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("players", ["alice", 2]),
        ("players", ("alice", "bob")),
        ("players", ["alice", " "]),
        ("initial_stack", "1000"),
        ("initial_stack", True),
        ("small_blind", 5.0),
        ("big_blind", False),
        ("hands_per_match", "2"),
        ("memory_max_entries", 1.0),
        ("memory_max_chars", True),
        ("invalid_turn_policy", "ignore"),
    ],
)
def test_invalid_poker_values_are_rejected(field: str, value: object) -> None:
    raw = _valid_config()
    raw[field] = value

    with pytest.raises(ValidationError):
        PokerConfig.model_validate(raw)


@pytest.mark.parametrize(
    "raw",
    [
        {key: value for key, value in _valid_config().items() if key != "players"},
        {**_valid_config(), "players": ["alice", "alice"]},
        {**_valid_config(), "players": ["alice"]},
        {**_valid_config(), "small_blind": 10, "big_blind": 10},
        {**_valid_config(), "initial_stack": 9},
        {**_valid_config(), "memory_max_entries": -1},
        {**_valid_config(), "memory_max_chars": -1},
        {**_valid_config(), "unknown": 1},
    ],
)
def test_invalid_poker_config_is_rejected(raw: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        PokerConfig.model_validate(raw)


def test_player_ids_from_config_does_not_fallback_for_malformed_players() -> None:
    game = PokerGame()

    with pytest.raises(ValidationError):
        game.player_ids_from_config({"players": ["alice", 2]})


def test_player_ids_from_config_requires_players() -> None:
    game = PokerGame()

    with pytest.raises(ValidationError):
        game.player_ids_from_config({})


def test_poker_config_rejects_tables_that_exceed_deck_capacity() -> None:
    raw = _valid_config()
    raw["players"] = [f"player-{index}" for index in range(23)]

    with pytest.raises(ValidationError, match="52-card deck"):
        PokerConfig.model_validate(raw)
