"""Regression tests for strict Spyfall configuration validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from benchtable.games.spyfall.config import (
    SpyfallConfig,
    min_required_turns,
    parse_spyfall_config,
)


def _valid_config() -> dict[str, object]:
    return {
        "players": ["player-1", "player-2", "player-3"],
        "rounds_per_match": 5,
        "question_rounds": 3,
        "accusation_vote_threshold": 0.75,
        "judge_id": "spyfall-leak-judge",
        "judge_leak_threshold": 0.85,
        "locations": [{"name": "airport"}],
    }


def test_valid_config_preserves_strict_values() -> None:
    config = SpyfallConfig.model_validate(_valid_config())

    assert config.players == ["player-1", "player-2", "player-3"]
    assert config.rounds_per_match == 5
    assert config.question_rounds == 3
    assert config.accusation_vote_threshold == 0.75
    assert config.judge_id == "spyfall-leak-judge"
    assert config.judge_leak_threshold == 0.85
    assert [location.name for location in config.locations] == ["airport"]


def test_parse_spyfall_config_matches_model_validate() -> None:
    config = parse_spyfall_config(_valid_config())

    assert config == SpyfallConfig.model_validate(_valid_config())


@pytest.mark.parametrize(
    "players",
    [
        ["player-1", "player-2"],
        [f"player-{index}" for index in range(1, 10)],
    ],
)
def test_rejects_player_count_outside_three_to_eight(players: list[str]) -> None:
    data = _valid_config() | {"players": players}

    with pytest.raises(ValidationError):
        SpyfallConfig.model_validate(data)


def test_rejects_duplicate_players() -> None:
    data = _valid_config() | {"players": ["player-1", "player-1", "player-2"]}

    with pytest.raises(ValidationError):
        SpyfallConfig.model_validate(data)


def test_rejects_blank_player_entry() -> None:
    data = _valid_config() | {"players": ["player-1", "   ", "player-3"]}

    with pytest.raises(ValidationError):
        SpyfallConfig.model_validate(data)


def test_rejects_non_list_players() -> None:
    data = _valid_config() | {"players": "player-1"}

    with pytest.raises(ValidationError):
        SpyfallConfig.model_validate(data)


@pytest.mark.parametrize("field", ["rounds_per_match", "question_rounds"])
@pytest.mark.parametrize("value", [0, -1, True, "3", 2.5])
def test_rejects_invalid_round_counts(field: str, value: object) -> None:
    data = _valid_config() | {field: value}

    with pytest.raises(ValidationError):
        SpyfallConfig.model_validate(data)


@pytest.mark.parametrize(
    "value",
    [0, -0.5, 1.01, 2.0, "0.75", float("inf"), float("nan")],
)
def test_rejects_invalid_accusation_vote_threshold(value: object) -> None:
    data = _valid_config() | {"accusation_vote_threshold": value}

    with pytest.raises(ValidationError):
        SpyfallConfig.model_validate(data)


def test_accepts_accusation_vote_threshold_upper_boundary() -> None:
    data = _valid_config() | {"accusation_vote_threshold": 1.0}

    config = SpyfallConfig.model_validate(data)

    assert config.accusation_vote_threshold == 1.0


@pytest.mark.parametrize(
    "value",
    [-0.1, 1.5, "0.5", float("inf"), float("nan")],
)
def test_rejects_invalid_judge_leak_threshold(value: object) -> None:
    data = _valid_config() | {"judge_leak_threshold": value}

    with pytest.raises(ValidationError):
        SpyfallConfig.model_validate(data)


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_accepts_judge_leak_threshold_boundaries(value: float) -> None:
    data = _valid_config() | {"judge_leak_threshold": value}

    config = SpyfallConfig.model_validate(data)

    assert config.judge_leak_threshold == value


def test_rejects_empty_locations() -> None:
    data = _valid_config() | {"locations": []}

    with pytest.raises(ValidationError):
        SpyfallConfig.model_validate(data)


def test_rejects_duplicate_location_names() -> None:
    data = _valid_config() | {"locations": [{"name": "airport"}, {"name": "airport"}]}

    with pytest.raises(ValidationError):
        SpyfallConfig.model_validate(data)


def test_rejects_blank_location_name() -> None:
    data = _valid_config() | {"locations": [{"name": "   "}]}

    with pytest.raises(ValidationError):
        SpyfallConfig.model_validate(data)


def test_rejects_location_extra_fields() -> None:
    data = _valid_config() | {"locations": [{"name": "airport", "deck": "base"}]}

    with pytest.raises(ValidationError):
        SpyfallConfig.model_validate(data)


def test_rejects_unknown_top_level_fields() -> None:
    data = _valid_config() | {"spy_master": True}

    with pytest.raises(ValidationError):
        SpyfallConfig.model_validate(data)


def test_min_required_turns_multiplies_rounds_players_and_rotations() -> None:
    config = SpyfallConfig.model_validate(_valid_config())

    assert min_required_turns(config) == 5 * 3**2 * 3


def test_min_required_turns_single_rotation_four_players() -> None:
    data = _valid_config() | {
        "rounds_per_match": 1,
        "question_rounds": 1,
        "players": ["p1", "p2", "p3", "p4"],
    }
    config = SpyfallConfig.model_validate(data)

    assert min_required_turns(config) == 16
