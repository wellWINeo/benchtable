"""Unit tests for the bunker session core."""

from __future__ import annotations

import re

import pytest
from pydantic import ValidationError

from benchtable.errors import InvalidActionError
from benchtable.games.bunker.session import BunkerGame, BunkerSession

PLAYERS = ["p1", "p2", "p3", "p4"]


def _session(seed: int = 7) -> BunkerSession:
    return BunkerSession(
        players=list(PLAYERS), scenario="sealed shelter", shelter_capacity=2, seed=seed
    )


def _speak(
    session: BunkerSession,
    actor_id: str,
    text: str = "a public statement",
    reveal_attribute: str | None = None,
) -> None:
    arguments: dict[str, str] = {"text": text}
    if reveal_attribute is not None:
        arguments["reveal_attribute"] = reveal_attribute
    session.apply_action(actor_id, "bunker_speak", arguments)


def _dossier_values(observation_text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for category in ("profession", "health", "skill", "trait"):
        match = re.search(
            rf"^\s+{category}(?: \(revealed\))?: (.+)$", observation_text, re.M
        )
        assert match is not None, f"missing {category} in observation"
        values[category] = match.group(1)
    return values


def test_dossiers_deterministic_for_seed_and_private() -> None:
    first = _session(seed=7)
    second = BunkerSession(
        players=list(PLAYERS), scenario="sealed shelter", shelter_capacity=2, seed=7
    )
    for actor_id in PLAYERS:
        assert _dossier_values(first.get_observation(actor_id).text) == (
            _dossier_values(second.get_observation(actor_id).text)
        )

    other = _session(seed=8)
    differing = any(
        _dossier_values(first.get_observation(actor_id).text)
        != _dossier_values(other.get_observation(actor_id).text)
        for actor_id in PLAYERS
    )
    assert differing, "different seeds must eventually assign different dossiers"

    dossiers = {
        actor_id: _dossier_values(first.get_observation(actor_id).text)
        for actor_id in PLAYERS
    }
    for actor_id in PLAYERS:
        assert len(set(dossiers[actor_id].values())) == 4
    assert dossiers["p1"] != dossiers["p2"], (
        "pools allow cross-actor dossier differences"
    )


def test_delta_delivery_and_no_redelivery() -> None:
    session = _session()
    before = session.get_observation("p2")
    assert "[1]" not in before.text
    assert before.metadata["public_seq"] == 0

    _speak(session, "p1", "hello group")

    after = session.get_observation("p2")
    assert "[1] p1 said: hello group" in after.text
    assert after.metadata["public_seq"] == 1

    _speak(session, "p2", "my own reply")

    p1_view = session.get_observation("p1")
    assert "[1] p1 said: hello group" in p1_view.text
    assert "[2] p2 said: my own reply" in p1_view.text
    assert p1_view.metadata["public_seq"] == 2

    repeat = session.get_observation("p1")
    assert "[3]" not in repeat.text
    assert repeat.metadata["public_seq"] == 2
    assert len(repeat.text) <= len(p1_view.text)


def test_unrevealed_values_isolated_and_reveals_become_public() -> None:
    session = _session()
    p1_dossier = _dossier_values(session.get_observation("p1").text)

    p2_view = session.get_observation("p2")
    for category, value in p1_dossier.items():
        assert value not in p2_view.text, f"{category} leaked to p2"

    _speak(session, "p1", "I bring practical value", reveal_attribute="skill")

    revealed_value = p1_dossier["skill"]
    for actor_id in PLAYERS:
        view = session.get_observation(actor_id)
        assert f"p1 revealed skill: {revealed_value}" in view.text
    assert "skill (revealed)" in session.get_observation("p1").text

    p2_after = session.get_observation("p2")
    for category, value in p1_dossier.items():
        if category == "skill":
            continue
        assert value not in p2_after.text, f"unrevealed {category} leaked to p2"


def test_speak_validation_rejects_bad_actions() -> None:
    session = _session()

    with pytest.raises(InvalidActionError):
        _speak(session, "p1", "   ")
    with pytest.raises(InvalidActionError):
        _speak(session, "p1", "x" * 501)
    with pytest.raises(InvalidActionError):
        _speak(session, "p1", "x" * 500, reveal_attribute="luck")
    with pytest.raises(InvalidActionError):
        _speak(session, "p2")

    _speak(session, "p1", "x" * 500)
    with pytest.raises(InvalidActionError):
        _speak(session, "p1", "second time")
    with pytest.raises(InvalidActionError):
        session.apply_action(
            "p2",
            "bunker_vote_eliminate",
            {"target_id": "p3"},
        )

    _speak(session, "p2")
    _speak(session, "p3")
    _speak(session, "p4")
    assert session.get_observation("p1").metadata["phase"] == "ballot"
    with pytest.raises(InvalidActionError):
        _speak(session, "p1", "speaking during ballots")
    with pytest.raises(InvalidActionError):
        session.apply_action("p1", "bunker_vote_eliminate", {"target_id": "p2"})


def test_rotation_and_ballot_phase_switch() -> None:
    session = _session()
    for expected_speaker in PLAYERS:
        assert session.current_actor_id == expected_speaker
        tools = session.get_tools(expected_speaker)
        assert [tool.name for tool in tools] == ["bunker_speak"]
        _speak(session, expected_speaker, f"statement by {expected_speaker}")

    assert session.current_actor_id == "p1"
    tools = session.get_tools("p1")
    assert [tool.name for tool in tools] == ["bunker_vote_eliminate"]

    observation = session.get_observation("p3")
    assert observation.metadata["phase"] == "ballot"
    assert observation.metadata["round_index"] == 0
    assert observation.metadata["survivors"] == PLAYERS
    assert "bunker_vote_eliminate" in observation.text


def test_speak_without_reveal_always_legal_until_ballot() -> None:
    session = _session(seed=3)
    for index, actor_id in enumerate(PLAYERS):
        assert session.get_observation(actor_id).metadata["phase"] == "discuss"
        _speak(session, actor_id, f"round statement {index}")
    assert session.get_observation("p1").metadata["phase"] == "ballot"


def test_observation_text_bounded_without_new_events() -> None:
    session = _session()
    _speak(session, "p1", "first")
    _speak(session, "p2", "second")

    first_view = session.get_observation("p3")
    second_view = session.get_observation("p3")
    assert len(second_view.text) <= len(first_view.text)
    assert second_view.metadata["public_seq"] == first_view.metadata["public_seq"]


def test_plugin_factory_contract() -> None:
    plugin = BunkerGame()
    assert plugin.name == "bunker"
    assert plugin.version == "0.1.0"
    assert plugin.requires_exact_agent_ids is True
    game_config = {
        "players": PLAYERS,
        "scenario": "sealed shelter",
        "shelter_capacity": 2,
    }
    assert plugin.player_ids_from_config(game_config) == PLAYERS
    plugin.validate_config(game_config)
    with pytest.raises(ValidationError):
        plugin.validate_config({"players": PLAYERS})

    prompt = plugin.system_prompt("p1")
    assert "bunker_speak" in prompt
    assert "bunker_vote_eliminate" in prompt
    assert "untrusted" in prompt

    session = plugin.create_session(seed=11, game_config=game_config)
    assert isinstance(session, BunkerSession)
    assert session.current_actor_id == "p1"
    assert "sealed shelter" in session.get_observation("p1").text
