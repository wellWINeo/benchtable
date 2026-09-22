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


def _vote(session: BunkerSession, actor_id: str, target_id: str) -> None:
    session.apply_action(actor_id, "bunker_vote_eliminate", {"target_id": target_id})


def _run_discussion(session: BunkerSession, speakers: list[str]) -> None:
    for actor_id in speakers:
        _speak(session, actor_id)


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

    # Ballots are legal in the ballot phase for the scheduled voter.
    _vote(session, "p1", "p2")
    assert session.current_actor_id == "p2"


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


def test_ballot_rotation_and_plurality_elimination() -> None:
    session = _session()
    _run_discussion(session, PLAYERS)

    for expected_voter in PLAYERS:
        assert session.current_actor_id == expected_voter
        tools = session.get_tools(expected_voter)
        assert [tool.name for tool in tools] == ["bunker_vote_eliminate"]
        target = "p3" if expected_voter == "p2" else "p2"
        _vote(session, expected_voter, target)

    # votes: p1->p2, p2->p3, p3->p2, p4->p2 => p2 has the plurality
    assert not session.is_terminal
    view = session.get_observation("p1")
    assert view.metadata["phase"] == "discuss"
    assert view.metadata["round_index"] == 1
    assert view.metadata["survivors"] == ["p1", "p3", "p4"]
    assert "vote totals: p2: 3, p3: 1" in view.text
    assert "p2 eliminated (no tie break)" in view.text


def test_tie_break_deterministic_for_seed_and_recorded() -> None:

    def run(seed: int) -> BunkerSession:
        tie = BunkerSession(
            players=list(PLAYERS),
            scenario="sealed shelter",
            shelter_capacity=3,
            seed=seed,
        )
        _run_discussion(tie, PLAYERS)
        _vote(tie, "p1", "p2")
        _vote(tie, "p2", "p3")
        _vote(tie, "p3", "p2")
        _vote(tie, "p4", "p3")
        return tie

    first = run(seed=11)
    second = run(seed=11)
    assert first.is_terminal and second.is_terminal
    assert first._eliminations == second._eliminations
    assert first._eliminations[0]["tie_break"] is True
    assert first._eliminations[0]["target"] in {"p2", "p3"}

    other_seed = run(seed=12)
    assert other_seed._eliminations[0]["target"] in {"p2", "p3"}

    view = first.get_observation("p1")
    assert "vote totals: p2: 2, p3: 2" in view.text
    assert re.search(
        rf"\[\d+\] {first._eliminations[0]['target']} eliminated \(tie break\)",
        view.text,
    )


def test_vote_validation_rejects_bad_ballots() -> None:
    session = _session()

    with pytest.raises(InvalidActionError):
        _vote(session, "p1", "p2")  # discuss phase

    _run_discussion(session, PLAYERS)

    with pytest.raises(InvalidActionError):
        _vote(session, "p2", "p3")  # out of turn
    with pytest.raises(InvalidActionError):
        _vote(session, "p1", "p1")  # self-vote
    with pytest.raises(InvalidActionError):
        _vote(session, "p1", "   ")  # blank target
    with pytest.raises(InvalidActionError):
        _vote(session, "p1", "p9")  # unknown target
    with pytest.raises(InvalidActionError):
        _speak(session, "p1", "wrong tool for ballots")

    _vote(session, "p1", "p2")
    with pytest.raises(InvalidActionError):
        _vote(session, "p1", "p3")  # double vote


def test_ballot_secrecy_until_closure() -> None:
    session = _session()
    _run_discussion(session, PLAYERS)

    _vote(session, "p1", "p3")

    events_before = session.drain_hand_events()
    for event in events_before:
        assert "target_id" not in event.payload

    for observer in ("p2", "p3", "p4"):
        view = session.get_observation(observer)
        assert "vote totals" not in view.text
        assert "eliminated" not in view.text
        assert "against p3" not in view.text

    _vote(session, "p2", "p3")
    _vote(session, "p3", "p2")
    _vote(session, "p4", "p3")

    view = session.get_observation("p1")
    assert "vote totals: p3: 3, p2: 1" in view.text
    assert "p3 eliminated (no tie break)" in view.text


def test_capacity_termination_and_terminal_rejection() -> None:
    session = _session()

    _run_discussion(session, PLAYERS)
    _vote(session, "p1", "p2")
    _vote(session, "p2", "p3")
    _vote(session, "p3", "p2")
    _vote(session, "p4", "p2")

    assert not session.is_terminal
    assert session.get_observation("p1").metadata["survivors"] == ["p1", "p3", "p4"]

    for speaker in ("p1", "p3", "p4"):
        assert session.current_actor_id == speaker
        _speak(session, speaker)

    with pytest.raises(InvalidActionError):
        _vote(session, "p1", "p2")  # eliminated player

    _vote(session, "p1", "p3")
    _vote(session, "p3", "p1")
    _vote(session, "p4", "p3")

    assert session.is_terminal
    assert session.get_observation("p1").metadata["survivors"] == ["p1", "p4"]

    with pytest.raises(InvalidActionError):
        _speak(session, "p1", "after the end")
    with pytest.raises(InvalidActionError):
        _vote(session, "p1", "p4")
    with pytest.raises(InvalidActionError):
        _vote(session, "p1", "p2")


def _scripted_full_match(seed: int = 5) -> BunkerSession:
    """4 players, capacity 2: p2 eliminated in round 1, p3 in round 2."""
    session = _session(seed=seed)
    _run_discussion(session, PLAYERS)
    _vote(session, "p1", "p2")
    _vote(session, "p2", "p3")
    _vote(session, "p3", "p2")
    _vote(session, "p4", "p2")
    _run_discussion(session, ["p1", "p3", "p4"])
    _vote(session, "p1", "p3")
    _vote(session, "p3", "p1")
    _vote(session, "p4", "p3")
    return session


def test_completed_result_contract() -> None:
    session = _scripted_full_match()
    assert session.is_terminal

    result = session.get_result()
    assert result.completed is True
    assert result.outcome["admitted"] == ["p1", "p4"]
    assert result.outcome["excluded"] == ["p2", "p3"]
    assert result.outcome["elimination_order"] == ["p2", "p3"]
    assert result.outcome["scenario"] == "sealed shelter"
    assert result.outcome["capacity"] == 2
    assert result.outcome["completion_reason"] == "capacity_reached"

    metrics = result.metrics
    assert metrics["elimination_rounds"] == 2
    assert metrics["speeches"] == 7
    assert metrics["ballots_cast"] == 7
    assert metrics["reveals_total"] == 0
    assert metrics["reveal_counts"] == {pid: 0 for pid in PLAYERS}
    assert metrics["failed_turn_eliminations"] == 0
    assert metrics["completed"] is True


def test_in_progress_result_contract() -> None:
    session = _session()
    _speak(session, "p1")

    result = session.get_result()
    assert result.completed is False
    assert result.outcome["admitted"] == PLAYERS
    assert result.outcome["excluded"] == []
    assert result.outcome["elimination_order"] == []
    assert result.outcome["scenario"] == "sealed shelter"
    assert result.outcome["capacity"] == 2
    assert result.outcome["completion_reason"] == "in_progress"
    assert result.metrics["completed"] is False
    assert result.metrics["elimination_rounds"] == 0
    assert result.metrics["speeches"] == 1


def test_reveals_counted_in_result_metrics() -> None:
    session = _session()
    _speak(session, "p1", "my contribution", reveal_attribute="skill")
    _speak(session, "p2", "no reveal from me")

    result = session.get_result()
    assert result.metrics["reveals_total"] == 1
    assert result.metrics["reveal_counts"]["p1"] == 1
    assert result.metrics["reveal_counts"]["p2"] == 0
    assert result.metrics["speeches"] == 2


def test_plugin_events_lifecycle_and_public_safety() -> None:
    session = _scripted_full_match()

    events = session.drain_hand_events()
    types = [event.event_type for event in events]

    assert types[0] == "round_start"
    assert types.count("round_start") == 2
    assert types.count("bunker_speak") == 7
    assert types.count("bunker_reveal") == 0
    assert types.count("bunker_vote_totals") == 2
    assert types.count("bunker_elimination") == 2
    assert types[-1] == "bunker_match_end"
    assert "match_end" not in types  # only the bunker-prefixed name

    round_start = events[0]
    assert round_start.payload["round_index"] == 0
    assert round_start.payload["survivors"] == PLAYERS

    vote_totals = [e for e in events if e.event_type == "bunker_vote_totals"]
    assert vote_totals[0].payload["round_index"] == 0
    assert vote_totals[0].payload["totals"] == {"p2": 3, "p3": 1}
    assert vote_totals[1].payload["round_index"] == 1
    assert vote_totals[1].payload["totals"] == {"p3": 2, "p1": 1}

    eliminations = [e for e in events if e.event_type == "bunker_elimination"]
    assert eliminations[0].payload["eliminated"] == "p2"
    assert eliminations[0].payload["tie_break"] is False
    assert eliminations[0].payload["reason"] == "vote"
    assert eliminations[1].payload["eliminated"] == "p3"

    match_end = events[-1]
    assert match_end.event_type == "bunker_match_end"
    assert match_end.payload["admitted"] == ["p1", "p4"]
    assert match_end.payload["excluded"] == ["p2", "p3"]
    assert match_end.payload["completion_reason"] == "capacity_reached"

    # Drain clears the queue.
    assert session.drain_hand_events() == []

    # Public safety: no unrevealed dossier values, no individual ballots.
    reference = _session(seed=5)
    hidden = {
        _dossier_values(reference.get_observation(pid).text)[category]
        for pid in PLAYERS
        for category in ("profession", "health", "skill", "trait")
    }
    for event in events:
        assert "target_id" not in event.payload
        assert "ballots" not in event.payload
        blob = str(event.payload)
        for value in hidden:
            assert value not in blob, f"dossier value {value!r} leaked into trace"


def test_failed_turn_during_discussion_eliminates_and_advances_round() -> None:
    session = _session()
    _speak(session, "p1")
    _speak(session, "p2")

    transition = session.handle_failed_turn("p3", "invalid_attempts_exhausted")

    assert transition is not None
    assert session._survivors == ["p1", "p2", "p4"]
    assert session._round_index == 1
    assert session._phase == "discuss"
    assert session._speaker_pointer == 0
    assert session.current_actor_id == "p1"
    assert session._eliminations == [
        {
            "round_index": 0,
            "target": "p3",
            "reason": "failed_turn_invalid_attempts_exhausted",
            "tie_break": False,
        }
    ]
    assert "p3 eliminated (no tie break)" in session.get_observation("p1").text


def test_failed_turn_during_ballot_discards_partial_ballots() -> None:
    session = _session()
    _run_discussion(session, PLAYERS)
    _vote(session, "p1", "p3")  # one partial ballot is collected
    assert session._ballots_cast == 1

    transition = session.handle_failed_turn("p2", "provider_retries_exhausted")

    assert transition is not None
    assert session._pending_ballots == []
    assert session._round_index == 1
    assert session._phase == "discuss"
    assert session.current_actor_id == "p1"
    assert session._eliminations == [
        {
            "round_index": 0,
            "target": "p2",
            "reason": "failed_turn_provider_retries_exhausted",
            "tie_break": False,
        }
    ]

    # The partial ballot never becomes a fabricated vote: no vote totals in
    # any observation, and no bunker_vote_totals event in the drained queue.
    view = session.get_observation("p1")
    assert "vote totals" not in view.text
    events = session.drain_hand_events()
    assert all(event.event_type != "bunker_vote_totals" for event in events)
    assert any(
        event.event_type == "bunker_elimination"
        and event.payload["eliminated"] == "p2"
        and event.payload["reason"] == "failed_turn_provider_retries_exhausted"
        for event in events
    )


def test_failed_turn_reaching_capacity_completes_match() -> None:
    session = BunkerSession(
        players=list(PLAYERS), scenario="sealed shelter", shelter_capacity=3, seed=9
    )

    transition = session.handle_failed_turn("p4", "invalid_attempts_exhausted")

    assert transition is not None
    assert session.is_terminal
    result = session.get_result()
    assert result.completed is True
    assert result.outcome["completion_reason"] == "capacity_reached"
    assert result.outcome["admitted"] == ["p1", "p2", "p3"]
    assert result.outcome["excluded"] == ["p4"]
    assert result.metrics["failed_turn_eliminations"] == 1

    events = session.drain_hand_events()
    end_events = [e for e in events if e.event_type == "bunker_match_end"]
    assert len(end_events) == 1
    assert end_events[0].payload["completion_reason"] == "capacity_reached"
    assert end_events[0].payload["admitted"] == ["p1", "p2", "p3"]


def test_failed_turn_transition_is_public_safe() -> None:
    session = _session()
    _speak(session, "p1")
    hidden = _dossier_values(session.get_observation("p2").text)

    transition = session.handle_failed_turn("p2", "invalid_attempts_exhausted")

    assert transition is not None
    assert transition.summary == (
        "round 1: p2 eliminated by failed turn (invalid_attempts_exhausted)"
    )
    for value in hidden.values():
        assert value not in transition.summary
        assert value not in str(transition.metrics)


def test_failed_turn_for_eliminated_actor_returns_none() -> None:
    session = _session()
    _speak(session, "p1")
    session.handle_failed_turn("p2", "invalid_attempts_exhausted")

    assert session.handle_failed_turn("p2", "provider_retries_exhausted") is None
    assert session.get_result().metrics["elimination_rounds"] == 1
    assert session._survivors == ["p1", "p3", "p4"]


def test_failed_turn_after_match_terminal_returns_none() -> None:
    session = _session()
    for target in ("p2", "p3"):
        survivors = list(session._survivors)
        for speaker in survivors:
            _speak(session, speaker)
        for voter in survivors:
            session.apply_action(
                voter,
                "bunker_vote_eliminate",
                {"target_id": target if voter != target else survivors[0]},
            )

    assert session.is_terminal
    assert session.handle_failed_turn("p1", "invalid_attempts_exhausted") is None
