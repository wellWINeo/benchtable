"""Unit tests for the Spyfall session state machine."""

from __future__ import annotations

import json

import pytest

from benchtable.contracts import JudgmentOutcome
from benchtable.errors import InvalidActionError
from benchtable.games.spyfall.session import (
    LEAK_RUBRIC,
    SpyfallGame,
    SpyfallSession,
)

PLAYERS = ["p1", "p2", "p3"]
LOCATIONS = ["airport", "library"]

TOOL_NAMES = {
    "spyfall_question",
    "spyfall_answer",
    "spyfall_accuse",
    "spyfall_guess_location",
    "spyfall_vote",
}


def make_session(**overrides: object) -> SpyfallSession:
    options: dict[str, object] = {
        "players": list(PLAYERS),
        "rounds_per_match": 1,
        "question_rounds": 1,
        "accusation_vote_threshold": 0.75,
        "locations": list(LOCATIONS),
        "seed": 42,
        "judge_id": "leak-judge",
        "judge_leak_threshold": 0.85,
    }
    options.update(overrides)
    return SpyfallSession(**options)


def known_location(session: SpyfallSession, actor_id: str) -> str | None:
    """Return the location shown to a non-Spy actor, else None."""
    observation = session.get_observation(actor_id)
    for location in LOCATIONS:
        if f"You know the secret location: {location}" in observation.text:
            return location
    return None


def spy_of(session: SpyfallSession) -> str:
    for player_id in PLAYERS:
        if known_location(session, player_id) is None:
            return player_id
    raise AssertionError("no Spy found")


def play_exchange(session: SpyfallSession, *, question: str = "question?") -> None:
    questioner = session.current_actor_id
    target = next(p for p in PLAYERS if p != questioner)
    session.apply_action(
        questioner,
        "spyfall_question",
        {"target_id": target, "text": f"{question} -> {target}"},
    )
    session.apply_action(target, "spyfall_answer", {"text": "an answer"})


def game_config() -> dict[str, object]:
    return {
        "players": list(PLAYERS),
        "rounds_per_match": 5,
        "question_rounds": 3,
        "accusation_vote_threshold": 0.75,
        "judge_id": "leak-judge",
        "judge_leak_threshold": 0.85,
        "locations": [{"name": LOCATIONS[0]}],
    }


def test_seeded_selection_is_deterministic_and_varies() -> None:
    first = make_session(seed=42)
    second = make_session(seed=42)
    first_state = (known_location(first, "p1"), spy_of(first))
    second_state = (known_location(second, "p1"), spy_of(second))
    assert first_state == second_state
    assert first_state[0] in LOCATIONS

    states = set()
    for seed in range(10):
        session = make_session(seed=seed)
        states.add((known_location(session, "p1"), spy_of(session)))
    assert len(states) > 1, "different seeds must eventually differ"


def test_full_rotation_ends_round_for_spy_and_completes_match() -> None:
    session = make_session()
    assert session.is_terminal is False

    for _ in range(3):
        play_exchange(session)

    assert session.is_terminal is True
    result = session.get_result()
    assert result.completed is True
    assert result.outcome["rounds"] == [
        {"round_index": 0, "winner_side": "spy", "reason": "question_rotation_complete"}
    ]
    points = result.outcome["points"]
    assert isinstance(points, dict)
    spy = spy_of(session)
    assert points[spy] == 1
    assert sum(points.values()) == 1  # the Spy side is a single player
    assert result.metrics["round_count"] == 1
    assert result.metrics["question_count"] == 3
    assert result.metrics["completed"] is True


def test_multi_round_starts_fresh_round_and_accumulates_scores() -> None:
    session = make_session(rounds_per_match=2)
    for _ in range(3):
        play_exchange(session)

    assert session.is_terminal is False
    assert session.conversation_scope_id == "round-1"
    observation = session.get_observation(PLAYERS[0])
    assert "Public dialogue this round:" not in observation.text

    for _ in range(3):
        play_exchange(session)

    assert session.is_terminal is True
    result = session.get_result()
    assert result.completed is True
    rounds = result.outcome["rounds"]
    assert isinstance(rounds, list)
    assert len(rounds) == 2
    points = result.outcome["points"]
    assert isinstance(points, dict)
    assert sum(points.values()) == 2  # one Spy-side point per completed round


def test_observation_isolation_between_spy_and_non_spies() -> None:
    session = make_session()
    play_exchange(session)
    spy = spy_of(session)
    witness = next(p for p in PLAYERS if p != spy)
    location = known_location(session, witness)
    assert location is not None

    spy_observation = session.get_observation(spy)
    assert location not in spy_observation.text
    assert "You are the Spy" in spy_observation.text
    assert set(spy_observation.metadata) == {
        "round_index",
        "phase",
        "rotation_step",
    }

    for player_id in PLAYERS:
        if player_id == spy:
            continue
        observation = session.get_observation(player_id)
        assert location in observation.text
        assert "You are the Spy" not in observation.text


def test_dialogue_is_shared_through_observations() -> None:
    session = make_session()
    questioner = session.current_actor_id
    target = next(p for p in PLAYERS if p != questioner)
    session.apply_action(
        questioner,
        "spyfall_question",
        {"target_id": target, "text": "favorite snack?"},
    )
    session.apply_action(target, "spyfall_answer", {"text": "dumplings"})

    for player_id in PLAYERS:
        text = session.get_observation(player_id).text
        assert "favorite snack?" in text
        assert "dumplings" in text


def test_action_validation_rejects_illegal_moves() -> None:
    session = make_session()
    questioner = session.current_actor_id
    other = next(p for p in PLAYERS if p != questioner)

    with pytest.raises(InvalidActionError):
        session.apply_action(questioner, "bogus_tool", {})
    with pytest.raises(InvalidActionError):
        session.apply_action(questioner, "spyfall_question", {"text": "hi"})
    with pytest.raises(InvalidActionError):
        session.apply_action(
            questioner, "spyfall_question", {"target_id": questioner, "text": "self?"}
        )
    with pytest.raises(InvalidActionError):
        session.apply_action(
            questioner, "spyfall_question", {"target_id": "ghost", "text": "hi?"}
        )
    with pytest.raises(InvalidActionError):
        session.apply_action(
            questioner, "spyfall_question", {"target_id": other, "text": "   "}
        )
    with pytest.raises(InvalidActionError):
        session.apply_action(
            questioner,
            "spyfall_question",
            {"target_id": other, "text": "x" * 501},
        )
    with pytest.raises(InvalidActionError):
        session.apply_action(
            other, "spyfall_question", {"target_id": questioner, "text": "not mine"}
        )
    with pytest.raises(InvalidActionError):
        session.apply_action(other, "spyfall_answer", {"text": "too early"})

    session.apply_action(
        questioner, "spyfall_question", {"target_id": other, "text": "real q?"}
    )
    with pytest.raises(InvalidActionError):
        session.apply_action(questioner, "spyfall_answer", {"text": "wrong actor"})
    with pytest.raises(InvalidActionError):
        session.apply_action(
            other, "spyfall_question", {"target_id": questioner, "text": "wrong phase"}
        )
    with pytest.raises(InvalidActionError):
        session.apply_action(other, "spyfall_answer", {"text": " " * 5})
    with pytest.raises(InvalidActionError):
        session.apply_action(other, "spyfall_answer", {"text": "y" * 501})


def test_match_over_rejects_actions() -> None:
    session = make_session()
    for _ in range(3):
        play_exchange(session)
    with pytest.raises(InvalidActionError):
        session.apply_action(
            PLAYERS[0],
            "spyfall_question",
            {"target_id": PLAYERS[1], "text": "after the end?"},
        )


def test_round_completing_transition_reports_original_round() -> None:
    session = make_session(rounds_per_match=2)
    for _ in range(2):
        play_exchange(session)
    questioner = session.current_actor_id
    target = next(p for p in PLAYERS if p != questioner)
    session.apply_action(
        questioner, "spyfall_question", {"target_id": target, "text": "last q?"}
    )
    transition = session.apply_action(target, "spyfall_answer", {"text": "last answer"})
    assert transition.metrics == {"round_index": 0, "rotation_step": 3}
    assert session.conversation_scope_id == "round-1"


def test_conversation_scope_and_turn_context() -> None:
    session = make_session(rounds_per_match=2)
    assert session.conversation_scope_id == "round-0"
    assert session.get_turn_context() == {"hand_index": 0, "in_hand_turn": 0}

    play_exchange(session)
    assert session.get_turn_context() == {"hand_index": 0, "in_hand_turn": 1}

    for _ in range(2):
        play_exchange(session)
    assert session.conversation_scope_id == "round-1"
    assert session.get_turn_context()["hand_index"] == 1


def test_tools_expose_all_five_actions() -> None:
    session = make_session()
    for player_id in PLAYERS:
        names = {tool.name for tool in session.get_tools(player_id)}
        assert names == TOOL_NAMES


def test_apply_judged_action_with_none_judgment_applies_normally() -> None:
    session = make_session()
    questioner = session.current_actor_id
    target = next(p for p in PLAYERS if p != questioner)
    arguments: dict[str, object] = {"target_id": target, "text": "judged?"}

    transition = session.apply_judged_action(
        questioner, "spyfall_question", arguments, None
    )
    assert "asks" in transition.summary
    assert session.current_actor_id == target


def test_plugin_events_are_ordered_and_public_safe() -> None:
    session = make_session()
    for _ in range(3):
        play_exchange(session)

    events = session.drain_hand_events()
    assert [event.event_type for event in events] == [
        "round_start",
        "public_question",
        "public_answer",
        "public_question",
        "public_answer",
        "public_question",
        "public_answer",
        "round_end",
        "score_update",
    ]
    start = events[0]
    assert start.payload == {"round_index": 0, "first_questioner": "p1"}
    question = events[1]
    assert question.payload["questioner"] == "p1"
    serialized = json.dumps([event.payload for event in events])
    assert "airport" not in serialized
    assert "library" not in serialized

    assert session.drain_hand_events() == []


def test_factory_surface() -> None:
    game = SpyfallGame()
    config = game_config()

    assert game.name == "spyfall"
    assert game.version == "0.1.0"
    assert game.requires_exact_agent_ids is True
    assert game.player_ids_from_config(config) == PLAYERS
    game.validate_config(config)
    with pytest.raises(Exception):
        game.validate_config({"players": PLAYERS})
    assert game.min_max_turns(config) == 135
    assert "untrusted" in game.system_prompt("p1")
    assert "500" in game.system_prompt("p1")

    session = game.create_session(seed=7, game_config=config)
    assert session.conversation_scope_id == "round-0"
    assert {tool.name for tool in session.get_tools("p1")} == TOOL_NAMES


# ---------------------------------------------------------------------------
# Leak judging policy (Task 9)
# ---------------------------------------------------------------------------


def judgment_outcome(probability: float | None, *, ok: bool = True) -> JudgmentOutcome:
    if not ok:
        return JudgmentOutcome(ok=False, failure_reason="provider_error")
    assert probability is not None
    return JudgmentOutcome(
        ok=True,
        decision={"answers": {"leak": {"probability": probability}}},
        model="typesafe/jev-1.13",
    )


def any_known_location(session: SpyfallSession) -> str:
    for player_id in PLAYERS:
        location = known_location(session, player_id)
        if location is not None:
            return location
    raise AssertionError("no non-Spy player found")


def test_leak_at_threshold_ends_round_for_spy_without_publishing_text() -> None:
    session = make_session(judge_leak_threshold=0.85)
    spy = spy_of(session)
    questioner = session.current_actor_id
    target = next(p for p in PLAYERS if p != questioner)
    text = "the venue is definitely the airport, right?"

    transition = session.apply_judged_action(
        questioner,
        "spyfall_question",
        {"target_id": target, "text": text},
        judgment_outcome(0.85),
    )

    assert transition.metrics["reason"] == "spy_leak_judged"
    assert session.is_terminal is True
    result = session.get_result()
    rounds = result.outcome["rounds"]
    assert isinstance(rounds, list)
    assert rounds[-1] == {
        "round_index": 0,
        "winner_side": "spy",
        "reason": "spy_leak_judged",
    }
    points = result.outcome["points"]
    assert isinstance(points, dict)
    assert points[spy] == 1
    assert text not in session.get_observation(spy).text
    assert result.metrics["question_count"] == 0
    assert result.metrics["judge_invocations"] == 1
    assert result.metrics["leak_confirmations"] == 1


def test_leak_below_threshold_applies_action_normally() -> None:
    session = make_session(judge_leak_threshold=0.85)
    questioner = session.current_actor_id
    target = next(p for p in PLAYERS if p != questioner)
    text = "how crowded does it get on weekends?"

    session.apply_judged_action(
        questioner,
        "spyfall_question",
        {"target_id": target, "text": text},
        judgment_outcome(0.84),
    )

    assert session.current_actor_id == target
    assert text in session.get_observation(target).text
    metrics = session.get_result().metrics
    assert metrics["judge_invocations"] == 1
    assert metrics["leak_confirmations"] == 0


def test_malformed_decision_applies_action_defensively() -> None:
    session = make_session()
    questioner = session.current_actor_id
    target = next(p for p in PLAYERS if p != questioner)

    session.apply_judged_action(
        questioner,
        "spyfall_question",
        {"target_id": target, "text": "any question"},
        JudgmentOutcome(
            ok=True, decision={"answers": {"leak": {"probability": "very high"}}}
        ),
    )

    assert session.current_actor_id == target
    metrics = session.get_result().metrics
    assert metrics["judge_malformed_decisions"] == 1
    assert metrics["judge_invocations"] == 1


def test_judge_failure_fails_open_and_records_event() -> None:
    session = make_session()
    questioner = session.current_actor_id
    target = next(p for p in PLAYERS if p != questioner)

    session.apply_judged_action(
        questioner,
        "spyfall_question",
        {"target_id": target, "text": "an innocent question?"},
        judgment_outcome(None, ok=False),
    )

    events = session.drain_hand_events()
    fail_open = [e for e in events if e.event_type == "judge_fail_open"]
    assert len(fail_open) == 1
    assert fail_open[0].payload["failure_reason"] == "provider_error"
    assert fail_open[0].payload["judgment_kind"] == "spyfall_leak"
    assert fail_open[0].payload["round_index"] == 0
    assert session.current_actor_id == target
    metrics = session.get_result().metrics
    assert metrics["judge_failures"] == 1
    assert metrics["judge_invocations"] == 0


def test_judgment_request_rejects_illegal_action_without_judgment() -> None:
    session = make_session()
    questioner = session.current_actor_id

    with pytest.raises(InvalidActionError):
        session.judgment_request(
            questioner,
            "spyfall_question",
            {"target_id": "p2", "text": "x" * 601},
        )

    with pytest.raises(InvalidActionError):
        session.judgment_request(
            questioner,
            "spyfall_question",
            {"target_id": questioner, "text": "self question?"},
        )


def test_judgment_request_payload_shape() -> None:
    session = make_session()
    questioner = session.current_actor_id
    location = any_known_location(session)

    request = session.judgment_request(
        questioner,
        "spyfall_question",
        {"target_id": "p2", "text": "is it noisy there?"},
    )
    assert request is not None
    assert request.judge_id == "leak-judge"
    assert request.judgment_kind == "spyfall_leak"
    assert request.payload["questions"] == {
        "leak": {"type": "noul", "instructions": LEAK_RUBRIC}
    }
    state = request.payload["state"]
    assert isinstance(state, dict)
    assert state["secret_location"] == location
    assert state["candidate_kind"] == "question"
    assert state["candidate_speaker"] == questioner
    assert state["candidate_text"] == "is it noisy there?"

    session.apply_action(
        questioner,
        "spyfall_question",
        {"target_id": "p2", "text": "is it noisy there?"},
    )
    answer_request = session.judgment_request(
        "p2", "spyfall_answer", {"text": "sometimes, in summer"}
    )
    assert answer_request is not None
    answer_state = answer_request.payload["state"]
    assert isinstance(answer_state, dict)
    assert answer_state["candidate_kind"] == "answer"
    assert answer_state["candidate_text"] == "sometimes, in summer"

    assert (
        session.judgment_request(questioner, "spyfall_accuse", {"target": "p2"}) is None
    )
    assert (
        session.judgment_request(
            questioner, "spyfall_guess_location", {"location": "airport"}
        )
        is None
    )
    assert session.judgment_request(questioner, "spyfall_vote", {"vote": "yes"}) is None


def test_required_judge_ids_declares_configured_judge() -> None:
    session = make_session(judge_id="my-judge")
    assert session.required_judge_ids() == ["my-judge"]


# ---------------------------------------------------------------------------
# Accusations, ballots, and the spy guess (Task 10)
# ---------------------------------------------------------------------------


def seed_where_spy(spy_id: str) -> int:
    """Return a deterministic seed whose round-0 Spy is ``spy_id``."""
    for seed in range(500):
        if spy_of(make_session(seed=seed)) == spy_id:
            return seed
    raise AssertionError(f"no seed found with spy {spy_id}")


def open_ballot(session: SpyfallSession, *, accuser: str, accused: str) -> None:
    session.apply_action(accuser, "spyfall_accuse", {"target": accused})


def cast_ballot(session: SpyfallSession, vote: str) -> None:
    session.apply_action(session.current_actor_id, "spyfall_vote", {"vote": vote})


def test_accusation_of_spy_convicts_non_spies() -> None:
    session = make_session(seed=seed_where_spy("p2"))
    assert session.current_actor_id == "p1"

    open_ballot(session, accuser="p1", accused="p2")
    assert session.current_actor_id == "p1"  # accuser is eligible and first

    cast_ballot(session, "yes")
    cast_ballot(session, "yes")

    assert session.is_terminal is True
    result = session.get_result()
    rounds = result.outcome["rounds"]
    assert isinstance(rounds, list)
    assert rounds[-1] == {
        "round_index": 0,
        "winner_side": "non_spies",
        "reason": "accusation_correct",
    }
    points = result.outcome["points"]
    assert isinstance(points, dict)
    assert points["p1"] == 1
    assert points["p2"] == 0
    assert points["p3"] == 1

    events = session.drain_hand_events()
    assert [event.event_type for event in events][-3:] == [
        "vote_resolution",
        "round_end",
        "score_update",
    ]
    opened = events[1]
    assert opened.event_type == "accusation_opened"
    assert opened.payload == {
        "round_index": 0,
        "accuser": "p1",
        "accused": "p2",
        "eligible_count": 2,
        "required_votes": 2,
    }
    resolution = events[-3]
    assert resolution.payload == {
        "round_index": 0,
        "yes": 2,
        "no": 0,
        "required": 2,
        "convicted": True,
    }


def test_wrong_conviction_awards_spy() -> None:
    session = make_session(seed=seed_where_spy("p1"))

    open_ballot(session, accuser="p1", accused="p2")
    cast_ballot(session, "yes")
    cast_ballot(session, "yes")

    result = session.get_result()
    rounds = result.outcome["rounds"]
    assert isinstance(rounds, list)
    assert rounds[-1] == {
        "round_index": 0,
        "winner_side": "spy",
        "reason": "accusation_incorrect",
    }
    points = result.outcome["points"]
    assert isinstance(points, dict)
    assert points["p1"] == 1  # the spy side is the Spy alone
    assert points["p2"] == 0
    assert points["p3"] == 0


def test_required_votes_use_ceil_arithmetic() -> None:
    session = make_session(
        players=["p1", "p2", "p3", "p4"],
        accusation_vote_threshold=0.34,
    )
    roster = ["p1", "p2", "p3", "p4"]
    spy = next(p for p in roster if known_location(session, p) is None)
    accused = "p2" if spy != "p2" else "p3"

    open_ballot(session, accuser="p1", accused=accused)
    events = session.drain_hand_events()
    opened = events[-1]
    assert opened.event_type == "accusation_opened"
    assert opened.payload["eligible_count"] == 3
    assert opened.payload["required_votes"] == 2  # ceil(0.34 * 3)

    cast_ballot(session, "yes")
    cast_ballot(session, "no")
    assert session.is_terminal is False  # 1 of 2 required votes
    cast_ballot(session, "yes")

    result = session.get_result()
    rounds = result.outcome["rounds"]
    assert isinstance(rounds, list)
    assert rounds[-1] == {
        "round_index": 0,
        "winner_side": "non_spies" if accused == spy else "spy",
        "reason": "accusation_correct" if accused == spy else "accusation_incorrect",
    }


def test_non_convicting_ballot_consumes_accuser_slot() -> None:
    session = make_session(rounds_per_match=2)

    open_ballot(session, accuser="p1", accused="p2")
    cast_ballot(session, "yes")
    cast_ballot(session, "no")

    assert session.is_terminal is False
    assert session.current_actor_id == "p2"  # the player after the accuser
    assert session.get_turn_context()["in_hand_turn"] == 1

    events = session.drain_hand_events()
    resolution = [e for e in events if e.event_type == "vote_resolution"]
    assert len(resolution) == 1
    assert resolution[0].payload == {
        "round_index": 0,
        "yes": 1,
        "no": 1,
        "required": 2,
        "convicted": False,
    }

    play_exchange(session)
    assert session.get_turn_context()["in_hand_turn"] == 2
    play_exchange(session)

    result = session.get_result()
    rounds = result.outcome["rounds"]
    assert isinstance(rounds, list)
    assert rounds[0] == {
        "round_index": 0,
        "winner_side": "spy",
        "reason": "question_rotation_complete",
    }


def test_vote_eligibility_and_validation() -> None:
    session = make_session(seed=seed_where_spy("p2"))

    open_ballot(session, accuser="p1", accused="p2")

    with pytest.raises(InvalidActionError):  # the accused cannot vote
        session.apply_action("p2", "spyfall_vote", {"vote": "yes"})
    with pytest.raises(InvalidActionError):  # unknown vote value
        session.apply_action("p1", "spyfall_vote", {"vote": "maybe"})
    with pytest.raises(InvalidActionError):  # out of turn
        session.apply_action("p3", "spyfall_vote", {"vote": "yes"})

    session.apply_action("p1", "spyfall_vote", {"vote": "yes"})
    with pytest.raises(InvalidActionError):  # double voting
        session.apply_action("p1", "spyfall_vote", {"vote": "yes"})

    assert session.current_actor_id == "p3"
    session.apply_action("p3", "spyfall_vote", {"vote": "no"})
    assert session.current_actor_id == "p2"


def test_ballot_secrecy_and_aggregate_only_publication() -> None:
    session = make_session(seed=seed_where_spy("p2"))

    open_ballot(session, accuser="p1", accused="p2")
    voter = session.current_actor_id
    assert voter == "p1"
    observation = session.get_observation(voter)
    assert "p1" in observation.text
    assert "p2" in observation.text
    assert "0 of 2 ballots cast" in observation.text
    assert "yes votes are required" in observation.text

    cast_ballot(session, "yes")
    assert "1 of 2 ballots cast" in session.get_observation("p3").text
    for player_id in PLAYERS:
        text = session.get_observation(player_id).text
        assert "voted yes" not in text.lower()
        assert "voted no" not in text.lower()
        assert "your vote" not in text.lower()

    cast_ballot(session, "no")
    events = session.drain_hand_events()
    serialized = json.dumps([event.payload for event in events])
    assert '"vote"' not in serialized
    resolution = [e for e in events if e.event_type == "vote_resolution"]
    assert len(resolution) == 1
    assert resolution[0].payload["yes"] == 1
    assert resolution[0].payload["no"] == 1


def test_guess_is_spy_only_and_phase_gated() -> None:
    session = make_session(seed=seed_where_spy("p1"))

    with pytest.raises(InvalidActionError):  # non-Spy callers are rejected
        session.apply_action("p2", "spyfall_guess_location", {"location": "airport"})

    session.apply_action("p1", "spyfall_question", {"target_id": "p2", "text": "q?"})
    with pytest.raises(InvalidActionError):  # not a question turn
        session.apply_action("p1", "spyfall_guess_location", {"location": "airport"})


def test_correct_guess_with_case_and_whitespace_awards_spy() -> None:
    session = make_session(seed=seed_where_spy("p1"))
    location = known_location(session, "p2")
    assert location is not None

    transition = session.apply_action(
        "p1", "spyfall_guess_location", {"location": f"  {location.upper()} "}
    )

    assert session.is_terminal is True
    result = session.get_result()
    rounds = result.outcome["rounds"]
    assert isinstance(rounds, list)
    assert rounds[-1] == {
        "round_index": 0,
        "winner_side": "spy",
        "reason": "guess_correct",
    }
    points = result.outcome["points"]
    assert isinstance(points, dict)
    assert points["p1"] == 1
    assert points["p2"] == 0
    assert points["p3"] == 0
    assert "guess" in transition.summary.lower()


def test_wrong_guess_awards_non_spies() -> None:
    session = make_session(seed=seed_where_spy("p1"))

    session.apply_action("p1", "spyfall_guess_location", {"location": "submarine"})

    assert session.is_terminal is True
    result = session.get_result()
    rounds = result.outcome["rounds"]
    assert isinstance(rounds, list)
    assert rounds[-1] == {
        "round_index": 0,
        "winner_side": "non_spies",
        "reason": "guess_incorrect",
    }
    points = result.outcome["points"]
    assert isinstance(points, dict)
    assert points["p2"] == 1
    assert points["p3"] == 1
    assert points["p1"] == 0


def test_accuse_validation() -> None:
    session = make_session()
    questioner = session.current_actor_id
    other = next(p for p in PLAYERS if p != questioner)

    session.apply_action(
        questioner, "spyfall_question", {"target_id": other, "text": "q?"}
    )
    with pytest.raises(InvalidActionError):  # not a question turn
        session.apply_action(questioner, "spyfall_accuse", {"target": other})

    fresh = make_session()
    with pytest.raises(InvalidActionError):  # not the scheduled questioner
        fresh.apply_action("p2", "spyfall_accuse", {"target": "p3"})
    with pytest.raises(InvalidActionError):  # self-accusation
        fresh.apply_action("p1", "spyfall_accuse", {"target": "p1"})
    with pytest.raises(InvalidActionError):  # unknown target
        fresh.apply_action("p1", "spyfall_accuse", {"target": "ghost"})
    with pytest.raises(InvalidActionError):  # missing target
        fresh.apply_action("p1", "spyfall_accuse", {})
