"""Unit tests for the Spyfall session state machine."""

from __future__ import annotations

import json

import pytest

from benchtable.errors import InvalidActionError
from benchtable.games.spyfall.session import SpyfallGame, SpyfallSession

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


def test_judged_capability_exists_and_defers_to_task_nine() -> None:
    session = make_session()
    questioner = session.current_actor_id
    target = next(p for p in PLAYERS if p != questioner)
    arguments: dict[str, object] = {"target_id": target, "text": "judged?"}

    assert session.judgment_request(questioner, "spyfall_question", arguments) is None

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
