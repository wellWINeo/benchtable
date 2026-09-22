"""Engine judge integration tests: judged actions, events, preflight."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from tests.fixtures.fake_agent import FakeAgent
from tests.fixtures.fake_judge import FakeJudge
from tests.fixtures.judged_tiny_game import JudgedTinyGame
from tests.fixtures.tiny_game import TinyGame

from benchtable.contracts import JsonObject
from benchtable.engine import RunEngine
from benchtable.errors import (
    JudgeMalformedResponseError,
    JudgeProviderError,
    JudgeTimeoutError,
)
from benchtable.games.protocol import JudgedActionSession


def _read_events(path: Path) -> list[dict[str, Any]]:
    raw = (path / "events.jsonl").read_text()
    return [json.loads(line) for line in raw.splitlines() if line]


def _events_of_type(
    events: list[dict[str, Any]], event_type: str
) -> list[dict[str, Any]]:
    return [event for event in events if event["event_type"] == event_type]


def _event_types(events: list[dict[str, Any]]) -> list[str]:
    return [event["event_type"] for event in events]


def _make_engine(
    game: Any,
    *,
    run_dir: Path,
    judges: dict[str, FakeJudge] | None = None,
    judge_max_retries: dict[str, int] | None = None,
    judge_metadata: list[JsonObject] | None = None,
    max_turns: int = 12,
) -> RunEngine:
    return RunEngine(
        game=game,
        agents={"a": FakeAgent(tool_name="act"), "b": FakeAgent(tool_name="act")},
        game_config={},
        run_dir=run_dir,
        seed=7,
        matches=1,
        max_turns=max_turns,
        judges=judges,
        judge_metadata=judge_metadata,
        judge_max_retries=judge_max_retries,
    )


@pytest.mark.asyncio
async def test_happy_path_judged_actions(tmp_path: Path) -> None:
    game = JudgedTinyGame(max_actions=2)
    judge = FakeJudge(
        judge_id="tiny-judge",
        decision={"answers": {"leak": {"probability": 0.1}}},
        raw_provider_request={"model": "typesafe/jev-1.13", "state": {"v": 1}},
        raw_provider_response={"answers": {"leak": {"probability": 0.1}}},
    )
    engine = _make_engine(
        game,
        run_dir=tmp_path,
        judges={"tiny-judge": judge},
        judge_max_retries={"tiny-judge": 1},
    )
    result = await engine.run()
    assert result.success

    session = game.sessions[0]
    assert isinstance(session, JudgedActionSession)
    assert len(session.received_judgments) == 2
    for judgment in session.received_judgments:
        assert judgment is not None and judgment.ok
        assert judgment.decision == {"answers": {"leak": {"probability": 0.1}}}
        assert judgment.model == "typesafe/jev-1.13"

    events = _read_events(tmp_path)
    judge_requests = _events_of_type(events, "judge_request")
    judge_responses = _events_of_type(events, "judge_response")
    assert len(judge_requests) == 2 and len(judge_responses) == 2
    for request_event, response_event in zip(judge_requests, judge_responses):
        assert request_event["payload"]["judgment_kind"] == "tiny_check"
        assert request_event["payload"]["payload"] == {"value": 1}
        assert response_event["payload"]["decision"] == {
            "answers": {"leak": {"probability": 0.1}}
        }
        assert response_event["payload"]["raw_provider_request"] == {
            "model": "typesafe/jev-1.13",
            "state": {"v": 1},
        }

    # Per-turn ordering: every judge_request is followed by judge_response
    # and both precede the next turn's transition event.
    ordered: list[str] = []
    for event in events:
        if event["event_type"] in {"judge_request", "judge_response", "transition"}:
            ordered.append(event["event_type"])
    assert ordered == ["judge_request", "judge_response", "transition"] * 2

    # Judging does not add provider requests: two turns, each with one action
    # request and one finalization request.
    assert len(_events_of_type(events, "model_request")) == 4

    match_end = _events_of_type(events, "match_end")[0]
    assert match_end["payload"]["metrics"]["judge_calls"] == 2
    assert match_end["payload"]["metrics"]["judge_failures"] == 0

    run_config = _events_of_type(events, "run_config")[0]
    assert run_config["payload"]["judges"] == []

    # Credential-shaped values planted in raw payloads are redacted.
    judge = FakeJudge(
        judge_id="tiny-judge",
        raw_provider_response={"api_key": "sk-super-secret-value"},
    )
    engine = _make_engine(
        JudgedTinyGame(max_actions=1),
        run_dir=tmp_path / "redaction",
        judges={"tiny-judge": judge},
    )
    assert (await engine.run()).success
    raw_text = (tmp_path / "redaction" / "events.jsonl").read_text()
    assert "sk-super-secret-value" not in raw_text


@pytest.mark.asyncio
async def test_judge_provider_failure_fails_open(tmp_path: Path) -> None:
    game = JudgedTinyGame(max_actions=1)
    judge = FakeJudge(
        judge_id="tiny-judge",
        raise_error=JudgeProviderError(
            "boom",
            provider="openrouter",
            model="typesafe/jev-1.13",
            raw_provider_request={"state": {"v": 1}},
            raw_provider_response={"status": 500},
        ),
    )
    engine = _make_engine(
        game,
        run_dir=tmp_path,
        judges={"tiny-judge": judge},
        judge_max_retries={"tiny-judge": 1},
    )
    result = await engine.run()
    assert result.success

    judgment = game.sessions[0].received_judgments[0]
    assert judgment is not None and not judgment.ok
    assert judgment.failure_reason == "provider_error"
    assert len(game.sessions[0].received_judgments) == 1

    events = _read_events(tmp_path)
    judge_errors = _events_of_type(events, "judge_error")
    assert [event["payload"]["attempt"] for event in judge_errors] == [1, 2]
    assert all(
        event["payload"]["failure_reason"] == "provider_error" for event in judge_errors
    )
    assert judge_errors[0]["payload"]["provider"] == "openrouter"
    assert judge_errors[0]["payload"]["raw_provider_response"] == {"status": 500}

    match_end = _events_of_type(events, "match_end")[0]
    assert match_end["payload"]["metrics"]["judge_failures"] == 2
    assert match_end["payload"]["result"]["completed"] is True


@pytest.mark.asyncio
async def test_malformed_response_raw_body_survives(tmp_path: Path) -> None:
    game = JudgedTinyGame(max_actions=1)
    judge = FakeJudge(
        judge_id="tiny-judge",
        raise_error=JudgeMalformedResponseError(
            "not json",
            raw_provider_response={"status": 200, "raw_body": "not-json-body"},
        ),
    )
    engine = _make_engine(
        game,
        run_dir=tmp_path,
        judges={"tiny-judge": judge},
        judge_max_retries={"tiny-judge": 0},
    )
    assert (await engine.run()).success

    events = _read_events(tmp_path)
    judge_errors = _events_of_type(events, "judge_error")
    assert len(judge_errors) == 1
    assert judge_errors[0]["payload"]["failure_reason"] == "malformed_response"
    assert judge_errors[0]["payload"]["raw_provider_response"] == {
        "status": 200,
        "raw_body": "not-json-body",
    }
    judgment = game.sessions[0].received_judgments[0]
    assert judgment is not None and not judgment.ok


@pytest.mark.asyncio
async def test_retry_then_success(tmp_path: Path) -> None:
    game = JudgedTinyGame(max_actions=1)
    judge = FakeJudge(
        judge_id="tiny-judge",
        outcomes=[
            JudgeTimeoutError("slow"),
            {"answers": {"leak": {"probability": 0.2}}},
        ],
    )
    engine = _make_engine(
        game,
        run_dir=tmp_path,
        judges={"tiny-judge": judge},
        judge_max_retries={"tiny-judge": 1},
    )
    assert (await engine.run()).success

    events = _read_events(tmp_path)
    assert len(_events_of_type(events, "judge_error")) == 1
    judge_error = _events_of_type(events, "judge_error")[0]
    assert judge_error["payload"]["failure_reason"] == "timeout"
    assert judge_error["payload"]["attempt"] == 1
    assert len(_events_of_type(events, "judge_response")) == 1
    assert judge.call_count == 2
    judgment = game.sessions[0].received_judgments[0]
    assert judgment is not None and judgment.ok
    assert judgment.decision == {"answers": {"leak": {"probability": 0.2}}}


@pytest.mark.asyncio
async def test_missing_judge_dependency_fails_before_requests(
    tmp_path: Path,
) -> None:
    game = JudgedTinyGame(max_actions=2, required_judges=["missing-judge"])
    engine = _make_engine(
        game,
        run_dir=tmp_path,
        judges={"tiny-judge": FakeJudge(judge_id="tiny-judge")},
    )
    result = await engine.run()
    assert not result.success

    events = _read_events(tmp_path)
    assert _events_of_type(events, "model_request") == []
    assert _events_of_type(events, "judge_request") == []
    engine_errors = _events_of_type(events, "engine_error")
    assert any(
        event["payload"].get("phase") == "judge_dependency"
        and event["payload"].get("missing") == ["missing-judge"]
        for event in engine_errors
    )
    match_end = _events_of_type(events, "match_end")[0]
    assert match_end["payload"]["failure_reason"] == "judge_dependency_missing"


@pytest.mark.asyncio
async def test_plain_game_backwards_compatible(tmp_path: Path) -> None:
    engine = RunEngine(
        game=TinyGame(max_actions=2),
        agents={"a": FakeAgent(tool_name="act"), "b": FakeAgent(tool_name="act")},
        game_config={},
        run_dir=tmp_path,
        seed=3,
        matches=1,
        max_turns=12,
    )
    result = await engine.run()
    assert result.success

    events = _read_events(tmp_path)
    assert _events_of_type(events, "judge_request") == []
    run_config = _events_of_type(events, "run_config")[0]
    assert run_config["payload"]["judges"] == []


@pytest.mark.asyncio
async def test_no_judgment_request_passes_none(tmp_path: Path) -> None:
    game = JudgedTinyGame(max_actions=2, request_judgment=False)
    engine = _make_engine(
        game,
        run_dir=tmp_path,
        judges={"tiny-judge": FakeJudge(judge_id="tiny-judge")},
    )
    assert (await engine.run()).success

    assert game.sessions[0].received_judgments == [None, None]
    events = _read_events(tmp_path)
    assert _events_of_type(events, "judge_request") == []
    assert _events_of_type(events, "judge_response") == []


@pytest.mark.asyncio
async def test_unresolved_judge_id_fails_open(tmp_path: Path) -> None:
    # Defensive branch: preflight validates the declared dependency, but the
    # session requests a judgment from a different, unconfigured judge id.
    game = JudgedTinyGame(
        max_actions=1,
        judge_id="ghost-judge",
        required_judges=["tiny-judge"],
    )
    engine = _make_engine(
        game,
        run_dir=tmp_path,
        judges={"tiny-judge": FakeJudge(judge_id="tiny-judge")},
    )
    assert (await engine.run()).success

    events = _read_events(tmp_path)
    judge_errors = _events_of_type(events, "judge_error")
    assert len(judge_errors) == 1
    assert judge_errors[0]["payload"]["failure_reason"] == "judge_unresolved"
    judgment = game.sessions[0].received_judgments[0]
    assert judgment is not None and not judgment.ok
    assert judgment.failure_reason == "judge_unresolved"


@pytest.mark.asyncio
async def test_judge_metadata_recorded_in_run_config(tmp_path: Path) -> None:
    game = JudgedTinyGame(max_actions=1)
    engine = _make_engine(
        game,
        run_dir=tmp_path,
        judges={"tiny-judge": FakeJudge(judge_id="tiny-judge")},
        judge_metadata=[
            {"id": "tiny-judge", "model": "typesafe/jev-1.13", "api_key_env": "X"}
        ],
    )
    assert (await engine.run()).success
    events = _read_events(tmp_path)
    run_config = _events_of_type(events, "run_config")[0]
    assert run_config["payload"]["judges"] == [
        {"id": "tiny-judge", "model": "typesafe/jev-1.13", "api_key_env": "X"}
    ]
