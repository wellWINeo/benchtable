"""Integration tests for offline Spyfall match traces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from tests.fixtures.fake_agent import FakeAgent
from tests.fixtures.fake_judge import FakeJudge
from typer.testing import CliRunner

from benchtable import cli
from benchtable.cli import app
from benchtable.contracts import JsonObject, ModelResponse, ToolCall, Usage
from benchtable.engine import RunEngine
from benchtable.errors import JudgeProviderError, ProviderError
from benchtable.games.registry import GameRegistry
from benchtable.games.spyfall.session import SpyfallGame

JUDGE_ID = "spyfall-leak-judge"
CONFIG: JsonObject = {
    "players": ["p1", "p2", "p3"],
    "rounds_per_match": 2,
    "question_rounds": 1,
    "accusation_vote_threshold": 0.75,
    "judge_id": JUDGE_ID,
    "judge_leak_threshold": 0.85,
    "locations": [{"name": "airport"}],
}


def _read_events(path: Path) -> list[dict[str, Any]]:
    raw = (path / "events.jsonl").read_text()
    return [json.loads(line) for line in raw.splitlines() if line]


def _action(actor: str, index: int, name: str, arguments: JsonObject) -> ModelResponse:
    return ModelResponse(
        assistant_text="Playing my part.",
        tool_calls=[
            ToolCall(
                call_id=f"{actor}-{index}",
                name=name,
                arguments=arguments,
            )
        ],
        finish_reason="tool_calls",
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _question(actor: str, index: int, target: str) -> ModelResponse:
    return _action(
        actor,
        index,
        "spyfall_question",
        {"target_id": target, "text": "Is it busy where you are?"},
    )


def _answer(actor: str, index: int) -> ModelResponse:
    return _action(actor, index, "spyfall_answer", {"text": "Sometimes, yes."})


def _rotation_queues() -> dict[str, list[ModelResponse]]:
    """Two rounds, one questioner rotation each, first questioner p1 then p2."""
    return {
        "p1": [
            _question("p1", 1, "p2"),
            _answer("p1", 2),
            _answer("p1", 3),
            _question("p1", 4, "p2"),
        ],
        "p2": [
            _answer("p2", 1),
            _question("p2", 2, "p3"),
            _question("p2", 3, "p3"),
            _answer("p2", 4),
        ],
        "p3": [
            _answer("p3", 1),
            _question("p3", 2, "p1"),
            _answer("p3", 3),
            _question("p3", 4, "p1"),
        ],
    }


def _agents(
    queues: dict[str, list[ModelResponse]],
) -> dict[str, FakeAgent]:
    return {pid: FakeAgent(responses=list(queue)) for pid, queue in queues.items()}


def _engine(
    tmp_path: Path,
    agents: dict[str, FakeAgent],
    *,
    judge: FakeJudge | None = None,
    judge_max_retries: dict[str, int] | None = None,
    max_provider_retries: int = 2,
    config: JsonObject = CONFIG,
    matches: int = 1,
) -> RunEngine:
    kwargs: dict[str, Any] = {}
    if judge is not None:
        kwargs["judges"] = {JUDGE_ID: judge}
        kwargs["judge_max_retries"] = judge_max_retries or {JUDGE_ID: 1}
    player_ids = [str(pid) for pid in config["players"]]
    return RunEngine(
        game=SpyfallGame(),
        agents=dict(agents),
        agent_metadata=[{"id": pid} for pid in player_ids],
        game_config=config,
        run_dir=tmp_path,
        run_id=f"run-{tmp_path.name}",
        seed=7,
        matches=matches,
        max_turns=400,
        max_provider_retries=max_provider_retries,
        **kwargs,
    )


def _match0(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [event for event in events if event.get("match_id") == "match-0"]


def _window(events: list[dict[str, Any]], turn_index: int) -> list[dict[str, Any]]:
    return [event for event in events if event.get("turn_index") == turn_index]


def _types(window: list[dict[str, Any]]) -> list[str]:
    return [str(event.get("event_type")) for event in window]


async def test_full_rotation_match_trace(tmp_path: Path) -> None:
    engine = _engine(
        tmp_path,
        _agents(_rotation_queues()),
        judge=FakeJudge(judge_id=JUDGE_ID),
    )
    result = await engine.run()
    assert result.success

    events = _read_events(tmp_path)
    match = _match0(events)
    match_end = next(e for e in match if e["event_type"] == "match_end")
    assert match_end["payload"]["result"]["completed"] is True
    rounds = match_end["payload"]["result"]["outcome"]["rounds"]
    assert rounds == [
        {
            "round_index": 0,
            "winner_side": "spy",
            "reason": "question_rotation_complete",
        },
        {
            "round_index": 1,
            "winner_side": "spy",
            "reason": "question_rotation_complete",
        },
    ]
    # One point per round, awarded to the round-winning spy (a fresh spy is
    # seeded per round, so the totals need not concentrate on one player).
    assert sum(match_end["payload"]["result"]["outcome"]["points"].values()) == 2

    assert _types(_window(match, 0)) == [
        "turn_start",
        "observation",
        "tool_schemas",
        "model_request",
        "model_response",
        "judge_request",
        "judge_response",
        "validation",
        "transition",
        "public_question",
        "model_request",
        "model_response",
        "turn_end",
    ]
    assert _types(_window(match, 5)) == [
        "turn_start",
        "observation",
        "tool_schemas",
        "model_request",
        "model_response",
        "judge_request",
        "judge_response",
        "validation",
        "transition",
        "public_answer",
        "round_end",
        "score_update",
        "round_start",
        "model_request",
        "model_response",
        "turn_end",
    ]
    by_type: dict[str, int] = {}
    for event in match:
        by_type[event["event_type"]] = by_type.get(event["event_type"], 0) + 1
    assert by_type["round_start"] == 2
    assert by_type["public_question"] == 6
    assert by_type["public_answer"] == 6
    assert by_type["round_end"] == 2
    assert by_type["score_update"] == 2
    assert by_type["judge_request"] == 12
    assert by_type["judge_response"] == 12


async def test_leak_confirmed_ends_round_immediately(tmp_path: Path) -> None:
    leaking = dict(CONFIG)
    leaking["rounds_per_match"] = 1
    agents = {
        "p1": FakeAgent(responses=[_question("p1", 1, "p2")]),
        "p2": FakeAgent(responses=[]),
        "p3": FakeAgent(responses=[]),
    }
    engine = _engine(
        tmp_path,
        agents,
        config=leaking,
        judge=FakeJudge(
            judge_id=JUDGE_ID,
            decision={"answers": {"leak": {"probability": 0.99}}},
        ),
    )
    result = await engine.run()
    assert result.success

    match = _match0(_read_events(tmp_path))
    round_end = next(e for e in match if e["event_type"] == "round_end")
    assert round_end["payload"]["reason"] == "spy_leak_judged"
    assert round_end["payload"]["winner_side"] == "spy"
    assert not [e for e in match if e["event_type"] == "public_question"]
    assert not [e for e in match if e["event_type"] == "public_answer"]
    assert {e.get("actor_id") for e in match if e.get("actor_id")} == {"p1"}
    assert len([e for e in match if e["event_type"] == "judge_request"]) == 1


async def test_judge_failure_fails_open(tmp_path: Path) -> None:
    engine = _engine(
        tmp_path,
        _agents(_rotation_queues()),
        judge=FakeJudge(
            judge_id=JUDGE_ID,
            raise_error=JudgeProviderError(
                "boom", provider="openrouter", model="typesafe/jev-1.13"
            ),
        ),
        judge_max_retries={JUDGE_ID: 1},
    )
    result = await engine.run()
    assert result.success

    match = _match0(_read_events(tmp_path))
    assert len([e for e in match if e["event_type"] == "judge_error"]) == 24
    assert not [e for e in match if e["event_type"] == "judge_response"]
    fail_open = [e for e in match if e["event_type"] == "judge_fail_open"]
    assert len(fail_open) == 12
    window = _window(match, 0)
    types = _types(window)
    assert types.index("model_response") < types.index("judge_error")
    assert types.index("judge_error") < types.index("validation")
    assert "transition" in types
    match_end = next(e for e in match if e["event_type"] == "match_end")
    rounds = match_end["payload"]["result"]["outcome"]["rounds"]
    assert [r["reason"] for r in rounds] == [
        "question_rotation_complete",
        "question_rotation_complete",
    ]


async def test_missing_judge_dependency_fails_before_requests(
    tmp_path: Path,
) -> None:
    engine = _engine(tmp_path, _agents(_rotation_queues()), judge=None)
    result = await engine.run()
    assert not result.success

    match = _match0(_read_events(tmp_path))
    match_end = next(e for e in match if e["event_type"] == "match_end")
    assert match_end["payload"]["failure_reason"] == "judge_dependency_missing"
    assert not [e for e in match if e["event_type"] == "model_request"]
    assert not [e for e in match if e["event_type"] == "judge_request"]


async def test_private_information_isolated_across_trace(tmp_path: Path) -> None:
    engine = _engine(
        tmp_path,
        _agents(_rotation_queues()),
        judge=FakeJudge(judge_id=JUDGE_ID),
    )
    await engine.run()

    # The spy and location are re-seeded each round; walk a probe session
    # through the same seed to learn each round's private values.
    players = [str(p) for p in CONFIG["players"]]
    probe = SpyfallGame().create_session(
        seed=engine._derive_match_seed(0), game_config=CONFIG
    )
    spies: dict[int, str] = {}
    locations: dict[int, str] = {}
    for round_index in range(2):
        spies[round_index] = probe._spy_id
        locations[round_index] = probe._location
        if round_index == 1:
            break
        for _ in range(3):
            questioner = probe.current_actor_id
            target = next(p for p in players if p != questioner)
            probe.apply_judged_action(
                questioner,
                "spyfall_question",
                {"target_id": target, "text": "Is it crowded?"},
                None,
            )
            answerer = probe.current_actor_id
            probe.apply_judged_action(
                answerer, "spyfall_answer", {"text": "Sometimes."}, None
            )

    match = _match0(_read_events(tmp_path))
    for event in match:
        if event["event_type"] == "observation":
            round_index = int(event["payload"]["metadata"]["round_index"])
            spy = spies[round_index]
            location = locations[round_index]
            text = event["payload"]["text"]
            if event.get("actor_id") == spy:
                assert location not in text
                assert "You know the secret location" not in text
            else:
                assert "You are the Spy" not in text
        if event["event_type"] == "model_request":
            first_user = next(
                message
                for message in event["payload"]["messages"]
                if message["role"] == "user"
            )
            round_index = (
                int(str(first_user["content"]).split("Round ")[1].split(" of")[0]) - 1
            )
            spy = spies[round_index]
            location = locations[round_index]
            dump = json.dumps(event["payload"]["messages"])
            if event.get("actor_id") == spy:
                assert location not in dump
            else:
                assert "You are the Spy" not in dump


async def test_provider_failures_lose_round_not_match(tmp_path: Path) -> None:
    queues = _rotation_queues()
    agents = {
        "p1": FakeAgent(responses=[_question("p1", 1, "p2")]),
        "p2": FakeAgent(
            responses=[],
            raise_on_respond=ProviderError("down", provider="openai", model="m"),
        ),
        "p3": FakeAgent(responses=list(queues["p3"])),
    }
    engine = _engine(
        tmp_path,
        agents,
        judge=FakeJudge(judge_id=JUDGE_ID),
        max_provider_retries=1,
    )
    result = await engine.run()
    assert result.success

    match = _match0(_read_events(tmp_path))
    match_end = next(e for e in match if e["event_type"] == "match_end")
    assert match_end["payload"]["result"]["completed"] is True
    rounds = match_end["payload"]["result"]["outcome"]["rounds"]
    assert [r["reason"] for r in rounds] == [
        "failed_turn_provider_retries_exhausted",
        "failed_turn_provider_retries_exhausted",
    ]
    assert len([e for e in match if e["event_type"] == "public_question"]) == 1
    responses = [event for event in match if event["event_type"] == "model_response"]
    for event in responses:
        for call in event["payload"]["tool_calls"] or []:
            assert call["name"] != "spyfall_vote"
    assert [
        e
        for e in match
        if e["event_type"] == "transition" and e["payload"].get("recovery")
    ]
    failed_ends = [
        e
        for e in match
        if e["event_type"] == "turn_end" and e["payload"].get("status") == "failed"
    ]
    assert len(failed_ends) == 2


def test_cli_run_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry = GameRegistry()
    registry.register(SpyfallGame())
    cli.set_registry(registry)

    def agent_factory(agent_config: Any) -> FakeAgent:
        queues = {
            "p1": [_question("p1", 1, "p2"), _answer("p1", 2)],
            "p2": [_answer("p2", 1), _question("p2", 2, "p3")],
            "p3": [_answer("p3", 1), _question("p3", 2, "p1")],
        }
        return FakeAgent(responses=list(queues[str(agent_config.id)]))

    cli.set_agent_factory(agent_factory)
    judge = FakeJudge(judge_id=JUDGE_ID)
    cli.set_judge_factory(lambda judge_config: judge)

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                "[run]",
                'game = "spyfall"',
                "matches = 1",
                "seed = 7",
                "max_turns = 135",
                "",
                "[[agents]]",
                'id = "p1"',
                'model = "fake-model"',
                "",
                "[[agents]]",
                'id = "p2"',
                'model = "fake-model"',
                "",
                "[[agents]]",
                'id = "p3"',
                'model = "fake-model"',
                "",
                "[run.game_config]",
                'players = ["p1", "p2", "p3"]',
                "rounds_per_match = 1",
                "question_rounds = 1",
                "accusation_vote_threshold = 0.75",
                'judge_id = "spyfall-leak-judge"',
                "judge_leak_threshold = 0.85",
                "",
                "[[run.game_config.locations]]",
                'name = "airport"',
                "",
                "[[judges]]",
                'id = "spyfall-leak-judge"',
                'model = "typesafe/jev-1.13"',
                'api_key_env = "OPENROUTER_API_KEY"',
                "",
            ]
        )
    )
    output = tmp_path / "out"
    runner = CliRunner()
    try:
        result = runner.invoke(
            app,
            ["run", "--config", str(config_path), "--output", str(output)],
        )
        assert result.exit_code == 0
        assert (output / "events.jsonl").exists()
        events = _read_events(output)
        match_end = next(e for e in events if e["event_type"] == "match_end")
        assert match_end["payload"]["result"]["completed"] is True
        judges_meta = next(e for e in events if e["event_type"] == "run_config")[
            "payload"
        ]["judges"]
        assert judges_meta[0]["api_key_env"] == "OPENROUTER_API_KEY"

        low_budget = tmp_path / "low-budget.toml"
        low_budget.write_text(
            config_path.read_text().replace("max_turns = 135", "max_turns = 8")
        )
        rejected = runner.invoke(
            app,
            ["run", "--config", str(low_budget), "--output", str(tmp_path / "o2")],
        )
        assert rejected.exit_code == 1
        assert "below the minimum" in rejected.output
        assert "9" in rejected.output
    finally:
        cli.set_registry(None)
        cli.set_agent_factory(None)
        cli.set_judge_factory(None)
