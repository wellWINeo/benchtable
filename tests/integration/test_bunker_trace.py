"""Integration tests for offline bunker match traces."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from tests.fixtures.fake_agent import FakeAgent

from benchtable.contracts import ModelResponse, ToolCall, Usage
from benchtable.engine import RunEngine
from benchtable.errors import ProviderError
from benchtable.games.bunker.session import BunkerGame

_CONFIG = {
    "players": ["player-1", "player-2", "player-3", "player-4"],
    "scenario": "sealed shelter after a solar storm",
    "shelter_capacity": 2,
}

_P1_R1 = "player-1: I secured the water supply"
_P1_R2 = "player-1: I can repair the air filters"
_P2_R1 = "player-2: I patched the eastern wall"
_P3_R1 = "player-3: I stocked the medical cabinet"
_P3_R2 = "player-3: I mapped the tunnel exits"
_P4_R1 = "player-4: I rationed the canned food"
_P4_R2 = "player-4: I wired the backup generator"

_SEQUENCE = {
    "player-1": [
        ("bunker_speak", {"text": _P1_R1}),
        ("bunker_vote_eliminate", {"target_id": "player-2"}),
        ("bunker_speak", {"text": _P1_R2}),
        ("bunker_vote_eliminate", {"target_id": "player-3"}),
    ],
    "player-2": [
        ("bunker_speak", {"text": _P2_R1}),
        ("bunker_vote_eliminate", {"target_id": "player-3"}),
    ],
    "player-3": [
        ("bunker_speak", {"text": _P3_R1}),
        ("bunker_vote_eliminate", {"target_id": "player-2"}),
        ("bunker_speak", {"text": _P3_R2}),
        ("bunker_vote_eliminate", {"target_id": "player-4"}),
    ],
    "player-4": [
        ("bunker_speak", {"text": _P4_R1}),
        ("bunker_vote_eliminate", {"target_id": "player-2"}),
        ("bunker_speak", {"text": _P4_R2}),
        ("bunker_vote_eliminate", {"target_id": "player-3"}),
    ],
}


_FAILURE_SEQUENCE = {
    "player-1": [
        ("bunker_speak", {"text": _P1_R1}),
        ("bunker_speak", {"text": _P1_R2}),
        ("bunker_vote_eliminate", {"target_id": "player-2"}),
    ],
    "player-2": [
        ("bunker_speak", {"text": _P2_R1}),
        ("bunker_speak", {"text": _P2_R1}),
        ("bunker_vote_eliminate", {"target_id": "player-4"}),
    ],
    "player-4": [
        ("bunker_speak", {"text": _P4_R1}),
        ("bunker_speak", {"text": _P4_R2}),
        ("bunker_vote_eliminate", {"target_id": "player-2"}),
    ],
}


def _read_events(path: Path) -> list[dict[str, Any]]:
    raw = (path / "events.jsonl").read_text()
    return [json.loads(line) for line in raw.splitlines() if line]


def _normalize(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop volatile fields so two identical runs compare equal."""

    def strip(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: strip(item)
                for key, item in value.items()
                if key
                not in ("occurred_at", "sequence", "run_id", "latency_seconds", "usage")
            }
        if isinstance(value, list):
            return [strip(item) for item in value]
        return value

    return [strip(copy.deepcopy(event)) for event in events]


def _response(name: str, call_id: str, **arguments: Any) -> ModelResponse:
    return ModelResponse(
        assistant_text="Taking my turn.",
        tool_calls=[ToolCall(call_id=call_id, name=name, arguments=dict(arguments))],
        finish_reason="stop",
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _queued_agents(
    sequence: dict[str, list[tuple[str, dict[str, Any]]]],
    *,
    prefix: str,
) -> dict[str, FakeAgent]:
    agents: dict[str, FakeAgent] = {}
    counter = 0
    for actor_id, actions in sequence.items():
        responses: list[ModelResponse] = []
        for name, arguments in actions:
            counter += 1
            responses.append(_response(name, f"{prefix}-{counter}", **arguments))
        agents[actor_id] = FakeAgent(responses=responses)
    return agents


async def _run_async(
    tmp_path: Path,
    *,
    agents: dict[str, Any],
    run_id: str,
    matches: int = 1,
    seed: int = 11,
    max_invalid_attempts: int = 2,
    max_provider_retries: int = 2,
) -> list[dict[str, Any]]:
    engine = RunEngine(
        game=BunkerGame(),
        agents=agents,
        agent_metadata=[{"id": actor_id} for actor_id in _CONFIG["players"]],
        game_config=dict(_CONFIG),
        run_dir=tmp_path,
        run_id=run_id,
        seed=seed,
        matches=matches,
        max_invalid_attempts=max_invalid_attempts,
        max_provider_retries=max_provider_retries,
    )
    result = await engine.run()
    assert result.success, f"run failed: {result.status}"
    return _read_events(tmp_path)


def _types_for_turn(
    events: list[dict[str, Any]], match_id: str, turn_index: int
) -> list[str]:
    return [
        event["event_type"]
        for event in events
        if event.get("match_id") == match_id and event.get("turn_index") == turn_index
    ]


class TestBunkerTrace:
    async def test_full_match_trace_order_and_counts(self, tmp_path: Path) -> None:
        events = await _run_async(
            tmp_path, agents=_queued_agents(_SEQUENCE, prefix="a"), run_id="run-full"
        )

        turn_starts = [
            event
            for event in events
            if event["event_type"] == "turn_start"
            and event.get("match_id") == "match-0"
        ]
        assert len(turn_starts) == 14

        speaks = [
            event
            for event in events
            if event["event_type"] == "bunker_speak"
            and event.get("match_id") == "match-0"
        ]
        assert len(speaks) == 7
        totals = [
            event
            for event in events
            if event["event_type"] == "bunker_vote_totals"
            and event.get("match_id") == "match-0"
        ]
        assert len(totals) == 2
        eliminations = [
            event
            for event in events
            if event["event_type"] == "bunker_elimination"
            and event.get("match_id") == "match-0"
        ]
        assert [event["payload"]["eliminated"] for event in eliminations] == [
            "player-2",
            "player-3",
        ]

        assert _types_for_turn(events, "match-0", 0) == [
            "turn_start",
            "observation",
            "tool_schemas",
            "model_request",
            "model_response",
            "validation",
            "transition",
            "bunker_speak",
            "model_request",
            "model_response",
            "turn_end",
        ]

        closure_types = _types_for_turn(events, "match-0", 7)
        assert closure_types[closure_types.index("transition") :] == [
            "transition",
            "bunker_vote_totals",
            "bunker_elimination",
            "round_start",
            "model_request",
            "model_response",
            "turn_end",
        ]

        final_types = _types_for_turn(events, "match-0", 13)
        assert final_types[final_types.index("transition") :] == [
            "transition",
            "bunker_vote_totals",
            "bunker_elimination",
            "bunker_match_end",
            "model_request",
            "model_response",
            "turn_end",
        ]

        match_end = [
            event
            for event in events
            if event["event_type"] == "match_end" and event.get("match_id") == "match-0"
        ]
        assert len(match_end) == 1
        result = match_end[0]["payload"]["result"]
        assert result["completed"] is True
        assert result["outcome"]["admitted"] == ["player-1", "player-4"]
        assert result["outcome"]["excluded"] == ["player-2", "player-3"]
        assert result["outcome"]["completion_reason"] == "capacity_reached"

        run_end = [event for event in events if event["event_type"] == "run_end"]
        assert run_end[-1]["payload"]["status"] == "success"

    async def test_observation_deltas_deliver_unseen_events_only(
        self, tmp_path: Path
    ) -> None:
        events = await _run_async(
            tmp_path, agents=_queued_agents(_SEQUENCE, prefix="b"), run_id="run-delta"
        )
        observations = [
            event
            for event in events
            if event["event_type"] == "observation"
            and event.get("match_id") == "match-0"
            and event.get("actor_id") == "player-2"
        ]
        assert len(observations) == 2

        first_text = observations[0]["payload"]["text"]
        assert _P1_R1 in first_text

        second_text = observations[1]["payload"]["text"]
        assert _P3_R1 in second_text
        assert _P4_R1 in second_text
        assert _P1_R1 not in second_text

    async def test_reveal_becomes_public_to_all_observers(self, tmp_path: Path) -> None:
        sequence = copy.deepcopy(_SEQUENCE)
        sequence["player-1"][0] = (
            "bunker_speak",
            {"text": _P1_R1, "reveal_attribute": "skill"},
        )
        events = await _run_async(
            tmp_path, agents=_queued_agents(sequence, prefix="c"), run_id="run-reveal"
        )

        reveals = [
            event
            for event in events
            if event["event_type"] == "bunker_reveal"
            and event.get("match_id") == "match-0"
        ]
        assert len(reveals) == 1
        revealed_value = reveals[0]["payload"]["value"]
        assert reveals[0]["payload"]["attribute"] == "skill"
        assert reveals[0]["payload"]["speaker"] == "player-1"

        later_observations = [
            event
            for event in events
            if event["event_type"] == "observation"
            and event.get("match_id") == "match-0"
            and event.get("actor_id") == "player-2"
            and event.get("turn_index") == 1
        ]
        assert len(later_observations) == 1
        assert (
            f"player-1 revealed skill: {revealed_value}"
            in (later_observations[0]["payload"]["text"])
        )

        revealer_observation = [
            event
            for event in events
            if event["event_type"] == "observation"
            and event.get("match_id") == "match-0"
            and event.get("actor_id") == "player-1"
            and event.get("turn_index") == 4
        ]
        assert len(revealer_observation) == 1
        assert (
            f"skill (revealed): {revealed_value}"
            in (revealer_observation[0]["payload"]["text"])
        )

    async def test_provider_failure_eliminates_only_failed_player(
        self, tmp_path: Path
    ) -> None:
        agents: dict[str, Any] = _queued_agents(_FAILURE_SEQUENCE, prefix="d")
        agents["player-3"] = FakeAgent(
            raise_on_respond=ProviderError(
                "simulated outage", provider="openai", model="fake-model"
            )
        )

        events = await _run_async(
            tmp_path,
            agents=agents,
            run_id="run-provider-fail",
            max_provider_retries=1,
        )

        eliminations = [
            event
            for event in events
            if event["event_type"] == "bunker_elimination"
            and event.get("match_id") == "match-0"
        ]
        assert [event["payload"]["eliminated"] for event in eliminations] == [
            "player-3",
            "player-2",
        ]
        assert (
            eliminations[0]["payload"]["reason"]
            == "failed_turn_provider_retries_exhausted"
        )

        failed_turn_end = [
            event
            for event in events
            if event["event_type"] == "turn_end"
            and event.get("match_id") == "match-0"
            and event.get("actor_id") == "player-3"
            and event["payload"].get("status") == "failed"
        ]
        assert len(failed_turn_end) == 1
        assert (
            failed_turn_end[0]["payload"]["failure_reason"]
            == "provider_retries_exhausted"
        )

        recoveries = [
            event
            for event in events
            if event["event_type"] == "transition"
            and event.get("match_id") == "match-0"
            and event["payload"].get("recovery") is True
        ]
        assert len(recoveries) == 1
        assert (
            "player-3 eliminated by failed turn" in recoveries[0]["payload"]["summary"]
        )

        round_one_totals = [
            event
            for event in events
            if event["event_type"] == "bunker_vote_totals"
            and event.get("match_id") == "match-0"
            and event["payload"]["round_index"] == 0
        ]
        assert round_one_totals == []

        match_end = [
            event
            for event in events
            if event["event_type"] == "match_end" and event.get("match_id") == "match-0"
        ]
        assert match_end[0]["payload"]["result"]["completed"] is True
        assert match_end[0]["payload"]["result"]["outcome"]["admitted"] == [
            "player-1",
            "player-4",
        ]

    async def test_invalid_action_exhaustion_eliminates_only_failed_player(
        self, tmp_path: Path
    ) -> None:
        agents: dict[str, Any] = _queued_agents(_FAILURE_SEQUENCE, prefix="e")
        agents["player-3"] = FakeAgent(
            responses=[
                ModelResponse(
                    assistant_text="I stay quiet.",
                    tool_calls=[],
                    finish_reason="stop",
                )
                for _ in range(6)
            ]
        )

        events = await _run_async(
            tmp_path,
            agents=agents,
            run_id="run-invalid-fail",
            max_invalid_attempts=1,
        )

        eliminations = [
            event
            for event in events
            if event["event_type"] == "bunker_elimination"
            and event.get("match_id") == "match-0"
        ]
        assert [event["payload"]["eliminated"] for event in eliminations] == [
            "player-3",
            "player-2",
        ]
        assert (
            eliminations[0]["payload"]["reason"]
            == "failed_turn_invalid_attempts_exhausted"
        )

        recoveries = [
            event
            for event in events
            if event["event_type"] == "transition"
            and event.get("match_id") == "match-0"
            and event["payload"].get("recovery") is True
        ]
        assert len(recoveries) == 1

        match_end = [
            event
            for event in events
            if event["event_type"] == "match_end" and event.get("match_id") == "match-0"
        ]
        assert match_end[0]["payload"]["result"]["completed"] is True

    async def test_ballot_targets_stay_in_actor_context_only(
        self, tmp_path: Path
    ) -> None:
        events = await _run_async(
            tmp_path, agents=_queued_agents(_SEQUENCE, prefix="f"), run_id="run-ballot"
        )

        expected_mapping = {
            ("player-1", "player-2"),
            ("player-2", "player-3"),
            ("player-3", "player-2"),
            ("player-4", "player-2"),
            ("player-1", "player-3"),
            ("player-3", "player-4"),
            ("player-4", "player-3"),
        }
        found_mapping: set[tuple[str, str]] = set()
        for event in events:
            if event["event_type"] != "model_request":
                continue
            actor_id = event.get("actor_id")
            for message in event["payload"]["messages"]:
                if message.get("role") != "assistant":
                    continue
                for call in message.get("tool_calls", []):
                    function = call.get("function", {})
                    if function.get("name") != "bunker_vote_eliminate":
                        continue
                    arguments = json.loads(function["arguments"])
                    found_mapping.add((actor_id, arguments["target_id"]))

        assert found_mapping == expected_mapping

        for event in events:
            if event["event_type"] == "bunker_vote_totals":
                assert set(event["payload"].keys()) == {"round_index", "totals"}
            if event["event_type"] == "transition":
                assert "bunker_vote_eliminate" not in json.dumps(event["payload"])
            if event["event_type"] == "observation":
                assert "target_id" not in event["payload"]["text"]

    async def test_identical_scripts_produce_identical_normalized_traces(
        self, tmp_path: Path
    ) -> None:
        first = await _run_async(
            tmp_path / "one",
            agents=_queued_agents(_SEQUENCE, prefix="g"),
            run_id="run-det-1",
        )
        second = await _run_async(
            tmp_path / "two",
            agents=_queued_agents(_SEQUENCE, prefix="g"),
            run_id="run-det-2",
        )
        assert _normalize(first) == _normalize(second)

    async def test_transcripts_and_memory_never_leak_between_actors(
        self, tmp_path: Path
    ) -> None:
        events = await _run_async(
            tmp_path, agents=_queued_agents(_SEQUENCE, prefix="i"), run_id="run-leak"
        )

        def _dossier_values(actor_id: str) -> list[str]:
            observation = next(
                event
                for event in events
                if event["event_type"] == "observation"
                and event.get("match_id") == "match-0"
                and event.get("actor_id") == actor_id
            )
            lines = observation["payload"]["text"].splitlines()
            start = lines.index("Your dossier:")
            return [line.split(": ", 1)[1] for line in lines[start + 1 : start + 5]]

        other_values = _dossier_values("player-2") + _dossier_values("player-3")
        own_values = _dossier_values("player-1")

        requests = [
            event
            for event in events
            if event["event_type"] == "model_request"
            and event.get("match_id") == "match-0"
            and event.get("actor_id") == "player-1"
        ]
        assert requests
        for event in requests:
            dumped = json.dumps(event["payload"]["messages"])
            for value in other_values:
                assert value not in dumped
        for event in requests:
            dumped = json.dumps(event["payload"]["messages"])
            assert any(value in dumped for value in own_values)

        tool_schemas = [
            event
            for event in events
            if event["event_type"] == "tool_schemas"
            and event.get("match_id") == "match-0"
            and event.get("actor_id") == "player-1"
        ]
        assert tool_schemas
        names = {
            tool["function"]["name"] for tool in tool_schemas[0]["payload"]["tools"]
        }
        assert {"read_memory", "write_memory"} <= names

    async def test_cross_match_memory_and_transcript_reset(
        self, tmp_path: Path
    ) -> None:
        agents: dict[str, Any] = _queued_agents(_SEQUENCE, prefix="j")
        note = "match one private note alpha"
        p1_queue: list[ModelResponse] = [
            _response(
                "write_memory",
                "j-mem-write",
                text=note,
            )
        ]
        p1_queue.extend(agents["player-1"]._responses)
        p1_queue.append(_response("read_memory", "j-mem-read"))
        agents["player-1"]._responses = p1_queue

        events = await _run_async(
            tmp_path, agents=agents, run_id="run-reset", matches=2
        )

        reads = [
            event
            for event in events
            if event["event_type"] == "memory_operation"
            and event.get("match_id") == "match-1"
            and event["payload"].get("operation") == "read"
        ]
        assert len(reads) == 1
        assert note not in json.dumps(reads[0]["payload"]["notes"])

        match_two_requests = [
            event
            for event in events
            if event["event_type"] == "model_request"
            and event.get("match_id") == "match-1"
            and event.get("actor_id") == "player-1"
        ]
        assert match_two_requests
        first_dump = json.dumps(match_two_requests[0]["payload"]["messages"])
        assert _P1_R1 not in first_dump
        assert _P1_R2 not in first_dump

        match_ends = [event for event in events if event["event_type"] == "match_end"]
        assert len(match_ends) == 2
        assert all(
            event["payload"]["result"]["completed"] is True for event in match_ends
        )

    def test_cli_run_writes_complete_offline_trace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from typer.testing import CliRunner

        from benchtable import cli

        monkeypatch.setattr(cli, "_agent_factory", None)
        cli.set_registry(None)
        scripted = _queued_agents(_SEQUENCE, prefix="k")
        cli.set_agent_factory(lambda agent_config: scripted[agent_config.id])

        config_path = tmp_path / "bunker.toml"
        agents_toml = "\n".join(
            f"""
            [[agents]]
            id = "player-{index}"
            model = "fake-model"
            api_key_env = "FAKE_KEY"
            """.strip()
            for index in range(1, 5)
        )
        config_path.write_text(
            f"""
            [run]
            game = "bunker"
            matches = 1
            seed = 11
            max_turns = 30

            [run.game_config]
            players = ["player-1", "player-2", "player-3", "player-4"]
            scenario = "sealed shelter after a solar storm"
            shelter_capacity = 2

            {agents_toml}
            """
        )

        output_dir = tmp_path / "out"
        runner = CliRunner()
        try:
            result = runner.invoke(
                cli.app,
                ["run", "--config", str(config_path), "--output", str(output_dir)],
            )

            assert result.exit_code == 0, result.output
            trace = output_dir / "events.jsonl"
            assert trace.exists()
            events = _read_events(output_dir)
            assert events[-1]["payload"]["status"] == "success"
        finally:
            cli.set_agent_factory(None)
