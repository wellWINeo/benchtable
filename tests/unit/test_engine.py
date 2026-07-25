"""Tests for the run engine."""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from typing import Any

import pytest
from tests.fixtures.fake_agent import FakeAgent
from tests.fixtures.tiny_game import TinyGame

from benchtable.contracts import ModelResponse, ToolCall, Usage
from benchtable.engine import RunEngine
from benchtable.errors import ProviderError
from benchtable.games.poker.session import PokerGame


def _read_events(path: Path) -> list[dict[str, Any]]:
    raw = (path / "events.jsonl").read_text()
    return [json.loads(line) for line in raw.splitlines() if line]


class TestRunEngine:
    async def test_emits_ordered_events(self, tmp_path: Path) -> None:
        engine = RunEngine(
            game=TinyGame(),
            agent=FakeAgent(tool_name="act"),
            run_dir=tmp_path / "run",
            run_id="test-run",
            seed=42,
            matches=1,
            max_turns=20,
        )
        result = await engine.run()
        assert result.success

        events = _read_events(tmp_path / "run")
        event_types = [e["event_type"] for e in events]

        assert event_types[0] == "run_config"
        assert "match_start" in event_types
        assert "turn_start" in event_types
        assert "observation" in event_types
        assert "model_request" in event_types
        assert "model_response" in event_types
        assert "validation" in event_types
        assert "transition" in event_types
        assert "turn_end" in event_types
        assert "match_end" in event_types
        assert "run_end" in event_types

    async def test_agent_receives_only_current_actor_observation(
        self, tmp_path: Path
    ) -> None:
        agent = FakeAgent(tool_name="act")
        engine = RunEngine(
            game=TinyGame(),
            agent=agent,
            run_dir=tmp_path / "run",
            run_id="test-run",
            seed=42,
            matches=1,
            max_turns=20,
        )
        await engine.run()

        for req in agent.requests:
            # Each request should have a system prompt and messages
            assert req.system_prompt
            assert req.messages
            # The observation in the messages should be for a single actor
            assert req.actor_id in ("a", "b")

    async def test_valid_tool_call_advances_game(self, tmp_path: Path) -> None:
        engine = RunEngine(
            game=TinyGame(),
            agent=FakeAgent(tool_name="act"),
            run_dir=tmp_path / "run",
            run_id="test-run",
            seed=42,
            matches=1,
            max_turns=20,
        )
        await engine.run()

        events = _read_events(tmp_path / "run")
        transitions = [e for e in events if e["event_type"] == "transition"]
        assert len(transitions) > 0

    async def test_assistant_text_retained_in_response(self, tmp_path: Path) -> None:
        def make_response() -> ModelResponse:
            return ModelResponse(
                assistant_text="I'll call act now.",
                tool_calls=[ToolCall(call_id="c1", name="act", arguments={})],
                finish_reason="stop",
                usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )

        agent = FakeAgent(responses=[make_response() for _ in range(10)])
        engine = RunEngine(
            game=TinyGame(),
            agent=agent,
            run_dir=tmp_path / "run",
            run_id="test-run",
            seed=42,
            matches=1,
            max_turns=20,
        )
        await engine.run()

        events = _read_events(tmp_path / "run")
        responses = [e for e in events if e["event_type"] == "model_response"]
        assert len(responses) > 0
        assert responses[0]["payload"]["assistant_text"] == "I'll call act now."

    async def test_missing_tool_call_retries_and_fails(self, tmp_path: Path) -> None:
        bad_response = ModelResponse(
            assistant_text="I don't want to call a tool.",
            tool_calls=[],
            finish_reason="stop",
        )
        agent = FakeAgent(responses=[bad_response for _ in range(5)])
        engine = RunEngine(
            game=TinyGame(),
            agent=agent,
            run_dir=tmp_path / "run",
            run_id="test-run",
            seed=42,
            matches=1,
            max_turns=20,
            max_invalid_attempts=2,
        )
        result = await engine.run()
        assert not result.success

        events = _read_events(tmp_path / "run")
        match_ends = [e for e in events if e["event_type"] == "match_end"]
        assert len(match_ends) == 1
        assert match_ends[0]["payload"]["result"]["completed"] is False

    async def test_unknown_tool_retries_and_fails(self, tmp_path: Path) -> None:
        bad_response = ModelResponse(
            assistant_text="Wrong tool.",
            tool_calls=[ToolCall(call_id="c1", name="nonexistent", arguments={})],
            finish_reason="stop",
        )
        agent = FakeAgent(responses=[bad_response for _ in range(5)])
        engine = RunEngine(
            game=TinyGame(),
            agent=agent,
            run_dir=tmp_path / "run",
            run_id="test-run",
            seed=42,
            matches=1,
            max_turns=20,
            max_invalid_attempts=2,
        )
        await engine.run()

        events = _read_events(tmp_path / "run")
        match_ends = [e for e in events if e["event_type"] == "match_end"]
        assert match_ends[0]["payload"]["result"]["completed"] is False

    async def test_malformed_arguments_retries(self, tmp_path: Path) -> None:
        bad_response = ModelResponse(
            assistant_text="Bad args.",
            tool_calls=[
                ToolCall(
                    call_id="c1",
                    name="act",
                    arguments=None,
                    raw_arguments="not-json",
                    parse_error="invalid json",
                )
            ],
            finish_reason="stop",
        )
        good_response = ModelResponse(
            assistant_text="Fixed.",
            tool_calls=[ToolCall(call_id="c2", name="act", arguments={})],
            finish_reason="stop",
        )
        agent = FakeAgent(
            responses=[bad_response, good_response]
            + [
                ModelResponse(
                    assistant_text="filler",
                    tool_calls=[ToolCall(call_id=f"c{i}", name="act", arguments={})],
                    finish_reason="stop",
                )
                for i in range(3, 10)
            ]
        )
        engine = RunEngine(
            game=TinyGame(),
            agent=agent,
            run_dir=tmp_path / "run",
            run_id="test-run",
            seed=42,
            matches=1,
            max_turns=20,
            max_invalid_attempts=3,
        )
        await engine.run()

        events = _read_events(tmp_path / "run")
        invalid_events = [
            e
            for e in events
            if e["event_type"] == "validation" and not e["payload"].get("valid", True)
        ]
        assert len(invalid_events) >= 1

    async def test_provider_exception_retries(self, tmp_path: Path) -> None:
        nonlocal_counter = {"n": 0}

        class FlakyAgent:
            async def respond(self, request: Any) -> ModelResponse:
                nonlocal_counter["n"] += 1
                if nonlocal_counter["n"] <= 2:
                    raise ProviderError("timeout", provider="test", model="test")
                return ModelResponse(
                    assistant_text="OK",
                    tool_calls=[
                        ToolCall(
                            call_id=f"c{nonlocal_counter['n']}",
                            name="act",
                            arguments={},
                        )
                    ],
                    finish_reason="stop",
                )

        agent = FlakyAgent()
        engine = RunEngine(
            game=TinyGame(),
            agent=agent,
            run_dir=tmp_path / "run",
            run_id="test-run",
            seed=42,
            matches=1,
            max_turns=20,
            max_provider_retries=3,
        )
        result = await engine.run()
        assert result.success

        events = _read_events(tmp_path / "run")
        provider_errors = [e for e in events if e["event_type"] == "provider_error"]
        assert len(provider_errors) >= 2

    async def test_exhausted_provider_budget_produces_failed_match(
        self, tmp_path: Path
    ) -> None:
        class AlwaysFailAgent:
            async def respond(self, request: Any) -> ModelResponse:
                raise ProviderError("always fails", provider="test", model="test")

        engine = RunEngine(
            game=TinyGame(),
            agent=AlwaysFailAgent(),
            run_dir=tmp_path / "run",
            run_id="test-run",
            seed=42,
            matches=1,
            max_turns=20,
            max_provider_retries=2,
        )
        result = await engine.run()
        assert not result.success

        events = _read_events(tmp_path / "run")
        match_ends = [e for e in events if e["event_type"] == "match_end"]
        assert match_ends[0]["payload"]["result"]["completed"] is False

    async def test_max_turns_terminates_game(self, tmp_path: Path) -> None:
        engine = RunEngine(
            game=TinyGame(),
            agent=FakeAgent(tool_name="act"),
            run_dir=tmp_path / "run",
            run_id="test-run",
            seed=42,
            matches=1,
            max_turns=3,
        )
        await engine.run()

        events = _read_events(tmp_path / "run")
        match_ends = [e for e in events if e["event_type"] == "match_end"]
        assert match_ends[0]["payload"]["result"]["completed"] is False
        assert (
            "max_turns" in str(match_ends[0]["payload"]).lower()
            or "turn" in str(match_ends[0]["payload"].get("failure_reason", "")).lower()
        )

    async def test_match_seeds_are_deterministic(self, tmp_path: Path) -> None:
        engine = RunEngine(
            game=TinyGame(),
            agent=FakeAgent(tool_name="act"),
            run_dir=tmp_path / "run",
            run_id="test-run",
            seed=42,
            matches=3,
            max_turns=20,
        )
        await engine.run()

        events = _read_events(tmp_path / "run")
        match_starts = [e for e in events if e["event_type"] == "match_start"]
        seeds = [e["payload"]["seed"] for e in match_starts]

        assert len(seeds) == 3
        assert len(set(seeds)) == 3  # all different
        assert all(isinstance(s, int) for s in seeds)

    async def test_common_metrics_included(self, tmp_path: Path) -> None:
        engine = RunEngine(
            game=TinyGame(),
            agent=FakeAgent(tool_name="act"),
            run_dir=tmp_path / "run",
            run_id="test-run",
            seed=42,
            matches=1,
            max_turns=20,
        )
        await engine.run()

        events = _read_events(tmp_path / "run")
        match_ends = [e for e in events if e["event_type"] == "match_end"]
        metrics = match_ends[0]["payload"]["metrics"]
        assert "turn_count" in metrics
        assert "invalid_actions" in metrics
        assert "provider_failures" in metrics
        assert "total_tokens" in metrics
        assert "latency_seconds" in metrics

    async def test_multiple_matches(self, tmp_path: Path) -> None:
        engine = RunEngine(
            game=TinyGame(),
            agent=FakeAgent(tool_name="act"),
            run_dir=tmp_path / "run",
            run_id="test-run",
            seed=42,
            matches=3,
            max_turns=20,
        )
        result = await engine.run()
        assert result.success

        events = _read_events(tmp_path / "run")
        match_starts = [e for e in events if e["event_type"] == "match_start"]
        match_ends = [e for e in events if e["event_type"] == "match_end"]
        assert len(match_starts) == 3
        assert len(match_ends) == 3

    async def test_run_end_event(self, tmp_path: Path) -> None:
        engine = RunEngine(
            game=TinyGame(),
            agent=FakeAgent(tool_name="act"),
            run_dir=tmp_path / "run",
            run_id="test-run",
            seed=42,
            matches=1,
            max_turns=20,
        )
        await engine.run()

        events = _read_events(tmp_path / "run")
        assert events[-1]["event_type"] == "run_end"

    async def test_run_can_execute_inside_an_active_event_loop(
        self, tmp_path: Path
    ) -> None:
        running_loop = asyncio.get_running_loop()
        engine = RunEngine(
            game=TinyGame(),
            agent=FakeAgent(tool_name="act"),
            run_dir=tmp_path / "run",
            run_id="test-run",
            max_turns=20,
        )

        result = await engine.run()

        assert result.success
        assert asyncio.get_running_loop() is running_loop

    async def test_actor_agent_mapping_dispatches_by_current_actor(
        self, tmp_path: Path
    ) -> None:
        agent_a = FakeAgent(tool_name="act")
        agent_b = FakeAgent(tool_name="act")
        engine = RunEngine(
            game=TinyGame(),
            agents={"a": agent_a, "b": agent_b},
            run_dir=tmp_path / "run",
            run_id="test-run",
            max_turns=20,
        )

        result = await engine.run()

        assert result.success
        assert {request.actor_id for request in agent_a.requests} == {"a"}
        assert {request.actor_id for request in agent_b.requests} == {"b"}

    async def test_missing_actor_agent_fails_explicitly(self, tmp_path: Path) -> None:
        engine = RunEngine(
            game=TinyGame(),
            agents={"a": FakeAgent(tool_name="act")},
            run_dir=tmp_path / "run",
            run_id="test-run",
            max_turns=20,
        )

        result = await engine.run()
        events = _read_events(tmp_path / "run")

        assert not result.success
        assert any(
            event["event_type"] == "engine_error"
            and event["payload"].get("error") == "missing_agent"
            for event in events
        )
        assert not any(event["event_type"] == "provider_error" for event in events)

    async def test_exact_mapping_is_enforced_for_direct_engine_use(
        self, tmp_path: Path
    ) -> None:
        agent = FakeAgent(tool_name="poker_action")
        engine = RunEngine(
            game=PokerGame(),
            agent=agent,
            game_config={"players": ["alice", "bob"]},
            run_dir=tmp_path / "run",
            run_id="test-run",
            max_turns=1,
        )

        result = await engine.run()

        assert not result.success
        assert agent.requests == []
        events = _read_events(tmp_path / "run")
        assert any(
            event["event_type"] == "engine_error"
            and event["payload"].get("error") == "exact_agent_mapping"
            for event in events
        )
        assert not any(event["event_type"] == "model_request" for event in events)

    async def test_game_config_is_forwarded_to_session(self, tmp_path: Path) -> None:
        game = TinyGame()
        engine = RunEngine(
            game=game,
            agent=FakeAgent(tool_name="act"),
            game_config={"max_actions": 1, "label": "configured"},
            run_dir=tmp_path / "run",
            run_id="test-run",
            max_turns=20,
        )

        result = await engine.run()

        assert result.success
        assert game.created_game_configs == [{"max_actions": 1, "label": "configured"}]

    async def test_terminal_action_on_final_allowed_turn_completes(
        self, tmp_path: Path
    ) -> None:
        game = TinyGame(max_actions=4)
        engine = RunEngine(
            game=game,
            agent=FakeAgent(tool_name="act"),
            run_dir=tmp_path / "run",
            run_id="test-run",
            max_turns=4,
        )

        result = await engine.run()
        events = _read_events(tmp_path / "run")
        match_end = next(
            event for event in events if event["event_type"] == "match_end"
        )

        assert result.success
        assert match_end["payload"]["result"]["completed"] is True
        assert "failure_reason" not in match_end["payload"]

    async def test_retry_history_uses_original_tool_call_id(
        self, tmp_path: Path
    ) -> None:
        invalid = ModelResponse(
            assistant_text="Wrong tool.",
            tool_calls=[
                ToolCall(call_id="original-id", name="nonexistent", arguments={})
            ],
            finish_reason="stop",
        )
        fixed = ModelResponse(
            assistant_text="Fixed.",
            tool_calls=[ToolCall(call_id="fixed-id", name="act", arguments={})],
            finish_reason="stop",
        )
        agent = FakeAgent(responses=[invalid, fixed])
        engine = RunEngine(
            game=TinyGame(max_actions=1),
            agent=agent,
            run_dir=tmp_path / "run",
            run_id="test-run",
            max_turns=2,
            max_invalid_attempts=1,
        )

        result = await engine.run()

        assert result.success
        retry_messages = agent.requests[1].messages
        assert retry_messages[1]["role"] == "assistant"
        assert retry_messages[1]["tool_calls"][0]["id"] == "original-id"
        assert retry_messages[2]["role"] == "tool"
        assert retry_messages[2]["tool_call_id"] == "original-id"

    async def test_multiple_tool_retry_has_tool_response_for_each_call(
        self, tmp_path: Path
    ) -> None:
        invalid = ModelResponse(
            assistant_text="I called two tools.",
            tool_calls=[
                ToolCall(call_id="first-id", name="act", arguments={}),
                ToolCall(call_id="second-id", name="act", arguments={}),
            ],
            finish_reason="stop",
        )
        fixed = ModelResponse(
            assistant_text="Fixed.",
            tool_calls=[ToolCall(call_id="fixed-id", name="act", arguments={})],
            finish_reason="stop",
        )
        agent = FakeAgent(responses=[invalid, fixed])
        engine = RunEngine(
            game=TinyGame(max_actions=1),
            agent=agent,
            run_dir=tmp_path / "run",
            run_id="test-run",
            max_turns=2,
            max_invalid_attempts=1,
        )

        result = await engine.run()

        assert result.success
        retry_messages = agent.requests[1].messages
        assert [message["role"] for message in retry_messages] == [
            "user",
            "assistant",
            "tool",
            "tool",
        ]
        assert [message["tool_call_id"] for message in retry_messages[2:]] == [
            "first-id",
            "second-id",
        ]

    async def test_retry_without_tool_id_uses_user_feedback(
        self, tmp_path: Path
    ) -> None:
        invalid = ModelResponse(
            assistant_text="No action.", tool_calls=[], finish_reason="stop"
        )
        fixed = ModelResponse(
            assistant_text="Acting.",
            tool_calls=[ToolCall(call_id="good-id", name="act", arguments={})],
            finish_reason="stop",
        )
        agent = FakeAgent(responses=[invalid, fixed])
        engine = RunEngine(
            game=TinyGame(max_actions=1),
            agent=agent,
            run_dir=tmp_path / "run",
            run_id="test-run",
            max_turns=2,
            max_invalid_attempts=1,
        )

        result = await engine.run()

        assert result.success
        retry_messages = agent.requests[1].messages
        assert retry_messages[1]["role"] == "assistant"
        assert retry_messages[-1]["role"] == "user"
        assert "validation" in str(retry_messages[-1]["content"]).lower()

    async def test_validation_is_not_marked_valid_before_action_application(
        self, tmp_path: Path
    ) -> None:
        bad = ModelResponse(
            assistant_text="Bad action.",
            tool_calls=[ToolCall(call_id="bad-id", name="act", arguments={})],
            finish_reason="stop",
        )
        engine = RunEngine(
            game=TinyGame(reject_actions=True),
            agent=FakeAgent(responses=bad),
            run_dir=tmp_path / "run",
            run_id="test-run",
            max_turns=1,
            max_invalid_attempts=0,
        )

        result = await engine.run()
        events = _read_events(tmp_path / "run")
        validations = [event for event in events if event["event_type"] == "validation"]

        assert not result.success
        assert validations
        assert all(event["payload"]["valid"] is False for event in validations)

    async def test_exhausted_invalid_budget_emits_recovery_transition(
        self, tmp_path: Path
    ) -> None:
        bad = ModelResponse(
            assistant_text="No action.", tool_calls=[], finish_reason="stop"
        )
        engine = RunEngine(
            game=TinyGame(failed_turn_recovery=True),
            agent=FakeAgent(responses=bad),
            run_dir=tmp_path / "run",
            run_id="test-run",
            max_turns=1,
            max_invalid_attempts=0,
        )

        result = await engine.run()
        events = _read_events(tmp_path / "run")

        assert not result.success
        assert any(
            event["event_type"] == "transition"
            and event["payload"].get("recovery") is True
            for event in events
        )

    async def test_nonterminal_recovery_continues_to_later_turns(
        self, tmp_path: Path
    ) -> None:
        bad = ModelResponse(
            assistant_text="No action.", tool_calls=[], finish_reason="stop"
        )
        good = ModelResponse(
            assistant_text="Acting.",
            tool_calls=[ToolCall(call_id="good", name="act", arguments={})],
            finish_reason="stop",
        )
        game = TinyGame(max_actions=2, failed_turn_recovery=True)
        engine = RunEngine(
            game=game,
            agent=FakeAgent(responses=[bad, good, good]),
            run_dir=tmp_path / "run",
            run_id="test-run",
            max_turns=3,
            max_invalid_attempts=0,
        )

        result = await engine.run()

        assert result.success
        assert (
            len(
                [
                    event
                    for event in _read_events(tmp_path / "run")
                    if event["event_type"] == "transition"
                ]
            )
            >= 3
        )

    async def test_provider_failure_is_not_reported_as_plugin_failure(
        self, tmp_path: Path
    ) -> None:
        engine = RunEngine(
            game=TinyGame(),
            agent=FakeAgent(
                raise_on_respond=ProviderError("down", provider="test", model="test")
            ),
            run_dir=tmp_path / "run",
            run_id="test-run",
            max_turns=1,
            max_provider_retries=0,
        )

        result = await engine.run()
        events = _read_events(tmp_path / "run")

        assert not result.success
        assert any(event["event_type"] == "provider_error" for event in events)
        assert not any(
            event["event_type"] == "engine_error"
            and event["payload"].get("phase") == "session_creation"
            for event in events
        )

    async def test_timeout_is_reported_as_provider_failure(
        self, tmp_path: Path
    ) -> None:
        engine = RunEngine(
            game=TinyGame(),
            agent=FakeAgent(raise_on_respond=TimeoutError("timed out")),
            run_dir=tmp_path / "run",
            run_id="test-run",
            max_turns=1,
            max_provider_retries=0,
        )

        result = await engine.run()
        events = _read_events(tmp_path / "run")

        assert not result.success
        assert any(event["event_type"] == "provider_error" for event in events)
        assert not any(
            event["event_type"] == "engine_error"
            and event["payload"].get("phase") == "agent_respond"
            for event in events
        )

    async def test_plugin_failure_is_terminal_engine_failure(
        self, tmp_path: Path
    ) -> None:
        class BrokenGame(TinyGame):
            def create_session(self, *, seed: int, game_config: Any = None) -> Any:
                raise RuntimeError("plugin exploded")

        engine = RunEngine(
            game=BrokenGame(),
            agent=FakeAgent(tool_name="act"),
            run_dir=tmp_path / "run",
            run_id="test-run",
            max_turns=1,
        )

        result = await engine.run()
        events = _read_events(tmp_path / "run")

        assert not result.success
        assert any(
            event["event_type"] == "engine_error"
            and event["payload"].get("phase") == "session_creation"
            for event in events
        )
        assert not any(event["event_type"] == "provider_error" for event in events)

    async def test_invalid_plugin_memory_limits_emit_match_failure(
        self, tmp_path: Path
    ) -> None:
        class InvalidMemoryGame(TinyGame):
            def create_session(self, *, seed: int, game_config: Any = None) -> Any:
                session = super().create_session(seed=seed, game_config=game_config)
                session.memory_max_entries = -1
                return session

        engine = RunEngine(
            game=InvalidMemoryGame(),
            agent=FakeAgent(tool_name="act"),
            run_dir=tmp_path / "run",
            run_id="invalid-memory-limits",
            max_turns=1,
        )

        result = await engine.run()

        assert not result.success
        events = _read_events(tmp_path / "run")
        assert any(
            event["event_type"] == "engine_error"
            and event["payload"].get("phase") == "memory_initialization"
            for event in events
        )
        assert any(
            event["event_type"] == "match_end"
            and event["payload"].get("failure_reason") == "memory_initialization_failed"
            for event in events
        )

    async def test_turn_setup_failure_retains_known_actor_context(
        self, tmp_path: Path
    ) -> None:
        class BrokenSetupSession:
            @property
            def current_actor_id(self) -> str:
                return "a"

            @property
            def is_terminal(self) -> bool:
                return False

            def get_observation(self, actor_id: str) -> Any:
                raise RuntimeError("observation failed")

        class BrokenSetupGame(TinyGame):
            def create_session(self, *, seed: int, game_config: Any = None) -> Any:
                return BrokenSetupSession()

        engine = RunEngine(
            game=BrokenSetupGame(),
            agent=FakeAgent(tool_name="act"),
            run_dir=tmp_path / "run",
            run_id="test-run",
            max_turns=1,
        )

        result = await engine.run()
        events = _read_events(tmp_path / "run")
        setup_error = next(
            event
            for event in events
            if event["event_type"] == "engine_error"
            and event["payload"].get("phase") == "turn_setup"
        )
        turn_end = next(event for event in events if event["event_type"] == "turn_end")

        assert not result.success
        assert setup_error["actor_id"] == "a"
        assert turn_end["actor_id"] == "a"

    async def test_model_request_and_response_retain_provider_payloads(
        self, tmp_path: Path
    ) -> None:
        response = ModelResponse(
            assistant_text="Acting.",
            tool_calls=[ToolCall(call_id="call-id", name="act", arguments={})],
            finish_reason="stop",
            usage=Usage(prompt_tokens=2, completion_tokens=3, total_tokens=5),
            raw_provider_response={"id": "provider-response"},
        )
        agent = FakeAgent(responses=response)
        engine = RunEngine(
            game=TinyGame(max_actions=1),
            agent=agent,
            run_dir=tmp_path / "run",
            run_id="test-run",
            max_turns=1,
        )

        result = await engine.run()
        events = _read_events(tmp_path / "run")
        request_event = next(
            event for event in events if event["event_type"] == "model_request"
        )
        response_event = next(
            event for event in events if event["event_type"] == "model_response"
        )

        assert result.success
        assert request_event["payload"]["messages"][0]["role"] == "system"
        assert request_event["payload"]["tools"][0]["function"]["name"] == "act"
        assert response_event["payload"]["raw_provider_response"] == {
            "id": "provider-response"
        }

    async def test_run_end_reports_honest_status_and_counts(
        self, tmp_path: Path
    ) -> None:
        bad = ModelResponse(
            assistant_text="No action.", tool_calls=[], finish_reason="stop"
        )
        engine = RunEngine(
            game=TinyGame(),
            agent=FakeAgent(responses=bad),
            run_dir=tmp_path / "run",
            run_id="test-run",
            matches=2,
            max_turns=1,
            max_invalid_attempts=0,
        )

        result = await engine.run()
        events = _read_events(tmp_path / "run")
        run_end = events[-1]

        assert not result.success
        assert run_end["event_type"] == "run_end"
        assert run_end["payload"] == {
            "status": "failed",
            "matches_total": 2,
            "matches_completed": 0,
            "matches_failed": 2,
        }

    async def test_cancellation_emits_failure_events(self, tmp_path: Path) -> None:
        game = TinyGame(failed_turn_recovery=True)
        engine = RunEngine(
            game=game,
            agent=FakeAgent(raise_on_respond=asyncio.CancelledError()),
            run_dir=tmp_path / "run",
            run_id="test-run",
            max_turns=1,
        )

        result = await engine.run()
        events = _read_events(tmp_path / "run")

        assert not result.success
        assert any(event["event_type"] == "cancellation" for event in events)
        assert any(event["event_type"] == "match_end" for event in events)

    async def test_cancellation_stops_later_matches_and_finishes_run(
        self, tmp_path: Path
    ) -> None:
        game = TinyGame(failed_turn_recovery=True)
        engine = RunEngine(
            game=game,
            agent=FakeAgent(raise_on_respond=asyncio.CancelledError()),
            run_dir=tmp_path / "run",
            matches=2,
            max_turns=1,
        )

        result = await engine.run()
        events = _read_events(tmp_path / "run")
        run_end = events[-1]

        assert not result.success
        assert result.matches_failed == 1
        assert (
            len([event for event in events if event["event_type"] == "match_start"])
            == 1
        )
        assert game.failed_turn_calls == []
        assert run_end["event_type"] == "run_end"
        assert run_end["payload"]["matches_failed"] == 1

        turn_end = next(event for event in events if event["event_type"] == "turn_end")
        assert turn_end["payload"]["status"] == "failed"
        assert turn_end["match_id"] == "match-0"
        assert turn_end["turn_index"] == 0
        assert turn_end["actor_id"] == "a"

    async def test_provider_exhaustion_does_not_invoke_failed_turn_handler(
        self, tmp_path: Path
    ) -> None:
        game = TinyGame(failed_turn_recovery=True)
        engine = RunEngine(
            game=game,
            agent=FakeAgent(
                raise_on_respond=ProviderError("down", provider="test", model="test")
            ),
            run_dir=tmp_path / "run",
            run_id="test-run",
            max_turns=1,
            max_provider_retries=0,
        )

        result = await engine.run()

        assert not result.success
        assert game.failed_turn_calls == []

    async def test_missing_failed_turn_handler_is_an_ordinary_failure(
        self, tmp_path: Path
    ) -> None:
        class SessionWithoutFailedTurn:
            def __init__(self) -> None:
                self._session = TinyGame().create_session(seed=1, game_config={})

            @property
            def current_actor_id(self) -> str:
                return self._session.current_actor_id

            def get_observation(self, actor_id: str) -> Any:
                return self._session.get_observation(actor_id)

            def get_tools(self, actor_id: str) -> Any:
                return self._session.get_tools(actor_id)

            def apply_action(
                self, actor_id: str, tool_name: str, arguments: Any
            ) -> Any:
                return self._session.apply_action(actor_id, tool_name, arguments)

            @property
            def is_terminal(self) -> bool:
                return self._session.is_terminal

            def get_result(self) -> Any:
                return self._session.get_result()

        class GameWithoutFailedTurn(TinyGame):
            def create_session(self, *, seed: int, game_config: Any = None) -> Any:
                return SessionWithoutFailedTurn()

        bad = ModelResponse(
            assistant_text="No action.", tool_calls=[], finish_reason="stop"
        )
        engine = RunEngine(
            game=GameWithoutFailedTurn(),
            agent=FakeAgent(responses=bad),
            run_dir=tmp_path / "run",
            run_id="test-run",
            max_turns=1,
            max_invalid_attempts=0,
        )

        result = await engine.run()
        events = _read_events(tmp_path / "run")

        assert not result.success
        assert not any(
            event["event_type"] == "engine_error"
            and event["payload"].get("phase") == "failed_turn_recovery"
            for event in events
        )

    async def test_recovery_transition_has_turn_context_and_failed_metrics(
        self, tmp_path: Path
    ) -> None:
        bad = ModelResponse(
            assistant_text="No action.", tool_calls=[], finish_reason="stop"
        )
        game = TinyGame(failed_turn_recovery=True)
        engine = RunEngine(
            game=game,
            agent=FakeAgent(responses=bad),
            run_dir=tmp_path / "run",
            run_id="test-run",
            max_turns=1,
            max_invalid_attempts=0,
        )

        result = await engine.run()
        events = _read_events(tmp_path / "run")
        recovery = next(
            event
            for event in events
            if event["event_type"] == "transition"
            and event["payload"].get("recovery") is True
        )
        turn_end = next(event for event in events if event["event_type"] == "turn_end")
        match_end = next(
            event for event in events if event["event_type"] == "match_end"
        )

        assert not result.success
        assert recovery["match_id"] == "match-0"
        assert recovery["turn_index"] == 0
        assert recovery["actor_id"] == "a"
        assert turn_end["payload"]["status"] == "failed"
        assert match_end["payload"]["metrics"]["turn_count"] == 1

    async def test_terminal_recovery_emits_completed_result(
        self, tmp_path: Path
    ) -> None:
        bad = ModelResponse(
            assistant_text="No action.", tool_calls=[], finish_reason="stop"
        )
        engine = RunEngine(
            game=TinyGame(
                failed_turn_recovery=True,
                recovery_terminates=True,
            ),
            agent=FakeAgent(responses=bad),
            run_dir=tmp_path / "run",
            run_id="test-run",
            max_turns=1,
            max_invalid_attempts=0,
        )

        result = await engine.run()
        events = _read_events(tmp_path / "run")
        match_end = next(
            event for event in events if event["event_type"] == "match_end"
        )

        assert result.success
        assert match_end["payload"]["result"]["completed"] is True
        assert "failure_reason" not in match_end["payload"]

    async def test_direct_engine_run_generates_non_empty_run_id(
        self, tmp_path: Path
    ) -> None:
        engine = RunEngine(
            game=TinyGame(),
            agent=FakeAgent(tool_name="act"),
            run_dir=tmp_path / "run",
            max_turns=1,
        )

        await engine.run()
        events = _read_events(tmp_path / "run")
        run_ids = {event["run_id"] for event in events}

        assert len(run_ids) == 1
        assert next(iter(run_ids))

    async def test_single_agent_metadata_uses_configured_agent_id(
        self, tmp_path: Path
    ) -> None:
        engine = RunEngine(
            game=TinyGame(),
            agent=FakeAgent(tool_name="act"),
            agent_id="player-a",
            run_dir=tmp_path / "run",
            run_id="test-run",
            max_turns=1,
        )

        await engine.run()
        events = _read_events(tmp_path / "run")
        config = next(event for event in events if event["event_type"] == "run_config")

        assert config["payload"]["agent_ids"] == ["player-a"]
        assert config["payload"]["agents"] == [{"id": "player-a"}]

    async def test_game_config_is_deep_copied_for_each_session(
        self, tmp_path: Path
    ) -> None:
        class MutatingGame(TinyGame):
            def __init__(self) -> None:
                super().__init__()
                self.seen_configs: list[dict[str, Any]] = []

            def create_session(
                self,
                *,
                seed: int,
                game_config: dict[str, Any] | None = None,
            ) -> Any:
                assert game_config is not None
                self.seen_configs.append(copy.deepcopy(game_config))
                game_config["nested"]["value"] = 99
                return super().create_session(seed=seed, game_config=game_config)

        game = MutatingGame()
        engine = RunEngine(
            game=game,
            agent=FakeAgent(tool_name="act"),
            game_config={"nested": {"value": 1}},
            run_dir=tmp_path / "run",
            run_id="test-run",
            matches=2,
            max_turns=1,
        )

        result = await engine.run()

        assert not result.success
        assert game.seen_configs == [
            {"nested": {"value": 1}},
            {"nested": {"value": 1}},
        ]

    async def test_max_memory_operations_per_turn_recorded_in_run_config(
        self, tmp_path: Path
    ) -> None:
        engine = RunEngine(
            game=TinyGame(),
            agent=FakeAgent(tool_name="act"),
            run_dir=tmp_path / "run",
            run_id="test-run",
            max_turns=1,
            max_memory_operations_per_turn=6,
        )

        await engine.run()
        events = _read_events(tmp_path / "run")
        config = next(event for event in events if event["event_type"] == "run_config")

        assert config["payload"]["max_memory_operations_per_turn"] == 6

    async def test_max_memory_operations_per_turn_stored_on_engine(
        self, tmp_path: Path
    ) -> None:
        engine = RunEngine(
            game=TinyGame(),
            agent=FakeAgent(tool_name="act"),
            run_dir=tmp_path / "run",
            run_id="test-run",
            max_turns=1,
            max_memory_operations_per_turn=3,
        )

        assert engine._max_memory_operations_per_turn == 3

    @pytest.mark.parametrize("value", [True, "4", 4.0, -1])
    def test_direct_engine_rejects_invalid_memory_budget(
        self, tmp_path: Path, value: object
    ) -> None:
        with pytest.raises(ValueError, match="max_memory_operations_per_turn"):
            RunEngine(
                game=TinyGame(),
                agent=FakeAgent(tool_name="act"),
                run_dir=tmp_path / "run",
                max_memory_operations_per_turn=value,  # type: ignore[arg-type]
            )
