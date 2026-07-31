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

from benchtable.agents.protocol import AgentRequest
from benchtable.contracts import (
    GameResult,
    ModelResponse,
    Observation,
    ToolCall,
    ToolSpec,
    Transition,
    Usage,
)
from benchtable.engine import RunEngine
from benchtable.errors import InvalidActionError, ProviderError
from benchtable.games.poker.session import PokerGame


def _read_events(path: Path) -> list[dict[str, Any]]:
    raw = (path / "events.jsonl").read_text()
    return [json.loads(line) for line in raw.splitlines() if line]


class _ResponseSequenceAgent:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[Any] = []

    async def respond(self, request: Any) -> ModelResponse:
        self.requests.append(request)
        return self.responses.pop(0)


class _ScopedTranscriptAgent:
    def __init__(self, actor_id: str) -> None:
        self.actor_id = actor_id
        self.requests: list[Any] = []
        self._actions = 0

    async def respond(self, request: Any) -> ModelResponse:
        self.requests.append(request)
        if not request.tools:
            return ModelResponse(
                assistant_text=f"{self.actor_id} final {self._actions - 1}",
                tool_calls=[],
                finish_reason="stop",
            )

        action_index = self._actions
        self._actions += 1
        return ModelResponse(
            assistant_text=f"{self.actor_id} action {action_index}",
            tool_calls=[
                ToolCall(
                    call_id=f"{self.actor_id}-call-{action_index}",
                    name="act",
                    arguments={},
                )
            ],
            finish_reason="tool_calls",
        )


class _ScopedTranscriptSession:
    def __init__(self) -> None:
        self._actions = 0
        self._actors = ("a", "b", "a", "b")

    @property
    def current_actor_id(self) -> str:
        return self._actors[min(self._actions, len(self._actors) - 1)]

    @property
    def conversation_scope_id(self) -> str:
        return "match"

    def get_observation(self, actor_id: str):
        return Observation(
            actor_id=actor_id,
            text=f"{actor_id} observation {self._actions}",
        )

    def get_tools(self, actor_id: str):
        return [ToolSpec(name="act", description="Act", parameters={"type": "object"})]

    def apply_action(self, actor_id: str, tool_name: str, arguments: dict[str, Any]):
        assert actor_id == self.current_actor_id
        self._actions += 1
        return Transition(summary=f"{actor_id} action {self._actions - 1}")

    @property
    def is_terminal(self) -> bool:
        return self._actions >= len(self._actors)

    def get_result(self):
        from benchtable.contracts import GameResult

        return GameResult(completed=True, outcome={"actions": self._actions})


class _ScopedTranscriptGame:
    @property
    def name(self) -> str:
        return "scoped"

    @property
    def version(self) -> str:
        return "1"

    def system_prompt(self, actor_id: str) -> str:
        return "Act."

    def create_session(self, *, seed: int, game_config: dict[str, Any] | None = None):
        return _ScopedTranscriptSession()


class TestRunEngine:
    async def test_observation_actor_mismatch_fails_before_provider_request(
        self, tmp_path: Path
    ) -> None:
        class WrongActorSession:
            @property
            def current_actor_id(self) -> str:
                return "a"

            def get_observation(self, actor_id: str) -> Observation:
                return Observation(actor_id="b", text="This belongs to b.")

            def get_tools(self, actor_id: str) -> list[ToolSpec]:
                return [
                    ToolSpec(
                        name="act",
                        description="Act",
                        parameters={"type": "object"},
                    )
                ]

            @property
            def is_terminal(self) -> bool:
                return False

        class WrongActorGame:
            @property
            def name(self) -> str:
                return "wrong-actor"

            @property
            def version(self) -> str:
                return "1"

            def system_prompt(self, actor_id: str) -> str:
                return "Act."

            def create_session(
                self, *, seed: int, game_config: dict[str, Any] | None = None
            ) -> WrongActorSession:
                return WrongActorSession()

        agent = FakeAgent(tool_name="act")
        engine = RunEngine(
            game=WrongActorGame(),
            agent=agent,
            run_dir=tmp_path / "run",
            run_id="wrong-actor-test",
            max_turns=1,
        )

        result = await engine.run()

        assert not result.success
        assert agent.requests == []
        events = _read_events(tmp_path / "run")
        assert not any(event["event_type"] == "observation" for event in events)
        assert not any(event["event_type"] == "model_request" for event in events)
        assert any(
            event["event_type"] == "engine_error"
            and event["payload"].get("phase") == "turn_setup"
            and "actor_id" in event["payload"].get("error", "")
            for event in events
        )

    async def test_initial_plugin_event_error_emits_match_end(
        self, tmp_path: Path
    ) -> None:
        class BrokenEventSession:
            @property
            def current_actor_id(self) -> str:
                return "a"

            @property
            def is_terminal(self) -> bool:
                return False

            def drain_hand_events(self) -> list[dict[str, object]]:
                raise RuntimeError("malformed initial event")

        class BrokenEventGame:
            @property
            def name(self) -> str:
                return "broken-events"

            @property
            def version(self) -> str:
                return "1"

            def system_prompt(self, actor_id: str) -> str:
                return "Act."

            def create_session(
                self, *, seed: int, game_config: dict[str, Any] | None = None
            ) -> BrokenEventSession:
                return BrokenEventSession()

        agent = FakeAgent(tool_name="act")
        engine = RunEngine(
            game=BrokenEventGame(),
            agent=agent,
            run_dir=tmp_path / "run",
            run_id="initial-event-error-test",
            max_turns=1,
        )

        result = await engine.run()

        assert not result.success
        assert agent.requests == []
        events = _read_events(tmp_path / "run")
        assert any(
            event["event_type"] == "engine_error"
            and event["payload"].get("error") == "malformed initial event"
            for event in events
        )
        match_end = next(
            event for event in events if event["event_type"] == "match_end"
        )
        assert match_end["payload"]["failure_reason"] == "engine_failure"

    async def test_invalid_action_detail_is_in_retry_tool_result(
        self, tmp_path: Path
    ) -> None:
        class DetailedInvalidActionSession:
            def __init__(self) -> None:
                self.actions = 0
                self.attempts = 0

            @property
            def current_actor_id(self) -> str:
                return "a"

            def get_observation(self, actor_id: str) -> Observation:
                return Observation(actor_id=actor_id, text="Choose an action.")

            def get_tools(self, actor_id: str) -> list[ToolSpec]:
                return [
                    ToolSpec(
                        name="act",
                        description="Act",
                        parameters={"type": "object", "properties": {}},
                    )
                ]

            def apply_action(
                self, actor_id: str, tool_name: str, arguments: dict[str, Any]
            ) -> Transition:
                if self.attempts == 0:
                    self.attempts += 1
                    raise InvalidActionError(
                        "Action is not legal. Legal actions: check, fold",
                        details={"legal_actions": ["check", "fold"]},
                    )
                self.actions += 1
                return Transition(summary="acted")

            @property
            def is_terminal(self) -> bool:
                return self.actions >= 1

            def get_result(self) -> GameResult:
                return GameResult(completed=True, outcome={})

        class DetailedInvalidActionGame:
            @property
            def name(self) -> str:
                return "detailed-invalid-action"

            @property
            def version(self) -> str:
                return "1"

            def system_prompt(self, actor_id: str) -> str:
                return "Act."

            def create_session(
                self, *, seed: int, game_config: dict[str, Any] | None = None
            ) -> DetailedInvalidActionSession:
                return DetailedInvalidActionSession()

        agent = _ResponseSequenceAgent(
            [
                ModelResponse(
                    assistant_text="Try action.",
                    tool_calls=[ToolCall(call_id="first", name="act", arguments={})],
                    finish_reason="tool_calls",
                ),
                ModelResponse(
                    assistant_text="Try legal action.",
                    tool_calls=[ToolCall(call_id="second", name="act", arguments={})],
                    finish_reason="tool_calls",
                ),
                ModelResponse(
                    assistant_text="Finished.",
                    tool_calls=[],
                    finish_reason="stop",
                ),
            ]
        )
        engine = RunEngine(
            game=DetailedInvalidActionGame(),
            agent=agent,
            run_dir=tmp_path / "run",
            run_id="invalid-action-detail-test",
            max_turns=1,
            max_invalid_attempts=1,
        )

        result = await engine.run()

        assert result.success
        retry_messages = agent.requests[1].messages
        retry_tool_result = next(
            message
            for message in retry_messages
            if message.get("role") == "tool" and message.get("tool_call_id") == "first"
        )
        assert "Action is not legal" in retry_tool_result["content"]
        assert '"legal_actions": ["check", "fold"]' in retry_tool_result["content"]

    async def test_invalid_action_retry_feedback_redacts_and_handles_non_json_details(
        self, tmp_path: Path
    ) -> None:
        class SensitiveInvalidActionSession:
            def __init__(self) -> None:
                self.attempts = 0
                self.actions = 0

            @property
            def current_actor_id(self) -> str:
                return "a"

            def get_observation(self, actor_id: str) -> Observation:
                return Observation(actor_id=actor_id, text="Choose an action.")

            def get_tools(self, actor_id: str) -> list[ToolSpec]:
                return [
                    ToolSpec(
                        name="act",
                        description="Act",
                        parameters={"type": "object", "properties": {}},
                    )
                ]

            def apply_action(
                self, actor_id: str, tool_name: str, arguments: dict[str, Any]
            ) -> Transition:
                if self.attempts == 0:
                    self.attempts += 1
                    raise InvalidActionError(
                        "api_key=sk-detail-secret authorization=Bearer-detail-secret",
                        details={
                            "api_key": "sk-details-secret",
                            "authorization": "Bearer details-secret",
                            "private_key": "top-detail",
                            "credentials": "credential-detail",
                            "non_json": object(),
                        },
                    )
                self.actions += 1
                return Transition(summary="acted")

            @property
            def is_terminal(self) -> bool:
                return self.actions >= 1

            def get_result(self) -> GameResult:
                return GameResult(completed=True, outcome={})

        class SensitiveInvalidActionGame:
            @property
            def name(self) -> str:
                return "sensitive-invalid-action"

            @property
            def version(self) -> str:
                return "1"

            def system_prompt(self, actor_id: str) -> str:
                return "Act."

            def create_session(
                self, *, seed: int, game_config: dict[str, Any] | None = None
            ) -> SensitiveInvalidActionSession:
                return SensitiveInvalidActionSession()

        agent = _ResponseSequenceAgent(
            [
                ModelResponse(
                    assistant_text="Try action.",
                    tool_calls=[ToolCall(call_id="first", name="act", arguments={})],
                    finish_reason="tool_calls",
                ),
                ModelResponse(
                    assistant_text="Try legal action.",
                    tool_calls=[ToolCall(call_id="second", name="act", arguments={})],
                    finish_reason="tool_calls",
                ),
                ModelResponse(
                    assistant_text="Finished.",
                    tool_calls=[],
                    finish_reason="stop",
                ),
            ]
        )
        engine = RunEngine(
            game=SensitiveInvalidActionGame(),
            agent=agent,
            run_dir=tmp_path / "run",
            run_id="sensitive-invalid-action-test",
            max_turns=1,
            max_invalid_attempts=1,
        )

        result = await engine.run()

        assert result.success
        retry_tool_result = next(
            message
            for message in agent.requests[1].messages
            if message.get("role") == "tool" and message.get("tool_call_id") == "first"
        )
        feedback = str(retry_tool_result["content"])
        assert feedback.startswith("Validation error: invalid_action")
        assert "Details:" in feedback
        assert feedback.endswith("Please try again.")
        for secret in (
            "sk-detail-secret",
            "Bearer-detail-secret",
            "sk-details-secret",
            "Bearer details-secret",
            "top-detail",
            "credential-detail",
        ):
            assert secret not in feedback

    async def test_transcripts_accumulate_within_scope_and_isolate_actors(
        self, tmp_path: Path
    ) -> None:
        agent_a = _ScopedTranscriptAgent("a")
        agent_b = _ScopedTranscriptAgent("b")
        engine = RunEngine(
            game=_ScopedTranscriptGame(),
            agents={"a": agent_a, "b": agent_b},
            run_dir=tmp_path / "run",
            run_id="scope-test",
            seed=42,
            matches=1,
            max_turns=10,
        )

        result = await engine.run()

        assert result.success
        for actor, agent, other_actor, first_index, second_index in (
            ("a", agent_a, "b", 0, 2),
            ("b", agent_b, "a", 1, 3),
        ):
            action_requests = [request for request in agent.requests if request.tools]
            assert [len(request.messages) for request in action_requests] == [1, 5]

            second_action_messages = action_requests[1].messages
            assert [message["role"] for message in second_action_messages] == [
                "user",
                "assistant",
                "tool",
                "assistant",
                "user",
            ]
            assert second_action_messages[0]["content"] == (
                f"{actor} observation {first_index}"
            )
            assert second_action_messages[1]["tool_calls"][0]["function"]["name"] == (
                "act"
            )
            assert second_action_messages[2]["tool_call_id"] == f"{actor}-call-0"
            assert second_action_messages[3]["content"] == f"{actor} final 0"
            assert second_action_messages[4]["content"] == (
                f"{actor} observation {second_index}"
            )

            other_markers = (
                f"{other_actor} observation",
                f"{other_actor} action",
                f"{other_actor} final",
                f"{other_actor}-call",
            )
            assert all(
                not any(marker in json.dumps(message) for marker in other_markers)
                for request in agent.requests
                for message in request.messages
            )

        events = _read_events(tmp_path / "run")
        model_requests = [e for e in events if e["event_type"] == "model_request"]
        action_model_requests = [
            event for event in model_requests if event["payload"]["tools"]
        ]
        assert [event["actor_id"] for event in action_model_requests] == [
            "a",
            "b",
            "a",
            "b",
        ]
        assert [event["payload"]["messages"] for event in action_model_requests] == [
            [
                {"role": "system", "content": "Act."},
                {"role": "user", "content": "a observation 0"},
            ],
            [
                {"role": "system", "content": "Act."},
                {"role": "user", "content": "b observation 1"},
            ],
            [
                {"role": "system", "content": "Act."},
                {"role": "user", "content": "a observation 0"},
                {
                    "role": "assistant",
                    "content": "a action 0",
                    "tool_calls": [
                        {
                            "id": "a-call-0",
                            "type": "function",
                            "function": {
                                "name": "act",
                                "arguments": "{}",
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": (
                        '{"summary": "a action 0", "metrics": {}, "terminal": false}'
                    ),
                    "tool_call_id": "a-call-0",
                },
                {"role": "assistant", "content": "a final 0"},
                {"role": "user", "content": "a observation 2"},
            ],
            [
                {"role": "system", "content": "Act."},
                {"role": "user", "content": "b observation 1"},
                {
                    "role": "assistant",
                    "content": "b action 0",
                    "tool_calls": [
                        {
                            "id": "b-call-0",
                            "type": "function",
                            "function": {
                                "name": "act",
                                "arguments": "{}",
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": (
                        '{"summary": "b action 1", "metrics": {}, "terminal": false}'
                    ),
                    "tool_call_id": "b-call-0",
                },
                {"role": "assistant", "content": "b final 0"},
                {"role": "user", "content": "b observation 3"},
            ],
        ]

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
        agent_a = FakeAgent(tool_name="act")
        agent_b = FakeAgent(tool_name="act")
        engine = RunEngine(
            game=TinyGame(),
            agents={"a": agent_a, "b": agent_b},
            run_dir=tmp_path / "run",
            run_id="test-run",
            seed=42,
            matches=1,
            max_turns=20,
        )
        await engine.run()

        assert agent_a.requests and agent_b.requests
        assert all(req.system_prompt for req in agent_a.requests + agent_b.requests)
        assert all(
            req.actor_id in ("a", "b") for req in agent_a.requests + agent_b.requests
        )
        assert all("secret_a" in req.messages[0]["content"] for req in agent_a.requests)
        assert all("secret_b" in req.messages[0]["content"] for req in agent_b.requests)
        assert all(
            "secret_b" not in msg.get("content", "")
            for req in agent_a.requests
            for msg in req.messages
            if isinstance(msg, dict)
        )
        assert all(
            "secret_a" not in msg.get("content", "")
            for req in agent_b.requests
            for msg in req.messages
            if isinstance(msg, dict)
        )

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

    async def test_action_tool_call_is_in_transcript_before_apply_action(
        self, tmp_path: Path
    ) -> None:
        engine: RunEngine | None = None

        class InspectingSession:
            def __init__(self) -> None:
                self.actions = 0

            @property
            def current_actor_id(self) -> str:
                return "a"

            def get_observation(self, actor_id: str) -> Observation:
                return Observation(actor_id=actor_id, text="Take an action.")

            def get_tools(self, actor_id: str) -> list[ToolSpec]:
                return [
                    ToolSpec(
                        name="act",
                        description="Act",
                        parameters={"type": "object", "properties": {}},
                    )
                ]

            def apply_action(
                self, actor_id: str, tool_name: str, arguments: dict[str, Any]
            ) -> Transition:
                assert engine is not None
                assistant_message = engine._actor_transcripts[actor_id][-1]
                assert assistant_message["role"] == "assistant"
                assert assistant_message["tool_calls"][0]["id"] == "action"
                self.actions += 1
                return Transition(summary="acted")

            @property
            def is_terminal(self) -> bool:
                return self.actions >= 1

            def get_result(self):
                from benchtable.contracts import GameResult

                return GameResult(completed=True, outcome={})

        class InspectingGame:
            @property
            def name(self) -> str:
                return "inspecting"

            @property
            def version(self) -> str:
                return "1"

            def system_prompt(self, actor_id: str) -> str:
                return "Act."

            def create_session(
                self, *, seed: int, game_config: dict[str, Any] | None = None
            ) -> InspectingSession:
                return InspectingSession()

        agent = _ResponseSequenceAgent(
            [
                ModelResponse(
                    assistant_text="Acting.",
                    tool_calls=[ToolCall(call_id="action", name="act", arguments={})],
                    finish_reason="tool_calls",
                ),
                ModelResponse(
                    assistant_text="Finished.",
                    tool_calls=[],
                    finish_reason="stop",
                ),
            ]
        )
        engine = RunEngine(
            game=InspectingGame(),
            agent=agent,
            run_dir=tmp_path / "run",
            run_id="action-transcript-order-test",
            max_turns=1,
        )

        result = await engine.run()

        assert result.success

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
                if not request.tools:
                    return ModelResponse(
                        assistant_text="", tool_calls=[], finish_reason="stop"
                    )
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

    async def test_provider_error_retains_redacted_raw_provider_response(
        self, tmp_path: Path
    ) -> None:
        raw_provider_response = {
            "client_token": "engine-client-secret",
            "error_text": "api_secret=engine-api-secret",
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "session_token=stable-call-id",
                                "type": "function",
                                "function": {"name": "act", "arguments": "{}"},
                            }
                        ]
                    }
                }
            ],
        }

        engine = RunEngine(
            game=TinyGame(),
            agent=FakeAgent(
                raise_on_respond=ProviderError(
                    "malformed provider response",
                    provider="test",
                    model="test",
                    raw_provider_response=raw_provider_response,
                )
            ),
            run_dir=tmp_path / "run",
            run_id="raw-provider-error-test",
            max_turns=1,
            max_provider_retries=0,
        )

        result = await engine.run()

        assert not result.success
        events = _read_events(tmp_path / "run")
        provider_error = next(
            event for event in events if event["event_type"] == "provider_error"
        )
        assert provider_error["payload"]["raw_provider_response"] == {
            "client_token": "[REDACTED]",
            "error_text": "api_secret=[REDACTED]",
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "session_token=stable-call-id",
                                "type": "function",
                                "function": {"name": "act", "arguments": "{}"},
                            }
                        ]
                    }
                }
            ],
        }

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
        result = await engine.run()

        events = _read_events(tmp_path / "run")
        match_ends = [e for e in events if e["event_type"] == "match_end"]
        assert not result.success
        assert match_ends[0]["payload"]["result"]["completed"] is False
        assert match_ends[0]["payload"]["result"]["outcome"]["total_actions"] == 3
        assert match_ends[0]["payload"]["result"]["metrics"]["seed"]
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

    async def test_exact_mapping_capability_failure_emits_match_end(
        self, tmp_path: Path
    ) -> None:
        class BrokenMappingGame(TinyGame):
            @property
            def requires_exact_agent_ids(self) -> bool:
                raise RuntimeError("mapping capability unavailable")

        agent = FakeAgent(tool_name="act")
        engine = RunEngine(
            game=BrokenMappingGame(),
            agent=agent,
            run_dir=tmp_path / "run",
            run_id="broken-mapping-test",
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
        match_end = next(
            event for event in events if event["event_type"] == "match_end"
        )
        assert match_end["payload"]["failure_reason"] == "exact_agent_mapping"

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

    @pytest.mark.parametrize(
        ("assistant_text", "finish_reason", "succeeds"),
        [
            ("", "stop", True),
            ("", None, False),
            ("", "", False),
            ("", "   ", False),
            ("Finalized.", None, True),
        ],
    )
    async def test_finalization_requires_text_or_finish_reason(
        self,
        tmp_path: Path,
        assistant_text: str,
        finish_reason: str | None,
        succeeds: bool,
    ) -> None:
        agent = _ResponseSequenceAgent(
            [
                ModelResponse(
                    assistant_text="Acting.",
                    tool_calls=[ToolCall(call_id="action", name="act", arguments={})],
                    finish_reason="tool_calls",
                ),
                ModelResponse(
                    assistant_text=assistant_text,
                    tool_calls=[],
                    finish_reason=finish_reason,
                ),
            ]
        )
        engine = RunEngine(
            game=TinyGame(max_actions=1),
            agent=agent,
            run_dir=tmp_path / "run",
            run_id="finalization-content-test",
            max_turns=1,
        )

        result = await engine.run()

        assert result.success is succeeds
        if not succeeds:
            events = _read_events(tmp_path / "run")
            assert any(
                event["event_type"] == "match_end"
                and event["payload"].get("failure_reason")
                == "finalization_empty_response"
                for event in events
            )

    async def test_read_memory_with_game_action_reports_generic_validation_error(
        self, tmp_path: Path
    ) -> None:
        mixed_response = ModelResponse(
            assistant_text="I'll read and act.",
            tool_calls=[
                ToolCall(call_id="read", name="read_memory", arguments={}),
                ToolCall(call_id="action", name="act", arguments={}),
            ],
            finish_reason="tool_calls",
        )
        action_response = ModelResponse(
            assistant_text="Acting now.",
            tool_calls=[ToolCall(call_id="fixed", name="act", arguments={})],
            finish_reason="tool_calls",
        )
        final_response = ModelResponse(
            assistant_text="Finished.",
            tool_calls=[],
            finish_reason="stop",
        )
        agent = _ResponseSequenceAgent(
            [mixed_response, action_response, final_response]
        )
        engine = RunEngine(
            game=TinyGame(max_actions=1),
            agent=agent,
            run_dir=tmp_path / "run",
            run_id="mixed-memory-action-test",
            max_turns=1,
            max_invalid_attempts=1,
        )

        result = await engine.run()

        assert result.success
        events = _read_events(tmp_path / "run")
        invalid_validations = [
            event
            for event in events
            if event["event_type"] == "validation"
            and event["payload"].get("valid") is False
        ]
        assert invalid_validations[0]["payload"]["error"] == ("memory_with_game_action")

    async def test_finalization_tool_call_fails_without_retry(
        self, tmp_path: Path
    ) -> None:
        agent = _ResponseSequenceAgent(
            [
                ModelResponse(
                    assistant_text="Acting.",
                    tool_calls=[ToolCall(call_id="action", name="act", arguments={})],
                    finish_reason="tool_calls",
                ),
                ModelResponse(
                    assistant_text="I will act again.",
                    tool_calls=[ToolCall(call_id="final", name="act", arguments={})],
                    finish_reason="tool_calls",
                ),
                ModelResponse(
                    assistant_text="Should not be requested.",
                    tool_calls=[],
                    finish_reason="stop",
                ),
            ]
        )
        engine = RunEngine(
            game=TinyGame(max_actions=1),
            agent=agent,
            run_dir=tmp_path / "run",
            run_id="finalization-tool-test",
            max_turns=1,
            max_invalid_attempts=2,
        )

        result = await engine.run()

        assert not result.success
        assert len(agent.requests) == 2
        events = _read_events(tmp_path / "run")
        assert any(
            event["event_type"] == "validation"
            and event["payload"].get("error") == "finalization_tool_calls"
            for event in events
        )

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

    async def test_retry_history_redacts_invalid_assistant_and_tool_arguments(
        self, tmp_path: Path
    ) -> None:
        invalid = ModelResponse(
            assistant_text=(
                "I tried api_key=sk-assistant-secret "
                "authorization=Bearer-assistant-secret"
            ),
            tool_calls=[
                ToolCall(
                    call_id="invalid-id",
                    name="nonexistent",
                    arguments={},
                    raw_arguments=(
                        '{"api_key":"sk-argument-secret",'
                        '"authorization":"Bearer argument-secret"}'
                    ),
                )
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
            run_id="redacted-retry-history-test",
            max_turns=1,
            max_invalid_attempts=1,
        )

        result = await engine.run()

        assert result.success
        retry_messages = agent.requests[1].messages
        retry_assistant = retry_messages[1]
        assert retry_assistant["role"] == "assistant"
        assert retry_assistant["tool_calls"][0]["id"] == "invalid-id"
        assert retry_assistant["tool_calls"][0]["type"] == "function"
        assert retry_assistant["tool_calls"][0]["function"]["name"] == "nonexistent"
        retry_text = json.dumps(retry_messages)
        for secret in (
            "sk-assistant-secret",
            "Bearer-assistant-secret",
            "sk-argument-secret",
            "Bearer argument-secret",
        ):
            assert secret not in retry_text
        assert retry_assistant["content"] == (
            "I tried api_key=[REDACTED] authorization=[REDACTED]"
        )
        assert retry_assistant["tool_calls"][0]["function"]["arguments"] == (
            '{"api_key":"[REDACTED]","authorization":"[REDACTED]"}'
        )

    async def test_retry_history_redacts_prefix_tokens_after_internal_quotes(
        self, tmp_path: Path
    ) -> None:
        invalid = ModelResponse(
            assistant_text='Bearer assistant-prefix"assistant-secret-suffix',
            tool_calls=[
                ToolCall(
                    call_id="invalid-id",
                    name="nonexistent",
                    arguments={},
                    raw_arguments='sk-argument-prefix"argument-secret-suffix',
                )
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
            run_id="prefix-retry-history-test",
            max_turns=1,
            max_invalid_attempts=1,
        )

        result = await engine.run()

        assert result.success
        retry_assistant = agent.requests[1].messages[1]
        assert retry_assistant["content"] == "Bearer [REDACTED]"
        assert retry_assistant["tool_calls"][0]["function"]["arguments"] == (
            "sk-[REDACTED]"
        )

    def test_retry_history_redacts_content_but_preserves_protocol_values(
        self, tmp_path: Path
    ) -> None:
        call_id = "private_key=call-id-secret"
        tool_name = "credentials=tool-name-secret"
        request = AgentRequest(
            actor_id="a",
            system_prompt="Act.",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"text": "private_key=prior-content-secret"},
                        {"nested": {"credentials": "nested-content-secret"}},
                    ],
                }
            ],
            tools=[],
        )
        response = ModelResponse(
            assistant_text="credentials=assistant-secret",
            tool_calls=[
                ToolCall(
                    call_id=call_id,
                    name=tool_name,
                    arguments={"nested": {"private_key": "argument-secret"}},
                    raw_arguments='{"nested":{"private_key":"argument-secret"}}',
                )
            ],
            finish_reason="stop",
        )
        engine = RunEngine(
            game=TinyGame(),
            agent=FakeAgent(tool_name="act"),
            run_dir=tmp_path / "run",
        )

        retry_request = engine._retry_request(request, response, "invalid_action")

        retry_assistant = retry_request.messages[1]
        retry_tool = retry_request.messages[2]
        assert retry_request.messages[0]["content"] == [
            {"text": "private_key=[REDACTED]"},
            {"nested": {"credentials": "[REDACTED]"}},
        ]
        assert retry_assistant["role"] == "assistant"
        assert retry_assistant["tool_calls"][0]["id"] == call_id
        assert retry_assistant["tool_calls"][0]["function"]["name"] == tool_name
        assert retry_assistant["content"] == "credentials=[REDACTED]"
        assert retry_assistant["tool_calls"][0]["function"]["arguments"] == (
            '{"nested":{"private_key":"[REDACTED]"}}'
        )
        assert retry_tool["role"] == "tool"
        assert retry_tool["tool_call_id"] == call_id

    def test_retry_current_assistant_json_content_remains_parseable(
        self, tmp_path: Path
    ) -> None:
        request = AgentRequest(
            actor_id="a",
            system_prompt="Act.",
            messages=[],
            tools=[],
        )
        response = ModelResponse(
            assistant_text='{"message":"api_key=current-secret","safe":"keep"}',
            tool_calls=[],
        )
        engine = RunEngine(
            game=TinyGame(),
            agent=FakeAgent(tool_name="act"),
            run_dir=tmp_path / "run",
        )

        retry_request = engine._retry_request(request, response, "no_tool_calls")

        assert json.loads(retry_request.messages[-2]["content"]) == {
            "message": "api_key=[REDACTED]",
            "safe": "keep",
        }

    def test_retry_history_fails_closed_for_malformed_quoted_credentials(
        self, tmp_path: Path
    ) -> None:
        message = {
            "role": "assistant",
            "content": r"Bearer \"assistant-secret trailing text",
            "tool_calls": [
                {
                    "id": "call-id",
                    "type": "function",
                    "function": {
                        "name": "act",
                        "arguments": r'api_key=abc"argument-secret trailing text',
                    },
                }
            ],
        }

        redacted = RunEngine._redact_retry_message(message)

        assert redacted["content"] == "Bearer [REDACTED]"
        assert redacted["tool_calls"][0]["function"]["arguments"] == (
            "api_key=[REDACTED]"
        )
        retry_text = json.dumps(redacted)
        assert "assistant-secret" not in retry_text
        assert "argument-secret" not in retry_text

    def test_redact_retry_message_recurses_through_structured_arguments(
        self, tmp_path: Path
    ) -> None:
        message = {
            "role": "assistant",
            "content": "api_key=content-secret",
            "tool_calls": [
                {
                    "id": "api_key=call-id-secret",
                    "type": "function",
                    "function": {
                        "name": "credentials=tool-name-secret",
                        "arguments": {
                            "api_key": "argument-secret",
                            "nested": [
                                {"access_token": "nested-secret"},
                                "password=string-secret",
                            ],
                            "safe": "preserved",
                        },
                    },
                }
            ],
        }

        redacted = RunEngine._redact_retry_message(message)

        assert redacted == {
            "role": "assistant",
            "content": "api_key=[REDACTED]",
            "tool_calls": [
                {
                    "id": "api_key=call-id-secret",
                    "type": "function",
                    "function": {
                        "name": "credentials=tool-name-secret",
                        "arguments": {
                            "api_key": "[REDACTED]",
                            "nested": [
                                {"access_token": "[REDACTED]"},
                                "password=[REDACTED]",
                            ],
                            "safe": "preserved",
                        },
                    },
                }
            ],
        }

    def test_redact_retry_message_redacts_nested_raw_json_arguments(self) -> None:
        message = {
            "role": "assistant",
            "content": "Choose.",
            "tool_calls": [
                {
                    "id": "call-id",
                    "type": "function",
                    "function": {
                        "name": "act",
                        "arguments": (
                            '{"api_key":{"nested":{"password":123}},'
                            '"safe":[1,{"access_token":456}]}'
                        ),
                    },
                }
            ],
        }

        redacted = RunEngine._redact_retry_message(message)
        function = redacted["tool_calls"][0]["function"]

        assert json.loads(function["arguments"]) == {
            "api_key": "[REDACTED]",
            "safe": [1, {"access_token": "[REDACTED]"}],
        }
        assert redacted["role"] == "assistant"
        assert redacted["tool_calls"][0]["id"] == "call-id"
        assert function["name"] == "act"

    def test_redact_retry_message_preserves_valid_json_string_content(self) -> None:
        message = {
            "role": "tool",
            "content": ('{"message":"api_key=content-secret","safe":"preserved"}'),
        }

        redacted = RunEngine._redact_retry_message(message)

        assert json.loads(redacted["content"]) == {
            "message": "api_key=[REDACTED]",
            "safe": "preserved",
        }

    def test_assistant_tool_message_redacts_raw_arguments_as_valid_json(self) -> None:
        response = ModelResponse(
            assistant_text='{"message":"api_key=assistant-secret","safe":"keep"}',
            tool_calls=[
                ToolCall(
                    call_id="call-id",
                    name="act",
                    arguments={"api_key": {"nested": 123}},
                    raw_arguments='{"api_key":{"nested":{"password":123}}}',
                )
            ],
        )

        message = RunEngine._assistant_tool_message(response, redact=True)
        raw_arguments = message["tool_calls"][0]["function"]["arguments"]

        assert json.loads(message["content"]) == {
            "message": "api_key=[REDACTED]",
            "safe": "keep",
        }
        assert json.loads(raw_arguments) == {
            "api_key": "[REDACTED]",
        }

    async def test_non_finite_retry_arguments_do_not_become_engine_failure(
        self, tmp_path: Path
    ) -> None:
        invalid = ModelResponse(
            assistant_text="Invalid arguments.",
            tool_calls=[
                ToolCall(
                    call_id="invalid-call",
                    name="act",
                    arguments=None,
                    raw_arguments=('{"api_key":"retry-secret","value":NaN}'),
                    parse_error="Tool arguments must be a JSON object",
                )
            ],
            finish_reason="tool_calls",
        )
        valid = ModelResponse(
            assistant_text="Valid arguments.",
            tool_calls=[ToolCall(call_id="valid-call", name="act", arguments={})],
            finish_reason="tool_calls",
        )
        agent = FakeAgent(responses=[invalid, valid])
        engine = RunEngine(
            game=TinyGame(max_actions=1),
            agent=agent,
            run_dir=tmp_path / "run",
            run_id="non-finite-retry-test",
            max_turns=1,
            max_invalid_attempts=1,
        )

        result = await engine.run()

        assert result.success
        retry_text = json.dumps(agent.requests[1].messages)
        assert "retry-secret" not in retry_text
        assert not any(
            event["event_type"] == "engine_error"
            for event in _read_events(tmp_path / "run")
        )

    @pytest.mark.parametrize("escaped_key", [r"\u0061pi_key", r"ref\u0072esh_token"])
    async def test_retry_redacts_unicode_escaped_keys_in_malformed_json(
        self, tmp_path: Path, escaped_key: str
    ) -> None:
        invalid = ModelResponse(
            assistant_text="Invalid arguments.",
            tool_calls=[
                ToolCall(
                    call_id="invalid-call",
                    name="act",
                    arguments=None,
                    raw_arguments=rf'{{"{escaped_key}":"retry-secret",}}',
                    parse_error="Tool arguments must be a JSON object",
                )
            ],
            finish_reason="tool_calls",
        )
        valid = ModelResponse(
            assistant_text="Valid arguments.",
            tool_calls=[ToolCall(call_id="valid-call", name="act", arguments={})],
            finish_reason="tool_calls",
        )
        agent = FakeAgent(responses=[invalid, valid])
        engine = RunEngine(
            game=TinyGame(max_actions=1),
            agent=agent,
            run_dir=tmp_path / "run",
            run_id="unicode-escaped-retry-test",
            max_turns=1,
            max_invalid_attempts=1,
        )

        result = await engine.run()

        assert result.success
        retry_text = json.dumps(agent.requests[1].messages)
        assert "retry-secret" not in retry_text
        retry_arguments = agent.requests[1].messages[1]["tool_calls"][0]["function"][
            "arguments"
        ]
        assert retry_arguments == rf'{{"{escaped_key}":"[REDACTED]",}}'

    def test_tool_call_payload_redacts_raw_arguments_as_valid_json(self) -> None:
        tool_call = ToolCall(
            call_id="call-id",
            name="act",
            arguments={"api_key": {"nested": 123}},
            raw_arguments='{"api_key":{"nested":{"password":123}}}',
        )

        payload = RunEngine._tool_call_payload(tool_call)

        assert json.loads(payload["raw_arguments"]) == {
            "api_key": "[REDACTED]",
        }

    def test_retry_details_remain_valid_json_after_redaction(
        self, tmp_path: Path
    ) -> None:
        request = AgentRequest(
            actor_id="a",
            system_prompt="Act.",
            messages=[],
            tools=[],
        )
        response = ModelResponse(assistant_text="Try again.", tool_calls=[])
        engine = RunEngine(
            game=TinyGame(),
            agent=FakeAgent(tool_name="act"),
            run_dir=tmp_path / "run",
        )

        retry_request = engine._retry_request(
            request,
            response,
            "invalid_action",
            details={"api_key": {"nested": {"password": 123}}},
        )
        content = retry_request.messages[-1]["content"]
        serialized_details = content.split(" Details: ", 1)[1].removesuffix(
            ". Please try again."
        )

        assert json.loads(serialized_details) == {
            "api_key": "[REDACTED]",
        }

    def test_retry_history_redacts_prior_messages_without_changing_shape(
        self, tmp_path: Path
    ) -> None:
        prior_messages = [
            {
                "role": "user",
                "content": "Remember api_key=prior-secret.",
            },
            {
                "role": "assistant",
                "content": "Prior response.",
                "tool_calls": [
                    {
                        "id": "prior-call",
                        "type": "function",
                        "function": {
                            "name": "act",
                            "arguments": '{"api_key":"prior-argument-secret"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "content": "Prior result.",
                "tool_call_id": "prior-call",
            },
        ]
        request = AgentRequest(
            actor_id="a",
            system_prompt="Act.",
            messages=prior_messages,
            tools=[],
        )
        response = ModelResponse(
            assistant_text="Current api_key=current-secret.",
            tool_calls=[
                ToolCall(
                    call_id="current-call",
                    name="unknown",
                    arguments={},
                    raw_arguments='{"api_key":"current-argument-secret"}',
                )
            ],
            finish_reason="stop",
        )
        engine = RunEngine(
            game=TinyGame(),
            agent=FakeAgent(tool_name="act"),
            run_dir=tmp_path / "run",
        )

        retry_request = engine._retry_request(request, response, "invalid_action")

        assert request.messages == prior_messages
        assert [message["role"] for message in retry_request.messages] == [
            "user",
            "assistant",
            "tool",
            "assistant",
            "tool",
        ]
        retry_text = json.dumps(retry_request.messages)
        for secret in (
            "prior-secret",
            "prior-argument-secret",
            "current-secret",
            "current-argument-secret",
        ):
            assert secret not in retry_text
        assert retry_request.messages[1]["tool_calls"][0]["id"] == "prior-call"
        assert retry_request.messages[2]["tool_call_id"] == "prior-call"
        assert retry_request.messages[3]["tool_calls"][0]["id"] == "current-call"
        assert retry_request.messages[4]["tool_call_id"] == "current-call"

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

    async def test_invalid_action_recovery_preserves_matching_tool_result_history(
        self, tmp_path: Path
    ) -> None:
        class RecoverySession:
            def __init__(self) -> None:
                self.turn = 0

            @property
            def current_actor_id(self) -> str:
                return "a"

            def get_observation(self, actor_id: str) -> Observation:
                return Observation(
                    actor_id=actor_id,
                    text=f"observation {self.turn}",
                )

            def get_tools(self, actor_id: str) -> list[ToolSpec]:
                return [
                    ToolSpec(
                        name="act",
                        description="Act",
                        parameters={"type": "object", "properties": {}},
                    )
                ]

            def apply_action(
                self, actor_id: str, tool_name: str, arguments: dict[str, Any]
            ) -> Transition:
                if self.turn == 0:
                    raise InvalidActionError("recover this action")
                self.turn = 2
                return Transition(summary="action succeeded")

            def handle_failed_turn(self, actor_id: str, reason: str) -> Transition:
                self.turn = 1
                return Transition(summary="recovered")

            @property
            def is_terminal(self) -> bool:
                return self.turn >= 2

            def get_result(self) -> GameResult:
                return GameResult(completed=True, outcome={})

        class RecoveryGame:
            @property
            def name(self) -> str:
                return "recovery"

            @property
            def version(self) -> str:
                return "1"

            def system_prompt(self, actor_id: str) -> str:
                return "Act."

            def create_session(
                self, *, seed: int, game_config: dict[str, Any] | None = None
            ) -> RecoverySession:
                return RecoverySession()

        agent = _ResponseSequenceAgent(
            [
                ModelResponse(
                    assistant_text="Invalid action.",
                    tool_calls=[
                        ToolCall(call_id="failed-call", name="act", arguments={})
                    ],
                    finish_reason="tool_calls",
                ),
                ModelResponse(
                    assistant_text="Recovered action.",
                    tool_calls=[
                        ToolCall(call_id="recovered-call", name="act", arguments={})
                    ],
                    finish_reason="tool_calls",
                ),
                ModelResponse(
                    assistant_text="Finished.",
                    tool_calls=[],
                    finish_reason="stop",
                ),
            ]
        )
        engine = RunEngine(
            game=RecoveryGame(),
            agent=agent,
            run_dir=tmp_path / "run",
            run_id="invalid-action-recovery-history-test",
            max_turns=2,
            max_invalid_attempts=0,
        )

        result = await engine.run()

        assert result.success
        next_request_messages = agent.requests[1].messages
        assert [message["role"] for message in next_request_messages] == [
            "user",
            "assistant",
            "tool",
            "user",
        ]
        assert next_request_messages[0]["content"] == "observation 0"
        assert next_request_messages[3]["content"] == "observation 1"
        assistant_messages = [
            message
            for message in next_request_messages
            if message["role"] == "assistant"
        ]
        assert len(assistant_messages) == 1
        failed_assistant = next_request_messages[1]
        failed_tool_result = next_request_messages[2]
        assert failed_assistant["tool_calls"][0]["id"] == "failed-call"
        assert failed_tool_result["tool_call_id"] == "failed-call"
        assert "Validation error" in failed_tool_result["content"]

    @pytest.mark.parametrize(
        ("invalid_response", "call_ids"),
        [
            (
                ModelResponse(
                    assistant_text="Malformed action.",
                    tool_calls=[
                        ToolCall(
                            call_id="malformed-call",
                            name="act",
                            arguments=None,
                            raw_arguments="not-json",
                            parse_error="invalid json",
                        )
                    ],
                    finish_reason="tool_calls",
                ),
                ["malformed-call"],
            ),
            (
                ModelResponse(
                    assistant_text="Two actions.",
                    tool_calls=[
                        ToolCall(call_id="first-call", name="act", arguments={}),
                        ToolCall(call_id="second-call", name="act", arguments={}),
                    ],
                    finish_reason="tool_calls",
                ),
                ["first-call", "second-call"],
            ),
            (
                ModelResponse(
                    assistant_text="Unknown action.",
                    tool_calls=[
                        ToolCall(call_id="unknown-call", name="unknown", arguments={})
                    ],
                    finish_reason="tool_calls",
                ),
                ["unknown-call"],
            ),
            (
                ModelResponse(
                    assistant_text="Memory and action.",
                    tool_calls=[
                        ToolCall(
                            call_id="memory-call",
                            name="read_memory",
                            arguments={},
                        ),
                        ToolCall(call_id="mixed-action-call", name="act", arguments={}),
                    ],
                    finish_reason="tool_calls",
                ),
                ["memory-call", "mixed-action-call"],
            ),
        ],
        ids=["malformed", "multiple", "unknown", "mixed"],
    )
    async def test_exhausted_pre_action_recovery_preserves_tool_history(
        self,
        tmp_path: Path,
        invalid_response: ModelResponse,
        call_ids: list[str],
    ) -> None:
        agent = _ResponseSequenceAgent(
            [
                invalid_response,
                ModelResponse(
                    assistant_text="Recovered action.",
                    tool_calls=[
                        ToolCall(call_id="recovered", name="act", arguments={})
                    ],
                    finish_reason="tool_calls",
                ),
                ModelResponse(
                    assistant_text="Finished.",
                    tool_calls=[],
                    finish_reason="stop",
                ),
            ]
        )
        engine = RunEngine(
            game=TinyGame(max_actions=1, failed_turn_recovery=True),
            agent=agent,
            run_dir=tmp_path / "run",
            run_id="exhausted-pre-action-recovery-test",
            max_turns=2,
            max_invalid_attempts=0,
        )

        result = await engine.run()

        assert result.success
        next_request_messages = agent.requests[1].messages
        assert [message["role"] for message in next_request_messages] == [
            "user",
            "assistant",
            "tool",
            *(["tool"] if len(call_ids) == 2 else []),
            "user",
        ]
        assistant_messages = [
            message
            for message in next_request_messages
            if message["role"] == "assistant"
        ]
        tool_messages = [
            message for message in next_request_messages if message["role"] == "tool"
        ]
        assert len(assistant_messages) == 1
        assert [
            tool_call["id"] for tool_call in assistant_messages[0]["tool_calls"]
        ] == call_ids
        assert [message["tool_call_id"] for message in tool_messages] == call_ids
        assert {
            tool_call["id"] for tool_call in assistant_messages[0]["tool_calls"]
        } == {message["tool_call_id"] for message in tool_messages}
        assert next_request_messages[-1]["role"] == "user"

    async def test_exhausted_no_tool_recovery_preserves_assistant_user_history(
        self, tmp_path: Path
    ) -> None:
        agent = _ResponseSequenceAgent(
            [
                ModelResponse(
                    assistant_text="No action.",
                    tool_calls=[],
                    finish_reason="stop",
                ),
                ModelResponse(
                    assistant_text="Recovered action.",
                    tool_calls=[
                        ToolCall(call_id="recovered", name="act", arguments={})
                    ],
                    finish_reason="tool_calls",
                ),
                ModelResponse(
                    assistant_text="Finished.",
                    tool_calls=[],
                    finish_reason="stop",
                ),
            ]
        )
        engine = RunEngine(
            game=TinyGame(max_actions=1, failed_turn_recovery=True),
            agent=agent,
            run_dir=tmp_path / "run",
            run_id="exhausted-no-tool-recovery-test",
            max_turns=2,
            max_invalid_attempts=0,
        )

        result = await engine.run()

        assert result.success
        next_request_messages = agent.requests[1].messages
        assert [message["role"] for message in next_request_messages] == [
            "user",
            "assistant",
            "user",
            "user",
        ]
        assert next_request_messages[1]["content"] == "No action."
        assert "Validation error" in next_request_messages[2]["content"]

    async def test_invalid_output_budget_is_shared_across_response_phases(
        self, tmp_path: Path
    ) -> None:
        agent = _ResponseSequenceAgent(
            [
                ModelResponse(
                    assistant_text="No action.",
                    tool_calls=[],
                    finish_reason="stop",
                ),
                ModelResponse(
                    assistant_text="Memory and action.",
                    tool_calls=[
                        ToolCall(call_id="memory", name="read_memory", arguments={}),
                        ToolCall(call_id="mixed-action", name="act", arguments={}),
                    ],
                    finish_reason="tool_calls",
                ),
                ModelResponse(
                    assistant_text="Unknown action.",
                    tool_calls=[
                        ToolCall(call_id="unknown", name="unknown", arguments={})
                    ],
                    finish_reason="tool_calls",
                ),
                ModelResponse(
                    assistant_text="Recovered action.",
                    tool_calls=[
                        ToolCall(call_id="recovered", name="act", arguments={})
                    ],
                    finish_reason="tool_calls",
                ),
                ModelResponse(
                    assistant_text="Finished.",
                    tool_calls=[],
                    finish_reason="stop",
                ),
            ]
        )
        engine = RunEngine(
            game=TinyGame(max_actions=1, failed_turn_recovery=True),
            agent=agent,
            run_dir=tmp_path / "run",
            run_id="shared-invalid-budget-test",
            max_turns=2,
            max_invalid_attempts=1,
        )

        result = await engine.run()

        assert result.success
        events = _read_events(tmp_path / "run")
        validation_events = [
            event
            for event in events
            if event["event_type"] == "validation"
            and event["payload"].get("valid") is False
        ]
        recovery_index = next(
            index
            for index, event in enumerate(events)
            if event["event_type"] == "transition"
            and event["payload"].get("recovery") is True
        )
        unknown_validation_index = next(
            index
            for index, event in enumerate(events)
            if event["event_type"] == "validation"
            and event["payload"].get("error") == "invalid_action"
        )
        assert [event["payload"]["error"] for event in validation_events] == [
            "no_tool_calls",
            "memory_with_game_action",
            "invalid_action",
        ]
        assert recovery_index < unknown_validation_index

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
