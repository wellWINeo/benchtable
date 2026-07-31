"""Tests for the engine's generic memory tool interaction loop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tests.fixtures.fake_agent import FakeAgent
from tests.fixtures.tiny_game import TinyGame

from benchtable.contracts import (
    GameResult,
    MatchMemorySummary,
    ModelResponse,
    Observation,
    ToolCall,
    ToolSpec,
    Transition,
)
from benchtable.engine import RunEngine
from benchtable.games.poker.session import PokerGame


def _read_events(path: Path) -> list[dict[str, Any]]:
    raw = (path / "events.jsonl").read_text()
    return [json.loads(line) for line in raw.splitlines() if line]


class _InitialSummarySession:
    def __init__(self, *, terminal: bool = False) -> None:
        self.actions = 1 if terminal else 0
        self.pending = [
            MatchMemorySummary(
                actor_id="a",
                text="pre-existing public summary",
                hand=0,
                turn=0,
            )
        ]

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
        self, actor_id: str, tool_name: str, arguments: dict[str, object]
    ) -> Transition:
        self.actions = 1
        return Transition(summary="acted")

    def drain_match_memory_summaries(self) -> list[MatchMemorySummary]:
        summaries = list(self.pending)
        self.pending.clear()
        return summaries

    @property
    def is_terminal(self) -> bool:
        return self.actions >= 1

    def get_result(self) -> GameResult:
        return GameResult(completed=True, outcome={})


class _InitialSummaryGame:
    def __init__(self, *, terminal: bool = False) -> None:
        self.terminal = terminal

    @property
    def name(self) -> str:
        return "initial-summary"

    @property
    def version(self) -> str:
        return "1"

    def system_prompt(self, actor_id: str) -> str:
        return "Act."

    def create_session(
        self, *, seed: int, game_config: dict[str, object] | None = None
    ) -> _InitialSummarySession:
        return _InitialSummarySession(terminal=self.terminal)


class TestMemoryInteraction:
    async def test_initial_summary_is_readable_on_first_turn(
        self, tmp_path: Path
    ) -> None:
        agent = FakeAgent(
            responses=[
                ModelResponse(
                    assistant_text="Read memory.",
                    tool_calls=[
                        ToolCall(call_id="read", name="read_memory", arguments={})
                    ],
                    finish_reason="stop",
                ),
                ModelResponse(
                    assistant_text="Act.",
                    tool_calls=[ToolCall(call_id="act", name="act", arguments={})],
                    finish_reason="stop",
                ),
                ModelResponse(
                    assistant_text="Finished.",
                    tool_calls=[],
                    finish_reason="stop",
                ),
            ]
        )
        engine = RunEngine(
            game=_InitialSummaryGame(),
            agents={"a": agent},
            run_dir=tmp_path / "run",
            run_id="initial-summary-read-test",
            max_turns=1,
        )

        result = await engine.run()

        assert result.success
        memory_result = next(
            message
            for message in agent.requests[1].messages
            if message.get("role") == "tool" and message.get("tool_call_id") == "read"
        )
        entries = json.loads(str(memory_result["content"]))
        assert entries[0]["text"] == "pre-existing public summary"
        events = _read_events(tmp_path / "run")
        summary_event = next(
            event
            for event in events
            if event["event_type"] == "memory_operation"
            and event["payload"]["operation"] == "summary"
        )
        assert summary_event["actor_id"] == "a"
        assert summary_event.get("turn_index") is None

    async def test_initial_terminal_summary_is_stored_before_match_end(
        self, tmp_path: Path
    ) -> None:
        engine = RunEngine(
            game=_InitialSummaryGame(terminal=True),
            agents={"a": FakeAgent(tool_name="act")},
            run_dir=tmp_path / "run",
            run_id="initial-terminal-summary-test",
            max_turns=1,
        )

        result = await engine.run()

        assert result.success
        events = _read_events(tmp_path / "run")
        summary_index = next(
            index
            for index, event in enumerate(events)
            if event["event_type"] == "memory_operation"
            and event["payload"]["operation"] == "summary"
        )
        match_end_index = next(
            index
            for index, event in enumerate(events)
            if event["event_type"] == "match_end"
        )
        assert summary_index < match_end_index
        summary_event = events[summary_index]
        assert summary_event["actor_id"] == "a"
        assert summary_event["payload"]["text"] == "pre-existing public summary"

    async def test_memory_only_write_uses_session_turn_context(
        self, tmp_path: Path
    ) -> None:
        class ContextSession:
            def __init__(self) -> None:
                self.actions = 0

            @property
            def current_actor_id(self) -> str:
                return "a"

            def get_turn_context(self) -> dict[str, int]:
                return {"hand_index": 7, "in_hand_turn": self.actions + 9}

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
                self, actor_id: str, tool_name: str, arguments: dict[str, object]
            ) -> Transition:
                self.actions += 1
                return Transition(summary="acted")

            @property
            def is_terminal(self) -> bool:
                return self.actions >= 3

            def get_result(self) -> GameResult:
                return GameResult(completed=True, outcome={}, metrics={})

        class ContextGame:
            @property
            def name(self) -> str:
                return "context"

            @property
            def version(self) -> str:
                return "1"

            def system_prompt(self, actor_id: str) -> str:
                return "Act."

            def create_session(
                self, *, seed: int, game_config: dict[str, object] | None = None
            ) -> ContextSession:
                return ContextSession()

        responses = [
            ModelResponse(
                assistant_text="Remember directly.",
                tool_calls=[
                    ToolCall(
                        call_id="write-1",
                        name="write_memory",
                        arguments={"text": "direct"},
                    ),
                ],
                finish_reason="stop",
            ),
            ModelResponse(
                assistant_text="Act now.",
                tool_calls=[ToolCall(call_id="act-1", name="act", arguments={})],
                finish_reason="stop",
            ),
            ModelResponse(
                assistant_text="Remember another note.",
                tool_calls=[
                    ToolCall(
                        call_id="write-2",
                        name="write_memory",
                        arguments={"text": "memory-only"},
                    )
                ],
                finish_reason="stop",
            ),
            ModelResponse(
                assistant_text="Remember one more.",
                tool_calls=[
                    ToolCall(
                        call_id="write-3",
                        name="write_memory",
                        arguments={"text": "after-memory"},
                    )
                ],
                finish_reason="stop",
            ),
            ModelResponse(
                assistant_text="Read notes.",
                tool_calls=[
                    ToolCall(
                        call_id="read",
                        name="read_memory",
                        arguments={},
                    )
                ],
                finish_reason="stop",
            ),
            ModelResponse(
                assistant_text="Act and finish.",
                tool_calls=[
                    ToolCall(call_id="act-2", name="act", arguments={}),
                ],
                finish_reason="stop",
            ),
        ]
        engine = RunEngine(
            game=ContextGame(),
            agent=FakeAgent(responses=responses),
            run_dir=tmp_path / "run",
            run_id="memory-context-test",
            max_turns=5,
        )

        result = await engine.run()

        assert result.success
        events = _read_events(tmp_path / "run")
        read_event = next(
            event
            for event in events
            if event["event_type"] == "memory_operation"
            and event["payload"]["operation"] == "read"
        )
        notes = read_event["payload"]["notes"]
        assert [(note["text"], note["hand"], note["turn"]) for note in notes] == [
            ("direct", 7, 9),
            ("memory-only", 7, 10),
            ("after-memory", 7, 10),
        ]

    async def test_poker_memory_context_uses_hand_and_in_hand_turn(
        self, tmp_path: Path
    ) -> None:
        class ContextAgent:
            def __init__(self) -> None:
                self.calls = 0

            async def respond(self, request: Any) -> ModelResponse:
                if not request.tools:
                    return ModelResponse(
                        assistant_text="", tool_calls=[], finish_reason="stop"
                    )
                self.calls += 1
                if self.calls == 1:
                    return ModelResponse(
                        assistant_text="Remembering.",
                        tool_calls=[
                            ToolCall(
                                call_id="write",
                                name="write_memory",
                                arguments={"text": "first hand note"},
                            )
                        ],
                        finish_reason="stop",
                    )
                if self.calls == 3:
                    return ModelResponse(
                        assistant_text="Reading.",
                        tool_calls=[
                            ToolCall(
                                call_id="read",
                                name="read_memory",
                                arguments={},
                            )
                        ],
                        finish_reason="stop",
                    )
                action = "call" if self.calls == 2 else "fold"
                return ModelResponse(
                    assistant_text="Acting.",
                    tool_calls=[
                        ToolCall(
                            call_id=f"action-{self.calls}",
                            name="poker_action",
                            arguments={"action": action},
                        )
                    ],
                    finish_reason="stop",
                )

        class CheckAgent:
            async def respond(self, request: Any) -> ModelResponse:
                if not request.tools:
                    return ModelResponse(
                        assistant_text="", tool_calls=[], finish_reason="stop"
                    )
                observation = str(request.messages[0]["content"])
                if "check" in observation:
                    action = "check"
                elif "call" in observation:
                    action = "call"
                else:
                    action = "fold"
                return ModelResponse(
                    assistant_text="Checking.",
                    tool_calls=[
                        ToolCall(
                            call_id="check",
                            name="poker_action",
                            arguments={"action": action},
                        )
                    ],
                    finish_reason="stop",
                )

        agent_a = ContextAgent()
        agent_b = CheckAgent()
        engine = RunEngine(
            game=PokerGame(),
            agents={"player-1": agent_a, "player-2": agent_b},
            game_config={"players": ["player-1", "player-2"], "hands_per_match": 1},
            run_dir=tmp_path / "run",
            run_id="context-test",
            max_turns=20,
        )

        result = await engine.run()

        assert result.success
        events = _read_events(tmp_path / "run")
        read_event = next(
            event
            for event in events
            if event["event_type"] == "memory_operation"
            and event["payload"]["operation"] == "read"
        )
        assert read_event["payload"]["notes"][0]["hand"] == 0
        assert read_event["payload"]["notes"][0]["turn"] == 0

    async def test_read_memory_followed_by_game_action(self, tmp_path: Path) -> None:
        """A read_memory call followed by one valid game action."""
        responses = [
            ModelResponse(
                assistant_text="Let me read my memory.",
                tool_calls=[ToolCall(call_id="c1", name="read_memory", arguments={})],
                finish_reason="stop",
            ),
            ModelResponse(
                assistant_text="I'll act now.",
                tool_calls=[ToolCall(call_id="c2", name="act", arguments={})],
                finish_reason="stop",
            ),
        ]
        engine = RunEngine(
            game=TinyGame(max_actions=1),
            agent=FakeAgent(responses=responses),
            run_dir=tmp_path / "run",
            run_id="test",
            max_turns=5,
            max_memory_operations_per_turn=4,
        )
        result = await engine.run()
        assert result.success

        events = _read_events(tmp_path / "run")
        memory_events = [e for e in events if e["event_type"] == "memory_operation"]
        assert len(memory_events) == 1
        assert memory_events[0]["payload"]["operation"] == "read"

    async def test_write_memory_only_followed_by_game_action(
        self, tmp_path: Path
    ) -> None:
        """A memory-only write followed by another model response and a valid action."""
        responses = [
            ModelResponse(
                assistant_text="I'll remember this.",
                tool_calls=[
                    ToolCall(
                        call_id="c1",
                        name="write_memory",
                        arguments={"text": "rarely bluffs"},
                    )
                ],
                finish_reason="stop",
            ),
            ModelResponse(
                assistant_text="Now I act.",
                tool_calls=[ToolCall(call_id="c2", name="act", arguments={})],
                finish_reason="stop",
            ),
        ]
        engine = RunEngine(
            game=TinyGame(max_actions=1),
            agent=FakeAgent(responses=responses),
            run_dir=tmp_path / "run",
            run_id="test",
            max_turns=5,
            max_memory_operations_per_turn=4,
        )
        result = await engine.run()
        assert result.success

        events = _read_events(tmp_path / "run")
        memory_events = [e for e in events if e["event_type"] == "memory_operation"]
        assert len(memory_events) == 1
        assert memory_events[0]["payload"]["operation"] == "write"

    async def test_multiple_memory_only_calls_are_processed_in_response_order(
        self, tmp_path: Path
    ) -> None:
        responses = [
            ModelResponse(
                assistant_text="I will record both notes.",
                tool_calls=[
                    ToolCall(
                        call_id="c1",
                        name="write_memory",
                        arguments={"text": "first"},
                    ),
                    ToolCall(
                        call_id="c2",
                        name="write_memory",
                        arguments={"text": "second"},
                    ),
                ],
                finish_reason="stop",
            ),
            ModelResponse(
                assistant_text="Now I act.",
                tool_calls=[ToolCall(call_id="c3", name="act", arguments={})],
                finish_reason="stop",
            ),
        ]
        agent = FakeAgent(responses=responses)
        engine = RunEngine(
            game=TinyGame(max_actions=1),
            agent=agent,
            run_dir=tmp_path / "run",
            run_id="test",
            max_turns=5,
            max_memory_operations_per_turn=4,
        )

        result = await engine.run()

        assert result.success
        memory_events = [
            event
            for event in _read_events(tmp_path / "run")
            if event["event_type"] == "memory_operation"
        ]
        assert [event["payload"]["operation"] for event in memory_events] == [
            "write",
            "write",
        ]
        assert len(agent.requests) == 3
        assert [message["role"] for message in agent.requests[1].messages] == [
            "user",
            "assistant",
            "tool",
            "tool",
        ]

    async def test_read_memory_combined_with_game_action_retries(
        self, tmp_path: Path
    ) -> None:
        """A read_memory combined with a game action should retry."""
        bad_response = ModelResponse(
            assistant_text="I'll read and act.",
            tool_calls=[
                ToolCall(call_id="c1", name="read_memory", arguments={}),
                ToolCall(call_id="c2", name="act", arguments={}),
            ],
            finish_reason="stop",
        )
        good_response = ModelResponse(
            assistant_text="Now I act.",
            tool_calls=[ToolCall(call_id="c3", name="act", arguments={})],
            finish_reason="stop",
        )
        engine = RunEngine(
            game=TinyGame(max_actions=1),
            agent=FakeAgent(responses=[bad_response, good_response]),
            run_dir=tmp_path / "run",
            run_id="test",
            max_turns=5,
            max_invalid_attempts=1,
            max_memory_operations_per_turn=4,
        )
        result = await engine.run()
        assert result.success

    async def test_retry_response_with_only_memory_calls_is_processed(
        self, tmp_path: Path
    ) -> None:
        responses = [
            ModelResponse(
                assistant_text="I'll read and act.",
                tool_calls=[
                    ToolCall(call_id="read-1", name="read_memory", arguments={}),
                    ToolCall(call_id="act-1", name="act", arguments={}),
                ],
                finish_reason="stop",
            ),
            ModelResponse(
                assistant_text="I'll remember first.",
                tool_calls=[
                    ToolCall(
                        call_id="write-1",
                        name="write_memory",
                        arguments={"text": "retry note"},
                    )
                ],
                finish_reason="stop",
            ),
            ModelResponse(
                assistant_text="Now I act.",
                tool_calls=[ToolCall(call_id="act-2", name="act", arguments={})],
                finish_reason="stop",
            ),
        ]
        engine = RunEngine(
            game=TinyGame(max_actions=1),
            agent=FakeAgent(responses=responses),
            run_dir=tmp_path / "run",
            run_id="memory-retry-test",
            max_turns=5,
            max_invalid_attempts=1,
        )

        result = await engine.run()

        assert result.success
        events = _read_events(tmp_path / "run")
        assert any(
            event["event_type"] == "memory_operation"
            and event["payload"]["operation"] == "write"
            for event in events
        )

    async def test_system_summaries_are_stored_and_read_after_scope_change(
        self, tmp_path: Path
    ) -> None:
        class SummarySession:
            def __init__(self) -> None:
                self._hand = 0
                self._pending: list[MatchMemorySummary] = []

            @property
            def current_actor_id(self) -> str:
                return "a"

            @property
            def conversation_scope_id(self) -> str:
                return f"hand-{self._hand}"

            def get_observation(self, actor_id: str) -> Observation:
                return Observation(actor_id=actor_id, text=f"Hand {self._hand}.")

            def get_tools(self, actor_id: str) -> list[ToolSpec]:
                return [
                    ToolSpec(
                        name="act",
                        description="Act",
                        parameters={"type": "object", "properties": {}},
                    )
                ]

            def apply_action(
                self, actor_id: str, tool_name: str, arguments: dict[str, object]
            ) -> Transition:
                self._pending.append(
                    MatchMemorySummary(
                        actor_id=actor_id,
                        text=f"summary for hand {self._hand}",
                        hand=self._hand,
                        turn=0,
                    )
                )
                self._hand += 1
                return Transition(summary="resolved")

            def drain_match_memory_summaries(self) -> list[MatchMemorySummary]:
                pending = list(self._pending)
                self._pending.clear()
                return pending

            @property
            def is_terminal(self) -> bool:
                return self._hand >= 2

            def get_result(self) -> GameResult:
                return GameResult(completed=True, outcome={}, metrics={})

        class SummaryGame:
            @property
            def name(self) -> str:
                return "summary"

            @property
            def version(self) -> str:
                return "1"

            def system_prompt(self, actor_id: str) -> str:
                return "Act."

            def create_session(
                self, *, seed: int, game_config: dict[str, object] | None = None
            ) -> SummarySession:
                return SummarySession()

        responses = [
            ModelResponse(
                assistant_text="Act.",
                tool_calls=[ToolCall(call_id="c1", name="act", arguments={})],
                finish_reason="stop",
            ),
            ModelResponse(assistant_text="Done.", tool_calls=[], finish_reason="stop"),
            ModelResponse(
                assistant_text="Read memory.",
                tool_calls=[ToolCall(call_id="c2", name="read_memory", arguments={})],
                finish_reason="stop",
            ),
            ModelResponse(
                assistant_text="Act again.",
                tool_calls=[ToolCall(call_id="c3", name="act", arguments={})],
                finish_reason="stop",
            ),
            ModelResponse(assistant_text="Done.", tool_calls=[], finish_reason="stop"),
        ]

        agent = FakeAgent(responses=responses)
        engine = RunEngine(
            game=SummaryGame(),
            agent=agent,
            run_dir=tmp_path / "run",
            run_id="summary-test",
            seed=42,
            matches=1,
            max_turns=10,
            max_memory_operations_per_turn=4,
        )

        result = await engine.run()

        assert result.success
        assert len(agent.requests) >= 3
        assert len(agent.requests[0].messages) == 1
        assert len(agent.requests[2].messages) == 1

        events = _read_events(tmp_path / "run")
        read_event = next(
            event
            for event in events
            if event["event_type"] == "memory_operation"
            and event["payload"]["operation"] == "read"
        )
        assert read_event["payload"]["notes"]
        assert read_event["payload"]["notes"][0]["source"] == "system"

    async def test_multiple_game_actions_retries(self, tmp_path: Path) -> None:
        """Multiple game actions in one response should retry."""
        bad_response = ModelResponse(
            assistant_text="Two game actions.",
            tool_calls=[
                ToolCall(call_id="c1", name="act", arguments={}),
                ToolCall(call_id="c2", name="act", arguments={}),
            ],
            finish_reason="stop",
        )
        good_response = ModelResponse(
            assistant_text="Fixed.",
            tool_calls=[ToolCall(call_id="c3", name="act", arguments={})],
            finish_reason="stop",
        )
        engine = RunEngine(
            game=TinyGame(max_actions=1),
            agent=FakeAgent(responses=[bad_response, good_response]),
            run_dir=tmp_path / "run",
            run_id="test",
            max_turns=5,
            max_invalid_attempts=1,
            max_memory_operations_per_turn=4,
        )
        result = await engine.run()
        assert result.success

    async def test_memory_budget_exhaustion(self, tmp_path: Path) -> None:
        """Memory-operation budget exhaustion."""
        responses = [
            ModelResponse(
                assistant_text="Memory op 1.",
                tool_calls=[
                    ToolCall(
                        call_id=f"c{i}",
                        name="write_memory",
                        arguments={"text": f"note {i}"},
                    )
                ],
                finish_reason="stop",
            )
            for i in range(5)
        ]
        engine = RunEngine(
            game=TinyGame(max_actions=1),
            agent=FakeAgent(responses=responses),
            run_dir=tmp_path / "run",
            run_id="test",
            max_turns=5,
            max_invalid_attempts=0,
            max_memory_operations_per_turn=2,
        )
        result = await engine.run()
        assert not result.success

    async def test_memory_budget_rejects_each_over_budget_call_in_history(
        self, tmp_path: Path
    ) -> None:
        response = ModelResponse(
            assistant_text="I will write three notes.",
            tool_calls=[
                ToolCall(
                    call_id=f"memory-{index}",
                    name="write_memory",
                    arguments={"text": f"note {index}"},
                )
                for index in range(1, 4)
            ],
            finish_reason="stop",
        )
        engine = RunEngine(
            game=TinyGame(max_actions=1),
            agent=FakeAgent(responses=response),
            run_dir=tmp_path / "run",
            run_id="memory-budget-history-test",
            max_turns=1,
            max_memory_operations_per_turn=2,
        )

        result = await engine.run()

        assert not result.success
        events = _read_events(tmp_path / "run")
        match_end = next(
            event for event in events if event["event_type"] == "match_end"
        )
        assert match_end["payload"]["result"]["outcome"]["total_actions"] == 0

        transcript = engine._actor_transcripts["a"]
        assistant_message = next(
            message for message in transcript if message["role"] == "assistant"
        )
        tool_results = [message for message in transcript if message["role"] == "tool"]
        assert [call["id"] for call in assistant_message["tool_calls"]] == [
            "memory-1",
            "memory-2",
            "memory-3",
        ]
        assert [result["tool_call_id"] for result in tool_results] == [
            "memory-1",
            "memory-2",
            "memory-3",
        ]
        assert json.loads(str(tool_results[-1]["content"])) == {
            "error": "Memory operation budget exceeded"
        }

    async def test_memory_budget_rejects_remaining_announced_calls_in_history(
        self, tmp_path: Path
    ) -> None:
        response = ModelResponse(
            assistant_text="I will write four notes.",
            tool_calls=[
                ToolCall(
                    call_id=f"memory-{index}",
                    name="write_memory",
                    arguments={"text": f"note {index}"},
                )
                for index in range(1, 5)
            ],
            finish_reason="stop",
        )
        engine = RunEngine(
            game=TinyGame(max_actions=1),
            agent=FakeAgent(responses=response),
            run_dir=tmp_path / "run",
            run_id="memory-budget-remaining-history-test",
            max_turns=1,
            max_memory_operations_per_turn=2,
        )

        result = await engine.run()

        assert not result.success
        transcript = engine._actor_transcripts["a"]
        assistant_message = next(
            message for message in transcript if message["role"] == "assistant"
        )
        tool_results = [message for message in transcript if message["role"] == "tool"]
        assert [call["id"] for call in assistant_message["tool_calls"]] == [
            "memory-1",
            "memory-2",
            "memory-3",
            "memory-4",
        ]
        assert [result["tool_call_id"] for result in tool_results] == [
            "memory-1",
            "memory-2",
            "memory-3",
            "memory-4",
        ]

    async def test_write_with_invalid_action_discards_note(
        self, tmp_path: Path
    ) -> None:
        bad_action = ModelResponse(
            assistant_text="Bad action with note.",
            tool_calls=[
                ToolCall(call_id="c1", name="nonexistent", arguments={}),
                ToolCall(
                    call_id="c2",
                    name="write_memory",
                    arguments={"text": "should be discarded"},
                ),
            ],
            finish_reason="stop",
        )
        good_action = ModelResponse(
            assistant_text="Fixed.",
            tool_calls=[ToolCall(call_id="c3", name="act", arguments={})],
            finish_reason="stop",
        )
        engine = RunEngine(
            game=TinyGame(max_actions=1),
            agent=FakeAgent(responses=[bad_action, good_action]),
            run_dir=tmp_path / "run",
            run_id="test",
            max_turns=5,
            max_invalid_attempts=1,
            max_memory_operations_per_turn=4,
        )
        result = await engine.run()
        assert result.success

    async def test_mixed_memory_action_response_retries_without_applying_action(
        self, tmp_path: Path
    ) -> None:
        bad_response = ModelResponse(
            assistant_text="Invalid note with an action.",
            tool_calls=[
                ToolCall(call_id="c1", name="act", arguments={}),
                ToolCall(
                    call_id="c2",
                    name="write_memory",
                    arguments={"text": "   "},
                ),
            ],
            finish_reason="stop",
        )
        good_response = ModelResponse(
            assistant_text="Fixed.",
            tool_calls=[ToolCall(call_id="c3", name="act", arguments={})],
            finish_reason="stop",
        )
        engine = RunEngine(
            game=TinyGame(max_actions=1),
            agent=FakeAgent(responses=[bad_response, good_response]),
            run_dir=tmp_path / "run",
            run_id="test",
            max_turns=5,
            max_invalid_attempts=1,
            max_memory_operations_per_turn=4,
        )

        result = await engine.run()

        assert result.success
        events = _read_events(tmp_path / "run")
        validation_errors = [
            event["payload"]["error"]
            for event in events
            if event["event_type"] == "validation"
            and event["payload"].get("valid") is False
        ]
        assert "memory_with_game_action" in validation_errors
        assert not any(
            event["event_type"] == "memory_operation"
            and event["payload"]["operation"] == "write"
            for event in events
        )

    async def test_memory_tool_result_messages_valid_history(
        self, tmp_path: Path
    ) -> None:
        """Memory tool result messages use valid Chat Completions history."""
        responses = [
            ModelResponse(
                assistant_text="Reading memory.",
                tool_calls=[ToolCall(call_id="c1", name="read_memory", arguments={})],
                finish_reason="stop",
            ),
            ModelResponse(
                assistant_text="Acting.",
                tool_calls=[ToolCall(call_id="c2", name="act", arguments={})],
                finish_reason="stop",
            ),
        ]
        agent = FakeAgent(responses=responses)
        engine = RunEngine(
            game=TinyGame(max_actions=1),
            agent=agent,
            run_dir=tmp_path / "run",
            run_id="test",
            max_turns=5,
            max_memory_operations_per_turn=4,
        )
        await engine.run()

        # The second request should contain the memory tool call and result
        assert len(agent.requests) >= 2
        second_request = agent.requests[1]
        messages = second_request.messages
        # Should have: user observation, assistant tool call, tool result
        roles = [m["role"] for m in messages]
        assert "assistant" in roles
        assert "tool" in roles

    async def test_max_turns_counts_only_game_actions(self, tmp_path: Path) -> None:
        """max_turns counts only accepted game actions, not memory calls."""
        responses = [
            ModelResponse(
                assistant_text="Memory op.",
                tool_calls=[
                    ToolCall(
                        call_id="c1", name="write_memory", arguments={"text": "note"}
                    )
                ],
                finish_reason="stop",
            ),
            ModelResponse(
                assistant_text="Acting.",
                tool_calls=[ToolCall(call_id="c2", name="act", arguments={})],
                finish_reason="stop",
            ),
            ModelResponse(
                assistant_text="Memory op 2.",
                tool_calls=[
                    ToolCall(
                        call_id="c3", name="write_memory", arguments={"text": "note2"}
                    )
                ],
                finish_reason="stop",
            ),
            ModelResponse(
                assistant_text="Acting again.",
                tool_calls=[ToolCall(call_id="c4", name="act", arguments={})],
                finish_reason="stop",
            ),
        ]
        # max_actions=2 so game is terminal after 2 game actions
        engine = RunEngine(
            game=TinyGame(max_actions=2),
            agent=FakeAgent(responses=responses),
            run_dir=tmp_path / "run",
            run_id="test",
            max_turns=5,
            max_memory_operations_per_turn=4,
        )
        result = await engine.run()
        assert result.success

    async def test_agent_receives_only_own_memory_results(self, tmp_path: Path) -> None:
        """Each agent receives only its own memory results."""
        # This test uses two agents to verify isolation
        responses_a = [
            ModelResponse(
                assistant_text="A writes.",
                tool_calls=[
                    ToolCall(
                        call_id="c1",
                        name="write_memory",
                        arguments={"text": "A's note"},
                    )
                ],
                finish_reason="stop",
            ),
            ModelResponse(
                assistant_text="A acts.",
                tool_calls=[ToolCall(call_id="c2", name="act", arguments={})],
                finish_reason="stop",
            ),
        ]
        responses_b = [
            ModelResponse(
                assistant_text="B reads.",
                tool_calls=[ToolCall(call_id="c3", name="read_memory", arguments={})],
                finish_reason="stop",
            ),
            ModelResponse(
                assistant_text="B acts.",
                tool_calls=[ToolCall(call_id="c4", name="act", arguments={})],
                finish_reason="stop",
            ),
        ]
        agent_a = FakeAgent(responses=responses_a)
        agent_b = FakeAgent(responses=responses_b)
        engine = RunEngine(
            game=TinyGame(max_actions=2),
            agents={"a": agent_a, "b": agent_b},
            run_dir=tmp_path / "run",
            run_id="test",
            max_turns=10,
            max_memory_operations_per_turn=4,
        )
        result = await engine.run()
        assert result.success

        # B's read should return empty (no notes from A are visible to B)
        events = _read_events(tmp_path / "run")
        memory_events = [e for e in events if e["event_type"] == "memory_operation"]
        read_events = [e for e in memory_events if e["payload"]["operation"] == "read"]
        assert len(read_events) == 1
        assert read_events[0]["actor_id"] == "b"
        # The result should indicate no notes
        assert read_events[0]["payload"]["count"] == 0

    async def test_memory_tool_specs_included_in_tools(self, tmp_path: Path) -> None:
        """Memory tool specs are included alongside game tools."""
        engine = RunEngine(
            game=TinyGame(max_actions=1),
            agent=FakeAgent(tool_name="act"),
            run_dir=tmp_path / "run",
            run_id="test",
            max_turns=5,
            max_memory_operations_per_turn=4,
        )
        await engine.run()

        events = _read_events(tmp_path / "run")
        tool_events = [e for e in events if e["event_type"] == "tool_schemas"]
        assert len(tool_events) > 0
        tool_names = [t["function"]["name"] for t in tool_events[0]["payload"]["tools"]]
        assert "read_memory" in tool_names
        assert "write_memory" in tool_names
        assert "act" in tool_names

    async def test_memory_operation_event_emitted(self, tmp_path: Path) -> None:
        """Each memory operation emits a memory_operation event."""
        responses = [
            ModelResponse(
                assistant_text="Write.",
                tool_calls=[
                    ToolCall(
                        call_id="c1",
                        name="write_memory",
                        arguments={"text": "hello"},
                    )
                ],
                finish_reason="stop",
            ),
            ModelResponse(
                assistant_text="Read.",
                tool_calls=[ToolCall(call_id="c2", name="read_memory", arguments={})],
                finish_reason="stop",
            ),
            ModelResponse(
                assistant_text="Act.",
                tool_calls=[ToolCall(call_id="c3", name="act", arguments={})],
                finish_reason="stop",
            ),
        ]
        engine = RunEngine(
            game=TinyGame(max_actions=1),
            agent=FakeAgent(responses=responses),
            run_dir=tmp_path / "run",
            run_id="test",
            max_turns=5,
            max_memory_operations_per_turn=4,
        )
        await engine.run()

        events = _read_events(tmp_path / "run")
        memory_events = [e for e in events if e["event_type"] == "memory_operation"]
        assert len(memory_events) == 2
        ops = [e["payload"]["operation"] for e in memory_events]
        assert ops == ["write", "read"]
