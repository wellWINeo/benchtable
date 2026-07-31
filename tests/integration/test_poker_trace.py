"""Integration tests for offline multi-hand poker traces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tests.fixtures.fake_agent import FakeAgent

from benchtable.agents.protocol import AgentRequest
from benchtable.contracts import ModelResponse, ToolCall, Usage
from benchtable.engine import RunEngine
from benchtable.games.poker.session import PokerGame


def _read_events(path: Path) -> list[dict[str, Any]]:
    raw = (path / "events.jsonl").read_text()
    return [json.loads(line) for line in raw.splitlines() if line]


class _PokerFakeAgent:
    """Fake agent that always calls poker_action with a valid action."""

    def __init__(self, preferred: str = "check") -> None:
        self._preferred = preferred
        self.requests: list[AgentRequest] = []
        self._count = 0

    async def respond(self, request: AgentRequest) -> ModelResponse:
        self.requests.append(request)
        self._count += 1
        if not request.tools:
            return ModelResponse(assistant_text="", tool_calls=[], finish_reason="stop")
        # Parse legal actions from observation
        action = self._pick_action(request)
        return ModelResponse(
            assistant_text=f"I'll {action}.",
            tool_calls=[
                ToolCall(
                    call_id=f"call-{self._count}",
                    name="poker_action",
                    arguments={"action": action},
                )
            ],
            finish_reason="stop",
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    def _pick_action(self, request: AgentRequest) -> str:
        # Find the observation message to determine legal actions
        for msg in request.messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                text = str(msg.get("content", ""))
                if "Legal actions:" in text:
                    legal_part = text.split("Legal actions:")[1].strip()
                    legal = [a.strip().rstrip(".") for a in legal_part.split(",")]
                    # Prefer check/call, fall back to first legal
                    if self._preferred in legal:
                        return self._preferred
                    if "call" in legal:
                        return "call"
                    if "check" in legal:
                        return "check"
                    if legal:
                        return legal[0]
        return "fold"


class _RecoveryMemoryAgent:
    """Fail the first turn, then read the forced-fold summary on hand two."""

    def __init__(self) -> None:
        self.requests: list[AgentRequest] = []
        self._count = 0

    async def respond(self, request: AgentRequest) -> ModelResponse:
        self.requests.append(request)
        self._count += 1
        if self._count == 1:
            return ModelResponse(
                assistant_text="No action.",
                tool_calls=[],
                finish_reason="stop",
            )
        if not request.tools:
            return ModelResponse(assistant_text="", tool_calls=[], finish_reason="stop")

        if any(
            isinstance(message, dict)
            and message.get("role") == "user"
            and "Hand 2 of 2." in str(message.get("content", ""))
            for message in request.messages
        ) and not any(
            isinstance(message, dict) and message.get("role") == "tool"
            for message in request.messages
        ):
            return ModelResponse(
                assistant_text="Reading memory.",
                tool_calls=[
                    ToolCall(
                        call_id=f"read-{self._count}",
                        name="read_memory",
                        arguments={},
                    )
                ],
                finish_reason="stop",
            )

        action = self._pick_action(request)
        return ModelResponse(
            assistant_text=f"I'll {action}.",
            tool_calls=[
                ToolCall(
                    call_id=f"call-{self._count}",
                    name="poker_action",
                    arguments={"action": action},
                )
            ],
            finish_reason="stop",
        )

    @staticmethod
    def _pick_action(request: AgentRequest) -> str:
        for message in request.messages:
            if isinstance(message, dict) and message.get("role") == "user":
                text = str(message.get("content", ""))
                if "Legal actions:" in text:
                    legal = [
                        action.strip().rstrip(".")
                        for action in text.split("Legal actions:")[1].split(",")
                    ]
                    if "check" in legal:
                        return "check"
                    if "call" in legal:
                        return "call"
                    if legal:
                        return legal[0]
        return "fold"


class TestPokerTrace:
    async def test_two_player_multi_hand_trace(self, tmp_path: Path) -> None:
        game = PokerGame()
        agent_a = _PokerFakeAgent(preferred="check")
        agent_b = _PokerFakeAgent(preferred="call")
        engine = RunEngine(
            game=game,
            agents={"player-1": agent_a, "player-2": agent_b},
            game_config={
                "players": ["player-1", "player-2"],
                "hands_per_match": 2,
                "initial_stack": 1000,
                "small_blind": 5,
                "big_blind": 10,
            },
            run_dir=tmp_path / "run",
            run_id="poker-trace-test",
            seed=42,
            matches=1,
            max_turns=200,
            max_memory_operations_per_turn=4,
        )
        result = await engine.run()

        events = _read_events(tmp_path / "run")
        types = {e["event_type"] for e in events}

        assert result.success
        assert "run_config" in types
        assert "match_start" in types
        assert "turn_start" in types
        assert "observation" in types
        assert "tool_schemas" in types
        assert "model_request" in types
        assert "model_response" in types
        assert "validation" in types
        assert "transition" in types
        assert "turn_end" in types
        assert "match_end" in types
        assert "run_end" in types
        hand_starts = [e for e in events if e["event_type"] == "hand_start"]
        hand_ends = [e for e in events if e["event_type"] == "hand_end"]
        assert len(hand_starts) == len(hand_ends) == 2
        assert [e["payload"]["hand_index"] for e in hand_starts] == [0, 1]
        assert all(sum(e["payload"]["payouts"].values()) > 0 for e in hand_ends)
        assert all(
            "small_blind" in e["payload"] and "big_blind" in e["payload"]
            for e in hand_ends
        )
        for agent in (agent_a, agent_b):
            second_hand_request = next(
                request
                for request in agent.requests
                if any(
                    isinstance(message, dict)
                    and message.get("role") == "user"
                    and "Hand 2 of 2." in str(message.get("content", ""))
                    for message in request.messages
                )
            )
            assert [message["role"] for message in second_hand_request.messages] == [
                "user"
            ]
        match_end = next(e for e in events if e["event_type"] == "match_end")
        result_payload = match_end["payload"]["result"]
        assert result_payload["completed"] is True
        assert set(result_payload["outcome"]) >= {
            "winner",
            "stacks",
            "deltas",
            "hands_played",
            "wins",
            "finish_reason",
            "completed",
        } or set(result_payload["outcome"]) >= {
            "winners",
            "stacks",
            "deltas",
            "hands_played",
            "wins",
            "finish_reason",
            "completed",
        }
        assert set(result_payload["metrics"]) >= {
            "final_stacks",
            "deltas",
            "hands_played",
            "hand_summaries",
            "failure_count",
            "recovery_count",
            "completed",
        }
        hand_summaries = result_payload["metrics"]["hand_summaries"]
        compact_summary_fields = {
            "hand_index",
            "finish_reason",
            "summary",
            "seat_deltas",
            "payouts",
        }
        assert len(hand_summaries) == 2
        assert [summary["hand_index"] for summary in hand_summaries] == [0, 1]
        assert [list(summary["seat_deltas"]) for summary in hand_summaries] == [
            ["player-1", "player-2"],
            ["player-1", "player-2"],
        ]
        assert [summary["seat_deltas"] for summary in hand_summaries] == [
            {"player-1": -10, "player-2": 10},
            {"player-1": 10, "player-2": -10},
        ]
        assert [list(summary["payouts"]) for summary in hand_summaries] == [
            ["player-2"],
            ["player-1"],
        ]
        assert [summary["payouts"] for summary in hand_summaries] == [
            {"player-2": 20},
            {"player-1": 20},
        ]
        for hand_index, summary in enumerate(hand_summaries):
            assert set(summary) == compact_summary_fields
            assert summary["hand_index"] == hand_index
            assert summary["finish_reason"] in {"fold", "showdown"}
            assert set(summary["seat_deltas"]) == {"player-1", "player-2"}
            assert isinstance(summary["payouts"], dict)
            assert sum(summary["payouts"].values()) > 0
            summary_text = summary["summary"].lower()
            assert all(
                marker not in summary_text
                for marker in (
                    "board",
                    "action history",
                    "action_history",
                    "action-history",
                    "hole card",
                    "hole_card",
                    "hole-card",
                    "private",
                    "raw state",
                    "raw_state",
                    "raw-state",
                    "raw",
                )
            )

    async def test_poker_observation_isolation(self, tmp_path: Path) -> None:
        game = PokerGame()
        agent_a = _PokerFakeAgent(preferred="check")
        agent_b = _PokerFakeAgent(preferred="call")
        engine = RunEngine(
            game=game,
            agents={"player-1": agent_a, "player-2": agent_b},
            game_config={
                "players": ["player-1", "player-2"],
                "hands_per_match": 1,
                "initial_stack": 1000,
                "small_blind": 5,
                "big_blind": 10,
            },
            run_dir=tmp_path / "run",
            run_id="isolation-test",
            seed=42,
            matches=1,
            max_turns=100,
        )
        await engine.run()

        events = _read_events(tmp_path / "run")
        obs_events = [e for e in events if e["event_type"] == "observation"]
        for obs in obs_events:
            text = obs["payload"]["text"]
            assert "hole cards" in text.lower()

        first_a = next(
            request
            for request in agent_a.requests
            if request.messages and request.messages[0]["role"] == "user"
        )
        first_b = next(
            request
            for request in agent_b.requests
            if request.messages and request.messages[0]["role"] == "user"
        )
        cards_a = (
            first_a.messages[0]["content"]
            .split("Your hole cards: ", 1)[1]
            .split(".", 1)[0]
            .split()
        )
        cards_b = (
            first_b.messages[0]["content"]
            .split("Your hole cards: ", 1)[1]
            .split(".", 1)[0]
            .split()
        )
        assert cards_a and cards_b and cards_a != cards_b
        assert all(card not in first_b.messages[0]["content"] for card in cards_a)
        assert all(card not in first_a.messages[0]["content"] for card in cards_b)

    async def test_forced_fold_on_failure(self, tmp_path: Path) -> None:
        game = PokerGame()

        class _CrashAgent:
            async def respond(self, request: AgentRequest) -> ModelResponse:
                raise Exception("agent crashed")

        agent = _CrashAgent()
        engine = RunEngine(
            game=game,
            agents={"player-1": agent, "player-2": agent},
            game_config={
                "players": ["player-1", "player-2"],
                "hands_per_match": 1,
                "initial_stack": 1000,
                "small_blind": 5,
                "big_blind": 10,
                "invalid_turn_policy": "forced_fold",
            },
            run_dir=tmp_path / "run",
            run_id="failure-test",
            seed=42,
            matches=1,
            max_turns=10,
            max_provider_retries=0,
        )
        result = await engine.run()
        assert not result.success

        events = _read_events(tmp_path / "run")
        assert any(e["event_type"] == "match_end" for e in events)

    async def test_forced_fold_summary_is_readable_on_the_next_hand(
        self, tmp_path: Path
    ) -> None:
        agent = _RecoveryMemoryAgent()
        engine = RunEngine(
            game=PokerGame(),
            agents={"player-1": agent, "player-2": agent},
            game_config={
                "players": ["player-1", "player-2"],
                "hands_per_match": 2,
                "initial_stack": 1000,
                "small_blind": 5,
                "big_blind": 10,
                "invalid_turn_policy": "forced_fold",
            },
            run_dir=tmp_path / "run",
            run_id="recovery-memory-test",
            seed=42,
            matches=1,
            max_turns=100,
            max_invalid_attempts=0,
        )

        result = await engine.run()

        assert result.success
        recovery_transition_index = next(
            index
            for index, event in enumerate(_read_events(tmp_path / "run"))
            if event["event_type"] == "transition"
            and event["payload"].get("recovery") is True
        )
        events = _read_events(tmp_path / "run")
        summary_indices = [
            index
            for index, event in enumerate(events)
            if event["event_type"] == "memory_operation"
            and event["payload"].get("operation") == "summary"
            and event["payload"].get("hand") == 0
        ]
        next_model_request_index = next(
            index
            for index, event in enumerate(events)
            if event["event_type"] == "model_request" and event["turn_index"] == 1
        )
        assert summary_indices
        assert recovery_transition_index < min(summary_indices)
        assert max(summary_indices) < next_model_request_index
        assert all(
            isinstance(event["payload"].get("text"), str) and event["payload"]["text"]
            for event in events
            if event["event_type"] == "memory_operation"
            and event["payload"].get("operation") == "summary"
        )

        read_request = next(
            request
            for request in agent.requests
            if any(
                isinstance(message, dict)
                and message.get("role") == "tool"
                and "source" in str(message.get("content", ""))
                for message in request.messages
            )
        )
        memory_results = [
            json.loads(str(message["content"]))
            for message in read_request.messages
            if isinstance(message, dict)
            and message.get("role") == "tool"
            and "source" in str(message.get("content", ""))
        ]
        assert memory_results
        assert any(
            entry["source"] == "system" and entry["hand"] == 0
            for entry in memory_results[0]
        )

    async def test_terminal_recovery_summary_is_persisted_before_match_completion(
        self, tmp_path: Path
    ) -> None:
        bad = ModelResponse(
            assistant_text="No action.", tool_calls=[], finish_reason="stop"
        )
        engine = RunEngine(
            game=PokerGame(),
            agents={
                "player-1": FakeAgent(responses=bad),
                "player-2": FakeAgent(responses=bad),
            },
            game_config={
                "players": ["player-1", "player-2"],
                "hands_per_match": 1,
                "initial_stack": 1000,
                "small_blind": 5,
                "big_blind": 10,
                "invalid_turn_policy": "forced_fold",
            },
            run_dir=tmp_path / "run",
            run_id="terminal-recovery-summary-test",
            seed=42,
            matches=1,
            max_turns=1,
            max_invalid_attempts=0,
        )

        result = await engine.run()

        assert result.success
        events = _read_events(tmp_path / "run")
        summary_events = [
            event
            for event in events
            if event["event_type"] == "memory_operation"
            and event["payload"].get("operation") == "summary"
        ]
        recovery_transition_index = next(
            index
            for index, event in enumerate(events)
            if event["event_type"] == "transition"
            and event["payload"].get("recovery") is True
        )
        summary_indices = [
            index
            for index, event in enumerate(events)
            if event["event_type"] == "memory_operation"
            and event["payload"].get("operation") == "summary"
        ]
        turn_end_index = next(
            index
            for index, event in enumerate(events)
            if event["event_type"] == "turn_end"
        )
        match_end_index = next(
            index
            for index, event in enumerate(events)
            if event["event_type"] == "match_end"
        )
        assert summary_events
        assert summary_indices
        assert recovery_transition_index < min(summary_indices)
        assert max(summary_indices) < turn_end_index < match_end_index

    async def test_fail_match_is_not_reported_as_a_completed_poker_result(
        self, tmp_path: Path
    ) -> None:
        bad = ModelResponse(
            assistant_text="No action.", tool_calls=[], finish_reason="stop"
        )
        engine = RunEngine(
            game=PokerGame(),
            agents={
                "player-1": FakeAgent(responses=bad),
                "player-2": FakeAgent(responses=bad),
            },
            game_config={
                "players": ["player-1", "player-2"],
                "hands_per_match": 1,
                "invalid_turn_policy": "fail_match",
            },
            run_dir=tmp_path / "run",
            run_id="fail-match-test",
            max_turns=1,
            max_invalid_attempts=0,
        )

        result = await engine.run()

        assert not result.success
        events = _read_events(tmp_path / "run")
        match_end = next(
            event for event in events if event["event_type"] == "match_end"
        )
        assert match_end["payload"]["result"]["completed"] is False
        outcome = match_end["payload"]["result"]["outcome"]
        metrics = match_end["payload"]["result"]["metrics"]
        assert "winner" not in outcome
        assert set(outcome) >= {"stacks", "deltas", "completed"}
        assert outcome["completed"] is False
        assert metrics["failure_count"] == 1
        assert metrics["completed"] is False
        assert match_end["payload"]["failure_reason"] == "invalid_attempts_exhausted"

    async def test_multiple_matches_persist_stacks(self, tmp_path: Path) -> None:
        game = PokerGame()
        agent_a = _PokerFakeAgent(preferred="check")
        agent_b = _PokerFakeAgent(preferred="call")
        engine = RunEngine(
            game=game,
            agents={"player-1": agent_a, "player-2": agent_b},
            game_config={
                "players": ["player-1", "player-2"],
                "hands_per_match": 3,
                "initial_stack": 1000,
                "small_blind": 5,
                "big_blind": 10,
            },
            run_dir=tmp_path / "run",
            run_id="multi-hand-test",
            seed=42,
            matches=2,
            max_turns=500,
        )
        result = await engine.run()
        assert result.success

        events = _read_events(tmp_path / "run")
        match_ends = [e for e in events if e["event_type"] == "match_end"]
        assert len(match_ends) == 2
        assert all(end["payload"]["result"]["completed"] is True for end in match_ends)
        hand_ends = [e for e in events if e["event_type"] == "hand_end"]
        assert len(hand_ends) == 6
        assert all(sum(event["payload"]["payouts"].values()) > 0 for event in hand_ends)

    async def test_poker_tool_schemas_include_poker_action(
        self, tmp_path: Path
    ) -> None:
        game = PokerGame()
        engine = RunEngine(
            game=game,
            agents={
                "player-1": _PokerFakeAgent(preferred="check"),
                "player-2": _PokerFakeAgent(preferred="call"),
            },
            game_config={
                "players": ["player-1", "player-2"],
                "hands_per_match": 1,
                "initial_stack": 1000,
                "small_blind": 5,
                "big_blind": 10,
            },
            run_dir=tmp_path / "run",
            run_id="tools-test",
            seed=42,
            matches=1,
            max_turns=10,
        )
        await engine.run()

        events = _read_events(tmp_path / "run")
        tool_events = [e for e in events if e["event_type"] == "tool_schemas"]
        assert len(tool_events) > 0
        tool_names = [t["function"]["name"] for t in tool_events[0]["payload"]["tools"]]
        assert "poker_action" in tool_names
        assert "read_memory" in tool_names
        assert "write_memory" in tool_names
