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
