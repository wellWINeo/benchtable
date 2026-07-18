"""Integration tests for the engine trace output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tests.fixtures.fake_agent import FakeAgent
from tests.fixtures.tiny_game import TinyGame

from benchtable.engine import RunEngine


def _read_events(path: Path) -> list[dict[str, Any]]:
    raw = (path / "events.jsonl").read_text()
    return [json.loads(line) for line in raw.splitlines() if line]


class TestEngineTrace:
    async def test_complete_trace_has_all_event_types(self, tmp_path: Path) -> None:
        engine = RunEngine(
            game=TinyGame(),
            agent=FakeAgent(tool_name="act"),
            run_dir=tmp_path / "run",
            run_id="trace-test",
            seed=42,
            matches=1,
            max_turns=20,
        )
        result = await engine.run()
        assert result.success

        events = _read_events(tmp_path / "run")
        types = {e["event_type"] for e in events}

        required = {
            "run_config",
            "match_start",
            "turn_start",
            "observation",
            "model_request",
            "model_response",
            "validation",
            "transition",
            "turn_end",
            "match_end",
            "run_end",
        }
        assert required.issubset(types)

    async def test_trace_schema_version_is_one(self, tmp_path: Path) -> None:
        engine = RunEngine(
            game=TinyGame(),
            agent=FakeAgent(tool_name="act"),
            run_dir=tmp_path / "run",
            run_id="trace-test",
            seed=42,
            matches=1,
            max_turns=20,
        )
        await engine.run()

        events = _read_events(tmp_path / "run")
        for e in events:
            assert e["schema_version"] == 1

    async def test_sequences_are_monotonic(self, tmp_path: Path) -> None:
        engine = RunEngine(
            game=TinyGame(),
            agent=FakeAgent(tool_name="act"),
            run_dir=tmp_path / "run",
            run_id="trace-test",
            seed=42,
            matches=1,
            max_turns=20,
        )
        await engine.run()

        events = _read_events(tmp_path / "run")
        seqs = [e["sequence"] for e in events]
        assert seqs == sorted(seqs)
        assert seqs == list(range(1, len(seqs) + 1))

    async def test_run_config_has_no_secrets(self, tmp_path: Path) -> None:
        engine = RunEngine(
            game=TinyGame(),
            agent=FakeAgent(tool_name="act"),
            run_dir=tmp_path / "run",
            run_id="trace-test",
            seed=42,
            matches=1,
            max_turns=20,
        )
        await engine.run()

        events = _read_events(tmp_path / "run")
        config_events = [e for e in events if e["event_type"] == "run_config"]
        config_str = json.dumps(config_events)
        assert "sk-" not in config_str
        assert "api_key" not in config_str.lower()

    async def test_observation_events_contain_only_actor_observation(
        self, tmp_path: Path
    ) -> None:
        engine = RunEngine(
            game=TinyGame(),
            agent=FakeAgent(tool_name="act"),
            run_dir=tmp_path / "run",
            run_id="trace-test",
            seed=42,
            matches=1,
            max_turns=20,
        )
        await engine.run()

        events = _read_events(tmp_path / "run")
        obs_events = [e for e in events if e["event_type"] == "observation"]
        for e in obs_events:
            assert e["actor_id"] in ("a", "b")
            obs_text = e["payload"]["text"].lower()
            # Should not contain other actor's private data
            if e["actor_id"] == "a":
                assert "secret_b" not in obs_text
            else:
                assert "secret_a" not in obs_text
