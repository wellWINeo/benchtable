"""Tests for the per-match memory store."""

from __future__ import annotations

import pytest

from benchtable.contracts import MatchMemorySummary, ToolSpec
from benchtable.memory import MatchMemory


class TestMatchMemory:
    def test_note_stored_under_current_actor(self) -> None:
        store = MatchMemory()
        store.write("a", text="hello", hand=1, turn=0)
        notes = store.read("a")
        assert len(notes) == 1
        assert notes[0]["text"] == "hello"

    def test_read_returns_notes_in_insertion_order(self) -> None:
        store = MatchMemory()
        store.write("a", text="first", hand=1, turn=0)
        store.write("a", text="second", hand=1, turn=1)
        store.write("a", text="third", hand=2, turn=0)
        notes = store.read("a")
        assert [n["text"] for n in notes] == ["first", "second", "third"]

    def test_read_merges_agent_notes_and_system_summaries_in_order(self) -> None:
        store = MatchMemory()
        store.write("a", text="agent-1", hand=1, turn=0)
        store.append_summary(
            MatchMemorySummary(actor_id="a", text="system-1", hand=1, turn=1)
        )
        store.write("a", text="agent-2", hand=1, turn=2)

        notes = store.read("a")

        assert [n["source"] for n in notes] == ["agent", "system", "agent"]
        assert [n["text"] for n in notes] == ["agent-1", "system-1", "agent-2"]

    def test_system_summary_does_not_consume_agent_limits(self) -> None:
        store = MatchMemory(max_entries=1)
        store.write("a", text="agent", hand=1, turn=0)
        store.append_summary(
            MatchMemorySummary(actor_id="a", text="system", hand=1, turn=1)
        )

        with pytest.raises(ValueError, match="Entry limit"):
            store.write("a", text="overflow", hand=1, turn=2)

        assert [note["text"] for note in store.read("a")] == ["agent", "system"]

    def test_read_includes_hand_and_turn_context(self) -> None:
        store = MatchMemory()
        store.write("a", text="note", hand=3, turn=5)
        notes = store.read("a")
        assert notes[0]["hand"] == 3
        assert notes[0]["turn"] == 5
        assert notes[0]["sequence"] == 1

    def test_actor_cannot_read_other_actor_notes(self) -> None:
        store = MatchMemory()
        store.write("a", text="private_a", hand=1, turn=0)
        store.write("b", text="private_b", hand=1, turn=0)
        notes_a = store.read("a")
        notes_b = store.read("b")
        assert all(n["text"] == "private_a" for n in notes_a)
        assert all(n["text"] == "private_b" for n in notes_b)

    def test_new_match_store_is_empty(self) -> None:
        store1 = MatchMemory()
        store1.write("a", text="old_note", hand=1, turn=0)
        store2 = MatchMemory()
        assert store2.read("a") == []

    def test_entry_limit_rejects_writes(self) -> None:
        store = MatchMemory(max_entries=2)
        store.write("a", text="n1", hand=1, turn=0)
        store.write("a", text="n2", hand=1, turn=1)
        with pytest.raises(ValueError, match="[Ee]ntry limit"):
            store.write("a", text="n3", hand=1, turn=2)

    def test_character_limit_rejects_writes(self) -> None:
        store = MatchMemory(max_chars=10)
        store.write("a", text="short", hand=1, turn=0)
        with pytest.raises(ValueError, match="[Cc]haracter limit"):
            store.write("a", text="this is way too long", hand=1, turn=1)

    def test_empty_note_rejected(self) -> None:
        store = MatchMemory()
        with pytest.raises(ValueError, match="empty"):
            store.write("a", text="", hand=1, turn=0)

    def test_whitespace_only_note_rejected(self) -> None:
        store = MatchMemory()
        with pytest.raises(ValueError, match="empty"):
            store.write("a", text="   ", hand=1, turn=0)

    def test_read_memory_tool_spec(self) -> None:
        spec = MatchMemory.read_tool_spec()
        assert isinstance(spec, ToolSpec)
        assert spec.name == "read_memory"
        assert "private notes" in spec.description
        assert "system summaries" in spec.description
        assert spec.parameters["type"] == "object"
        assert spec.parameters.get("properties") == {}
        assert spec.parameters["additionalProperties"] is False

    def test_write_memory_tool_spec(self) -> None:
        spec = MatchMemory.write_tool_spec()
        assert isinstance(spec, ToolSpec)
        assert spec.name == "write_memory"
        assert spec.parameters["type"] == "object"
        assert "text" in spec.parameters.get("properties", {})
        assert spec.parameters["additionalProperties"] is False

    def test_read_returns_empty_list_for_unknown_actor(self) -> None:
        store = MatchMemory()
        assert store.read("nonexistent") == []

    def test_system_summaries_are_actor_private(self) -> None:
        store = MatchMemory()
        store.append_summary(
            MatchMemorySummary(actor_id="a", text="summary-a", hand=1, turn=0)
        )
        assert store.read("b") == []

    def test_notes_are_copy_safe(self) -> None:
        store = MatchMemory()
        store.write("a", text="note", hand=1, turn=0)
        notes = store.read("a")
        notes[0]["text"] = "mutated"
        notes2 = store.read("a")
        assert notes2[0]["text"] == "note"

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("max_entries", True),
            ("max_entries", -1),
            ("max_chars", 1.0),
            ("max_chars", -1),
        ],
    )
    def test_limits_must_be_non_negative_strict_integers(
        self, field: str, value: object
    ) -> None:
        with pytest.raises(ValueError, match="non-negative integer"):
            MatchMemory(**{field: value})
