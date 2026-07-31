"""Per-match private memory store for actors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from benchtable.contracts import JsonObject, MatchMemorySummary, ToolSpec


@dataclass(frozen=True)
class _MemoryNote:
    sequence: int
    text: str
    hand: int
    turn: int


@dataclass(frozen=True)
class _MemorySummary:
    sequence: int
    text: str
    hand: int
    turn: int


class MatchMemory:
    """Engine-owned memory store keyed by actor ID.

    Each match gets a fresh instance. Notes are reset between matches.
    """

    def __init__(
        self,
        *,
        max_entries: int = 100,
        max_chars: int = 20000,
    ) -> None:
        if type(max_entries) is not int or max_entries < 0:
            raise ValueError("max_entries must be a non-negative integer")
        if type(max_chars) is not int or max_chars < 0:
            raise ValueError("max_chars must be a non-negative integer")
        self._max_entries = max_entries
        self._max_chars = max_chars
        self._notes: dict[str, list[_MemoryNote]] = {}
        self._summaries: dict[str, list[_MemorySummary]] = {}
        self._next_sequence = 1

    def append_summary(self, summary: MatchMemorySummary) -> None:
        """Append a public system summary for the target actor."""
        summaries = self._summaries.setdefault(summary.actor_id, [])
        summaries.append(
            _MemorySummary(
                sequence=self._next_sequence,
                text=summary.text,
                hand=summary.hand,
                turn=summary.turn,
            )
        )
        self._next_sequence += 1

    def write(self, actor_id: str, *, text: str, hand: int, turn: int) -> None:
        """Append a note for the given actor.

        Raises ValueError for empty text or when limits are exceeded.
        """
        if type(text) is not str or not text.strip():
            raise ValueError("Note text must not be empty or whitespace-only")

        actor_notes = self._notes.setdefault(actor_id, [])

        if len(actor_notes) >= self._max_entries:
            raise ValueError(
                f"Entry limit reached ({self._max_entries} entries per actor)"
            )

        total_chars = sum(len(note.text) for note in actor_notes) + len(text)
        if total_chars > self._max_chars:
            raise ValueError(
                f"Character limit reached ({self._max_chars} chars per actor)"
            )

        actor_notes.append(
            _MemoryNote(
                sequence=self._next_sequence,
                text=text,
                hand=hand,
                turn=turn,
            )
        )
        self._next_sequence += 1

    def read(self, actor_id: str) -> list[JsonObject]:
        """Return the actor's notes and summaries in insertion order."""
        entries: list[tuple[int, JsonObject]] = []
        for note in self._notes.get(actor_id, []):
            entries.append(
                (
                    note.sequence,
                    cast(
                        JsonObject,
                        {
                            "source": "agent",
                            "sequence": note.sequence,
                            "text": note.text,
                            "hand": note.hand,
                            "turn": note.turn,
                        },
                    ),
                )
            )
        for summary in self._summaries.get(actor_id, []):
            entries.append(
                (
                    summary.sequence,
                    cast(
                        JsonObject,
                        {
                            "source": "system",
                            "sequence": summary.sequence,
                            "text": summary.text,
                            "hand": summary.hand,
                            "turn": summary.turn,
                        },
                    ),
                )
            )
        return [entry for _sequence, entry in sorted(entries, key=lambda item: item[0])]

    @staticmethod
    def read_tool_spec() -> ToolSpec:
        """Return the reserved read_memory tool specification."""
        return ToolSpec(
            name="read_memory",
            description="Read your private notes and system summaries from memory.",
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        )

    @staticmethod
    def write_tool_spec() -> ToolSpec:
        """Return the reserved write_memory tool specification."""
        return ToolSpec(
            name="write_memory",
            description="Write a private note to memory.",
            parameters={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The note text to store.",
                    }
                },
                "required": ["text"],
                "additionalProperties": False,
            },
        )
