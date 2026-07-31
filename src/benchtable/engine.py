"""Benchmark run engine: orchestrates matches between game plugins and agents."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast
from uuid import uuid4

from pydantic import ValidationError

from benchtable.agents.protocol import Agent, AgentRequest
from benchtable.contracts import (
    JsonObject,
    MatchMemorySummary,
    ModelResponse,
    PluginEvent,
    ToolCall,
    Transition,
)
from benchtable.errors import InvalidActionError, ProviderError, RunError
from benchtable.events import (
    EventWriter,
    redact_json_text,
    redact_text,
    redact_value,
)
from benchtable.games.protocol import GamePlugin, GameSession
from benchtable.memory import MatchMemory


@dataclass(frozen=True)
class RunResult:
    """Explicit status and match counts for one engine run."""

    success: bool
    matches_total: int
    matches_completed: int
    matches_failed: int

    @property
    def status(self) -> str:
        """Return the serialized run status."""
        return "success" if self.success else "failed"


@dataclass(frozen=True)
class _TurnFailure:
    reason: str
    actor_id: str | None = None
    turn_index: int | None = None
    recover: bool = False


@dataclass(frozen=True)
class _ResponseFailure:
    reason: str


@dataclass(frozen=True)
class _ActionResponse:
    request: AgentRequest
    response: ModelResponse
    memory_calls: tuple[ToolCall, ...]
    game_calls: tuple[ToolCall, ...]
    memory_operations: int
    invalid_attempts: int


_RESERVED_MEMORY_TOOLS = frozenset({"read_memory", "write_memory"})


@dataclass(frozen=True)
class _MatchOutcome:
    success: bool
    cancelled: bool = False


class RunEngine:
    """Sequential match runner with retry policies and event logging."""

    def __init__(
        self,
        *,
        game: GamePlugin,
        agent: Agent | None = None,
        agents: Mapping[str, Agent] | None = None,
        agent_metadata: list[JsonObject] | None = None,
        agent_id: str | None = None,
        game_config: JsonObject | None = None,
        run_dir: str | Any,
        run_id: str = "",
        seed: int = 0,
        matches: int = 1,
        max_turns: int = 200,
        max_invalid_attempts: int = 2,
        max_provider_retries: int = 2,
        max_memory_operations_per_turn: int = 4,
    ) -> None:
        self._game = game
        self._agent = agent
        self._agents = agents
        self._agent_metadata = agent_metadata
        self._agent_id = agent_id
        self._game_config = game_config or {}
        self._run_dir = run_dir
        self._run_id = run_id or f"run-{uuid4().hex}"
        self._seed = seed
        self._matches = matches
        self._max_turns = max_turns
        self._max_invalid_attempts = max_invalid_attempts
        self._max_provider_retries = max_provider_retries
        if type(max_memory_operations_per_turn) is not int or (
            max_memory_operations_per_turn < 0
        ):
            raise ValueError(
                "max_memory_operations_per_turn must be a non-negative integer"
            )
        self._max_memory_operations_per_turn = max_memory_operations_per_turn
        self._actor_transcripts: dict[str, list[JsonObject]] = {}
        self._conversation_scope_id: str | None = None

    async def run(self) -> RunResult:
        """Execute all matches and write the trace."""
        matches_completed = 0
        matches_failed = 0

        with EventWriter(self._run_dir, run_id=self._run_id) as writer:
            self._emit_run_config(writer)
            for match_idx in range(self._matches):
                try:
                    outcome = await self._run_match(writer, match_idx)
                except asyncio.CancelledError:
                    writer.emit("cancellation", {"phase": "run"})
                    outcome = _MatchOutcome(False, cancelled=True)
                except Exception as exc:
                    writer.emit(
                        "engine_error",
                        {"phase": "match", "error": str(exc)},
                    )
                    outcome = _MatchOutcome(False)

                if outcome.success:
                    matches_completed += 1
                else:
                    matches_failed += 1
                if outcome.cancelled:
                    break

            success = matches_failed == 0
            writer.emit(
                "run_end",
                {
                    "status": "success" if success else "failed",
                    "matches_total": self._matches,
                    "matches_completed": matches_completed,
                    "matches_failed": matches_failed,
                },
            )

        return RunResult(
            success=success,
            matches_total=self._matches,
            matches_completed=matches_completed,
            matches_failed=matches_failed,
        )

    def _emit_run_config(self, writer: EventWriter) -> None:
        if self._agents is not None:
            agent_ids = list(self._agents)
        else:
            configured_id = self._agent_id
            if configured_id is None and self._agent_metadata:
                metadata_id = self._agent_metadata[0].get("id")
                if isinstance(metadata_id, str):
                    configured_id = metadata_id
            agent_ids = [configured_id or "direct"]
        agent_metadata = self._agent_metadata
        if agent_metadata is None:
            agent_metadata = [
                cast(JsonObject, {"id": agent_id}) for agent_id in agent_ids
            ]
        writer.emit(
            "run_config",
            cast(
                JsonObject,
                {
                    "game": self._game.name,
                    "game_version": self._game.version,
                    "game_config": self._game_config,
                    "agent_ids": agent_ids,
                    "agents": agent_metadata,
                    "seed": self._seed,
                    "matches": self._matches,
                    "max_turns": self._max_turns,
                    "max_invalid_attempts": self._max_invalid_attempts,
                    "max_provider_retries": self._max_provider_retries,
                    "max_memory_operations_per_turn": (
                        self._max_memory_operations_per_turn
                    ),
                },
            ),
        )

    def _derive_match_seed(self, match_idx: int) -> int:
        h = hashlib.sha256(f"{self._seed}:{match_idx}".encode()).hexdigest()
        return int(h[:16], 16)

    async def _run_match(self, writer: EventWriter, match_idx: int) -> _MatchOutcome:
        match_id = f"match-{match_idx}"
        seed = self._derive_match_seed(match_idx)
        self._actor_transcripts = {}
        self._conversation_scope_id = None
        metrics: dict[str, Any] = {
            "turn_count": 0,
            "invalid_actions": 0,
            "provider_failures": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_seconds": 0.0,
        }

        writer.emit(
            "match_start",
            {"match_index": match_idx, "seed": seed},
            match_id=match_id,
        )

        try:
            session = self._game.create_session(
                seed=seed,
                game_config=copy.deepcopy(self._game_config),
            )
        except asyncio.CancelledError:
            writer.emit(
                "cancellation",
                {"phase": "session_creation"},
                match_id=match_id,
            )
            self._emit_match_end(
                writer,
                match_id,
                metrics,
                completed=False,
                failure_reason="cancelled",
            )
            return _MatchOutcome(False, cancelled=True)
        except Exception as exc:
            writer.emit(
                "engine_error",
                {
                    "phase": "session_creation",
                    "error": str(exc),
                },
                match_id=match_id,
            )
            self._emit_match_end(
                writer,
                match_id,
                metrics,
                completed=False,
                failure_reason="session_creation_failed",
            )
            return _MatchOutcome(False)

        mapping_error = self._exact_agent_mapping_error()
        if mapping_error is not None:
            writer.emit(
                "engine_error",
                {"error": "exact_agent_mapping", "details": mapping_error},
                match_id=match_id,
            )
            self._emit_match_end(
                writer,
                match_id,
                metrics,
                completed=False,
                failure_reason="exact_agent_mapping",
            )
            return _MatchOutcome(False)

        try:
            memory = MatchMemory(
                max_entries=getattr(session, "memory_max_entries", 100),
                max_chars=getattr(session, "memory_max_chars", 20000),
            )
        except asyncio.CancelledError:
            writer.emit(
                "cancellation",
                {"phase": "memory_initialization"},
                match_id=match_id,
            )
            self._emit_match_end(
                writer,
                match_id,
                metrics,
                completed=False,
                failure_reason="cancelled",
            )
            return _MatchOutcome(False, cancelled=True)
        except Exception as exc:
            writer.emit(
                "engine_error",
                {"phase": "memory_initialization", "error": str(exc)},
                match_id=match_id,
            )
            self._emit_match_end(
                writer,
                match_id,
                metrics,
                completed=False,
                failure_reason="memory_initialization_failed",
            )
            return _MatchOutcome(False)

        try:
            self._drain_plugin_events(
                writer,
                session,
                match_id,
                turn_index=None,
                actor_id=getattr(session, "current_actor_id", None),
            )
            self._drain_match_memory_summaries(
                writer,
                session,
                memory,
                match_id,
                turn_idx=None,
                actor_id=getattr(session, "current_actor_id", None),
            )
            for turn_idx in range(self._max_turns):
                if session.is_terminal:
                    return self._complete_match(
                        writer, session, memory, match_id, metrics
                    )

                turn_result = await self._run_turn(
                    writer, session, memory, match_id, turn_idx, metrics
                )
                metrics["turn_count"] += 1
                if isinstance(turn_result, _TurnFailure):
                    failure_outcome = self._fail_match(
                        writer,
                        session,
                        memory,
                        match_id,
                        metrics,
                        turn_result,
                    )
                    if failure_outcome is None:
                        continue
                    return failure_outcome

                if turn_result == "terminal":
                    return self._complete_match(
                        writer, session, memory, match_id, metrics
                    )

            if session.is_terminal:
                return self._complete_match(writer, session, memory, match_id, metrics)

            outcome, game_metrics = self._failure_result(session)
            self._emit_match_end(
                writer,
                match_id,
                metrics,
                completed=False,
                failure_reason="max_turns_exceeded",
                outcome=outcome,
                game_metrics=game_metrics,
            )
            return _MatchOutcome(False)
        except asyncio.CancelledError:
            writer.emit("cancellation", {"phase": "match"}, match_id=match_id)
            outcome, game_metrics = self._failure_result(session)
            self._emit_match_end(
                writer,
                match_id,
                metrics,
                completed=False,
                failure_reason="cancelled",
                outcome=outcome,
                game_metrics=game_metrics,
            )
            return _MatchOutcome(False, cancelled=True)
        except Exception as exc:
            writer.emit(
                "engine_error",
                {"phase": "match", "error": str(exc)},
                match_id=match_id,
            )
            outcome, game_metrics = self._failure_result(session)
            self._emit_match_end(
                writer,
                match_id,
                metrics,
                completed=False,
                failure_reason="engine_failure",
                outcome=outcome,
                game_metrics=game_metrics,
            )
            return _MatchOutcome(False)

    async def _run_turn(
        self,
        writer: EventWriter,
        session: GameSession,
        memory: MatchMemory,
        match_id: str,
        turn_idx: int,
        metrics: dict[str, Any],
    ) -> str | _TurnFailure:
        """Run one turn with memory interaction loop."""
        actor_id: str | None = None
        try:
            actor_id = session.current_actor_id
            writer.emit(
                "turn_start",
                {},
                match_id=match_id,
                turn_index=turn_idx,
                actor_id=actor_id,
            )
            observation = session.get_observation(actor_id)
            if observation.actor_id != actor_id:
                raise RunError(
                    "Observation actor_id does not match current actor_id: "
                    f"{observation.actor_id!r} != {actor_id!r}"
                )
            game_tools = session.get_tools(actor_id)
            system_prompt = self._game.system_prompt(actor_id)
        except asyncio.CancelledError:
            writer.emit(
                "cancellation",
                {"phase": "turn_setup"},
                match_id=match_id,
                turn_index=turn_idx,
                actor_id=actor_id,
            )
            return _TurnFailure("cancelled", actor_id=actor_id, turn_index=turn_idx)
        except Exception as exc:
            writer.emit(
                "engine_error",
                {"phase": "turn_setup", "error": str(exc)},
                match_id=match_id,
                turn_index=turn_idx,
                actor_id=actor_id,
            )
            return _TurnFailure(
                "turn_setup_failed", actor_id=actor_id, turn_index=turn_idx
            )

        assert actor_id is not None

        # Check for reserved tool name collisions
        game_tool_names = {t.name for t in game_tools}
        reserved_collisions = game_tool_names & _RESERVED_MEMORY_TOOLS
        if reserved_collisions:
            writer.emit(
                "engine_error",
                {
                    "phase": "tool_validation",
                    "error": (
                        "Game tool name(s) collide with reserved memory tools: "
                        f"{', '.join(sorted(reserved_collisions))}"
                    ),
                },
                match_id=match_id,
                turn_index=turn_idx,
                actor_id=actor_id,
            )
            return _TurnFailure(
                "reserved_tool_name_collision",
                actor_id=actor_id,
                turn_index=turn_idx,
            )

        all_tools = list(game_tools) + [
            MatchMemory.read_tool_spec(),
            MatchMemory.write_tool_spec(),
        ]

        writer.emit(
            "observation",
            {"text": observation.text, "metadata": observation.metadata},
            match_id=match_id,
            turn_index=turn_idx,
            actor_id=actor_id,
        )
        writer.emit(
            "tool_schemas",
            {"tools": [tool.to_openai_tool() for tool in all_tools]},
            match_id=match_id,
            turn_index=turn_idx,
            actor_id=actor_id,
        )

        scope_id = self._conversation_scope(session, match_id)
        if self._conversation_scope_id != scope_id:
            self._actor_transcripts = {}
            self._conversation_scope_id = scope_id
        transcript = self._actor_transcripts.setdefault(actor_id, [])
        transcript.append(
            cast(JsonObject, {"role": "user", "content": observation.text})
        )
        request = self._request_from_transcript(
            actor_id,
            system_prompt,
            transcript,
            all_tools,
        )

        action_response = await self._get_action_response(
            writer,
            session,
            memory,
            request,
            all_tools,
            system_prompt,
            match_id,
            turn_idx,
            actor_id,
            metrics,
            transcript,
        )
        if isinstance(action_response, _TurnFailure):
            return action_response
        request = action_response.request
        response = action_response.response
        memory_calls = list(action_response.memory_calls)
        game_calls = list(action_response.game_calls)
        memory_ops_this_turn = action_response.memory_operations
        invalid_attempts = action_response.invalid_attempts

        # Phase 2: Handle game action with retry loop
        while True:
            # Memory calls cannot be combined with a game action.
            if memory_calls and game_calls:
                self._emit_validation_failure(
                    writer,
                    match_id,
                    turn_idx,
                    actor_id,
                    response,
                    "memory_with_game_action",
                )
                metrics["invalid_actions"] += 1
                invalid_attempts += 1
                request = self._retry_request(
                    request,
                    response,
                    "memory_with_game_action",
                    transcript,
                )
                if invalid_attempts > self._max_invalid_attempts:
                    return _TurnFailure(
                        "invalid_attempts_exhausted",
                        actor_id=actor_id,
                        turn_index=turn_idx,
                        recover=True,
                    )
                action_response = await self._get_action_response(
                    writer,
                    session,
                    memory,
                    request,
                    all_tools,
                    system_prompt,
                    match_id,
                    turn_idx,
                    actor_id,
                    metrics,
                    memory_operations=memory_ops_this_turn,
                    transcript=transcript,
                    invalid_attempts=invalid_attempts,
                )
                if isinstance(action_response, _TurnFailure):
                    return action_response
                request = action_response.request
                response = action_response.response
                memory_calls = list(action_response.memory_calls)
                game_calls = list(action_response.game_calls)
                memory_ops_this_turn = action_response.memory_operations
                invalid_attempts = action_response.invalid_attempts
                continue

            # Multiple game actions
            if len(game_calls) > 1:
                self._emit_validation_failure(
                    writer,
                    match_id,
                    turn_idx,
                    actor_id,
                    response,
                    "multiple_tool_calls",
                )
                metrics["invalid_actions"] += 1
                invalid_attempts += 1
                request = self._retry_request(
                    request, response, "multiple_tool_calls", transcript
                )
                if invalid_attempts > self._max_invalid_attempts:
                    return _TurnFailure(
                        "invalid_attempts_exhausted",
                        actor_id=actor_id,
                        turn_index=turn_idx,
                        recover=True,
                    )
                action_response = await self._get_action_response(
                    writer,
                    session,
                    memory,
                    request,
                    all_tools,
                    system_prompt,
                    match_id,
                    turn_idx,
                    actor_id,
                    metrics,
                    memory_operations=memory_ops_this_turn,
                    transcript=transcript,
                    invalid_attempts=invalid_attempts,
                )
                if isinstance(action_response, _TurnFailure):
                    return action_response
                request = action_response.request
                response = action_response.response
                memory_calls = list(action_response.memory_calls)
                game_calls = list(action_response.game_calls)
                memory_ops_this_turn = action_response.memory_operations
                invalid_attempts = action_response.invalid_attempts
                continue

            tool_call = game_calls[0]

            # Malformed arguments
            if tool_call.arguments is None:
                self._emit_validation_failure(
                    writer,
                    match_id,
                    turn_idx,
                    actor_id,
                    response,
                    "malformed_arguments",
                )
                metrics["invalid_actions"] += 1
                invalid_attempts += 1
                request = self._retry_request(
                    request, response, "malformed_arguments", transcript
                )
                if invalid_attempts > self._max_invalid_attempts:
                    return _TurnFailure(
                        "invalid_attempts_exhausted",
                        actor_id=actor_id,
                        turn_index=turn_idx,
                        recover=True,
                    )
                action_response = await self._get_action_response(
                    writer,
                    session,
                    memory,
                    request,
                    all_tools,
                    system_prompt,
                    match_id,
                    turn_idx,
                    actor_id,
                    metrics,
                    memory_operations=memory_ops_this_turn,
                    transcript=transcript,
                    invalid_attempts=invalid_attempts,
                )
                if isinstance(action_response, _TurnFailure):
                    return action_response
                request = action_response.request
                response = action_response.response
                memory_calls = list(action_response.memory_calls)
                game_calls = list(action_response.game_calls)
                memory_ops_this_turn = action_response.memory_operations
                invalid_attempts = action_response.invalid_attempts
                continue

            # Unknown game action
            if tool_call.name not in game_tool_names:
                self._emit_validation_failure(
                    writer,
                    match_id,
                    turn_idx,
                    actor_id,
                    response,
                    "invalid_action",
                )
                metrics["invalid_actions"] += 1
                invalid_attempts += 1
                request = self._retry_request(
                    request,
                    response,
                    "invalid_action",
                    transcript,
                )
                if invalid_attempts > self._max_invalid_attempts:
                    return _TurnFailure(
                        "invalid_attempts_exhausted",
                        actor_id=actor_id,
                        turn_index=turn_idx,
                        recover=True,
                    )
                action_response = await self._get_action_response(
                    writer,
                    session,
                    memory,
                    request,
                    all_tools,
                    system_prompt,
                    match_id,
                    turn_idx,
                    actor_id,
                    metrics,
                    memory_operations=memory_ops_this_turn,
                    transcript=transcript,
                    invalid_attempts=invalid_attempts,
                )
                if isinstance(action_response, _TurnFailure):
                    return action_response
                request = action_response.request
                response = action_response.response
                memory_calls = list(action_response.memory_calls)
                game_calls = list(action_response.game_calls)
                memory_ops_this_turn = action_response.memory_operations
                invalid_attempts = action_response.invalid_attempts
                continue

            transcript.append(self._assistant_tool_message(response))

            try:
                transition = session.apply_action(
                    actor_id,
                    tool_call.name,
                    tool_call.arguments,
                )
            except InvalidActionError as exc:
                self._emit_validation_failure(
                    writer,
                    match_id,
                    turn_idx,
                    actor_id,
                    response,
                    "invalid_action",
                    details=str(exc),
                )
                metrics["invalid_actions"] += 1
                invalid_attempts += 1
                request = self._retry_request(
                    request,
                    response,
                    "invalid_action",
                    transcript,
                    detail=str(exc),
                    details=exc.details,
                )
                if invalid_attempts > self._max_invalid_attempts:
                    return _TurnFailure(
                        "invalid_attempts_exhausted",
                        actor_id=actor_id,
                        turn_index=turn_idx,
                        recover=True,
                    )
                action_response = await self._get_action_response(
                    writer,
                    session,
                    memory,
                    request,
                    all_tools,
                    system_prompt,
                    match_id,
                    turn_idx,
                    actor_id,
                    metrics,
                    memory_operations=memory_ops_this_turn,
                    transcript=transcript,
                    invalid_attempts=invalid_attempts,
                )
                if isinstance(action_response, _TurnFailure):
                    return action_response
                request = action_response.request
                response = action_response.response
                memory_calls = list(action_response.memory_calls)
                game_calls = list(action_response.game_calls)
                memory_ops_this_turn = action_response.memory_operations
                invalid_attempts = action_response.invalid_attempts
                continue
            except asyncio.CancelledError:
                writer.emit(
                    "cancellation",
                    {"phase": "action_application"},
                    match_id=match_id,
                    turn_index=turn_idx,
                    actor_id=actor_id,
                )
                return _TurnFailure(
                    "cancelled",
                    actor_id=actor_id,
                    turn_index=turn_idx,
                )
            except Exception as exc:
                writer.emit(
                    "engine_error",
                    {"phase": "action_application", "error": str(exc)},
                    match_id=match_id,
                    turn_index=turn_idx,
                    actor_id=actor_id,
                )
                return _TurnFailure(
                    "action_application_failed",
                    actor_id=actor_id,
                    turn_index=turn_idx,
                )

            writer.emit(
                "validation",
                {"valid": True, "tool_name": tool_call.name},
                match_id=match_id,
                turn_index=turn_idx,
                actor_id=actor_id,
            )
            writer.emit(
                "transition",
                {"summary": transition.summary, "metrics": transition.metrics},
                match_id=match_id,
                turn_index=turn_idx,
                actor_id=actor_id,
            )
            self._drain_plugin_events(
                writer,
                session,
                match_id,
                turn_index=turn_idx,
                actor_id=actor_id,
            )

            transcript.append(
                cast(
                    JsonObject,
                    {
                        "role": "tool",
                        "content": json.dumps(
                            {
                                "summary": transition.summary,
                                "metrics": transition.metrics,
                                "terminal": session.is_terminal,
                            }
                        ),
                        "tool_call_id": tool_call.call_id,
                    },
                )
            )

            self._drain_match_memory_summaries(
                writer,
                session,
                memory,
                match_id,
                turn_idx,
                actor_id,
            )

            finalization_result = await self._finalize_turn(
                writer,
                request=self._request_from_transcript(
                    actor_id,
                    system_prompt,
                    transcript,
                    [],
                ),
                match_id=match_id,
                turn_idx=turn_idx,
                actor_id=actor_id,
                metrics=metrics,
                transcript=transcript,
            )
            if isinstance(finalization_result, _TurnFailure):
                return finalization_result

            writer.emit(
                "turn_end",
                {"status": "ok"},
                match_id=match_id,
                turn_index=turn_idx,
                actor_id=actor_id,
            )
            return "terminal" if session.is_terminal else "ok"

        return _TurnFailure(
            "invalid_attempts_exhausted",
            actor_id=actor_id,
            turn_index=turn_idx,
            recover=True,
        )

    async def _get_action_response(
        self,
        writer: EventWriter,
        session: GameSession,
        memory: MatchMemory,
        request: AgentRequest,
        all_tools: list[Any],
        system_prompt: str,
        match_id: str,
        turn_idx: int,
        actor_id: str,
        metrics: dict[str, Any],
        transcript: list[JsonObject],
        memory_operations: int = 0,
        invalid_attempts: int = 0,
    ) -> _ActionResponse | _TurnFailure:
        """Return a response containing a game call after memory-only calls."""
        while True:
            response_or_failure = await self._get_response(
                writer,
                request,
                match_id,
                turn_idx,
                actor_id,
                metrics,
            )
            if isinstance(response_or_failure, _ResponseFailure):
                return _TurnFailure(
                    response_or_failure.reason,
                    actor_id=actor_id,
                    turn_index=turn_idx,
                )
            response = response_or_failure
            if not response.tool_calls:
                self._emit_validation_failure(
                    writer,
                    match_id,
                    turn_idx,
                    actor_id,
                    response,
                    "no_tool_calls",
                )
                metrics["invalid_actions"] += 1
                invalid_attempts += 1
                request = self._retry_request(
                    request, response, "no_tool_calls", transcript
                )
                if invalid_attempts > self._max_invalid_attempts:
                    return _TurnFailure(
                        "invalid_attempts_exhausted",
                        actor_id=actor_id,
                        turn_index=turn_idx,
                        recover=True,
                    )
                continue

            memory_calls = tuple(
                call
                for call in response.tool_calls
                if call.name in _RESERVED_MEMORY_TOOLS
            )
            game_calls = tuple(
                call
                for call in response.tool_calls
                if call.name not in _RESERVED_MEMORY_TOOLS
            )
            if game_calls:
                return _ActionResponse(
                    request=self._request_from_transcript(
                        actor_id,
                        system_prompt,
                        transcript,
                        all_tools,
                    ),
                    response=response,
                    memory_calls=memory_calls,
                    game_calls=game_calls,
                    memory_operations=memory_operations,
                    invalid_attempts=invalid_attempts,
                )

            hand_idx, in_hand_turn = self._memory_context(session, turn_idx)
            transcript.append(self._assistant_tool_message(response))
            for memory_index, mem_call in enumerate(memory_calls):
                memory_operations += 1
                if memory_operations > self._max_memory_operations_per_turn:
                    for rejected_call in memory_calls[memory_index:]:
                        writer.emit(
                            "memory_operation",
                            {
                                "operation": "budget_exhausted",
                                "tool_name": rejected_call.name,
                            },
                            match_id=match_id,
                            turn_index=turn_idx,
                            actor_id=actor_id,
                        )
                        transcript.append(
                            {
                                "role": "tool",
                                "content": json.dumps(
                                    {"error": "Memory operation budget exceeded"}
                                ),
                                "tool_call_id": rejected_call.call_id,
                            }
                        )
                    return _TurnFailure(
                        "memory_budget_exhausted",
                        actor_id=actor_id,
                        turn_index=turn_idx,
                    )

                if mem_call.name == "read_memory" and mem_call.arguments is not None:
                    notes = memory.read(actor_id)
                    result_content = json.dumps(notes)
                    writer.emit(
                        "memory_operation",
                        cast(
                            JsonObject,
                            {"operation": "read", "count": len(notes), "notes": notes},
                        ),
                        match_id=match_id,
                        turn_index=turn_idx,
                        actor_id=actor_id,
                    )
                elif mem_call.name == "write_memory" and mem_call.arguments is not None:
                    text_value = mem_call.arguments.get("text")
                    if isinstance(text_value, str) and text_value.strip():
                        try:
                            memory.write(
                                actor_id,
                                text=text_value,
                                hand=hand_idx,
                                turn=in_hand_turn,
                            )
                        except ValueError as exc:
                            result_content = json.dumps({"error": str(exc)})
                            writer.emit(
                                "memory_operation",
                                {"operation": "write_rejected", "error": str(exc)},
                                match_id=match_id,
                                turn_index=turn_idx,
                                actor_id=actor_id,
                            )
                        else:
                            result_content = json.dumps({"status": "ok"})
                            writer.emit(
                                "memory_operation",
                                {"operation": "write", "text_length": len(text_value)},
                                match_id=match_id,
                                turn_index=turn_idx,
                                actor_id=actor_id,
                            )
                    else:
                        result_content = json.dumps({"error": "Invalid text"})
                        writer.emit(
                            "memory_operation",
                            {"operation": "write_rejected", "error": "Invalid text"},
                            match_id=match_id,
                            turn_index=turn_idx,
                            actor_id=actor_id,
                        )
                else:
                    result_content = json.dumps({"error": "Malformed memory arguments"})
                    writer.emit(
                        "memory_operation",
                        {
                            "operation": "rejected",
                            "tool_name": mem_call.name,
                            "error": "Malformed memory arguments",
                        },
                        match_id=match_id,
                        turn_index=turn_idx,
                        actor_id=actor_id,
                    )

                transcript.append(
                    {
                        "role": "tool",
                        "content": result_content,
                        "tool_call_id": mem_call.call_id,
                    }
                )
            request = self._request_from_transcript(
                actor_id,
                system_prompt,
                transcript,
                all_tools,
            )

    async def _finalize_turn(
        self,
        writer: EventWriter,
        request: AgentRequest,
        match_id: str,
        turn_idx: int,
        actor_id: str,
        metrics: dict[str, Any],
        transcript: list[JsonObject],
    ) -> _TurnFailure | None:
        response_or_failure = await self._get_response(
            writer,
            request,
            match_id,
            turn_idx,
            actor_id,
            metrics,
        )
        if isinstance(response_or_failure, _ResponseFailure):
            return _TurnFailure(
                response_or_failure.reason,
                actor_id=actor_id,
                turn_index=turn_idx,
            )

        response = response_or_failure
        if response.tool_calls:
            self._emit_validation_failure(
                writer,
                match_id,
                turn_idx,
                actor_id,
                response,
                "finalization_tool_calls",
            )
            return _TurnFailure(
                "finalization_tool_calls",
                actor_id=actor_id,
                turn_index=turn_idx,
            )

        if not response.assistant_text and not (
            response.finish_reason and response.finish_reason.strip()
        ):
            self._emit_validation_failure(
                writer,
                match_id,
                turn_idx,
                actor_id,
                response,
                "finalization_empty_response",
            )
            return _TurnFailure(
                "finalization_empty_response",
                actor_id=actor_id,
                turn_index=turn_idx,
            )

        if response.assistant_text:
            transcript.append(
                cast(
                    JsonObject,
                    {"role": "assistant", "content": response.assistant_text},
                )
            )
        return None

    @staticmethod
    def _memory_context(session: GameSession, fallback_turn: int) -> tuple[int, int]:
        context = getattr(session, "get_turn_context", None)
        if not callable(context):
            return fallback_turn, fallback_turn
        values = cast(JsonObject, context())
        hand_index = (
            cast(int, values["hand_index"])
            if type(values.get("hand_index")) is int
            else fallback_turn
        )
        in_hand_turn = (
            cast(int, values["in_hand_turn"])
            if type(values.get("in_hand_turn")) is int
            else fallback_turn
        )
        return hand_index, in_hand_turn

    def _drain_match_memory_summaries(
        self,
        writer: EventWriter,
        session: GameSession,
        memory: MatchMemory,
        match_id: str,
        turn_idx: int | None,
        actor_id: str | None,
    ) -> None:
        drain = getattr(session, "drain_match_memory_summaries", None)
        if not callable(drain):
            return
        summaries_raw: object = drain()
        if type(summaries_raw) is not list:
            raise RunError("Plugin summary drain must return a list")
        for summary_raw in cast(list[object], summaries_raw):
            summary = MatchMemorySummary.model_validate(summary_raw)
            memory.append_summary(summary)
            writer.emit(
                "memory_operation",
                {
                    "operation": "summary",
                    "actor_id": summary.actor_id,
                    "hand": summary.hand,
                    "turn": summary.turn,
                    "text": summary.text,
                    "text_length": len(summary.text),
                },
                match_id=match_id,
                turn_index=turn_idx,
                actor_id=summary.actor_id,
            )

    @staticmethod
    def _conversation_scope(session: GameSession, fallback_scope: str) -> str:
        scope = getattr(session, "conversation_scope_id", None)
        if scope is None:
            return fallback_scope
        if type(scope) is not str or not scope:
            raise RunError("conversation_scope_id must be a non-empty string")
        return scope

    @staticmethod
    def _request_from_transcript(
        actor_id: str,
        system_prompt: str,
        transcript: list[JsonObject],
        tools: list[Any],
    ) -> AgentRequest:
        return AgentRequest(
            actor_id=actor_id,
            system_prompt=system_prompt,
            messages=copy.deepcopy(transcript),
            tools=list(tools),
        )

    async def _get_response(
        self,
        writer: EventWriter,
        request: AgentRequest,
        match_id: str,
        turn_idx: int,
        actor_id: str,
        metrics: dict[str, Any],
    ) -> ModelResponse | _ResponseFailure:
        """Call the selected actor agent with provider retry logging."""
        try:
            agent = self._select_agent(actor_id)
        except RunError as exc:
            writer.emit(
                "engine_error",
                {"error": "missing_agent", "details": str(exc)},
                match_id=match_id,
                turn_index=turn_idx,
                actor_id=actor_id,
            )
            return _ResponseFailure("missing_agent")

        for attempt in range(self._max_provider_retries + 1):
            writer.emit(
                "model_request",
                cast(
                    JsonObject,
                    {
                        "actor_id": actor_id,
                        "system_prompt": request.system_prompt,
                        "messages": self._provider_messages(request),
                        "tools": [tool.to_openai_tool() for tool in request.tools],
                        "attempt": attempt + 1,
                    },
                ),
                match_id=match_id,
                turn_index=turn_idx,
                actor_id=actor_id,
            )
            started = time.monotonic()
            try:
                response = await agent.respond(request)
            except ProviderError as exc:
                metrics["provider_failures"] += 1
                metrics["latency_seconds"] += time.monotonic() - started
                provider_error_payload: dict[str, Any] = {
                    "error": str(exc),
                    "provider": exc.provider,
                    "model": exc.model,
                    "attempt": attempt + 1,
                }
                if exc.raw_provider_response is not None:
                    provider_error_payload["raw_provider_response"] = (
                        exc.raw_provider_response
                    )
                writer.emit(
                    "provider_error",
                    cast(JsonObject, provider_error_payload),
                    match_id=match_id,
                    turn_index=turn_idx,
                    actor_id=actor_id,
                )
                if attempt >= self._max_provider_retries:
                    return _ResponseFailure("provider_retries_exhausted")
                continue
            except TimeoutError as exc:
                metrics["provider_failures"] += 1
                metrics["latency_seconds"] += time.monotonic() - started
                writer.emit(
                    "provider_error",
                    {
                        "error": str(exc),
                        "attempt": attempt + 1,
                    },
                    match_id=match_id,
                    turn_index=turn_idx,
                    actor_id=actor_id,
                )
                if attempt >= self._max_provider_retries:
                    return _ResponseFailure("provider_retries_exhausted")
                continue
            except asyncio.CancelledError:
                metrics["latency_seconds"] += time.monotonic() - started
                writer.emit(
                    "cancellation",
                    {"phase": "provider_request", "attempt": attempt + 1},
                    match_id=match_id,
                    turn_index=turn_idx,
                    actor_id=actor_id,
                )
                return _ResponseFailure("cancelled")
            except Exception as exc:
                metrics["latency_seconds"] += time.monotonic() - started
                writer.emit(
                    "engine_error",
                    {
                        "phase": "agent_respond",
                        "error": str(exc),
                        "attempt": attempt + 1,
                    },
                    match_id=match_id,
                    turn_index=turn_idx,
                    actor_id=actor_id,
                )
                return _ResponseFailure("agent_failure")

            latency = time.monotonic() - started
            metrics["latency_seconds"] += latency
            if response.usage:
                metrics["prompt_tokens"] += response.usage.prompt_tokens
                metrics["completion_tokens"] += response.usage.completion_tokens
                metrics["total_tokens"] += response.usage.total_tokens

            writer.emit(
                "model_response",
                cast(
                    JsonObject,
                    {
                        "assistant_text": response.assistant_text,
                        "tool_calls": [
                            self._tool_call_payload(call)
                            for call in response.tool_calls
                        ],
                        "finish_reason": response.finish_reason,
                        "usage": (
                            response.usage.model_dump(mode="json")
                            if response.usage
                            else None
                        ),
                        "latency_seconds": latency,
                        "raw_provider_response": response.raw_provider_response,
                    },
                ),
                match_id=match_id,
                turn_index=turn_idx,
                actor_id=actor_id,
            )
            return response

        return _ResponseFailure("provider_retries_exhausted")

    def _select_agent(self, actor_id: str) -> Agent:
        if self._agents is not None:
            agent = self._agents.get(actor_id)
            if agent is None:
                raise RunError(f"No agent configured for actor {actor_id}")
            return agent
        if self._agent is None:
            raise RunError(f"No agent configured for actor {actor_id}")
        return self._agent

    def _exact_agent_mapping_error(self) -> str | None:
        try:
            if not bool(getattr(self._game, "requires_exact_agent_ids", False)):
                return None

            resolver = getattr(self._game, "player_ids_from_config", None)
            if callable(resolver):
                actor_ids_raw: object = cast(Callable[[JsonObject], object], resolver)(
                    copy.deepcopy(self._game_config)
                )
            else:
                actor_ids_raw = cast(object, getattr(self._game, "player_ids"))
        except Exception as exc:
            return f"Could not resolve exact game actors: {exc}"

        if type(actor_ids_raw) is not list:
            return "Exact-agent game returned invalid actor IDs"
        actor_ids = cast(list[object], actor_ids_raw)
        if not all(type(actor_id) is str for actor_id in actor_ids):
            return "Exact-agent game returned invalid actor IDs"
        expected = set(cast(list[str], actor_ids))
        if self._agents is not None:
            configured = set(self._agents)
            if configured != expected:
                return (
                    f"configured agent IDs {sorted(configured)} do not match "
                    f"game actors {sorted(expected)}"
                )
            return None

        if self._agent is None:
            return "No exact agent mapping was provided"
        if len(actor_ids) != 1 or self._agent_id not in expected:
            return (
                "A single direct agent is only valid for a one-actor game "
                "with a matching agent_id"
            )
        return None

    @staticmethod
    def _provider_messages(request: AgentRequest) -> list[JsonObject]:
        return [
            {"role": "system", "content": request.system_prompt},
            *request.messages,
        ]

    @staticmethod
    def _tool_call_payload(tool_call: ToolCall) -> JsonObject:
        arguments = redact_value(tool_call.arguments)
        raw_arguments = tool_call.raw_arguments
        if raw_arguments is None:
            raw_arguments = json.dumps(arguments or {})
        raw_arguments = redact_json_text(raw_arguments)
        return cast(
            JsonObject,
            {
                "call_id": tool_call.call_id,
                "name": tool_call.name,
                "arguments": arguments,
                "raw_arguments": raw_arguments,
                "parse_error": tool_call.parse_error,
            },
        )

    def _retry_request(
        self,
        request: AgentRequest,
        response: ModelResponse,
        error: str,
        transcript: list[JsonObject] | None = None,
        *,
        detail: str | None = None,
        details: object | None = None,
    ) -> AgentRequest:
        validation_message = f"Validation error: {error}"
        if detail:
            validation_message += f": {redact_text(detail)}"
        if details is not None:
            try:
                serialized_details = json.dumps(details, sort_keys=True)
            except (TypeError, ValueError, OverflowError):
                try:
                    serialized_details = repr(details)
                except Exception:
                    serialized_details = "<unserializable details>"
                serialized_details = redact_text(serialized_details)
            else:
                serialized_details = redact_json_text(serialized_details)
            validation_message += f" Details: {serialized_details}"
        validation_message += ". Please try again."
        messages = [self._redact_retry_message(message) for message in request.messages]
        if response.tool_calls:
            messages.append(self._assistant_tool_message(response, redact=True))
            for tool_call in response.tool_calls:
                messages.append(
                    {
                        "role": "tool",
                        "content": validation_message,
                        "tool_call_id": tool_call.call_id,
                    }
                )
        else:
            messages.append(
                {
                    "role": "assistant",
                    "content": redact_json_text(response.assistant_text),
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": validation_message,
                }
            )

        if transcript is not None:
            transcript.clear()
            transcript.extend(copy.deepcopy(messages))

        return request.model_copy(update={"messages": messages})

    @staticmethod
    def _redact_retry_message(message: JsonObject) -> JsonObject:
        """Redact model-visible content without changing protocol identifiers."""
        redacted = copy.deepcopy(message)
        if "content" in redacted:
            content = redacted["content"]
            redacted["content"] = (
                redact_json_text(content)
                if isinstance(content, str)
                else redact_value(content)
            )

        tool_calls = redacted.get("tool_calls")
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function")
                if not isinstance(function, dict):
                    continue
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    function["arguments"] = redact_json_text(arguments)
                elif isinstance(arguments, dict | list):
                    function["arguments"] = redact_value(arguments)
        return redacted

    @staticmethod
    def _assistant_tool_message(
        response: ModelResponse, *, redact: bool = False
    ) -> JsonObject:
        calls: list[JsonObject] = []
        for tool_call in response.tool_calls:
            raw_arguments = tool_call.raw_arguments
            if raw_arguments is None:
                raw_arguments = json.dumps(tool_call.arguments or {})
            if redact:
                raw_arguments = redact_json_text(raw_arguments)
            calls.append(
                {
                    "id": tool_call.call_id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": raw_arguments,
                    },
                }
            )
        return cast(
            JsonObject,
            {
                "role": "assistant",
                "content": (
                    redact_json_text(response.assistant_text)
                    if redact
                    else response.assistant_text
                ),
                "tool_calls": calls,
            },
        )

    def _emit_validation_failure(
        self,
        writer: EventWriter,
        match_id: str,
        turn_idx: int,
        actor_id: str,
        response: ModelResponse,
        error: str,
        *,
        details: str | None = None,
    ) -> None:
        payload: JsonObject = {
            "valid": False,
            "error": error,
            "assistant_text": response.assistant_text,
        }
        if error == "multiple_tool_calls":
            payload["count"] = len(response.tool_calls)
        if response.tool_calls:
            payload["tool_name"] = response.tool_calls[0].name
            if error == "malformed_arguments":
                payload["parse_error"] = response.tool_calls[0].parse_error
        if details is not None:
            payload["details"] = details
        writer.emit(
            "validation",
            payload,
            match_id=match_id,
            turn_index=turn_idx,
            actor_id=actor_id,
        )

    def _complete_match(
        self,
        writer: EventWriter,
        session: GameSession,
        memory: MatchMemory,
        match_id: str,
        metrics: dict[str, Any],
    ) -> _MatchOutcome:
        self._drain_plugin_events(
            writer,
            session,
            match_id,
            turn_index=None,
            actor_id=getattr(session, "current_actor_id", None),
        )
        self._drain_match_memory_summaries(
            writer,
            session,
            memory,
            match_id,
            turn_idx=None,
            actor_id=getattr(session, "current_actor_id", None),
        )
        try:
            result = session.get_result()
        except Exception as exc:
            writer.emit(
                "engine_error",
                {"phase": "result", "error": str(exc)},
                match_id=match_id,
            )
            self._emit_match_end(
                writer,
                match_id,
                metrics,
                completed=False,
                failure_reason="result_failed",
            )
            return _MatchOutcome(False)

        self._emit_match_end(
            writer,
            match_id,
            metrics,
            completed=result.completed,
            outcome=result.outcome,
            game_metrics=result.metrics,
        )
        return _MatchOutcome(result.completed)

    @staticmethod
    def _drain_plugin_events(
        writer: EventWriter,
        session: GameSession,
        match_id: str,
        *,
        turn_index: int | None,
        actor_id: str | None,
    ) -> None:
        drain = getattr(session, "drain_hand_events", None)
        if not callable(drain):
            return
        events_raw: object = drain()
        if type(events_raw) is not list:
            raise RunError("Plugin event drain must return a list")
        for event_raw in cast(list[object], events_raw):
            try:
                event = PluginEvent.model_validate(event_raw)
            except ValidationError as exc:
                raise RunError("Plugin event must match its typed contract") from exc
            writer.emit(
                event.event_type,
                event.payload,
                match_id=match_id,
                turn_index=turn_index,
                actor_id=actor_id,
            )

    def _fail_match(
        self,
        writer: EventWriter,
        session: GameSession,
        memory: MatchMemory,
        match_id: str,
        metrics: dict[str, Any],
        failure: _TurnFailure,
    ) -> _MatchOutcome | None:
        if (
            failure.reason == "invalid_attempts_exhausted"
            and failure.actor_id is not None
        ):
            handler = getattr(session, "handle_failed_turn", None)
            recovery: Transition | None = None
            cancelled = False
            try:
                if callable(handler):
                    recovery = cast(
                        Transition | None,
                        handler(failure.actor_id, failure.reason),
                    )
            except asyncio.CancelledError:
                writer.emit(
                    "cancellation",
                    {"phase": "failed_turn_recovery"},
                    match_id=match_id,
                    turn_index=failure.turn_index,
                    actor_id=failure.actor_id,
                )
                cancelled = True
            except Exception as exc:
                writer.emit(
                    "engine_error",
                    {"phase": "failed_turn_recovery", "error": str(exc)},
                    match_id=match_id,
                    turn_index=failure.turn_index,
                    actor_id=failure.actor_id,
                )
            else:
                if recovery is not None:
                    self._emit_recovery_transition(
                        writer,
                        match_id,
                        failure.actor_id,
                        failure.turn_index,
                        recovery,
                    )
                    self._drain_plugin_events(
                        writer,
                        session,
                        match_id,
                        turn_index=failure.turn_index,
                        actor_id=failure.actor_id,
                    )
                    self._drain_match_memory_summaries(
                        writer,
                        session,
                        memory,
                        match_id,
                        turn_idx=failure.turn_index,
                        actor_id=failure.actor_id,
                    )
                if callable(handler):
                    try:
                        if session.is_terminal:
                            if recovery is not None:
                                self._emit_failed_turn_end(writer, match_id, failure)
                                return self._complete_match(
                                    writer, session, memory, match_id, metrics
                                )
                    except Exception as exc:
                        writer.emit(
                            "engine_error",
                            {"phase": "failed_turn_recovery", "error": str(exc)},
                            match_id=match_id,
                            turn_index=failure.turn_index,
                            actor_id=failure.actor_id,
                        )

            if cancelled:
                self._emit_failed_turn_end(writer, match_id, failure)
                outcome, game_metrics = self._failure_result(session)
                self._emit_match_end(
                    writer,
                    match_id,
                    metrics,
                    completed=False,
                    failure_reason="cancelled",
                    outcome=outcome,
                    game_metrics=game_metrics,
                )
                return _MatchOutcome(False, cancelled=True)

            if recovery is not None:
                self._emit_failed_turn_end(writer, match_id, failure)
                return None

        self._emit_failed_turn_end(writer, match_id, failure)
        outcome, game_metrics = self._failure_result(session)
        self._emit_match_end(
            writer,
            match_id,
            metrics,
            completed=False,
            failure_reason=failure.reason,
            outcome=outcome,
            game_metrics=game_metrics,
        )
        return _MatchOutcome(False, cancelled=failure.reason == "cancelled")

    @staticmethod
    def _failure_result(session: GameSession) -> tuple[JsonObject, dict[str, Any]]:
        try:
            result = session.get_result()
        except Exception:
            return {}, {}
        outcome = copy.deepcopy(result.outcome)
        outcome.pop("winner", None)
        outcome.pop("winners", None)
        outcome["completed"] = False
        game_metrics = cast(dict[str, Any], copy.deepcopy(result.metrics))
        game_metrics["completed"] = False
        return outcome, game_metrics

    @staticmethod
    def _emit_failed_turn_end(
        writer: EventWriter,
        match_id: str,
        failure: _TurnFailure,
    ) -> None:
        writer.emit(
            "turn_end",
            {"status": "failed", "failure_reason": failure.reason},
            match_id=match_id,
            turn_index=failure.turn_index,
            actor_id=failure.actor_id,
        )

    @staticmethod
    def _emit_recovery_transition(
        writer: EventWriter,
        match_id: str,
        actor_id: str,
        turn_index: int | None,
        transition: Transition,
    ) -> None:
        writer.emit(
            "transition",
            {
                "summary": transition.summary,
                "metrics": transition.metrics,
                "recovery": True,
            },
            match_id=match_id,
            turn_index=turn_index,
            actor_id=actor_id,
        )

    @staticmethod
    def _emit_match_end(
        writer: EventWriter,
        match_id: str,
        metrics: dict[str, Any],
        *,
        completed: bool,
        failure_reason: str | None = None,
        outcome: JsonObject | None = None,
        game_metrics: dict[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "result": {
                "completed": completed,
                "outcome": outcome or {},
                "metrics": game_metrics or {},
            },
            "metrics": metrics,
        }
        if failure_reason is not None:
            payload["failure_reason"] = failure_reason
        writer.emit("match_end", payload, match_id=match_id)
