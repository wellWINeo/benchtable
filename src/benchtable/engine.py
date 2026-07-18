"""Benchmark run engine: orchestrates matches between game plugins and agents."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast
from uuid import uuid4

from benchtable.agents.protocol import Agent, AgentRequest
from benchtable.contracts import JsonObject, ModelResponse, ToolCall, Transition
from benchtable.errors import InvalidActionError, ProviderError, RunError
from benchtable.events import EventWriter
from benchtable.games.protocol import GamePlugin, GameSession


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
                },
            ),
        )

    def _derive_match_seed(self, match_idx: int) -> int:
        h = hashlib.sha256(f"{self._seed}:{match_idx}".encode()).hexdigest()
        return int(h[:16], 16)

    async def _run_match(self, writer: EventWriter, match_idx: int) -> _MatchOutcome:
        match_id = f"match-{match_idx}"
        seed = self._derive_match_seed(match_idx)
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

        try:
            for turn_idx in range(self._max_turns):
                if session.is_terminal:
                    return self._complete_match(writer, session, match_id, metrics)

                turn_result = await self._run_turn(
                    writer, session, match_id, turn_idx, metrics
                )
                metrics["turn_count"] += 1
                if isinstance(turn_result, _TurnFailure):
                    return self._fail_match(
                        writer,
                        session,
                        match_id,
                        metrics,
                        turn_result,
                    )

                if turn_result == "terminal":
                    return self._complete_match(writer, session, match_id, metrics)

            if session.is_terminal:
                return self._complete_match(writer, session, match_id, metrics)

            self._emit_match_end(
                writer,
                match_id,
                metrics,
                completed=False,
                failure_reason="max_turns_exceeded",
            )
            return _MatchOutcome(False)
        except asyncio.CancelledError:
            writer.emit("cancellation", {"phase": "match"}, match_id=match_id)
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
                {"phase": "match", "error": str(exc)},
                match_id=match_id,
            )
            self._emit_match_end(
                writer,
                match_id,
                metrics,
                completed=False,
                failure_reason="engine_failure",
            )
            return _MatchOutcome(False)

    async def _run_turn(
        self,
        writer: EventWriter,
        session: GameSession,
        match_id: str,
        turn_idx: int,
        metrics: dict[str, Any],
    ) -> str | _TurnFailure:
        """Run one turn and return whether it was accepted or terminal."""
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
            tools = session.get_tools(actor_id)
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

        writer.emit(
            "observation",
            {"text": observation.text, "metadata": observation.metadata},
            match_id=match_id,
            turn_index=turn_idx,
            actor_id=actor_id,
        )
        writer.emit(
            "tool_schemas",
            {"tools": [tool.to_openai_tool() for tool in tools]},
            match_id=match_id,
            turn_index=turn_idx,
            actor_id=actor_id,
        )

        request = AgentRequest(
            actor_id=actor_id,
            system_prompt=system_prompt,
            messages=[{"role": "user", "content": observation.text}],
            tools=tools,
        )

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
        for invalid_attempt in range(self._max_invalid_attempts + 1):
            structural_error = self._structural_action_error(response, tools)
            if structural_error is not None:
                self._emit_validation_failure(
                    writer,
                    match_id,
                    turn_idx,
                    actor_id,
                    response,
                    structural_error,
                )
                metrics["invalid_actions"] += 1
            else:
                tool_call = response.tool_calls[0]
                try:
                    transition = session.apply_action(
                        actor_id,
                        tool_call.name,
                        tool_call.arguments or {},
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
                    structural_error = "invalid_action"
                except asyncio.CancelledError:
                    writer.emit(
                        "cancellation",
                        {"phase": "action_application"},
                        match_id=match_id,
                        turn_index=turn_idx,
                        actor_id=actor_id,
                    )
                    return _TurnFailure(
                        "cancelled", actor_id=actor_id, turn_index=turn_idx
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
                else:
                    writer.emit(
                        "validation",
                        {
                            "valid": True,
                            "tool_name": tool_call.name,
                        },
                        match_id=match_id,
                        turn_index=turn_idx,
                        actor_id=actor_id,
                    )
                    writer.emit(
                        "transition",
                        {
                            "summary": transition.summary,
                            "metrics": transition.metrics,
                        },
                        match_id=match_id,
                        turn_index=turn_idx,
                        actor_id=actor_id,
                    )
                    writer.emit(
                        "turn_end",
                        {"status": "ok"},
                        match_id=match_id,
                        turn_index=turn_idx,
                        actor_id=actor_id,
                    )
                    return "terminal" if session.is_terminal else "ok"

            if invalid_attempt >= self._max_invalid_attempts:
                return _TurnFailure(
                    "invalid_attempts_exhausted",
                    actor_id=actor_id,
                    turn_index=turn_idx,
                    recover=True,
                )

            request = self._retry_request(
                request,
                response,
                structural_error or "invalid_action",
            )
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

        return _TurnFailure(
            "invalid_attempts_exhausted",
            actor_id=actor_id,
            turn_index=turn_idx,
            recover=True,
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
                writer.emit(
                    "provider_error",
                    {
                        "error": str(exc),
                        "provider": exc.provider,
                        "model": exc.model,
                        "attempt": attempt + 1,
                    },
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

    @staticmethod
    def _provider_messages(request: AgentRequest) -> list[JsonObject]:
        return [
            {"role": "system", "content": request.system_prompt},
            *request.messages,
        ]

    @staticmethod
    def _tool_call_payload(tool_call: ToolCall) -> JsonObject:
        return cast(
            JsonObject,
            {
                "call_id": tool_call.call_id,
                "name": tool_call.name,
                "arguments": tool_call.arguments,
                "raw_arguments": tool_call.raw_arguments,
                "parse_error": tool_call.parse_error,
            },
        )

    @staticmethod
    def _structural_action_error(
        response: ModelResponse, tools: list[Any]
    ) -> str | None:
        if not response.tool_calls:
            return "no_tool_calls"
        if len(response.tool_calls) > 1:
            return "multiple_tool_calls"
        tool_call = response.tool_calls[0]
        if tool_call.arguments is None:
            return "malformed_arguments"
        if tool_call.name not in {tool.name for tool in tools}:
            return "invalid_action"
        return None

    def _retry_request(
        self,
        request: AgentRequest,
        response: ModelResponse,
        error: str,
    ) -> AgentRequest:
        messages = list(request.messages)
        if response.tool_calls:
            messages.append(self._assistant_tool_message(response))
            for tool_call in response.tool_calls:
                messages.append(
                    {
                        "role": "tool",
                        "content": f"Validation error: {error}. Please try again.",
                        "tool_call_id": tool_call.call_id,
                    }
                )
        else:
            messages.append({"role": "assistant", "content": response.assistant_text})
            messages.append(
                {
                    "role": "user",
                    "content": f"Validation error: {error}. Please try again.",
                }
            )

        return request.model_copy(update={"messages": messages})

    @staticmethod
    def _assistant_tool_message(response: ModelResponse) -> JsonObject:
        calls: list[JsonObject] = []
        for tool_call in response.tool_calls:
            raw_arguments = tool_call.raw_arguments
            if raw_arguments is None:
                raw_arguments = json.dumps(tool_call.arguments or {})
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
                "content": response.assistant_text,
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
        match_id: str,
        metrics: dict[str, Any],
    ) -> _MatchOutcome:
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

    def _fail_match(
        self,
        writer: EventWriter,
        session: GameSession,
        match_id: str,
        metrics: dict[str, Any],
        failure: _TurnFailure,
    ) -> _MatchOutcome:
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
                if callable(handler):
                    try:
                        if session.is_terminal:
                            self._emit_failed_turn_end(writer, match_id, failure)
                            return self._complete_match(
                                writer, session, match_id, metrics
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
                self._emit_match_end(
                    writer,
                    match_id,
                    metrics,
                    completed=False,
                    failure_reason="cancelled",
                )
                return _MatchOutcome(False, cancelled=True)

        self._emit_failed_turn_end(writer, match_id, failure)
        self._emit_match_end(
            writer,
            match_id,
            metrics,
            completed=False,
            failure_reason=failure.reason,
        )
        return _MatchOutcome(False, cancelled=failure.reason == "cancelled")

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
