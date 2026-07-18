"""Agent protocol for model providers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from benchtable.contracts import JsonObject, ModelResponse, ToolSpec


class AgentRequest(BaseModel):
    """Input to an agent for a single model call."""

    model_config = ConfigDict(extra="forbid")

    actor_id: str
    system_prompt: str
    messages: list[JsonObject]
    tools: list[ToolSpec]


@runtime_checkable
class Agent(Protocol):
    """A model agent that can produce a response for a given request."""

    async def respond(self, request: AgentRequest) -> ModelResponse: ...
