"""Deterministic fake agent for testing."""

from __future__ import annotations

from benchtable.agents.protocol import AgentRequest
from benchtable.contracts import ModelResponse, ToolCall, Usage


class FakeAgent:
    """Deterministic agent that returns pre-configured responses.

    If `responses` is a list, each call to `respond` pops the next response.
    If `responses` is a single ModelResponse, it is returned every time.
    If `tool_name` is set, responses are auto-generated as single tool calls.
    """

    def __init__(
        self,
        *,
        responses: list[ModelResponse] | ModelResponse | None = None,
        tool_name: str | None = None,
        raise_on_respond: BaseException | None = None,
    ) -> None:
        self._raise = raise_on_respond
        self.requests: list[AgentRequest] = []

        if responses is not None:
            if isinstance(responses, ModelResponse):
                self._responses = [responses]
            else:
                self._responses = list(responses)
        elif tool_name is not None:
            self._responses = []
            self._tool_name = tool_name
            self._auto_generate = True
        else:
            self._responses = []
            self._tool_name = "act"
            self._auto_generate = True

        self._auto_generate = tool_name is not None or (responses is None and True)
        if hasattr(self, "_tool_name") is False:
            self._tool_name = "act"
        self._call_count = 0

    async def respond(self, request: AgentRequest) -> ModelResponse:
        self.requests.append(request)
        self._call_count += 1

        if self._raise is not None:
            raise self._raise

        if not request.tools:
            return ModelResponse(assistant_text="", tool_calls=[], finish_reason="stop")

        if self._responses:
            return self._responses.pop(0)

        # Auto-generate a single tool call response
        return ModelResponse(
            assistant_text=f"I'll call {self._tool_name}.",
            tool_calls=[
                ToolCall(
                    call_id=f"call-{self._call_count}",
                    name=self._tool_name,
                    arguments={},
                )
            ],
            finish_reason="stop",
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
