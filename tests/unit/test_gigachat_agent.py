"""Offline tests for the native GigaChat adapter."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from benchtable.agents.gigachat import GigaChatAgent
from benchtable.agents.protocol import AgentRequest
from benchtable.contracts import ToolSpec
from benchtable.errors import ProviderError


def _request() -> AgentRequest:
    return AgentRequest(
        actor_id="a",
        system_prompt="system",
        messages=[{"role": "user", "content": "go"}],
        tools=[ToolSpec(name="act", description="Act.", parameters={"type": "object"})],
    )


class _Response:
    choices = [
        {
            "message": {
                "content": "",
                "function_call": {"name": "act", "arguments": {"choice": "go"}},
                "functions_state_id": "state-1",
                "reasoning_content": "not retained",
            },
            "finish_reason": "function_call",
        }
    ]
    usage = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}

    def model_dump(self, *, mode: str = "json") -> dict[str, Any]:
        return {"choices": self.choices, "usage": self.usage}


class _Client:
    def __init__(self, request: dict[str, Any]) -> None:
        self.request = request

    async def achat(self, request: dict[str, Any]) -> _Response:
        self.request = request
        return _Response()


class _ContinuationClient:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    async def achat(self, request: dict[str, Any]) -> Any:
        self.requests.append(request)
        if len(self.requests) == 1:
            return _Response()
        return {
            "choices": [{"message": {"content": "complete"}, "finish_reason": "stop"}]
        }


def test_gigachat_maps_function_calls_and_client_settings(monkeypatch: Any) -> None:
    monkeypatch.setenv("GIGA_ENV", "configured-value")
    construction: dict[str, Any] = {}
    client = _Client({})

    def factory(**kwargs: Any) -> _Client:
        construction.update(kwargs)
        return client

    agent = GigaChatAgent(
        model="GigaChat-3-Ultra",
        credential_env="GIGA_ENV",
        scope="GIGACHAT_API_PERS",
        timeout=30.0,
        max_completion_tokens=128,
        _client_factory=factory,
    )
    response = asyncio.run(agent.respond(_request()))

    assert construction == {
        "credentials": "configured-value",
        "scope": "GIGACHAT_API_PERS",
        "model": "GigaChat-3-Ultra",
        "verify_ssl_certs": True,
        "max_retries": 0,
        "timeout": 30.0,
    }
    assert client.request["function_call"] == "auto"
    assert response.tool_calls[0].name == "act"
    assert response.tool_calls[0].raw_arguments == '{"choice":"go"}'
    assert response.raw_provider_response is not None
    assert "functions_state_id" not in str(response.raw_provider_response)
    assert "reasoning_content" not in str(response.raw_provider_response)


def test_gigachat_continuation_uses_object_arguments_and_sanitized_trace_request(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("GIGA_ENV", "configured-value")
    client = _ContinuationClient()
    agent = GigaChatAgent(
        model="GigaChat-3-Ultra",
        credential_env="GIGA_ENV",
        scope="GIGACHAT_API_PERS",
        _client_factory=lambda **kwargs: client,
    )

    first_response = asyncio.run(agent.respond(_request()))
    call_id = first_response.tool_calls[0].call_id
    second_response = asyncio.run(
        agent.respond(
            AgentRequest(
                actor_id="a",
                system_prompt="system",
                messages=[
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": "act",
                                    "arguments": '{"choice":"go"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": '{"summary":"acted"}',
                    },
                ],
                tools=[],
            )
        )
    )

    native_assistant_message = client.requests[1]["messages"][1]
    assert native_assistant_message == {
        "role": "assistant",
        "content": "",
        "function_call": {"name": "act", "arguments": {"choice": "go"}},
        "functions_state_id": "state-1",
    }
    assert second_response.raw_provider_request is not None
    assert "functions_state_id" not in str(second_response.raw_provider_request)
    assert "reasoning_content" not in str(second_response.raw_provider_request)


@pytest.mark.parametrize("arguments", ["not json", "[]", '{"choice":NaN}'])
def test_gigachat_rejects_invalid_historical_function_arguments(
    monkeypatch: Any, arguments: str
) -> None:
    monkeypatch.setenv("GIGA_ENV", "configured-value")
    agent = GigaChatAgent(
        model="GigaChat-3-Ultra",
        credential_env="GIGA_ENV",
        scope="GIGACHAT_API_PERS",
        _client_factory=lambda **kwargs: _Client({}),
    )
    request = AgentRequest(
        actor_id="a",
        system_prompt="system",
        messages=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "act", "arguments": arguments},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": '{"summary":"acted"}',
            },
        ],
        tools=[],
    )

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(agent.respond(request))

    assert exc_info.value.provider == "gigachat"
    assert exc_info.value.model == "GigaChat-3-Ultra"


def test_gigachat_malformed_history_content_error_includes_model(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("GIGA_ENV", "configured-value")
    agent = GigaChatAgent(
        model="GigaChat-3-Ultra",
        credential_env="GIGA_ENV",
        scope="GIGACHAT_API_PERS",
        _client_factory=lambda **kwargs: _Client({}),
    )

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(
            agent.respond(
                AgentRequest(
                    actor_id="a",
                    system_prompt="system",
                    messages=[{"role": "user", "content": {"not": "text"}}],
                    tools=[],
                )
            )
        )

    assert exc_info.value.model == "GigaChat-3-Ultra"
    assert exc_info.value.provider == "gigachat"
