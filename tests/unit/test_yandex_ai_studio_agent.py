"""Offline tests for the native Yandex AI Studio adapter."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import pytest

from benchtable.agents.protocol import AgentRequest
from benchtable.agents.yandex_ai_studio import YandexAIStudioAgent
from benchtable.contracts import ToolSpec
from benchtable.errors import ProviderError


class _Configured:
    def __init__(self, response: Any | None = None) -> None:
        self._response = response

    async def run(self, messages: Any, *, timeout: float | None) -> Any:
        if self._response is not None:
            return self._response
        return {
            "choices": [
                {
                    "message": {
                        "content": "done",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {
                                    "name": "act",
                                    "arguments": '{"choice":"go"}',
                                },
                            }
                        ],
                        "reasoning_content": "not retained",
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        }


class _Model:
    def __init__(self, response: Any | None = None) -> None:
        self.kwargs: dict[str, Any] = {}
        self._response = response

    def configure(self, **kwargs: Any) -> _Configured:
        self.kwargs = kwargs
        return _Configured(self._response)


class _SDK:
    def __init__(self, model: _Model) -> None:
        self.model = model
        self.chat = self
        self.completions = self

    def __call__(self, model: str) -> _Model:
        return self.model


def test_yandex_maps_openai_messages_and_native_response(monkeypatch: Any) -> None:
    monkeypatch.setenv("YC_ENV", "configured-value")
    model = _Model()
    sdk = _SDK(model)
    construction: dict[str, Any] = {}

    def factory(**kwargs: Any) -> _SDK:
        construction.update(kwargs)
        return sdk

    agent = YandexAIStudioAgent(
        model="aliceai-llm",
        folder_id="folder-123",
        credential_kind="oauth",
        credential_env="YC_ENV",
        timeout=30.0,
        max_completion_tokens=128,
        _client_factory=factory,
    )
    response = asyncio.run(
        agent.respond(
            AgentRequest(
                actor_id="a",
                system_prompt="system",
                messages=[{"role": "user", "content": "go"}],
                tools=[ToolSpec(name="act", parameters={"type": "object"})],
            )
        )
    )

    assert construction == {"folder_id": "folder-123", "auth": "configured-value"}
    assert model.kwargs["parallel_tool_calls"] is False
    assert model.kwargs["max_tokens"] == 128
    assert response.tool_calls[0].call_id == "call-1"
    assert response.usage is not None
    assert "reasoning_content" not in str(response.raw_provider_response)


@dataclass
class _SdkReasoningDetails:
    reasoning_content: str
    reasoning_text: str


@dataclass
class _SdkMessage:
    content: str
    details: _SdkReasoningDetails
    reasoning_text: str


@dataclass
class _SdkChoice:
    message: _SdkMessage
    finish_reason: str


@dataclass
class _SdkUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class _SdkResponse:
    choices: list[_SdkChoice]
    usage: _SdkUsage
    reasoning_text: str


def test_yandex_strips_nested_reasoning_from_sdk_dataclass_response(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("YC_ENV", "configured-value")
    response = _SdkResponse(
        choices=[
            _SdkChoice(
                message=_SdkMessage(
                    content="done",
                    details=_SdkReasoningDetails(
                        reasoning_content="hidden analysis",
                        reasoning_text="hidden trace",
                    ),
                    reasoning_text="hidden message",
                ),
                finish_reason="stop",
            )
        ],
        usage=_SdkUsage(prompt_tokens=2, completion_tokens=1, total_tokens=3),
        reasoning_text="hidden response",
    )
    model = _Model(response)
    sdk = _SDK(model)
    agent = YandexAIStudioAgent(
        model="aliceai-llm",
        folder_id="folder-123",
        credential_kind="oauth",
        credential_env="YC_ENV",
        _client_factory=lambda **kwargs: sdk,
    )

    normalized = asyncio.run(
        agent.respond(
            AgentRequest(
                actor_id="a",
                system_prompt="system",
                messages=[{"role": "user", "content": "go"}],
                tools=[],
            )
        )
    )

    assert normalized.raw_provider_response is not None
    assert normalized.raw_provider_response == {
        "choices": [
            {
                "message": {"content": "done", "details": {}},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
    }
    serialized = json.dumps(normalized.raw_provider_response)
    assert "reasoning_content" not in serialized
    assert "reasoning_text" not in serialized


def test_yandex_retains_sanitized_response_when_normalization_fails(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("YC_ENV", "configured-value")
    response = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "reasoning_content": "hidden content",
                    "tool_calls": [
                        {
                            "function": {
                                "arguments": "{}",
                                "reasoning_text": "hidden tool reasoning",
                            }
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "reasoning_text": "hidden response reasoning",
    }
    model = _Model(response)
    sdk = _SDK(model)
    agent = YandexAIStudioAgent(
        model="aliceai-llm",
        folder_id="folder-123",
        credential_kind="oauth",
        credential_env="YC_ENV",
        _client_factory=lambda **kwargs: sdk,
    )

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(
            agent.respond(
                AgentRequest(
                    actor_id="a",
                    system_prompt="system",
                    messages=[{"role": "user", "content": "go"}],
                    tools=[],
                )
            )
        )

    error = exc_info.value
    assert error.provider == "yandex_ai_studio"
    assert error.model == "aliceai-llm"
    assert error.raw_provider_response == {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [{"function": {"arguments": "{}"}}],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    serialized = json.dumps(error.raw_provider_response)
    assert "reasoning_content" not in serialized
    assert "reasoning_text" not in serialized
