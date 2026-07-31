"""Tests for the OpenAI-compatible agent adapter."""

from __future__ import annotations

import asyncio
import math
from typing import Any

import pytest

from benchtable.agents.openai_compatible import OpenAICompatibleAgent
from benchtable.agents.protocol import AgentRequest
from benchtable.contracts import ToolSpec
from benchtable.errors import ConfigurationError, ProviderError


def _make_request(**kwargs: Any) -> AgentRequest:
    defaults = {
        "actor_id": "player-1",
        "system_prompt": "You are a player.",
        "messages": [{"role": "user", "content": "Your turn."}],
        "tools": [
            ToolSpec(name="act", description="Act.", parameters={"type": "object"})
        ],
    }
    defaults.update(kwargs)
    return AgentRequest(**defaults)


def _make_choice(
    *,
    content: str = "I'll act.",
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    choice: dict[str, Any] = {
        "index": 0,
        "message": {"role": "assistant", "content": content},
        "finish_reason": finish_reason,
    }
    if tool_calls is not None:
        choice["message"]["tool_calls"] = tool_calls
    return choice


def _make_response(
    choices: list[dict[str, Any]],
    *,
    usage: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Any:
    """Return a mock that behaves like an OpenAI ChatCompletion."""
    data: dict[str, Any] = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": choices,
    }
    if usage is not None:
        data["usage"] = usage
    if extra is not None:
        data.update(extra)

    class _FakeCompletion:
        def __init__(self, d: dict[str, Any]) -> None:
            self._d = d
            self.choices = [_Choice(c) for c in d["choices"]]
            self.model = d["model"]
            self.usage = _Usage(d.get("usage")) if d.get("usage") else None

        def model_dump(self, *, mode: str = "json") -> dict[str, Any]:
            return self._d

    class _Choice:
        def __init__(self, d: dict[str, Any]) -> None:
            self.index = d["index"]
            self.finish_reason = d.get("finish_reason")
            self.message = _Message(d["message"])

    class _Message:
        def __init__(self, d: dict[str, Any]) -> None:
            self.role = d.get("role", "assistant")
            self.content = d.get("content")
            self.tool_calls = (
                [_ToolCall(tc) for tc in d["tool_calls"]]
                if d.get("tool_calls")
                else None
            )

    class _ToolCall:
        def __init__(self, d: dict[str, Any]) -> None:
            self.id = d["id"]
            self.type = d.get("type", "function")
            self.function = _Function(d["function"])

    class _Function:
        def __init__(self, d: dict[str, Any]) -> None:
            self.name = d["name"]
            self.arguments = d["arguments"]

    class _Usage:
        def __init__(self, d: dict[str, Any] | None) -> None:
            self.prompt_tokens = d.get("prompt_tokens", 0) if d else 0
            self.completion_tokens = d.get("completion_tokens", 0) if d else 0
            self.total_tokens = d.get("total_tokens", 0) if d else 0

    return _FakeCompletion(data)


class TestOpenAICompatibleAgent:
    @pytest.fixture(autouse=True)
    def _set_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_API_KEY", "sk-test123")

    def test_passes_model_messages_and_tools(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        async def fake_create(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return _make_response(
                [
                    _make_choice(
                        tool_calls=[
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {
                                    "name": "act",
                                    "arguments": "{}",
                                },
                            }
                        ]
                    )
                ],
            )

        agent = OpenAICompatibleAgent(
            model="gpt-4o",
            api_key_env="TEST_API_KEY",
            max_completion_tokens=128,
            _client_factory=lambda **kw: _FakeClient(fake_create, kw),
        )
        request = _make_request()

        import asyncio

        resp = asyncio.run(agent.respond(request))

        assert captured == {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are a player."},
                {"role": "user", "content": "Your turn."},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "act",
                        "description": "Act.",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "n": 1,
            "max_completion_tokens": 128,
        }
        assert resp.tool_calls[0].name == "act"

    @pytest.mark.parametrize(
        "overrides",
        [
            {"model": ""},
            {"base_url": ""},
            {"timeout": 0},
            {"timeout": -1},
        ],
    )
    def test_rejects_invalid_provider_settings(self, overrides: dict[str, Any]) -> None:
        kwargs: dict[str, Any] = {
            "model": "gpt-4o",
            "api_key_env": "TEST_API_KEY",
        }
        kwargs.update(overrides)

        with pytest.raises(ConfigurationError):
            OpenAICompatibleAgent(**kwargs)

    @pytest.mark.parametrize(
        "timeout",
        [True, "30", "   ", math.nan, math.inf, 0, -1],
    )
    def test_rejects_invalid_timeout_values(self, timeout: Any) -> None:
        with pytest.raises(ConfigurationError):
            OpenAICompatibleAgent(
                model="gpt-4o",
                api_key_env="TEST_API_KEY",
                timeout=timeout,
            )

    def test_rejects_astronomical_timeout_value(self) -> None:
        with pytest.raises(ConfigurationError):
            OpenAICompatibleAgent(
                model="gpt-4o",
                api_key_env="TEST_API_KEY",
                timeout=10**1000,
            )

    @pytest.mark.parametrize("timeout", [30, 30.5])
    def test_accepts_positive_numeric_timeout_values(
        self, timeout: int | float
    ) -> None:
        OpenAICompatibleAgent(
            model="gpt-4o",
            api_key_env="TEST_API_KEY",
            timeout=timeout,
        )

    @pytest.mark.parametrize(
        "overrides",
        [
            {"max_completion_tokens": True},
            {"max_completion_tokens": "128"},
            {"max_completion_tokens": 128.0},
            {"model": "   "},
            {"model": 123},
        ],
    )
    def test_direct_construction_rejects_invalid_values(
        self, overrides: dict[str, Any]
    ) -> None:
        kwargs: dict[str, Any] = {
            "model": "gpt-4o",
            "api_key_env": "TEST_API_KEY",
        }
        kwargs.update(overrides)

        with pytest.raises(ConfigurationError):
            OpenAICompatibleAgent(**kwargs)

    def test_client_factory_receives_exact_sdk_construction_kwargs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_create(**kwargs: Any) -> Any:
            return _make_response([_make_choice()])

        captured_kwargs: dict[str, Any] = {}

        def make_client(**kwargs: Any) -> Any:
            captured_kwargs.update(kwargs)
            return _FakeClient(fake_create, kwargs)

        agent = OpenAICompatibleAgent(
            model="gpt-4o",
            api_key_env="TEST_API_KEY",
            base_url="https://custom.api.com/v1",
            timeout=30.0,
            _client_factory=make_client,
        )

        asyncio.run(agent.respond(_make_request()))

        assert captured_kwargs == {
            "api_key": "sk-test123",
            "base_url": "https://custom.api.com/v1",
            "timeout": 30.0,
            "max_retries": 0,
        }

    def test_missing_api_key_env_raises_configuration_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MISSING_KEY", raising=False)

        with pytest.raises(ConfigurationError) as exc_info:
            OpenAICompatibleAgent(
                model="gpt-4o",
                api_key_env="MISSING_KEY",
            )
        assert "MISSING_KEY" not in str(exc_info.value)
        assert "API key environment variable" in str(exc_info.value)

    def test_empty_api_key_env_raises_configuration_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EMPTY_KEY", "")

        with pytest.raises(ConfigurationError) as exc_info:
            OpenAICompatibleAgent(
                model="gpt-4o",
                api_key_env="EMPTY_KEY",
            )
        assert "EMPTY_KEY" not in str(exc_info.value)
        assert "API key environment variable" in str(exc_info.value)

    def test_whitespace_api_key_value_raises_configuration_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WHITESPACE_KEY", "   ")

        with pytest.raises(ConfigurationError) as exc_info:
            OpenAICompatibleAgent(
                model="gpt-4o",
                api_key_env="WHITESPACE_KEY",
            )
        assert "WHITESPACE_KEY" not in str(exc_info.value)
        assert "API key environment variable" in str(exc_info.value)

    def test_key_shaped_api_key_env_is_not_echoed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        api_key_env = "sk-api-key-secret"
        monkeypatch.delenv(api_key_env, raising=False)

        with pytest.raises(ConfigurationError) as exc_info:
            OpenAICompatibleAgent(
                model="gpt-4o",
                api_key_env=api_key_env,
            )

        assert api_key_env not in str(exc_info.value)
        assert "API key environment variable" in str(exc_info.value)

    def test_normalizes_text_response_without_tool_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_create(**kwargs: Any) -> Any:
            return _make_response(
                [_make_choice(content="I think I'll wait.", finish_reason="stop")],
                usage={
                    "prompt_tokens": 50,
                    "completion_tokens": 10,
                    "total_tokens": 60,
                },
            )

        agent = OpenAICompatibleAgent(
            model="gpt-4o",
            api_key_env="TEST_API_KEY",
            _client_factory=lambda **kw: _FakeClient(fake_create, kw),
        )
        import asyncio

        resp = asyncio.run(agent.respond(_make_request()))

        assert resp.assistant_text == "I think I'll wait."
        assert resp.tool_calls == []
        assert resp.finish_reason == "stop"
        assert resp.usage is not None
        assert resp.usage.prompt_tokens == 50
        assert resp.raw_provider_response == {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "I think I'll wait.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 10,
                "total_tokens": 60,
            },
        }

    def test_malformed_tool_arguments_preserve_raw_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_create(**kwargs: Any) -> Any:
            return _make_response(
                [
                    _make_choice(
                        tool_calls=[
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {
                                    "name": "act",
                                    "arguments": "not-json",
                                },
                            }
                        ]
                    )
                ],
            )

        agent = OpenAICompatibleAgent(
            model="gpt-4o",
            api_key_env="TEST_API_KEY",
            _client_factory=lambda **kw: _FakeClient(fake_create, kw),
        )
        import asyncio

        resp = asyncio.run(agent.respond(_make_request()))

        assert resp.tool_calls[0].arguments is None
        assert resp.tool_calls[0].raw_arguments == "not-json"
        assert resp.tool_calls[0].parse_error is not None

    def test_duplicate_provider_tool_call_ids_are_provider_errors(self) -> None:
        provider_response = _make_response(
            [
                _make_choice(
                    tool_calls=[
                        {
                            "id": "duplicate",
                            "type": "function",
                            "function": {"name": "act", "arguments": "{}"},
                        },
                        {
                            "id": "duplicate",
                            "type": "function",
                            "function": {"name": "act", "arguments": "{}"},
                        },
                    ]
                )
            ],
            extra={"api_secret": "adapter-secret"},
        )

        async def fake_create(**kwargs: Any) -> Any:
            return provider_response

        agent = OpenAICompatibleAgent(
            model="gpt-4o",
            api_key_env="TEST_API_KEY",
            _client_factory=lambda **kw: _FakeClient(fake_create, kw),
        )

        with pytest.raises(ProviderError, match="duplicate call_id") as exc_info:
            asyncio.run(agent.respond(_make_request()))

        assert exc_info.value.provider == "openai"
        assert exc_info.value.model == "gpt-4o"
        assert exc_info.value.raw_provider_response == provider_response.model_dump(
            mode="json"
        )

    def test_invalid_provider_tool_call_id_is_a_provider_error_with_raw_response(
        self,
    ) -> None:
        provider_response = _make_response(
            [
                _make_choice(
                    tool_calls=[
                        {
                            "id": "",
                            "type": "function",
                            "function": {"name": "act", "arguments": "{}"},
                        }
                    ]
                )
            ],
            extra={"session_token": "adapter-session-secret"},
        )

        async def fake_create(**kwargs: Any) -> Any:
            return provider_response

        agent = OpenAICompatibleAgent(
            model="gpt-4o",
            api_key_env="TEST_API_KEY",
            _client_factory=lambda **kw: _FakeClient(fake_create, kw),
        )

        with pytest.raises(ProviderError, match="call_id") as exc_info:
            asyncio.run(agent.respond(_make_request()))

        assert exc_info.value.raw_provider_response == provider_response.model_dump(
            mode="json"
        )

    @pytest.mark.parametrize("raw_arguments", ["null", "[]", "1", '"text"'])
    def test_non_object_tool_arguments_preserve_raw_text(
        self, raw_arguments: str
    ) -> None:
        async def fake_create(**kwargs: Any) -> Any:
            return _make_response(
                [
                    _make_choice(
                        tool_calls=[
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {
                                    "name": "act",
                                    "arguments": raw_arguments,
                                },
                            }
                        ]
                    )
                ],
            )

        agent = OpenAICompatibleAgent(
            model="gpt-4o",
            api_key_env="TEST_API_KEY",
            _client_factory=lambda **kw: _FakeClient(fake_create, kw),
        )

        resp = asyncio.run(agent.respond(_make_request()))

        assert resp.tool_calls[0].arguments is None
        assert resp.tool_calls[0].raw_arguments == raw_arguments
        assert resp.tool_calls[0].parse_error is not None

    @pytest.mark.parametrize(
        "raw_arguments",
        [
            '{"value": NaN}',
            '{"value": Infinity}',
            '{"value": -Infinity}',
            '{"value": 1e999}',
        ],
    )
    def test_non_finite_tool_arguments_are_malformed(self, raw_arguments: str) -> None:
        async def fake_create(**kwargs: Any) -> Any:
            return _make_response(
                [
                    _make_choice(
                        tool_calls=[
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {
                                    "name": "act",
                                    "arguments": raw_arguments,
                                },
                            }
                        ]
                    )
                ],
            )

        agent = OpenAICompatibleAgent(
            model="gpt-4o",
            api_key_env="TEST_API_KEY",
            _client_factory=lambda **kw: _FakeClient(fake_create, kw),
        )

        resp = asyncio.run(agent.respond(_make_request()))

        assert resp.tool_calls[0].arguments is None
        assert resp.tool_calls[0].raw_arguments == raw_arguments
        assert resp.tool_calls[0].parse_error is not None

    def test_provider_exception_is_wrapped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_create(**kwargs: Any) -> Any:
            raise RuntimeError("connection refused")

        agent = OpenAICompatibleAgent(
            model="gpt-4o",
            api_key_env="TEST_API_KEY",
            _client_factory=lambda **kw: _FakeClient(fake_create, kw),
        )
        import asyncio

        with pytest.raises(ProviderError, match="connection refused"):
            asyncio.run(agent.respond(_make_request()))

    def test_client_construction_exception_is_wrapped_without_api_key(
        self,
    ) -> None:
        def fail_to_construct(**kwargs: Any) -> Any:
            raise RuntimeError("factory failed for sk-test123")

        agent = OpenAICompatibleAgent(
            model="gpt-4o",
            api_key_env="TEST_API_KEY",
            _client_factory=fail_to_construct,
        )

        with pytest.raises(ProviderError, match="factory failed") as exc_info:
            asyncio.run(agent.respond(_make_request()))

        assert "sk-test123" not in str(exc_info.value)

    def test_empty_provider_choices_are_wrapped(self) -> None:
        provider_response = _make_response(
            [], extra={"client_token": "adapter-client-secret"}
        )

        async def fake_create(**kwargs: Any) -> Any:
            return provider_response

        agent = OpenAICompatibleAgent(
            model="gpt-4o",
            api_key_env="TEST_API_KEY",
            _client_factory=lambda **kw: _FakeClient(fake_create, kw),
        )

        with pytest.raises(ProviderError) as exc_info:
            asyncio.run(agent.respond(_make_request()))

        assert exc_info.value.raw_provider_response == provider_response.model_dump(
            mode="json"
        )

    def test_api_key_not_in_error_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_create(**kwargs: Any) -> Any:
            raise RuntimeError("server error")

        agent = OpenAICompatibleAgent(
            model="gpt-4o",
            api_key_env="TEST_API_KEY",
            _client_factory=lambda **kw: _FakeClient(fake_create, kw),
        )
        import asyncio

        with pytest.raises(ProviderError) as exc_info:
            asyncio.run(agent.respond(_make_request()))

        assert "sk-test123" not in str(exc_info.value)


class _FakeClient:
    """Minimal mock of AsyncOpenAI for testing."""

    def __init__(self, create_fn: Any, kwargs: dict[str, Any]) -> None:
        self._create_fn = create_fn
        self._kwargs = kwargs
        self.chat = self
        self.completions = self

    async def create(self, **kwargs: Any) -> Any:
        return await self._create_fn(**kwargs)
