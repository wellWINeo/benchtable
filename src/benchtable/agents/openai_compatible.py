"""OpenAI-compatible async agent adapter."""

from __future__ import annotations

import json
import math
import os
from typing import Any, NoReturn, Protocol, cast

from benchtable.agents.protocol import AgentRequest
from benchtable.contracts import (
    JsonObject,
    ModelResponse,
    ToolCall,
    Usage,
)
from benchtable.errors import ConfigurationError, ProviderError


class _AsyncClientFactory(Protocol):
    """Factory that creates an async OpenAI-compatible client."""

    def __call__(self, **kwargs: Any) -> Any: ...


def _default_client_factory(**kwargs: Any) -> Any:
    from openai import AsyncOpenAI

    return AsyncOpenAI(**kwargs)


def _reject_non_finite_json_constant(value: str) -> NoReturn:
    raise ValueError(f"Non-finite JSON constant is not allowed: {value}")


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite JSON number is not allowed: {value}")
    return parsed


class OpenAICompatibleAgent:
    """Agent adapter for OpenAI-compatible Chat Completions endpoints."""

    def __init__(
        self,
        *,
        model: str,
        api_key_env: str,
        base_url: str | None = None,
        timeout: float | None = None,
        max_completion_tokens: int | None = None,
        _client_factory: _AsyncClientFactory | None = None,
    ) -> None:
        if type(model) is not str or not model.strip():
            raise ConfigurationError("Model must not be empty")
        if base_url is not None and (type(base_url) is not str or not base_url.strip()):
            raise ConfigurationError("Base URL must not be empty")
        if timeout is not None:
            try:
                if type(timeout) is int:
                    timeout_value = float(timeout)
                elif type(timeout) is float:
                    timeout_value = timeout
                else:
                    raise ConfigurationError("Timeout must be a finite positive number")
            except OverflowError as exc:
                raise ConfigurationError(
                    "Timeout must be a finite positive number"
                ) from exc
            if not math.isfinite(timeout_value) or timeout_value <= 0:
                raise ConfigurationError("Timeout must be a finite positive number")
        if max_completion_tokens is not None and (
            type(max_completion_tokens) is not int or max_completion_tokens < 1
        ):
            raise ConfigurationError("max_completion_tokens must be at least 1")
        if type(api_key_env) is not str or not api_key_env.strip():
            raise ConfigurationError("API key environment variable must not be empty")

        self._model = model
        self._api_key_env = api_key_env
        self._base_url = base_url
        self._timeout = timeout
        self._max_completion_tokens = max_completion_tokens
        self._client_factory = _client_factory or _default_client_factory
        self._client: Any | None = None

        # Validate API key exists at construction time
        api_key = os.environ.get(api_key_env)
        if not api_key or not api_key.strip():
            raise ConfigurationError("API key environment variable is missing or empty")

    def _get_client(self) -> Any:
        if self._client is None:
            kwargs: dict[str, Any] = {"api_key": os.environ[self._api_key_env]}
            if self._base_url is not None:
                kwargs["base_url"] = self._base_url
            if self._timeout is not None:
                kwargs["timeout"] = self._timeout
            kwargs["max_retries"] = 0
            self._client = self._client_factory(**kwargs)
        return self._client

    async def respond(self, request: AgentRequest) -> ModelResponse:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": request.system_prompt}
        ]
        for msg in request.messages:
            messages.append(dict(msg))

        tools = [t.to_openai_tool() for t in request.tools]

        request_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "tools": tools,
            "n": 1,
        }
        if self._max_completion_tokens is not None:
            request_kwargs["max_completion_tokens"] = self._max_completion_tokens

        try:
            client = self._get_client()
            response = await client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            raise self._provider_error(exc) from exc

        try:
            return self._normalize_response(response)
        except Exception as exc:
            raise self._provider_error(
                exc,
                raw_provider_response=self._raw_provider_response(response),
            ) from exc

    def _provider_error(
        self,
        exc: Exception,
        *,
        raw_provider_response: Any | None = None,
    ) -> ProviderError:
        message = str(exc)
        api_key = os.environ.get(self._api_key_env)
        if api_key:
            message = message.replace(api_key, "[REDACTED]")
        return ProviderError(
            message,
            provider="openai",
            model=self._model,
            raw_provider_response=raw_provider_response,
        )

    @staticmethod
    def _raw_provider_response(response: Any) -> Any | None:
        model_dump = getattr(response, "model_dump", None)
        if not callable(model_dump):
            return None
        try:
            return model_dump(mode="json")
        except Exception:
            return None

    def _normalize_response(self, response: Any) -> ModelResponse:
        choice = response.choices[0]
        message = choice.message

        # Parse tool calls
        tool_calls: list[ToolCall] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                raw_args = tc.function.arguments
                parsed_args: JsonObject | None = None
                parse_error = None
                try:
                    decoded_args = json.loads(
                        raw_args,
                        parse_float=_parse_finite_json_float,
                        parse_constant=_reject_non_finite_json_constant,
                    )
                except (ValueError, TypeError) as exc:
                    parse_error = str(exc)
                else:
                    if isinstance(decoded_args, dict):
                        parsed_args = cast(JsonObject, decoded_args)
                    else:
                        parse_error = "Tool arguments must be a JSON object"

                tool_calls.append(
                    ToolCall(
                        call_id=tc.id,
                        name=tc.function.name,
                        arguments=parsed_args,
                        raw_arguments=raw_args,
                        parse_error=parse_error,
                    )
                )

        # Parse usage
        usage = None
        if response.usage:
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

        # Raw provider response
        raw = None
        if hasattr(response, "model_dump"):
            raw = response.model_dump(mode="json")

        return ModelResponse(
            assistant_text=message.content or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            usage=usage,
            raw_provider_response=raw,
        )
