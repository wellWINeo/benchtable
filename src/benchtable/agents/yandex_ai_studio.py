"""Native asynchronous Yandex AI Studio agent adapter."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import Any, NoReturn, Protocol, cast
from uuid import uuid4

from benchtable.agents.protocol import AgentRequest
from benchtable.contracts import JsonObject, ModelResponse, ToolCall, Usage
from benchtable.errors import ConfigurationError, ProviderError


class _ClientFactory(Protocol):
    def __call__(self, **kwargs: Any) -> Any: ...


def _default_client_factory(**kwargs: Any) -> Any:
    from yandex_ai_studio_sdk import AsyncAIStudio

    return AsyncAIStudio(**kwargs)


def _attr(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return cast(Mapping[str, Any], value).get(name, default)
    return getattr(value, name, default)


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[Any, Any], value)
        return {str(key): _json_value(nested) for key, nested in mapping.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in cast(list[Any], value)]
    if isinstance(value, tuple):
        return [_json_value(item) for item in cast(tuple[Any, ...], value)]
    if is_dataclass(value):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in fields(value)
            if not field.name.startswith("_")
        }
    if hasattr(value, "value") and type(value).__module__ == "enum":
        return value.value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "model_dump"):
        try:
            return _json_value(value.model_dump(mode="json"))
        except Exception:
            return None
    return value


def _strip_reasoning(value: Any) -> Any:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[Any, Any], value)
        return {
            str(key): _strip_reasoning(nested)
            for key, nested in mapping.items()
            if str(key).casefold() not in {"reasoning_content", "reasoning_text"}
        }
    if isinstance(value, list):
        return [_strip_reasoning(item) for item in cast(list[Any], value)]
    return value


def _finite(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number")
    return parsed


def _reject_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant: {value}")


class YandexAIStudioAgent:
    """Adapt canonical requests to Yandex AI Studio chat completions."""

    def __init__(
        self,
        *,
        model: str,
        folder_id: str,
        credential_kind: str,
        credential_env: str,
        timeout: float | None = None,
        max_completion_tokens: int | None = None,
        _client_factory: _ClientFactory | None = None,
    ) -> None:
        if type(model) is not str or not model.strip():
            raise ConfigurationError("Model must not be empty")
        if type(folder_id) is not str or not folder_id.strip():
            raise ConfigurationError("Yandex folder ID must not be empty")
        if type(credential_env) is not str or not credential_env.strip():
            raise ConfigurationError(
                "Credential environment variable must not be empty"
            )
        if credential_kind not in {"oauth", "api_key"}:
            raise ConfigurationError("Yandex credential kind is invalid")
        if timeout is not None and (
            type(timeout) not in (int, float)
            or not math.isfinite(float(timeout))
            or timeout <= 0
        ):
            raise ConfigurationError("Timeout must be a finite positive number")
        if max_completion_tokens is not None and (
            type(max_completion_tokens) is not int or max_completion_tokens < 1
        ):
            raise ConfigurationError("max_completion_tokens must be at least 1")
        credential = os.environ.get(credential_env)
        if not credential or not credential.strip():
            raise ConfigurationError(
                "Yandex credential environment variable is missing"
            )
        self._model = model
        self._folder_id = folder_id
        self._credential_kind = credential_kind
        self._credential_env = credential_env
        self._timeout = timeout
        self._max_completion_tokens = max_completion_tokens
        self._client_factory = _client_factory or _default_client_factory
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = self._client_factory(
                folder_id=self._folder_id,
                auth=os.environ[self._credential_env],
            )
        return self._client

    async def respond(self, request: AgentRequest) -> ModelResponse:
        native_request = cast(
            JsonObject,
            {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": request.system_prompt},
                    *request.messages,
                ],
                "tools": [tool.to_openai_tool() for tool in request.tools],
                "parallel_tool_calls": False,
            },
        )
        if self._max_completion_tokens is not None:
            native_request["max_tokens"] = self._max_completion_tokens
        response: Any | None = None
        try:
            client = self._get_client()
            model = client.chat.completions(self._model)
            configure_kwargs: dict[str, Any] = {
                "tools": [
                    client.tools.function(
                        tool.parameters,
                        name=tool.name,
                        description=tool.description,
                    )
                    for tool in request.tools
                ],
                "parallel_tool_calls": False,
            }
            if self._max_completion_tokens is not None:
                configure_kwargs["max_tokens"] = self._max_completion_tokens
            configured = model.configure(**configure_kwargs)
            response = await configured.run(
                native_request["messages"], timeout=self._timeout
            )
            return self._normalize(response, native_request)
        except ProviderError as exc:
            if exc.raw_provider_request is None:
                exc.raw_provider_request = native_request
            raise
        except Exception as exc:
            credential = os.environ.get(self._credential_env)
            message = (
                str(exc).replace(credential, "[REDACTED]") if credential else str(exc)
            )
            raise ProviderError(
                message,
                provider="yandex_ai_studio",
                model=self._model,
                raw_provider_request=native_request,
                raw_provider_response=(
                    cast(JsonObject, _strip_reasoning(_json_value(response)))
                    if response is not None
                    else None
                ),
            ) from exc

    def _normalize(self, response: Any, native_request: JsonObject) -> ModelResponse:
        choices = _attr(response, "choices")
        if choices is None:
            choices = [response]
        if not isinstance(choices, (list, tuple)):
            raise ValueError("Yandex response must contain one choice")
        choice_values = cast(list[Any] | tuple[Any, ...], choices)
        if len(choice_values) != 1:
            raise ValueError("Yandex response must contain one choice")
        choice = list(choice_values)[0]
        message = _attr(choice, "message", choice)
        tool_calls_raw = list(_attr(message, "tool_calls") or [])
        tool_calls: list[ToolCall] = []
        for native_call in tool_calls_raw:
            function = _attr(native_call, "function", native_call)
            name = _attr(function, "name")
            arguments = _attr(function, "arguments", "")
            if not isinstance(name, str):
                raise ValueError("Yandex tool call is malformed")
            if isinstance(arguments, dict):
                raw_arguments = json.dumps(arguments, separators=(",", ":"))
            elif isinstance(arguments, str):
                raw_arguments = arguments
            else:
                raise ValueError("Yandex tool call is malformed")
            parsed: JsonObject | None = None
            parse_error: str | None = None
            try:
                decoded = json.loads(
                    raw_arguments, parse_float=_finite, parse_constant=_reject_constant
                )
                if isinstance(decoded, dict):
                    parsed = cast(JsonObject, decoded)
                else:
                    parse_error = "Tool arguments must be a JSON object"
            except (ValueError, TypeError) as exc:
                parse_error = str(exc)
            call_id = _attr(native_call, "id") or uuid4().hex
            tool_calls.append(
                ToolCall(
                    call_id=call_id,
                    name=name,
                    arguments=parsed,
                    raw_arguments=raw_arguments,
                    parse_error=parse_error,
                )
            )
        usage_raw = _attr(response, "usage")
        usage = None
        if usage_raw is not None:
            usage = Usage(
                prompt_tokens=_attr(usage_raw, "prompt_tokens", 0),
                completion_tokens=_attr(usage_raw, "completion_tokens", 0),
                total_tokens=_attr(usage_raw, "total_tokens", 0),
            )
        return ModelResponse(
            assistant_text=_attr(message, "content", "") or "",
            tool_calls=tool_calls,
            finish_reason=self._finish_reason(_attr(choice, "finish_reason")),
            usage=usage,
            raw_provider_request=native_request,
            raw_provider_response=cast(
                JsonObject, _strip_reasoning(_json_value(response))
            ),
        )

    @staticmethod
    def _finish_reason(value: Any) -> str | None:
        if value is None:
            return None
        return cast(str, getattr(value, "value", value))
