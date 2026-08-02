"""Native asynchronous GigaChat agent adapter."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from typing import Any, NoReturn, Protocol, cast
from uuid import uuid4

from benchtable.agents.protocol import AgentRequest
from benchtable.contracts import JsonObject, ModelResponse, ToolCall, Usage
from benchtable.errors import ConfigurationError, ProviderError


class _ClientFactory(Protocol):
    def __call__(self, **kwargs: Any) -> Any: ...


def _default_client_factory(**kwargs: Any) -> Any:
    from gigachat import GigaChat

    return GigaChat(**kwargs)


def _finite(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number")
    return parsed


def _reject_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant: {value}")


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[Any, Any], value)
        return {str(key): _json_value(nested) for key, nested in mapping.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in cast(list[Any], value)]
    if hasattr(value, "model_dump"):
        try:
            return _json_value(value.model_dump(mode="json"))
        except Exception:
            return None
    return value


def _without_sensitive_fields(value: Any) -> Any:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[Any, Any], value)
        return {
            str(key): _without_sensitive_fields(nested)
            for key, nested in mapping.items()
            if str(key).casefold() not in {"reasoning_content", "functions_state_id"}
        }
    if isinstance(value, list):
        return [_without_sensitive_fields(item) for item in cast(list[Any], value)]
    return value


def _attr(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return cast(Mapping[str, Any], value).get(name, default)
    return getattr(value, name, default)


class GigaChatAgent:
    """Adapt canonical requests to GigaChat's legacy function-call API."""

    def __init__(
        self,
        *,
        model: str,
        credential_env: str,
        scope: str,
        timeout: float | None = None,
        max_completion_tokens: int | None = None,
        _client_factory: _ClientFactory | None = None,
    ) -> None:
        if type(model) is not str or not model.strip():
            raise ConfigurationError("Model must not be empty")
        if type(credential_env) is not str or not credential_env.strip():
            raise ConfigurationError(
                "Credential environment variable must not be empty"
            )
        if type(scope) is not str or not scope.strip():
            raise ConfigurationError("GigaChat scope must not be empty")
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
                "GigaChat credential environment variable is missing"
            )

        self._model = model
        self._credential_env = credential_env
        self._scope = scope
        self._timeout = timeout
        self._max_completion_tokens = max_completion_tokens
        self._client_factory = _client_factory or _default_client_factory
        self._client: Any | None = None
        self._functions_state: dict[str, str] = {}

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = self._client_factory(
                credentials=os.environ[self._credential_env],
                scope=self._scope,
                model=self._model,
                verify_ssl_certs=True,
                max_retries=0,
                timeout=self._timeout,
            )
        return self._client

    def _to_gigachat_messages(self, request: AgentRequest) -> list[JsonObject]:
        messages: list[JsonObject] = [
            {"role": "system", "content": request.system_prompt}
        ]
        pending: dict[str, str] = {}
        for message in request.messages:
            role = message.get("role")
            if role == "tool":
                call_id = message.get("tool_call_id")
                if not isinstance(call_id, str) or call_id not in pending:
                    raise ProviderError(
                        "GigaChat tool result has no matching call",
                        provider="gigachat",
                        model=self._model,
                    )
                messages.append(
                    {
                        "role": "function",
                        "name": pending.pop(call_id),
                        "content": self._require_content(message),
                    }
                )
                self._functions_state.pop(call_id, None)
                continue
            if role == "assistant" and "tool_calls" in message:
                calls = message.get("tool_calls")
                if not isinstance(calls, list) or len(calls) != 1:
                    raise ProviderError(
                        "GigaChat supports one historical function call per message",
                        provider="gigachat",
                        model=self._model,
                    )
                call = calls[0]
                if not isinstance(call, dict):
                    raise ProviderError(
                        "Malformed historical function call",
                        provider="gigachat",
                        model=self._model,
                    )
                call_id = call.get("id")
                function = call.get("function")
                if not isinstance(call_id, str) or not isinstance(function, dict):
                    raise ProviderError(
                        "Malformed historical function call",
                        provider="gigachat",
                        model=self._model,
                    )
                name = function.get("name")
                arguments = function.get("arguments")
                if not isinstance(name, str) or not isinstance(arguments, str):
                    raise ProviderError(
                        "Malformed historical function call",
                        provider="gigachat",
                        model=self._model,
                    )
                try:
                    decoded_arguments = json.loads(
                        arguments,
                        parse_float=_finite,
                        parse_constant=_reject_constant,
                    )
                except (TypeError, ValueError) as exc:
                    raise ProviderError(
                        "Malformed historical function arguments",
                        provider="gigachat",
                        model=self._model,
                    ) from exc
                if not isinstance(decoded_arguments, dict):
                    raise ProviderError(
                        "Historical function arguments must be a JSON object",
                        provider="gigachat",
                        model=self._model,
                    )
                if call_id in pending:
                    raise ProviderError(
                        "Duplicated historical function call",
                        provider="gigachat",
                        model=self._model,
                    )
                pending[call_id] = name
                assistant_message: JsonObject = {
                    "role": "assistant",
                    "content": self._require_content(message),
                    "function_call": {
                        "name": name,
                        "arguments": cast(JsonObject, decoded_arguments),
                    },
                }
                state_id = self._functions_state.get(call_id)
                if state_id is not None:
                    assistant_message["functions_state_id"] = state_id
                messages.append(assistant_message)
                continue
            if role in {"user", "assistant", "system"}:
                messages.append(
                    {"role": role, "content": self._require_content(message)}
                )
                continue
            raise ProviderError(
                "Malformed GigaChat message history",
                provider="gigachat",
                model=self._model,
            )
        if pending:
            raise ProviderError(
                "GigaChat function call is missing its result",
                provider="gigachat",
                model=self._model,
            )
        return messages

    def _require_content(self, message: JsonObject) -> str:
        content = message.get("content", "")
        if not isinstance(content, str):
            raise ProviderError(
                "GigaChat message content must be text",
                provider="gigachat",
                model=self._model,
            )
        return content

    async def respond(self, request: AgentRequest) -> ModelResponse:
        native_request: JsonObject = {"model": self._model}
        try:
            native_request["messages"] = cast(
                list[Any], self._to_gigachat_messages(request)
            )
            native_request["functions"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
                for tool in request.tools
            ]
            native_request["function_call"] = "auto" if request.tools else "none"
            if self._max_completion_tokens is not None:
                native_request["max_tokens"] = self._max_completion_tokens
            response = await self._get_client().achat(native_request)
            return self._normalize(response, native_request)
        except ProviderError as exc:
            if exc.raw_provider_request is None:
                exc.raw_provider_request = self._trace_request(native_request)
            raise
        except Exception as exc:
            raise ProviderError(
                self._safe_message(exc),
                provider="gigachat",
                model=self._model,
                raw_provider_request=self._trace_request(native_request),
                raw_provider_response=self._raw_provider_response(
                    locals().get("response")
                ),
            ) from exc

    def _normalize(self, response: Any, native_request: JsonObject) -> ModelResponse:
        choices_raw: object = _attr(response, "choices")
        if not isinstance(choices_raw, list):
            raise ProviderError(
                "GigaChat response must contain one choice",
                provider="gigachat",
                model=self._model,
                raw_provider_request=self._trace_request(native_request),
            )
        choices = cast(list[Any], choices_raw)
        if len(choices) != 1:
            raise ProviderError(
                "GigaChat response must contain one choice",
                provider="gigachat",
                model=self._model,
                raw_provider_request=self._trace_request(native_request),
            )
        choice = choices[0]
        message = _attr(choice, "message")
        if message is None:
            raise ProviderError(
                "GigaChat response has no message",
                provider="gigachat",
                model=self._model,
                raw_provider_request=self._trace_request(native_request),
            )
        function_call = _attr(message, "function_call")
        tool_calls: list[ToolCall] = []
        if function_call is not None:
            name = _attr(function_call, "name")
            arguments = _attr(function_call, "arguments")
            if not isinstance(name, str):
                raise ValueError("GigaChat function name is invalid")
            if isinstance(arguments, dict):
                raw_arguments = json.dumps(arguments, separators=(",", ":"))
            elif isinstance(arguments, str):
                raw_arguments = arguments
            else:
                raise ValueError("GigaChat function arguments are invalid")
            call_id = uuid4().hex
            state_id = _attr(message, "functions_state_id")
            if isinstance(state_id, str):
                self._functions_state[call_id] = state_id
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
            finish_reason=_attr(choice, "finish_reason"),
            usage=usage,
            raw_provider_request=self._trace_request(native_request),
            raw_provider_response=cast(
                JsonObject, _without_sensitive_fields(_json_value(response))
            ),
        )

    @staticmethod
    def _trace_request(native_request: JsonObject) -> JsonObject:
        return cast(JsonObject, _without_sensitive_fields(native_request))

    @staticmethod
    def _raw_provider_response(response: Any) -> JsonObject | None:
        if response is None:
            return None
        value = _without_sensitive_fields(_json_value(response))
        return cast(JsonObject, value) if isinstance(value, dict) else None

    def _safe_message(self, exc: Exception) -> str:
        message = str(exc)
        credential = os.environ.get(self._credential_env)
        return message.replace(credential, "[REDACTED]") if credential else message
