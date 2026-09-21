"""OpenRouter System One/Decisions judge adapter for pinned models."""

from __future__ import annotations

import os
from typing import Any, Protocol, cast

import httpx

from benchtable.contracts import JsonObject
from benchtable.errors import (
    ConfigurationError,
    JudgeMalformedResponseError,
    JudgeProviderError,
    JudgeTimeoutError,
)
from benchtable.judges.protocol import JudgeRequest, JudgmentDecision

DEFAULT_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
DEFAULT_TIMEOUT_SECONDS = 30.0


class _PostTransport(Protocol):
    """Async transport performing one Decisions POST."""

    async def __call__(
        self,
        url: str,
        headers: dict[str, str],
        json_body: dict[str, Any],
        timeout: float,
    ) -> tuple[int, Any, str]: ...


async def _httpx_post(
    url: str,
    headers: dict[str, str],
    json_body: dict[str, Any],
    timeout: float,
) -> tuple[int, Any, str]:
    """POST via httpx and return (status, parsed JSON or None, raw text).

    Exception-to-typed-error mapping lives in ``decide`` so injected
    transports receive identical treatment.
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            url, json=json_body, headers=headers, timeout=timeout
        )
    try:
        parsed: Any = response.json()
    except ValueError:
        parsed = None
    return response.status_code, parsed, response.text


def _raw_response(status: int, parsed: Any, raw: str) -> dict[str, Any]:
    if parsed is None:
        return {"status": status, "raw_body": raw}
    return cast(dict[str, Any], parsed)


def _extract_answers(parsed: dict[str, object], status: int, raw: str) -> JsonObject:
    """Return the Decisions ``answers`` object from a parsed response."""
    raw_response: dict[str, Any] = {"status": status, "raw_body": raw}
    answers_raw: object = parsed.get("answers")
    if type(answers_raw) is not dict:
        raise JudgeMalformedResponseError(
            "Judge response must contain an 'answers' object",
            raw_provider_response=raw_response,
        )
    answers = cast("dict[str, object]", answers_raw)
    for question_id, answer in answers.items():
        if type(answer) is not dict:
            raise JudgeMalformedResponseError(
                f"Judge answer for {question_id!r} must be an object",
                raw_provider_response=raw_response,
            )
    return cast(JsonObject, answers)


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_usage(parsed: dict[str, object]) -> JsonObject | None:
    usage = parsed.get("usage")
    return cast(JsonObject, usage) if isinstance(usage, dict) else None


class OpenRouterDecisionsJudge:
    """Judge adapter for the OpenRouter System One/Decisions endpoint.

    The endpoint and request/response shape follow the OpenRouter Jev model
    page and the OpenRouter SystemOne SDK reference. The model must be a
    pinned ID; the adapter adds no fallback models and no alias rewriting.
    """

    def __init__(
        self,
        *,
        judge_id: str,
        model: str,
        api_key_env: str,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        _post: _PostTransport | None = None,
    ) -> None:
        if type(model) is not str or not model.strip():
            raise ConfigurationError("Judge model must not be empty")
        if base_url is not None and (type(base_url) is not str or not base_url.strip()):
            raise ConfigurationError("Judge base URL must not be empty")
        if type(api_key_env) is not str or not api_key_env.strip():
            raise ConfigurationError(
                "Judge API key environment variable name is required"
            )
        api_key = os.environ.get(api_key_env)
        if not api_key or not api_key.strip():
            raise ConfigurationError(
                f"Judge API key environment variable '{api_key_env}' "
                "is missing or empty"
            )
        self._judge_id = judge_id
        self._model = model
        self._api_key_env = api_key_env
        self._url = base_url or DEFAULT_DECISIONS_URL
        self._timeout = timeout
        self._post: _PostTransport = _post or _httpx_post

    @property
    def judge_id(self) -> str:
        return self._judge_id

    async def decide(self, request: JudgeRequest) -> JudgmentDecision:
        body: dict[str, Any] = {"model": self._model, **request.payload}
        headers = {
            "Authorization": f"Bearer {os.environ[self._api_key_env]}",
            "Content-Type": "application/json",
        }
        try:
            status, parsed, raw = await self._post(
                self._url, headers, body, self._timeout
            )
        except httpx.TimeoutException as exc:
            raise JudgeTimeoutError(str(exc)) from exc
        except Exception as exc:
            raise JudgeProviderError(
                str(exc),
                provider="openrouter",
                model=self._model,
                raw_provider_request=cast(JsonObject, dict(body)),
            ) from exc
        if status < 200 or status >= 300:
            raise JudgeProviderError(
                f"Judge request failed with HTTP {status}",
                provider="openrouter",
                model=self._model,
                raw_provider_request=cast(JsonObject, dict(body)),
                raw_provider_response=_raw_response(status, parsed, raw),
            )
        if parsed is None:
            raise JudgeMalformedResponseError(
                "Judge response was not valid JSON",
                raw_provider_response={"status": status, "raw_body": raw},
            )
        if type(parsed) is not dict:
            raise JudgeMalformedResponseError(
                "Judge response must be a JSON object",
                raw_provider_response={"status": status, "raw_body": raw},
            )
        response = cast("dict[str, object]", parsed)
        answers = _extract_answers(response, status, raw)
        return JudgmentDecision(
            decision=cast(JsonObject, {"answers": answers}),
            model=_optional_str(response.get("model")),
            usage=_optional_usage(response),
            raw_provider_request=cast(JsonObject, dict(body)),
            raw_provider_response=cast(JsonObject, response),
        )
