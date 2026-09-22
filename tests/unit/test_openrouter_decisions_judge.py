"""OpenRouter Decisions judge adapter tests with an injected fake transport."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from benchtable.config import JudgeConfig
from benchtable.errors import (
    ConfigurationError,
    JudgeMalformedResponseError,
    JudgeProviderError,
    JudgeTimeoutError,
)
from benchtable.judges import create_judge
from benchtable.judges.openrouter_decisions import (
    DEFAULT_DECISIONS_URL,
    OpenRouterDecisionsJudge,
)
from benchtable.judges.protocol import JudgeRequest

API_KEY_ENV = "OPENROUTER_API_KEY"
API_KEY = "test-key-123"
MODEL = "typesafe/jev-1.13"


class FakeTransport:
    """Async transport recording calls and returning a scripted response."""

    def __init__(
        self,
        *,
        status: int = 200,
        body: Any = None,
        raw_text: str | None = None,
        raise_exc: Exception | None = None,
    ) -> None:
        self.calls: list[tuple[str, dict[str, str], dict[str, Any], float]] = []
        self._status = status
        self._body = body
        self._raw_text = (
            raw_text if raw_text is not None else json.dumps(body, default=str)
        )
        self._raise = raise_exc

    async def __call__(
        self,
        url: str,
        headers: dict[str, str],
        json_body: dict[str, Any],
        timeout: float,
    ) -> tuple[int, Any, str]:
        self.calls.append((url, headers, json_body, timeout))
        if self._raise is not None:
            raise self._raise
        return self._status, self._body, self._raw_text


def _payload() -> dict[str, Any]:
    return {
        "state": {"secret_location": "airport", "candidate_text": "busy place"},
        "questions": {"leak": {"type": "noul", "instructions": "rubric"}},
    }


def _request() -> JudgeRequest:
    return JudgeRequest(judgment_kind="spyfall_leak", payload=_payload())


def _judge(transport: FakeTransport, **overrides: Any) -> OpenRouterDecisionsJudge:
    return OpenRouterDecisionsJudge(
        judge_id="spyfall-leak-judge",
        model=MODEL,
        api_key_env=API_KEY_ENV,
        _post=transport,
        **overrides,
    )


def test_construction_requires_present_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    with pytest.raises(ConfigurationError):
        _judge(FakeTransport())


def test_construction_rejects_blank_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "   ")
    with pytest.raises(ConfigurationError):
        _judge(FakeTransport())


def test_construction_stores_no_literal_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, API_KEY)
    judge = _judge(FakeTransport())
    stored = {key: repr(value) for key, value in vars(judge).items()}
    assert stored["_api_key_env"] == repr(API_KEY_ENV)
    assert API_KEY not in json.dumps(stored)


async def test_decide_posts_verbatim_and_normalizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(API_KEY_ENV, API_KEY)
    transport = FakeTransport(
        body={
            "answers": {"leak": {"probability": 0.42}},
            "model": MODEL,
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        }
    )
    judge = _judge(transport)

    decision = await judge.decide(_request())

    assert len(transport.calls) == 1
    url, headers, body, timeout = transport.calls[0]
    assert url == DEFAULT_DECISIONS_URL
    assert headers["Authorization"] == f"Bearer {API_KEY}"
    assert body == {"model": MODEL, **_payload()}
    assert timeout == 30.0
    assert decision.decision["answers"]["leak"]["probability"] == 0.42
    assert decision.model == MODEL
    assert decision.usage == {"prompt_tokens": 3, "completion_tokens": 1}
    assert decision.raw_provider_request == {"model": MODEL, **_payload()}
    assert decision.raw_provider_response is not None
    assert decision.raw_provider_response["answers"]["leak"]["probability"] == 0.42


async def test_explicit_base_url_replaces_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(API_KEY_ENV, API_KEY)
    transport = FakeTransport(body={"answers": {}})
    judge = _judge(transport, base_url="https://example.test/decisions")

    await judge.decide(_request())

    assert transport.calls[0][0] == "https://example.test/decisions"


@pytest.mark.parametrize("status", [429, 503])
async def test_http_error_raises_provider_error_with_raw_request(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    monkeypatch.setenv(API_KEY_ENV, API_KEY)
    transport = FakeTransport(status=status, body={"error": "slow down"})
    judge = _judge(transport)

    with pytest.raises(JudgeProviderError) as excinfo:
        await judge.decide(_request())

    error = excinfo.value
    assert error.provider == "openrouter"
    assert error.model == MODEL
    assert error.raw_provider_request == {"model": MODEL, **_payload()}
    assert error.raw_provider_response == {"error": "slow down"}


async def test_timeout_exception_raises_judge_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(API_KEY_ENV, API_KEY)
    transport = FakeTransport(raise_exc=httpx.TimeoutException("timed out"))
    judge = _judge(transport)

    with pytest.raises(JudgeTimeoutError):
        await judge.decide(_request())


async def test_invalid_json_raises_malformed_with_raw_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(API_KEY_ENV, API_KEY)
    transport = FakeTransport(raw_text="<html>gateway error</html>")
    judge = _judge(transport)

    with pytest.raises(JudgeMalformedResponseError) as excinfo:
        await judge.decide(_request())

    assert excinfo.value.raw_provider_response == {
        "status": 200,
        "raw_body": "<html>gateway error</html>",
    }


async def test_missing_answers_raises_malformed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(API_KEY_ENV, API_KEY)
    transport = FakeTransport(body={"model": MODEL})
    judge = _judge(transport)

    with pytest.raises(JudgeMalformedResponseError) as excinfo:
        await judge.decide(_request())

    assert excinfo.value.raw_provider_response is not None


async def test_non_object_answers_raise_malformed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(API_KEY_ENV, API_KEY)
    transport = FakeTransport(body={"answers": {"leak": "high"}})
    judge = _judge(transport)

    with pytest.raises(JudgeMalformedResponseError):
        await judge.decide(_request())


def test_create_judge_builds_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, API_KEY)
    config = JudgeConfig(id="spyfall-leak-judge", model=MODEL)
    judge = create_judge(config)
    assert isinstance(judge, OpenRouterDecisionsJudge)
    assert judge.judge_id == "spyfall-leak-judge"


def test_create_judge_rejects_unknown_adapter() -> None:
    with pytest.raises(Exception):
        JudgeConfig.model_validate(
            {"id": "j", "model": MODEL, "adapter": "unknown_adapter"}
        )


def test_construction_rejects_unpinned_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "test-key")
    for model in ("typesafe/jev-1.12", "openai/gpt-4o", " typesafe/jev-1.13"):
        with pytest.raises(ConfigurationError, match="pinned"):
            OpenRouterDecisionsJudge(
                judge_id="spyfall-leak-judge",
                model=model,
                api_key_env=API_KEY_ENV,
                _post=FakeTransport(),
            )


def test_construction_accepts_pinned_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "test-key")
    judge = _judge(FakeTransport())
    assert judge.judge_id == "spyfall-leak-judge"
