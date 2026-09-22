"""Generic judge contract tests."""

import pytest
from pydantic import ValidationError

from benchtable.contracts import JudgmentOutcome, JudgmentRequest
from benchtable.errors import (
    JudgeError,
    JudgeMalformedResponseError,
    JudgeProviderError,
    JudgeTimeoutError,
)


def test_judgment_request_requires_fields() -> None:
    with pytest.raises(ValidationError):
        JudgmentRequest(judge_id="", judgment_kind="k", payload={})
    with pytest.raises(ValidationError):
        JudgmentRequest(judge_id="j", judgment_kind="", payload={})


def test_judgment_outcome_success_shape() -> None:
    outcome = JudgmentOutcome(
        ok=True,
        decision={"answers": {"leak": {"probability": 0.9}}},
        model="typesafe/jev-1.13",
        latency_ms=120.0,
    )
    assert outcome.failure_reason is None
    assert outcome.usage is None


def test_judgment_outcome_failure_shape() -> None:
    outcome = JudgmentOutcome(ok=False, failure_reason="provider_error")
    assert outcome.decision is None


def test_judge_error_hierarchy() -> None:
    for exc in (
        JudgeProviderError("boom", provider="openrouter", model="m"),
        JudgeTimeoutError("slow"),
        JudgeMalformedResponseError("bad"),
    ):
        assert isinstance(exc, JudgeError)


@pytest.mark.asyncio
async def test_runtime_checkable_judge_protocol() -> None:
    class _Fake:
        @property
        def judge_id(self) -> str:
            return "fake"

        async def decide(self, request):  # noqa: ANN001
            raise AssertionError

    from benchtable.judges.protocol import Judge

    assert isinstance(_Fake(), Judge)
