"""Deterministic fake judge for offline tests."""

from __future__ import annotations

from benchtable.contracts import JsonObject
from benchtable.judges.protocol import JudgeRequest, JudgmentDecision


class FakeJudge:
    """Deterministic judge returning scripted decisions or errors.

    Behavior per `decide` call:
    - when `outcomes` is a non-empty list, pop the next entry; an `Exception`
      entry is raised, any other entry is returned as the decision data;
    - otherwise `raise_error` is raised when set;
    - otherwise a decision built from `decision` is returned.
    """

    def __init__(
        self,
        *,
        judge_id: str = "tiny-judge",
        decision: JsonObject | None = None,
        raise_error: Exception | None = None,
        outcomes: list[JsonObject | Exception] | None = None,
        raw_provider_request: JsonObject | None = None,
        raw_provider_response: JsonObject | None = None,
    ) -> None:
        self._judge_id = judge_id
        self._decision = decision or {"answers": {"leak": {"probability": 0.0}}}
        self._raise = raise_error
        self._outcomes = list(outcomes) if outcomes is not None else None
        self._raw_provider_request = raw_provider_request
        self._raw_provider_response = raw_provider_response
        self.requests: list[JudgeRequest] = []
        self.call_count = 0

    @property
    def judge_id(self) -> str:
        return self._judge_id

    async def decide(self, request: JudgeRequest) -> JudgmentDecision:
        self.requests.append(request)
        self.call_count += 1
        if self._outcomes:
            outcome = self._outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return JudgmentDecision(decision=dict(outcome), model="typesafe/jev-1.13")
        if self._raise is not None:
            raise self._raise
        return JudgmentDecision(
            decision=dict(self._decision),
            model="typesafe/jev-1.13",
            raw_provider_request=self._raw_provider_request,
            raw_provider_response=self._raw_provider_response,
        )
