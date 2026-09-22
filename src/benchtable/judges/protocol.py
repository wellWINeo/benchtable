"""Judge protocol for typed model decisions."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, StrictStr

from benchtable.contracts import JsonObject


class JudgeRequest(BaseModel):
    """Input to a judge for one decision."""

    model_config = ConfigDict(extra="forbid")

    judgment_kind: StrictStr = Field(..., min_length=1)
    payload: JsonObject


class JudgmentDecision(BaseModel):
    """Normalized typed decision from a judge adapter."""

    model_config = ConfigDict(extra="forbid")

    decision: JsonObject
    model: StrictStr | None = None
    usage: JsonObject | None = None
    raw_provider_request: JsonObject | None = None
    raw_provider_response: JsonObject | None = None


@runtime_checkable
class Judge(Protocol):
    """A typed-decision judge, separate from the Agent protocol."""

    @property
    def judge_id(self) -> str: ...

    async def decide(self, request: JudgeRequest) -> JudgmentDecision: ...
