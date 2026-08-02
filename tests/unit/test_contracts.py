from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from benchtable.contracts import (
    GameResult,
    MatchMemorySummary,
    ModelResponse,
    Observation,
    PluginEvent,
    ToolCall,
    ToolSpec,
    Transition,
    Usage,
)
from benchtable.errors import (
    ConfigurationError,
    InvalidActionError,
    MalformedModelError,
    ProviderError,
)


class TestToolSpec:
    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValidationError, match="name"):
            ToolSpec(name="", description="test", parameters={"type": "object"})

    def test_serializes_to_openai_function_tool_shape(self) -> None:
        spec = ToolSpec(
            name="place_bet",
            description="Place a bet",
            parameters={
                "type": "object",
                "properties": {"amount": {"type": "integer"}},
            },
        )
        raw = spec.to_openai_tool()
        assert raw == {
            "type": "function",
            "function": {
                "name": "place_bet",
                "description": "Place a bet",
                "parameters": {
                    "type": "object",
                    "properties": {"amount": {"type": "integer"}},
                },
            },
        }


class TestObservation:
    def test_contains_text_metadata_and_actor(self) -> None:
        obs = Observation(
            actor_id="player-1",
            text="You see a table.",
            metadata={"round": 3},
        )
        assert obs.actor_id == "player-1"
        assert obs.text == "You see a table."
        assert obs.metadata == {"round": 3}

    def test_has_no_raw_state_field(self) -> None:
        with pytest.raises(ValidationError):
            Observation(actor_id="a", text="b", raw_state={})  # type: ignore[call-arg]


class TestPluginEvent:
    def test_requires_event_type_and_json_payload(self) -> None:
        event = PluginEvent(event_type="hand_start", payload={"hand_index": 0})

        assert event.model_dump(mode="json") == {
            "event_type": "hand_start",
            "payload": {"hand_index": 0},
        }

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            PluginEvent(
                event_type="hand_start",
                payload={},
                unexpected=True,  # type: ignore[call-arg]
            )

    def test_rejects_non_finite_json_values(self) -> None:
        with pytest.raises(ValidationError, match="JSON-compatible"):
            PluginEvent(event_type="hand_start", payload={"value": math.nan})

    def test_rejects_nested_non_json_values(self) -> None:
        with pytest.raises(ValidationError, match="JSON-compatible"):
            PluginEvent(event_type="hand_start", payload={"value": (1, 2)})


class TestToolCall:
    def test_rejects_empty_call_id(self) -> None:
        with pytest.raises(ValidationError, match="call_id"):
            ToolCall(call_id="", name="place_bet", arguments={})

    def test_rejects_whitespace_only_call_id(self) -> None:
        with pytest.raises(ValidationError, match="whitespace"):
            ToolCall(call_id=" \t\n", name="place_bet", arguments={})

    @pytest.mark.parametrize("call_id", [123, True, None])
    def test_rejects_non_string_call_id(self, call_id: object) -> None:
        with pytest.raises(ValidationError, match="call_id"):
            ToolCall(call_id=call_id, name="place_bet", arguments={})  # type: ignore[arg-type]

    def test_retains_provider_fields(self) -> None:
        tc = ToolCall(
            call_id="call-1",
            name="place_bet",
            arguments={"amount": 10},
        )
        assert tc.call_id == "call-1"
        assert tc.name == "place_bet"
        assert tc.arguments == {"amount": 10}
        assert tc.parse_error is None

    def test_retains_raw_text_on_parse_failure(self) -> None:
        tc = ToolCall(
            call_id="call-2",
            name="place_bet",
            arguments=None,
            raw_arguments="not-json",
            parse_error="Expecting value: line 1 column 1",
        )
        assert tc.arguments is None
        assert tc.raw_arguments == "not-json"
        assert tc.parse_error is not None

    def test_normalizes_non_object_arguments_as_parse_failure(self) -> None:
        tc = ToolCall(
            call_id="call-3",
            name="place_bet",
            arguments=["not", "an", "object"],  # type: ignore[arg-type]
            raw_arguments='["not", "an", "object"]',
        )

        assert tc.arguments is None
        assert tc.raw_arguments == '["not", "an", "object"]'
        assert tc.parse_error == "Tool arguments must be a JSON object"

    def test_normalizes_json_null_arguments_as_parse_failure(self) -> None:
        tc = ToolCall(
            call_id="call-4",
            name="place_bet",
            arguments=None,
            raw_arguments="null",
        )

        assert tc.arguments is None
        assert tc.raw_arguments == "null"
        assert tc.parse_error == "Tool arguments must be a JSON object"

    def test_preserves_manual_none_arguments_without_new_parse_error(self) -> None:
        manual = ToolCall(call_id="call-5", name="place_bet", arguments=None)
        existing_error = ToolCall(
            call_id="call-6",
            name="place_bet",
            arguments=None,
            raw_arguments="not-json",
            parse_error="already recorded",
        )

        assert manual.parse_error is None
        assert existing_error.parse_error == "already recorded"


class TestModelResponse:
    def test_rejects_duplicate_tool_call_ids(self) -> None:
        with pytest.raises(ValidationError, match="duplicate call_id"):
            ModelResponse(
                assistant_text="I'll bet twice.",
                tool_calls=[
                    ToolCall(call_id="duplicate", name="place_bet", arguments={}),
                    ToolCall(call_id="duplicate", name="place_bet", arguments={}),
                ],
                finish_reason="stop",
            )

    def test_basic_response(self) -> None:
        resp = ModelResponse(
            assistant_text="I'll bet.",
            tool_calls=[
                ToolCall(call_id="c1", name="place_bet", arguments={"amount": 5})
            ],
            finish_reason="stop",
        )
        assert resp.assistant_text == "I'll bet."
        assert len(resp.tool_calls) == 1
        assert resp.finish_reason == "stop"
        assert resp.usage is None

    def test_with_usage(self) -> None:
        usage = Usage(prompt_tokens=100, completion_tokens=50)
        resp = ModelResponse(
            assistant_text="",
            tool_calls=[],
            finish_reason="stop",
            usage=usage,
        )
        assert resp.usage is not None
        assert resp.usage.prompt_tokens == 100

    def test_preserves_raw_provider_request(self) -> None:
        request = {"provider": "fake", "request": {"model": "test"}}
        response = ModelResponse(
            assistant_text="",
            tool_calls=[],
            raw_provider_request=request,
        )

        assert response.raw_provider_request == request


class TestGameResult:
    def test_completion_with_outcome(self) -> None:
        gr = GameResult(
            completed=True,
            outcome={"winner": "player-1"},
            metrics={"hands_played": 5},
        )
        assert gr.completed is True
        assert gr.outcome == {"winner": "player-1"}
        assert gr.metrics == {"hands_played": 5}


class TestMatchMemorySummary:
    def test_is_strict_and_json_compatible(self) -> None:
        summary = MatchMemorySummary(
            actor_id="player-1",
            text="Hand 3 ended in a split pot.",
            hand=3,
            turn=8,
        )

        assert summary.model_dump(mode="json") == {
            "actor_id": "player-1",
            "text": "Hand 3 ended in a split pot.",
            "hand": 3,
            "turn": 8,
        }

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            MatchMemorySummary(
                actor_id="player-1",
                text="Hand 3 ended.",
                hand=3,
                turn=8,
                board="secret",  # type: ignore[call-arg]
            )

    def test_rejects_whitespace_only_text(self) -> None:
        with pytest.raises(ValidationError, match="whitespace"):
            MatchMemorySummary(
                actor_id="player-1",
                text=" \t\n",
                hand=3,
                turn=8,
            )

    def test_rejects_whitespace_only_actor_id(self) -> None:
        with pytest.raises(ValidationError, match="whitespace"):
            MatchMemorySummary(
                actor_id=" \t\n",
                text="Hand 3 ended.",
                hand=3,
                turn=8,
            )

    @pytest.mark.parametrize("value", [b"player-1", True, 123, 3.0, None])
    def test_rejects_non_strict_actor_id(self, value: object) -> None:
        with pytest.raises(ValidationError):
            MatchMemorySummary(
                actor_id=value,  # type: ignore[arg-type]
                text="Hand 3 ended.",
                hand=3,
                turn=8,
            )

    @pytest.mark.parametrize("value", [True, "3", 3.0])
    def test_rejects_non_strict_hand_and_turn_values(self, value: object) -> None:
        with pytest.raises(ValidationError):
            MatchMemorySummary(
                actor_id="player-1",
                text="Hand 3 ended.",
                hand=value,  # type: ignore[arg-type]
                turn=8,
            )

        with pytest.raises(ValidationError):
            MatchMemorySummary(
                actor_id="player-1",
                text="Hand 3 ended.",
                hand=3,
                turn=value,  # type: ignore[arg-type]
            )


class TestTransition:
    def test_transition_fields(self) -> None:
        t = Transition(
            summary="player-1 bet 10",
            metrics={"pot": 10},
        )
        assert t.summary == "player-1 bet 10"
        assert t.metrics == {"pot": 10}


class TestDomainErrors:
    def test_configuration_error(self) -> None:
        err = ConfigurationError("missing api_key_env")
        assert str(err) == "missing api_key_env"

    def test_invalid_action_error(self) -> None:
        err = InvalidActionError("not a legal move", details={"action": "fold"})
        assert "not a legal move" in str(err)

    def test_malformed_model_error(self) -> None:
        err = MalformedModelError("no tool calls", response_text="I think")
        assert err.response_text == "I think"

    def test_provider_error(self) -> None:
        err = ProviderError("timeout", provider="openai", model="gpt-4o")
        assert err.provider == "openai"
        assert err.model == "gpt-4o"
