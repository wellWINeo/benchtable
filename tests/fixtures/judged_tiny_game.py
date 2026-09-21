"""A tiny judged test game exercising the engine judgment capabilities.

Two actors alternate calling "act". Every "act" action requests one judgment
(kind "tiny_check") unless judgments are disabled for the instance. The session
records every judgment outcome the engine delivers so tests can assert on the
normalized result. Not registered as an entry point.
"""

from __future__ import annotations

from benchtable.contracts import (
    GameResult,
    JsonObject,
    JudgmentOutcome,
    JudgmentRequest,
    Observation,
    ToolSpec,
    Transition,
)


class JudgedTinySession:
    """Session for the judged alternating-choice test game."""

    def __init__(
        self,
        *,
        seed: int,
        max_actions: int = 4,
        actors: list[str] | None = None,
        judge_id: str = "tiny-judge",
        required_judges: list[str] | None = None,
        request_judgment: bool = True,
    ) -> None:
        self._seed = seed
        self._actions_taken = 0
        self._max_actions = max_actions
        self._actors = tuple(actors or ("a", "b"))
        self._judge_id = judge_id
        self._required_judges = list(
            required_judges if required_judges is not None else [judge_id]
        )
        self._request_judgment = request_judgment
        self.received_judgments: list[JudgmentOutcome | None] = []
        self._private = {actor: f"secret_{actor}" for actor in self._actors}

    @property
    def current_actor_id(self) -> str:
        return self._actors[self._actions_taken % len(self._actors)]

    def get_observation(self, actor_id: str) -> Observation:
        return Observation(
            actor_id=actor_id,
            text=(
                f"You are player {actor_id}. "
                f"Your private marker is {self._private[actor_id]}. "
                f"Actions taken so far: {self._actions_taken}."
            ),
            metadata={"actions_taken": self._actions_taken, "seed": self._seed},
        )

    def get_tools(self, actor_id: str) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="act",
                description="Take an action in the game.",
                parameters={"type": "object", "properties": {}},
            )
        ]

    def required_judge_ids(self) -> list[str]:
        return list(self._required_judges)

    def judgment_request(
        self, actor_id: str, tool_name: str, arguments: JsonObject
    ) -> JudgmentRequest | None:
        if not self._request_judgment:
            return None
        if tool_name != "act":
            return None
        return JudgmentRequest(
            judge_id=self._judge_id,
            judgment_kind="tiny_check",
            payload={"value": 1},
        )

    def apply_judged_action(
        self,
        actor_id: str,
        tool_name: str,
        arguments: JsonObject,
        judgment: JudgmentOutcome | None,
    ) -> Transition:
        from benchtable.errors import InvalidActionError

        if tool_name != "act":
            raise InvalidActionError(f"Unknown tool: {tool_name}")
        self.received_judgments.append(judgment)
        self._actions_taken += 1
        return Transition(
            summary=f"player {actor_id} acted",
            metrics={"actions_taken": self._actions_taken},
        )

    @property
    def is_terminal(self) -> bool:
        return self._actions_taken >= self._max_actions

    def get_result(self) -> GameResult:
        return GameResult(
            completed=True,
            outcome={"total_actions": self._actions_taken},
            metrics={"seed": self._seed},
        )


class JudgedTinyGame:
    """A minimal judged test game plugin.  Not registered as an entry point."""

    def __init__(
        self,
        *,
        max_actions: int = 4,
        judge_id: str = "tiny-judge",
        required_judges: list[str] | None = None,
        request_judgment: bool = True,
    ) -> None:
        self._max_actions = max_actions
        self._judge_id = judge_id
        self._required_judges = required_judges
        self._request_judgment = request_judgment
        self.sessions: list[JudgedTinySession] = []

    @property
    def name(self) -> str:
        return "judged-tiny"

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def player_ids(self) -> list[str]:
        return ["a", "b"]

    def system_prompt(self, actor_id: str) -> str:
        return f"You are {actor_id}. Call 'act' when it is your turn."

    def create_session(
        self, *, seed: int, game_config: JsonObject | None = None
    ) -> JudgedTinySession:
        session = JudgedTinySession(
            seed=seed,
            max_actions=self._max_actions,
            judge_id=self._judge_id,
            required_judges=self._required_judges,
            request_judgment=self._request_judgment,
        )
        self.sessions.append(session)
        return session
