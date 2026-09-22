"""Judge construction."""

from benchtable.config import JudgeConfig
from benchtable.errors import ConfigurationError
from benchtable.judges.openrouter_decisions import OpenRouterDecisionsJudge
from benchtable.judges.protocol import Judge


def create_judge(config: JudgeConfig) -> Judge:
    """Build the configured judge adapter.

    Raises ConfigurationError for unknown adapter names; ``JudgeConfig``
    rejects unknown adapters earlier through its Literal field, so this
    guard covers hand-built configs and future adapter removal.
    """
    if config.adapter == "openrouter_decisions":
        return OpenRouterDecisionsJudge(
            judge_id=config.id,
            model=config.model,
            api_key_env=config.api_key_env or "OPENROUTER_API_KEY",
            base_url=config.base_url,
            timeout=config.timeout if config.timeout is not None else 30.0,
        )
    raise ConfigurationError(f"Unknown judge adapter: {config.adapter}")
