"""Benchtable CLI: list-games and run commands."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, cast
from uuid import uuid4

import typer

from benchtable.agents.protocol import Agent
from benchtable.config import AgentConfig, load_config
from benchtable.contracts import JsonObject
from benchtable.events import redact_value
from benchtable.games.registry import GameRegistry

app = typer.Typer(
    name="benchtable",
    help="Run language models as agents in games and record traces.",
)


_registry: GameRegistry | None = None
AgentFactory = Callable[[AgentConfig], Agent]
_agent_factory: AgentFactory | None = None


def _safe_error_message(error: BaseException) -> str:
    """Redact credential-shaped text before printing an error."""
    return cast(str, redact_value(str(error)))


def _report_match_progress(completed: int, total: int, success: bool) -> None:
    """Print progress after each match without mixing with the result path."""
    status = "completed" if success else "failed"
    typer.echo(f"Match {completed}/{total} {status}", err=True)


def _get_registry() -> GameRegistry:
    """Create and populate the game registry."""
    global _registry
    if _registry is not None:
        return _registry
    registry = GameRegistry()
    registry.discover()
    _registry = registry
    return registry


def set_registry(registry: GameRegistry | None) -> None:
    """Override the registry (for testing)."""
    global _registry
    _registry = registry


def set_agent_factory(factory: AgentFactory | None) -> None:
    """Override agent construction (for testing)."""
    global _agent_factory
    _agent_factory = factory


def _default_agent_factory(agent_config: AgentConfig) -> Agent:
    match agent_config.provider:
        case "openai_compatible":
            from benchtable.agents.openai_compatible import OpenAICompatibleAgent

            api_key_env = agent_config.api_key_env
            if api_key_env is None:
                raise AssertionError("validated OpenAI configuration lacks api_key_env")
            return OpenAICompatibleAgent(
                model=agent_config.model,
                api_key_env=api_key_env,
                base_url=agent_config.base_url,
                timeout=agent_config.timeout,
                max_completion_tokens=agent_config.max_completion_tokens,
            )
        case "gigachat":
            from benchtable.agents.gigachat import GigaChatAgent

            credential_env = agent_config.credential_env
            scope = agent_config.scope
            if credential_env is None or scope is None:
                raise AssertionError("validated GigaChat configuration is incomplete")
            return GigaChatAgent(
                model=agent_config.model,
                credential_env=credential_env,
                scope=scope,
                timeout=agent_config.timeout,
                max_completion_tokens=agent_config.max_completion_tokens,
                verify_ssl_certs=agent_config.verify_ssl_certs,
            )
        case "yandex_ai_studio":
            from benchtable.agents.yandex_ai_studio import YandexAIStudioAgent

            credential_env = agent_config.credential_env
            folder_id = agent_config.folder_id
            credential_kind = agent_config.credential_kind
            if credential_env is None or folder_id is None or credential_kind is None:
                raise AssertionError("validated Yandex configuration is incomplete")
            return YandexAIStudioAgent(
                model=agent_config.model,
                folder_id=folder_id,
                credential_kind=credential_kind,
                credential_env=credential_env,
                timeout=agent_config.timeout,
                max_completion_tokens=agent_config.max_completion_tokens,
            )


@app.command("list-games")
def list_games() -> None:
    """List installed game plugins and their versions."""
    try:
        registry = _get_registry()
    except Exception as exc:
        typer.echo(f"Plugin discovery error: {_safe_error_message(exc)}", err=True)
        raise typer.Exit(code=1) from exc
    try:
        plugins = registry.list()
        if not plugins:
            typer.echo("No game plugins installed.")
            return
        for plugin in plugins:
            typer.echo(f"{plugin.name}\t{plugin.version}")
    except Exception as exc:
        typer.echo(f"Plugin listing error: {_safe_error_message(exc)}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("run")
def run_experiment(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to TOML configuration."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory for trace."),
    ],
) -> None:
    """Validate configuration, execute matches, and write the trace."""
    try:
        cfg = load_config(config)
    except Exception as exc:
        typer.echo(f"Configuration error: {_safe_error_message(exc)}", err=True)
        raise typer.Exit(code=1) from exc

    try:
        registry = _get_registry()
    except Exception as exc:
        typer.echo(f"Plugin discovery error: {_safe_error_message(exc)}", err=True)
        raise typer.Exit(code=1) from exc
    try:
        game = registry.load(cfg.run.game)
    except Exception as exc:
        typer.echo(f"Plugin error: {_safe_error_message(exc)}", err=True)
        raise typer.Exit(code=1) from exc

    # Validate game-specific configuration before constructing agents.
    try:
        registry.validate_plugin_config(cfg.run.game, cfg.run.game_config)
    except Exception as exc:
        typer.echo(f"Game configuration error: {_safe_error_message(exc)}", err=True)
        raise typer.Exit(code=1) from exc

    try:
        actor_ids = set(registry.resolve_player_ids(cfg.run.game, cfg.run.game_config))
    except Exception as exc:
        typer.echo(f"Game configuration error: {_safe_error_message(exc)}", err=True)
        raise typer.Exit(code=1) from exc

    try:
        exact_mapping = registry.requires_exact_agent_ids(cfg.run.game)
    except Exception as exc:
        typer.echo(f"Plugin error: {_safe_error_message(exc)}", err=True)
        raise typer.Exit(code=1) from exc

    if exact_mapping or len(cfg.agents) > 1:
        configured_ids = {agent_cfg.id for agent_cfg in cfg.agents}
        unknown_ids = sorted(configured_ids - actor_ids)
        missing_ids = sorted(actor_ids - configured_ids)
        if unknown_ids or missing_ids:
            details: list[str] = []
            if unknown_ids:
                details.append(f"unknown agent IDs: {', '.join(unknown_ids)}")
            if missing_ids:
                details.append(f"missing agent IDs: {', '.join(missing_ids)}")
            typer.echo(
                "Configuration error: multi-agent IDs must match game actors ("
                + "; ".join(details)
                + ")",
                err=True,
            )
            raise typer.Exit(code=1)
    # Construct every configured agent so multi-actor games can dispatch by ID.
    agent_factory = _agent_factory or _default_agent_factory
    agents: dict[str, Agent] = {}
    agent_metadata: list[JsonObject] = []
    for agent_cfg in cfg.agents:
        try:
            agents[agent_cfg.id] = agent_factory(agent_cfg)
        except Exception as exc:
            typer.echo(f"Agent error: {_safe_error_message(exc)}", err=True)
            raise typer.Exit(code=1) from exc
        agent_metadata.append(cast(JsonObject, agent_cfg.model_dump(mode="json")))

    try:
        output.mkdir(parents=True, exist_ok=True)
        typer.echo(f"Running {cfg.run.matches} match(es)...", err=True)

        from benchtable.engine import RunEngine

        engine = RunEngine(
            game=game,
            agent=next(iter(agents.values())) if len(agents) == 1 else None,
            agents=agents if len(agents) > 1 else None,
            agent_id=cfg.agents[0].id if len(agents) == 1 else None,
            agent_metadata=agent_metadata,
            game_config=cfg.run.game_config,
            run_dir=output,
            run_id=f"run-{cfg.run.seed}-{uuid4().hex}",
            seed=cfg.run.seed,
            matches=cfg.run.matches,
            max_turns=cfg.run.max_turns,
            max_invalid_attempts=cfg.run.max_invalid_attempts,
            max_provider_retries=cfg.run.max_provider_retries,
            max_memory_operations_per_turn=cfg.run.max_memory_operations_per_turn,
            progress_callback=_report_match_progress,
        )
        result = asyncio.run(engine.run())
    except Exception as exc:
        typer.echo(f"Run error: {_safe_error_message(exc)}", err=True)
        raise typer.Exit(code=1) from exc

    try:
        trace_path = output / "events.jsonl"
        success = result.success
    except Exception as exc:
        typer.echo(f"Run error: {_safe_error_message(exc)}", err=True)
        raise typer.Exit(code=1) from exc
    if not success:
        typer.echo(f"Run failed; trace written to {trace_path}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"Trace written to {trace_path}")


if __name__ == "__main__":
    app()
