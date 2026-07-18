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
from benchtable.errors import (
    BenchtableError,
    ConfigurationError,
    PluginError,
)
from benchtable.games.registry import GameRegistry

app = typer.Typer(
    name="benchtable",
    help="Run language models as agents in games and record traces.",
)


_registry: GameRegistry | None = None
AgentFactory = Callable[[AgentConfig], Agent]
_agent_factory: AgentFactory | None = None


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
    from benchtable.agents.openai_compatible import OpenAICompatibleAgent

    return OpenAICompatibleAgent(
        model=agent_config.model,
        api_key_env=agent_config.api_key_env,
        base_url=agent_config.base_url,
        timeout=agent_config.timeout,
        max_completion_tokens=agent_config.max_completion_tokens,
    )


@app.command("list-games")
def list_games() -> None:
    """List installed game plugins and their versions."""
    try:
        registry = _get_registry()
    except PluginError as exc:
        typer.echo(f"Plugin discovery error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    plugins = registry.list()
    if not plugins:
        typer.echo("No game plugins installed.")
        return
    for plugin in plugins:
        typer.echo(f"{plugin.name}\t{plugin.version}")


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
    except ConfigurationError as exc:
        typer.echo(f"Configuration error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    try:
        registry = _get_registry()
    except PluginError as exc:
        typer.echo(f"Plugin discovery error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    try:
        game = registry.load(cfg.run.game)
    except PluginError as exc:
        typer.echo(f"Plugin error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if len(cfg.agents) > 1:
        configured_ids = {agent_cfg.id for agent_cfg in cfg.agents}
        actor_ids = set(game.player_ids)
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
        except ConfigurationError as exc:
            typer.echo(f"Agent error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        agent_metadata.append(cast(JsonObject, agent_cfg.model_dump(mode="json")))

    output.mkdir(parents=True, exist_ok=True)

    from benchtable.engine import RunEngine

    engine = RunEngine(
        game=game,
        agent=next(iter(agents.values())) if len(agents) == 1 else None,
        agents=agents if len(agents) > 1 else None,
        agent_metadata=agent_metadata,
        game_config=cfg.run.game_config,
        run_dir=output,
        run_id=f"run-{cfg.run.seed}-{uuid4().hex}",
        seed=cfg.run.seed,
        matches=cfg.run.matches,
        max_turns=cfg.run.max_turns,
        max_invalid_attempts=cfg.run.max_invalid_attempts,
        max_provider_retries=cfg.run.max_provider_retries,
    )

    try:
        result = asyncio.run(engine.run())
    except BenchtableError as exc:
        typer.echo(f"Run error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    trace_path = output / "events.jsonl"
    if not result.success:
        typer.echo(f"Run failed; trace written to {trace_path}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"Trace written to {trace_path}")


if __name__ == "__main__":
    app()
