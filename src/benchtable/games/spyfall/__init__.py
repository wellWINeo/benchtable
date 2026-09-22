"""Spyfall game plugin package."""

from benchtable.games.spyfall.config import (
    SpyfallConfig,
    SpyfallLocation,
    min_required_turns,
    parse_spyfall_config,
)

__all__ = [
    "SpyfallConfig",
    "SpyfallLocation",
    "min_required_turns",
    "parse_spyfall_config",
]
