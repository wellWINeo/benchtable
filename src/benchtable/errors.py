"""Domain error types for benchtable."""

from __future__ import annotations

from typing import Any


class BenchtableError(Exception):
    """Base error for all benchtable domain errors."""


class ConfigurationError(BenchtableError):
    """Invalid or missing configuration."""


class PluginError(BenchtableError):
    """Plugin discovery or loading failure."""


class MalformedModelError(BenchtableError):
    """Model output could not be parsed into a valid action."""

    def __init__(self, message: str, *, response_text: str | None = None) -> None:
        super().__init__(message)
        self.response_text = response_text


class InvalidActionError(BenchtableError):
    """The parsed action is not legal in the current game state."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details


class ProviderError(BenchtableError):
    """The model provider returned an error or timed out."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.model = model


class RunError(BenchtableError):
    """A run-level failure such as exhausted budgets or cancellation."""
