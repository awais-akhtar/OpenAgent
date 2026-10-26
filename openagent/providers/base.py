"""Base provider interface for external LLM integrations.

Supports OpenAI, Anthropic, Google, and other providers with:
- Authentication and key management
- Request/response mapping
- Error handling and fallback
- Rate limiting and quota tracking
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ProviderError(Exception):
    """Raised when a provider operation fails."""


class BaseProvider(ABC):
    """Base class for all LLM providers."""

    def __init__(self, name: str, api_key: str, base_url: str = "") -> None:
        """Initialize provider with credentials."""
        self.name = name
        self.api_key = api_key
        self.base_url = base_url
        self.request_count = 0
        self.error_count = 0

    @abstractmethod
    def validate_credentials(self) -> bool:
        """Validate that credentials are valid and provider is accessible."""

    @abstractmethod
    def create_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create a completion from the provider."""

    @abstractmethod
    def create_embedding(
        self,
        model: str,
        input: str | list[str],
    ) -> dict[str, Any]:
        """Create embeddings from the provider."""

    @abstractmethod
    def list_models(self) -> list[dict[str, Any]]:
        """List available models from this provider."""

    def track_request(self, success: bool = True) -> None:
        """Track request metrics."""
        self.request_count += 1
        if not success:
            self.error_count += 1

    def get_stats(self) -> dict[str, Any]:
        """Return request statistics."""
        return {
            "provider": self.name,
            "requests": self.request_count,
            "errors": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
        }
