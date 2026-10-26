"""Provider test fixtures with mocked responses and configuration.

Supports testing against local mock providers, consented test accounts,
and integration with real providers through environment configuration.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from unittest.mock import Mock


@dataclass(frozen=True)
class ProviderTestConfig:
    """Configuration for a provider test including credentials and behavior."""

    provider_name: str
    test_mode: str  # 'mock', 'sandbox', 'real'
    api_key: str | None
    base_url: str | None
    should_validate_creds: bool
    error_behavior: str | None  # 'timeout', 'auth_error', 'rate_limit', None for success


class ProviderFixture(ABC):
    """Base fixture for provider testing."""

    @abstractmethod
    def get_config(self) -> ProviderTestConfig:
        """Return provider configuration."""

    @abstractmethod
    def validate_credentials(self) -> bool:
        """Validate credentials against provider."""

    @abstractmethod
    def create_completion(self, model: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Create a completion."""

    @abstractmethod
    def handle_auth_refresh(self, refresh_token: str) -> dict[str, Any]:
        """Handle auth token refresh."""

    @abstractmethod
    def handle_error(self, error_type: str) -> dict[str, Any]:
        """Handle provider error scenarios."""

    @abstractmethod
    def redact_secrets(self, data: dict[str, Any]) -> dict[str, Any]:
        """Redact sensitive data from responses."""


class OpenAITestFixture(ProviderFixture):
    """OpenAI provider test fixture."""

    def get_config(self) -> ProviderTestConfig:
        """Get OpenAI test configuration."""
        api_key = os.environ.get("OPENAI_API_KEY", "sk-test-mock-key")
        test_mode = os.environ.get("OPENAI_TEST_MODE", "mock")
        return ProviderTestConfig(
            provider_name="openai",
            test_mode=test_mode,
            api_key=api_key if api_key.startswith("sk-") else None,
            base_url=os.environ.get("OPENAI_BASE_URL"),
            should_validate_creds=test_mode != "mock",
            error_behavior=None,
        )

    def validate_credentials(self) -> bool:
        """Validate OpenAI credentials."""
        config = self.get_config()
        if config.test_mode == "mock":
            return bool(config.api_key and config.api_key.startswith("sk-"))
        # In real/sandbox modes, would make actual API call
        return bool(config.api_key)

    def create_completion(self, model: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Create completion response."""
        config = self.get_config()
        if config.error_behavior == "timeout":
            raise TimeoutError("OpenAI request timed out")
        if config.error_behavior == "auth_error":
            return {"error": {"message": "Invalid authentication", "type": "invalid_request_error"}}
        if config.error_behavior == "rate_limit":
            return {"error": {"message": "Rate limit exceeded", "type": "server_error"}}

        return {
            "id": "chatcmpl-test",
            "object": "text_completion",
            "created": 1234567890,
            "model": model,
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Test response"},
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    def handle_auth_refresh(self, refresh_token: str) -> dict[str, Any]:
        """Handle auth refresh - not applicable to OpenAI."""
        return {"error": "OpenAI does not support token refresh"}

    def handle_error(self, error_type: str) -> dict[str, Any]:
        """Handle specific error scenarios."""
        errors = {
            "timeout": {"error": {"message": "Request timeout", "type": "server_error"}},
            "auth": {"error": {"message": "Invalid API key", "type": "authentication_error"}},
            "rate_limit": {"error": {"message": "Rate limit exceeded", "type": "server_error"}},
            "not_found": {"error": {"message": "Model not found", "type": "invalid_request_error"}},
        }
        return errors.get(error_type, {"error": {"message": "Unknown error", "type": "server_error"}})

    def redact_secrets(self, data: dict[str, Any]) -> dict[str, Any]:
        """Redact API keys and sensitive data."""
        redacted = data.copy()
        if "api_key" in redacted:
            api_key = redacted["api_key"]
            redacted["api_key"] = f"sk-{api_key[-4:]}" if len(api_key) > 4 else "sk-***"
        return redacted


class AnthropicTestFixture(ProviderFixture):
    """Anthropic provider test fixture."""

    def get_config(self) -> ProviderTestConfig:
        """Get Anthropic test configuration."""
        api_key = os.environ.get("ANTHROPIC_API_KEY", "sk-ant-test-mock-key")
        test_mode = os.environ.get("ANTHROPIC_TEST_MODE", "mock")
        return ProviderTestConfig(
            provider_name="anthropic",
            test_mode=test_mode,
            api_key=api_key if api_key.startswith("sk-ant-") else None,
            base_url=os.environ.get("ANTHROPIC_BASE_URL"),
            should_validate_creds=test_mode != "mock",
            error_behavior=None,
        )

    def validate_credentials(self) -> bool:
        """Validate Anthropic credentials."""
        config = self.get_config()
        if config.test_mode == "mock":
            return bool(config.api_key and config.api_key.startswith("sk-ant-"))
        return bool(config.api_key)

    def create_completion(self, model: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Create completion response."""
        config = self.get_config()
        if config.error_behavior == "timeout":
            raise TimeoutError("Anthropic request timed out")
        if config.error_behavior == "auth_error":
            return {"error": {"message": "Invalid x-api-key header", "type": "authentication_error"}}
        if config.error_behavior == "rate_limit":
            return {"error": {"message": "Rate limited", "type": "rate_limit_error"}}

        return {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Test response"}],
            "model": model,
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

    def handle_auth_refresh(self, refresh_token: str) -> dict[str, Any]:
        """Handle auth refresh - not applicable to Anthropic."""
        return {"error": "Anthropic does not support token refresh"}

    def handle_error(self, error_type: str) -> dict[str, Any]:
        """Handle specific error scenarios."""
        errors = {
            "timeout": {"error": {"message": "Request timeout", "type": "timeout_error"}},
            "auth": {"error": {"message": "Invalid API key", "type": "authentication_error"}},
            "rate_limit": {"error": {"message": "Rate limited", "type": "rate_limit_error"}},
            "overloaded": {"error": {"message": "Model overloaded", "type": "overloaded_error"}},
        }
        return errors.get(error_type, {"error": {"message": "Unknown error", "type": "api_error"}})

    def redact_secrets(self, data: dict[str, Any]) -> dict[str, Any]:
        """Redact API keys and sensitive data."""
        redacted = data.copy()
        if "api_key" in redacted:
            api_key = redacted["api_key"]
            redacted["api_key"] = f"sk-ant-{api_key[-4:]}" if len(api_key) > 8 else "sk-ant-***"
        return redacted


class GoogleTestFixture(ProviderFixture):
    """Google AI provider test fixture."""

    def get_config(self) -> ProviderTestConfig:
        """Get Google test configuration."""
        api_key = os.environ.get("GOOGLE_API_KEY", "test-mock-key")
        test_mode = os.environ.get("GOOGLE_TEST_MODE", "mock")
        return ProviderTestConfig(
            provider_name="google",
            test_mode=test_mode,
            api_key=api_key,
            base_url=os.environ.get("GOOGLE_BASE_URL"),
            should_validate_creds=test_mode != "mock",
            error_behavior=None,
        )

    def validate_credentials(self) -> bool:
        """Validate Google credentials."""
        config = self.get_config()
        if config.test_mode == "mock":
            return bool(config.api_key)
        return bool(config.api_key)

    def create_completion(self, model: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Create completion response."""
        config = self.get_config()
        if config.error_behavior == "timeout":
            raise TimeoutError("Google request timed out")
        if config.error_behavior == "auth_error":
            return {"error": {"code": 401, "message": "Invalid API key"}}
        if config.error_behavior == "rate_limit":
            return {"error": {"code": 429, "message": "Rate limit exceeded"}}

        return {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Test response"}],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
            },
        }

    def handle_auth_refresh(self, refresh_token: str) -> dict[str, Any]:
        """Handle OAuth refresh for Google."""
        if not refresh_token:
            return {"error": "No refresh token provided"}
        return {
            "access_token": "test-new-token",
            "expires_in": 3600,
            "refresh_token": refresh_token,
        }

    def handle_error(self, error_type: str) -> dict[str, Any]:
        """Handle specific error scenarios."""
        errors = {
            "timeout": {"error": {"code": 504, "message": "Gateway timeout"}},
            "auth": {"error": {"code": 401, "message": "Unauthenticated"}},
            "rate_limit": {"error": {"code": 429, "message": "Resource exhausted"}},
            "not_found": {"error": {"code": 404, "message": "Not found"}},
        }
        return errors.get(error_type, {"error": {"code": 500, "message": "Internal error"}})

    def redact_secrets(self, data: dict[str, Any]) -> dict[str, Any]:
        """Redact API keys and sensitive data."""
        redacted = data.copy()
        if "api_key" in redacted:
            api_key = redacted["api_key"]
            redacted["api_key"] = f"***{api_key[-4:]}" if len(api_key) > 4 else "***"
        if "refresh_token" in redacted:
            redacted["refresh_token"] = "***"
        return redacted


class VoiceProviderTestFixture(ProviderFixture):
    """Voice/STT provider test fixture."""

    def get_config(self) -> ProviderTestConfig:
        """Get voice provider test configuration."""
        provider = os.environ.get("VOICE_PROVIDER", "google")
        api_key = os.environ.get(f"{provider.upper()}_VOICE_KEY", "test-mock-key")
        test_mode = os.environ.get("VOICE_TEST_MODE", "mock")
        return ProviderTestConfig(
            provider_name=f"voice_{provider}",
            test_mode=test_mode,
            api_key=api_key,
            base_url=os.environ.get(f"{provider.upper()}_VOICE_BASE_URL"),
            should_validate_creds=test_mode != "mock",
            error_behavior=None,
        )

    def validate_credentials(self) -> bool:
        """Validate voice provider credentials."""
        config = self.get_config()
        if config.test_mode == "mock":
            return bool(config.api_key)
        return bool(config.api_key)

    def create_completion(self, model: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Create voice transcription response."""
        config = self.get_config()
        if config.error_behavior == "timeout":
            raise TimeoutError("Voice provider request timed out")
        if config.error_behavior == "auth_error":
            return {"error": "Invalid authentication"}

        return {
            "transcript": "Test voice transcription",
            "confidence": 0.95,
            "language": "en-US",
            "duration_ms": 1500,
        }

    def handle_auth_refresh(self, refresh_token: str) -> dict[str, Any]:
        """Handle auth refresh for OAuth-based voice providers."""
        if not refresh_token:
            return {"error": "No refresh token"}
        return {
            "access_token": "test-new-token",
            "expires_in": 3600,
            "refresh_token": refresh_token,
        }

    def handle_error(self, error_type: str) -> dict[str, Any]:
        """Handle voice provider error scenarios."""
        errors = {
            "timeout": {"error": "Request timeout"},
            "auth": {"error": "Authentication failed"},
            "rate_limit": {"error": "Rate limit exceeded"},
            "invalid_audio": {"error": "Invalid audio format"},
            "no_speech": {"error": "No speech detected"},
        }
        return errors.get(error_type, {"error": "Unknown error"})

    def redact_secrets(self, data: dict[str, Any]) -> dict[str, Any]:
        """Redact sensitive voice provider data."""
        redacted = data.copy()
        if "api_key" in redacted:
            redacted["api_key"] = "***"
        if "access_token" in redacted:
            redacted["access_token"] = "***"
        return redacted


class CalendarProviderTestFixture(ProviderFixture):
    """Calendar/OAuth provider test fixture."""

    def get_config(self) -> ProviderTestConfig:
        """Get calendar provider test configuration."""
        provider = os.environ.get("CALENDAR_PROVIDER", "google")
        return ProviderTestConfig(
            provider_name=f"calendar_{provider}",
            test_mode=os.environ.get("CALENDAR_TEST_MODE", "mock"),
            api_key=os.environ.get(f"{provider.upper()}_CALENDAR_KEY"),
            base_url=os.environ.get(f"{provider.upper()}_CALENDAR_BASE_URL"),
            should_validate_creds=os.environ.get("CALENDAR_TEST_MODE", "mock") != "mock",
            error_behavior=None,
        )

    def validate_credentials(self) -> bool:
        """Validate calendar provider credentials."""
        config = self.get_config()
        if config.test_mode == "mock":
            return True
        return bool(config.api_key)

    def create_completion(self, model: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Not used for calendar provider."""
        raise NotImplementedError("Calendar provider does not support completions")

    def handle_auth_refresh(self, refresh_token: str) -> dict[str, Any]:
        """Handle OAuth refresh for calendar provider."""
        if not refresh_token:
            return {"error": "No refresh token provided"}
        return {
            "access_token": "test-new-calendar-token",
            "expires_in": 3600,
            "refresh_token": refresh_token,
        }

    def handle_error(self, error_type: str) -> dict[str, Any]:
        """Handle calendar provider error scenarios."""
        errors = {
            "timeout": {"error": "Calendar sync timeout"},
            "auth": {"error": "Calendar authentication failed"},
            "rate_limit": {"error": "Calendar rate limit exceeded"},
            "permission": {"error": "Calendar access denied"},
        }
        return errors.get(error_type, {"error": "Calendar error"})

    def redact_secrets(self, data: dict[str, Any]) -> dict[str, Any]:
        """Redact sensitive calendar provider data."""
        redacted = data.copy()
        for field in ["api_key", "access_token", "refresh_token"]:
            if field in redacted:
                redacted[field] = "***"
        return redacted


def get_fixture(provider_name: str) -> ProviderFixture:
    """Get test fixture for a provider."""
    fixtures = {
        "openai": OpenAITestFixture,
        "anthropic": AnthropicTestFixture,
        "google": GoogleTestFixture,
        "voice": VoiceProviderTestFixture,
        "calendar": CalendarProviderTestFixture,
    }
    fixture_class = fixtures.get(provider_name.lower())
    if not fixture_class:
        raise ValueError(f"Unknown provider: {provider_name}")
    return fixture_class()
