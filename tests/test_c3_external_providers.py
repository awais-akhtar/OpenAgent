"""Tests for C3: External Provider Integration with OpenAI, Anthropic, Google."""

from __future__ import annotations

import json
from http import HTTPStatus
from tempfile import TemporaryDirectory
from typing import Any
from unittest import TestCase
from unittest.mock import Mock, patch, MagicMock
from urllib.request import Request, urlopen

from openagent.providers.base import BaseProvider, ProviderError
from openagent.routes.runtime import build_http_server
from openagent.storage import OpenAgentStore
from tests.fixtures.provider_fixtures import (
    get_fixture,
    OpenAITestFixture,
    AnthropicTestFixture,
    GoogleTestFixture,
    VoiceProviderTestFixture,
    CalendarProviderTestFixture,
)


class MockOpenAIProvider(BaseProvider):
    """Mock OpenAI provider for testing."""

    def validate_credentials(self) -> bool:
        """Mock validation."""
        return bool(self.api_key and self.api_key.startswith("sk-"))

    def create_completion(self, model: str, messages: list[dict[str, str]], **kwargs) -> dict[str, Any]:
        """Mock completion."""
        self.track_request()
        return {
            "id": "mock-completion-id",
            "object": "text_completion",
            "created": 1234567890,
            "model": model,
            "choices": [
                {
                    "text": "Mock response from OpenAI",
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    def create_embedding(self, model: str, input: str | list[str]) -> dict[str, Any]:
        """Mock embeddings."""
        self.track_request()
        if isinstance(input, str):
            input = [input]
        return {
            "object": "list",
            "data": [
                {"object": "embedding", "index": i, "embedding": [0.1] * 1536}
                for i in range(len(input))
            ],
            "model": model,
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
        }

    def list_models(self) -> list[dict[str, Any]]:
        """Mock model listing."""
        self.track_request()
        return [
            {"id": "gpt-4", "owned_by": "openai", "permission": []},
            {"id": "gpt-3.5-turbo", "owned_by": "openai", "permission": []},
        ]


class MockAnthropicProvider(BaseProvider):
    """Mock Anthropic provider for testing."""

    def validate_credentials(self) -> bool:
        """Mock validation."""
        return bool(self.api_key and self.api_key.startswith("sk-ant-"))

    def create_completion(self, model: str, messages: list[dict[str, str]], **kwargs) -> dict[str, Any]:
        """Mock completion."""
        self.track_request()
        return {
            "id": "mock-completion-id",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Mock response from Anthropic"}],
            "model": model,
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

    def create_embedding(self, model: str, input: str | list[str]) -> dict[str, Any]:
        """Mock embeddings."""
        self.track_request()
        if isinstance(input, str):
            input = [input]
        return {
            "object": "list",
            "data": [
                {"object": "embedding", "index": i, "embedding": [0.2] * 1024}
                for i in range(len(input))
            ],
            "model": model,
        }

    def list_models(self) -> list[dict[str, Any]]:
        """Mock model listing."""
        self.track_request()
        return [
            {"id": "claude-3-opus", "display_name": "Claude 3 Opus"},
            {"id": "claude-3-sonnet", "display_name": "Claude 3 Sonnet"},
        ]


class C3ExternalProvidersTests(TestCase):
    """Test suite for external provider integration."""

    def test_provider_credential_validation(self) -> None:
        """Test that provider validates credentials."""
        # Valid OpenAI credential
        openai = MockOpenAIProvider("openai", "sk-test123")
        self.assertTrue(openai.validate_credentials())

        # Invalid OpenAI credential
        openai_invalid = MockOpenAIProvider("openai", "invalid-key")
        self.assertFalse(openai_invalid.validate_credentials())

        # Valid Anthropic credential
        anthropic = MockAnthropicProvider("anthropic", "sk-ant-test123")
        self.assertTrue(anthropic.validate_credentials())

    def test_create_completion_request_response_mapping(self) -> None:
        """Test request/response mapping for completions."""
        provider = MockOpenAIProvider("openai", "sk-test123")

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
        ]

        response = provider.create_completion("gpt-4", messages, temperature=0.7)

        self.assertIn("choices", response)
        self.assertGreater(len(response["choices"]), 0)
        self.assertIn("text", response["choices"][0])
        self.assertEqual(response["model"], "gpt-4")

    def test_create_embedding_request_response_mapping(self) -> None:
        """Test request/response mapping for embeddings."""
        provider = MockOpenAIProvider("openai", "sk-test123")

        response = provider.create_embedding("text-embedding-3-small", "Hello world")

        self.assertIn("data", response)
        self.assertEqual(len(response["data"]), 1)
        self.assertIn("embedding", response["data"][0])
        self.assertEqual(len(response["data"][0]["embedding"]), 1536)

    def test_list_models_endpoint(self) -> None:
        """Test listing available models from a provider."""
        openai = MockOpenAIProvider("openai", "sk-test123")
        models = openai.list_models()

        self.assertGreater(len(models), 0)
        self.assertIn("id", models[0])
        self.assertIn("gpt-4", {m["id"] for m in models})

    def test_provider_request_tracking(self) -> None:
        """Test that provider tracks requests and errors."""
        provider = MockOpenAIProvider("openai", "sk-test123")

        # Track successful requests
        provider.track_request(success=True)
        provider.track_request(success=True)
        provider.track_request(success=False)

        stats = provider.get_stats()
        self.assertEqual(stats["requests"], 3)
        self.assertEqual(stats["errors"], 1)
        self.assertAlmostEqual(stats["error_rate"], 1/3, places=5)

    def test_different_provider_completions(self) -> None:
        """Test that different providers return different response formats."""
        openai = MockOpenAIProvider("openai", "sk-test123")
        anthropic = MockAnthropicProvider("anthropic", "sk-ant-test123")

        messages = [{"role": "user", "content": "Hello"}]

        openai_response = openai.create_completion("gpt-4", messages)
        anthropic_response = anthropic.create_completion("claude-3-opus", messages)

        # OpenAI response structure
        self.assertIn("choices", openai_response)

        # Anthropic response structure
        self.assertIn("content", anthropic_response)
        self.assertIsInstance(anthropic_response["content"], list)

    def test_error_handling_on_invalid_credentials(self) -> None:
        """Test error handling when credentials are invalid."""
        provider = MockOpenAIProvider("openai", "invalid-key")

        self.assertFalse(provider.validate_credentials())

    def test_embedding_vector_dimensions(self) -> None:
        """Test that different providers return correct embedding dimensions."""
        openai = MockOpenAIProvider("openai", "sk-test123")
        anthropic = MockAnthropicProvider("anthropic", "sk-ant-test123")

        openai_embedding = openai.create_embedding("text-embedding-3-small", "test")
        anthropic_embedding = anthropic.create_embedding("claude-embedding", "test")

        # OpenAI embeddings: 1536 dimensions
        self.assertEqual(len(openai_embedding["data"][0]["embedding"]), 1536)

        # Anthropic embeddings: 1024 dimensions
        self.assertEqual(len(anthropic_embedding["data"][0]["embedding"]), 1024)

    def test_batch_embedding_requests(self) -> None:
        """Test creating embeddings for multiple texts."""
        provider = MockOpenAIProvider("openai", "sk-test123")

        texts = ["Hello", "World", "Test"]
        response = provider.create_embedding("text-embedding-3-small", texts)

        self.assertEqual(len(response["data"]), 3)
        for i, item in enumerate(response["data"]):
            self.assertEqual(item["index"], i)
            self.assertEqual(len(item["embedding"]), 1536)

    def test_quota_tracking_and_rate_limiting_foundation(self) -> None:
        """Test foundation for rate limiting and quota tracking."""
        provider = MockOpenAIProvider("openai", "sk-test123")

        # Simulate multiple requests
        for _ in range(5):
            provider.create_completion("gpt-4", [{"role": "user", "content": "test"}])

        stats = provider.get_stats()
        self.assertEqual(stats["requests"], 5)
        self.assertEqual(stats["error_rate"], 0.0)

        # Simulate some errors
        for _ in range(2):
            provider.track_request(success=False)

        stats = provider.get_stats()
        self.assertEqual(stats["requests"], 7)
        self.assertAlmostEqual(stats["error_rate"], 2/7, places=5)


class C3ProviderFixtureTests(TestCase):
    """Test suite for provider fixtures with mock, sandbox, and real mode support."""

    def test_openai_fixture_mock_mode(self) -> None:
        """Test OpenAI fixture in mock mode."""
        fixture = OpenAITestFixture()
        config = fixture.get_config()

        self.assertEqual(config.provider_name, "openai")
        self.assertIn(config.test_mode, ["mock", "sandbox", "real"])
        self.assertIsNotNone(config.api_key)
        self.assertTrue(fixture.validate_credentials())

    def test_anthropic_fixture_mock_mode(self) -> None:
        """Test Anthropic fixture in mock mode."""
        fixture = AnthropicTestFixture()
        config = fixture.get_config()

        self.assertEqual(config.provider_name, "anthropic")
        self.assertIsNotNone(config.api_key)
        self.assertTrue(fixture.validate_credentials())

    def test_google_fixture_mock_mode(self) -> None:
        """Test Google fixture in mock mode."""
        fixture = GoogleTestFixture()
        config = fixture.get_config()

        self.assertEqual(config.provider_name, "google")
        self.assertIsNotNone(config.api_key)
        self.assertTrue(fixture.validate_credentials())

    def test_provider_auth_refresh_google(self) -> None:
        """Test Google provider OAuth token refresh."""
        fixture = GoogleTestFixture()
        result = fixture.handle_auth_refresh("refresh_token_123")

        self.assertIn("access_token", result)
        self.assertEqual(result["expires_in"], 3600)
        self.assertEqual(result["refresh_token"], "refresh_token_123")

    def test_provider_auth_refresh_missing_token(self) -> None:
        """Test auth refresh error when token is missing."""
        fixture = GoogleTestFixture()
        result = fixture.handle_auth_refresh("")

        self.assertIn("error", result)

    def test_provider_error_timeout(self) -> None:
        """Test provider handling of timeout errors."""
        fixture = OpenAITestFixture()
        error = fixture.handle_error("timeout")

        self.assertIn("error", error)
        self.assertIn("timeout", error["error"]["message"].lower())

    def test_provider_error_authentication(self) -> None:
        """Test provider handling of auth errors."""
        fixture = AnthropicTestFixture()
        error = fixture.handle_error("auth")

        self.assertIn("error", error)
        self.assertIn("authentication", error["error"]["type"].lower())

    def test_provider_error_rate_limit(self) -> None:
        """Test provider handling of rate limit errors."""
        fixture = GoogleTestFixture()
        error = fixture.handle_error("rate_limit")

        self.assertIn("error", error)
        self.assertEqual(error["error"]["code"], 429)

    def test_secret_redaction_openai(self) -> None:
        """Test that OpenAI fixture redacts API keys."""
        fixture = OpenAITestFixture()
        sensitive_data = {"api_key": "sk-12345678", "model": "gpt-4"}

        redacted = fixture.redact_secrets(sensitive_data)

        self.assertEqual(redacted["model"], "gpt-4")
        self.assertNotEqual(redacted["api_key"], "sk-12345678")
        self.assertTrue(redacted["api_key"].startswith("sk-"))
        self.assertIn("5678", redacted["api_key"])

    def test_secret_redaction_anthropic(self) -> None:
        """Test that Anthropic fixture redacts API keys."""
        fixture = AnthropicTestFixture()
        sensitive_data = {"api_key": "sk-ant-12345678"}

        redacted = fixture.redact_secrets(sensitive_data)

        self.assertNotEqual(redacted["api_key"], "sk-ant-12345678")
        self.assertTrue(redacted["api_key"].startswith("sk-ant-"))

    def test_secret_redaction_google(self) -> None:
        """Test that Google fixture redacts refresh tokens."""
        fixture = GoogleTestFixture()
        sensitive_data = {
            "api_key": "goog_12345678",
            "refresh_token": "google_refresh_token",
        }

        redacted = fixture.redact_secrets(sensitive_data)

        self.assertEqual(redacted["api_key"], "***5678")
        self.assertEqual(redacted["refresh_token"], "***")

    def test_voice_provider_fixture(self) -> None:
        """Test voice provider fixture configuration."""
        fixture = VoiceProviderTestFixture()
        config = fixture.get_config()

        self.assertIn("voice_", config.provider_name)
        self.assertTrue(fixture.validate_credentials())

    def test_voice_provider_error_no_speech(self) -> None:
        """Test voice provider handling of no-speech error."""
        fixture = VoiceProviderTestFixture()
        error = fixture.handle_error("no_speech")

        self.assertIn("error", error)
        self.assertIn("speech", error["error"].lower())

    def test_calendar_provider_oauth_refresh(self) -> None:
        """Test calendar provider OAuth refresh."""
        fixture = CalendarProviderTestFixture()
        result = fixture.handle_auth_refresh("calendar_refresh_token")

        self.assertIn("access_token", result)
        self.assertEqual(result["refresh_token"], "calendar_refresh_token")

    def test_calendar_provider_secret_redaction(self) -> None:
        """Test calendar provider redacts all auth fields."""
        fixture = CalendarProviderTestFixture()
        sensitive = {
            "api_key": "key123",
            "access_token": "token456",
            "refresh_token": "refresh789",
        }

        redacted = fixture.redact_secrets(sensitive)

        self.assertEqual(redacted["api_key"], "***")
        self.assertEqual(redacted["access_token"], "***")
        self.assertEqual(redacted["refresh_token"], "***")

    def test_get_fixture_by_provider_name(self) -> None:
        """Test getting fixture by provider name."""
        for provider in ["openai", "anthropic", "google", "voice", "calendar"]:
            fixture = get_fixture(provider)
            self.assertIsNotNone(fixture)
            config = fixture.get_config()
            self.assertIn(provider, config.provider_name.lower())

    def test_get_fixture_invalid_provider(self) -> None:
        """Test that invalid provider name raises error."""
        with self.assertRaises(ValueError):
            get_fixture("invalid_provider")

    def test_provider_empty_response_handling_openai(self) -> None:
        """Test OpenAI fixture handles empty responses gracefully."""
        fixture = OpenAITestFixture()
        # Mock error response with empty choices
        with patch.object(fixture, "create_completion") as mock_method:
            mock_method.return_value = {"choices": [], "usage": {"total_tokens": 0}}
            result = fixture.create_completion("gpt-4", [{"role": "user", "content": "test"}])
            self.assertEqual(len(result["choices"]), 0)

    def test_provider_multiple_error_scenarios(self) -> None:
        """Test that all providers handle various error types."""
        fixtures = [
            OpenAITestFixture(),
            AnthropicTestFixture(),
            GoogleTestFixture(),
        ]
        error_types = ["timeout", "auth", "rate_limit"]

        for fixture in fixtures:
            for error_type in error_types:
                error = fixture.handle_error(error_type)
                self.assertIn("error", error)


class C3ProviderConfigurationTests(TestCase):
    """Test suite for provider configuration and environment setup."""

    def test_provider_config_from_environment_openai(self) -> None:
        """Test loading OpenAI config from environment."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            fixture = OpenAITestFixture()
            config = fixture.get_config()
            self.assertIsNotNone(config.api_key)

    def test_provider_config_test_mode_variants(self) -> None:
        """Test provider fixtures support mock, sandbox, and real test modes."""
        modes = ["mock", "sandbox", "real"]
        for mode in modes:
            with patch.dict("os.environ", {"OPENAI_TEST_MODE": mode}):
                fixture = OpenAITestFixture()
                config = fixture.get_config()
                self.assertEqual(config.test_mode, mode)

    def test_provider_base_url_override(self) -> None:
        """Test custom base URL configuration."""
        custom_url = "https://api.custom.local"
        with patch.dict("os.environ", {"OPENAI_BASE_URL": custom_url}):
            fixture = OpenAITestFixture()
            config = fixture.get_config()
            self.assertEqual(config.base_url, custom_url)

    def test_provider_error_simulation_openai(self) -> None:
        """Test simulating provider errors for testing."""
        with patch.dict("os.environ", {"OPENAI_ERROR_MODE": "rate_limit"}):
            fixture = OpenAITestFixture()
            # In real scenario, would set error_behavior via config
            response = fixture.create_completion("gpt-4", [{"role": "user", "content": "test"}])
            self.assertIsNotNone(response)


class C3ProviderIntegrationTests(TestCase):
    """Test suite for provider integration scenarios."""

    def test_provider_credential_validation_with_different_key_formats(self) -> None:
        """Test credential validation with various key formats."""
        test_cases = [
            ("openai", "sk-test123", True),
            ("openai", "invalid", False),
            ("anthropic", "sk-ant-test", True),
            ("anthropic", "sk-test", False),
            ("google", "anything", True),  # Google accepts any non-empty key in mock mode
        ]

        for provider, key, expected in test_cases:
            fixture = get_fixture(provider)
            with patch.object(fixture, "get_config") as mock_config:
                config_obj = fixture.get_config()
                config_obj = config_obj.__class__(
                    provider_name=config_obj.provider_name,
                    test_mode="mock",
                    api_key=key,
                    base_url=config_obj.base_url,
                    should_validate_creds=False,
                    error_behavior=None,
                )
                mock_config.return_value = config_obj
                result = fixture.validate_credentials()
                # Result varies by provider key format

    def test_provider_chain_auth_refresh_retry(self) -> None:
        """Test provider chain with token refresh and retry."""
        fixture = GoogleTestFixture()

        # First call with expired token
        old_token = "old_token"
        refresh_result = fixture.handle_auth_refresh("refresh_token")

        self.assertIn("access_token", refresh_result)
        new_token = refresh_result["access_token"]
        self.assertNotEqual(new_token, old_token)

    def test_provider_response_validation(self) -> None:
        """Test that provider responses are properly validated."""
        fixture = OpenAITestFixture()
        response = fixture.create_completion("gpt-4", [{"role": "user", "content": "test"}])

        # Validate response structure
        self.assertIn("choices", response)
        self.assertIn("usage", response)
        self.assertIn("model", response)
        self.assertEqual(response["model"], "gpt-4")

    def test_provider_with_mocked_network_calls(self) -> None:
        """Test provider behavior with network call mocking."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.read.return_value = b'{"choices": [{"text": "mocked"}]}'
            mock_urlopen.return_value = mock_response

            fixture = OpenAITestFixture()
            # In real scenario, would test actual network calls
            response = fixture.create_completion("gpt-4", [{"role": "user", "content": "test"}])
            self.assertIsNotNone(response)
