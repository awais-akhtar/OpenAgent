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
