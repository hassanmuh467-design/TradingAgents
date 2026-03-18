"""Tests for tradingagents.llm_clients.factory.create_llm_client."""

import pytest
from unittest.mock import patch, MagicMock

from tradingagents.llm_clients.factory import create_llm_client
from tradingagents.llm_clients.openai_client import OpenAIClient
from tradingagents.llm_clients.anthropic_client import AnthropicClient
from tradingagents.llm_clients.google_client import GoogleClient


class TestCreateLLMClient:
    """Tests for the create_llm_client factory function."""

    def test_openai_provider_returns_openai_client(self):
        client = create_llm_client("openai", "gpt-5-mini")
        assert isinstance(client, OpenAIClient)

    def test_anthropic_provider_returns_anthropic_client(self):
        client = create_llm_client("anthropic", "claude-sonnet-4-5")
        assert isinstance(client, AnthropicClient)

    def test_google_provider_returns_google_client(self):
        client = create_llm_client("google", "gemini-2.5-flash")
        assert isinstance(client, GoogleClient)

    def test_xai_provider_returns_openai_client(self):
        client = create_llm_client("xai", "grok-4-fast-reasoning")
        assert isinstance(client, OpenAIClient)
        assert client.provider == "xai"

    def test_ollama_provider_returns_openai_client(self):
        client = create_llm_client("ollama", "llama3")
        assert isinstance(client, OpenAIClient)
        assert client.provider == "ollama"

    def test_openrouter_provider_returns_openai_client(self):
        client = create_llm_client("openrouter", "meta-llama/llama-3-70b")
        assert isinstance(client, OpenAIClient)
        assert client.provider == "openrouter"

    def test_invalid_provider_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            create_llm_client("invalid_provider", "some-model")

    def test_provider_is_case_insensitive(self):
        client = create_llm_client("OpenAI", "gpt-5-mini")
        assert isinstance(client, OpenAIClient)

        client = create_llm_client("ANTHROPIC", "claude-sonnet-4-5")
        assert isinstance(client, AnthropicClient)

        client = create_llm_client("Google", "gemini-2.5-flash")
        assert isinstance(client, GoogleClient)

    def test_kwargs_passed_through_to_openai(self):
        client = create_llm_client(
            "openai", "gpt-5-mini", timeout=30, max_retries=5
        )
        assert isinstance(client, OpenAIClient)
        assert client.kwargs["timeout"] == 30
        assert client.kwargs["max_retries"] == 5

    def test_kwargs_passed_through_to_anthropic(self):
        client = create_llm_client(
            "anthropic", "claude-sonnet-4-5", max_tokens=4096
        )
        assert isinstance(client, AnthropicClient)
        assert client.kwargs["max_tokens"] == 4096

    def test_kwargs_passed_through_to_google(self):
        client = create_llm_client(
            "google", "gemini-2.5-flash", thinking_level="high"
        )
        assert isinstance(client, GoogleClient)
        assert client.kwargs["thinking_level"] == "high"

    def test_base_url_passed_through(self):
        client = create_llm_client(
            "openai", "gpt-5-mini", base_url="http://custom:8080/v1"
        )
        assert client.base_url == "http://custom:8080/v1"

    def test_base_url_none_by_default(self):
        client = create_llm_client("openai", "gpt-5-mini")
        assert client.base_url is None
