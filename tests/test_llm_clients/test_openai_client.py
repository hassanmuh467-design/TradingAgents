"""Tests for tradingagents.llm_clients.openai_client."""

import pytest
from unittest.mock import patch, MagicMock

from tradingagents.llm_clients.openai_client import OpenAIClient, UnifiedChatOpenAI


class TestOpenAIClient:
    """Tests for OpenAIClient."""

    def test_constructor_stores_model(self):
        client = OpenAIClient("gpt-5-mini")
        assert client.model == "gpt-5-mini"

    def test_constructor_stores_base_url(self):
        client = OpenAIClient("gpt-5-mini", base_url="http://localhost:8080/v1")
        assert client.base_url == "http://localhost:8080/v1"

    def test_constructor_default_provider_is_openai(self):
        client = OpenAIClient("gpt-5-mini")
        assert client.provider == "openai"

    def test_constructor_stores_provider(self):
        client = OpenAIClient("grok-4-fast-reasoning", provider="xai")
        assert client.provider == "xai"

    def test_constructor_provider_lowercased(self):
        client = OpenAIClient("model", provider="XAI")
        assert client.provider == "xai"

    def test_validate_model_delegates_to_validators(self):
        with patch(
            "tradingagents.llm_clients.openai_client.validate_model"
        ) as mock_validate:
            mock_validate.return_value = True
            client = OpenAIClient("gpt-5-mini", provider="openai")
            result = client.validate_model()
            mock_validate.assert_called_once_with("openai", "gpt-5-mini")
            assert result is True

    def test_validate_model_with_xai_provider(self):
        with patch(
            "tradingagents.llm_clients.openai_client.validate_model"
        ) as mock_validate:
            mock_validate.return_value = True
            client = OpenAIClient("grok-4-fast-reasoning", provider="xai")
            client.validate_model()
            mock_validate.assert_called_once_with("xai", "grok-4-fast-reasoning")

    @patch("tradingagents.llm_clients.openai_client.UnifiedChatOpenAI")
    def test_get_llm_returns_unified_chat_openai(self, MockUnified):
        mock_instance = MagicMock()
        MockUnified.return_value = mock_instance
        client = OpenAIClient("gpt-5-mini")
        result = client.get_llm()
        assert result is mock_instance
        MockUnified.assert_called_once()
        call_kwargs = MockUnified.call_args[1]
        assert call_kwargs["model"] == "gpt-5-mini"

    @patch("tradingagents.llm_clients.openai_client.UnifiedChatOpenAI")
    def test_xai_provider_sets_correct_base_url(self, MockUnified):
        MockUnified.return_value = MagicMock()
        client = OpenAIClient("grok-4-fast-reasoning", provider="xai")
        client.get_llm()
        call_kwargs = MockUnified.call_args[1]
        assert call_kwargs["base_url"] == "https://api.x.ai/v1"

    @patch("tradingagents.llm_clients.openai_client.UnifiedChatOpenAI")
    def test_openrouter_provider_sets_correct_base_url(self, MockUnified):
        MockUnified.return_value = MagicMock()
        client = OpenAIClient("meta/llama-3", provider="openrouter")
        client.get_llm()
        call_kwargs = MockUnified.call_args[1]
        assert call_kwargs["base_url"] == "https://openrouter.ai/api/v1"

    @patch("tradingagents.llm_clients.openai_client.UnifiedChatOpenAI")
    def test_ollama_provider_sets_correct_base_url_and_api_key(self, MockUnified):
        MockUnified.return_value = MagicMock()
        client = OpenAIClient("llama3", provider="ollama")
        client.get_llm()
        call_kwargs = MockUnified.call_args[1]
        assert call_kwargs["base_url"] == "http://localhost:11434/v1"
        assert call_kwargs["api_key"] == "ollama"

    @patch("tradingagents.llm_clients.openai_client.UnifiedChatOpenAI")
    def test_custom_base_url_used_for_openai_provider(self, MockUnified):
        MockUnified.return_value = MagicMock()
        client = OpenAIClient(
            "gpt-5-mini", base_url="http://custom:9090/v1", provider="openai"
        )
        client.get_llm()
        call_kwargs = MockUnified.call_args[1]
        assert call_kwargs["base_url"] == "http://custom:9090/v1"

    @patch("tradingagents.llm_clients.openai_client.UnifiedChatOpenAI")
    def test_kwargs_forwarded_to_llm(self, MockUnified):
        MockUnified.return_value = MagicMock()
        client = OpenAIClient("gpt-5-mini", timeout=30, max_retries=5)
        client.get_llm()
        call_kwargs = MockUnified.call_args[1]
        assert call_kwargs["timeout"] == 30
        assert call_kwargs["max_retries"] == 5


class TestUnifiedChatOpenAI:
    """Tests for UnifiedChatOpenAI temperature stripping."""

    @patch("tradingagents.llm_clients.openai_client.ChatOpenAI.__init__", return_value=None)
    def test_strips_temperature_for_gpt5_models(self, mock_init):
        UnifiedChatOpenAI(model="gpt-5-mini", temperature=0.7, top_p=0.9)
        call_kwargs = mock_init.call_args[1]
        assert "temperature" not in call_kwargs
        assert "top_p" not in call_kwargs
        assert call_kwargs["model"] == "gpt-5-mini"

    @patch("tradingagents.llm_clients.openai_client.ChatOpenAI.__init__", return_value=None)
    def test_keeps_temperature_for_non_gpt5_models(self, mock_init):
        UnifiedChatOpenAI(model="gpt-4.1", temperature=0.7, top_p=0.9)
        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["top_p"] == 0.9

    @patch("tradingagents.llm_clients.openai_client.ChatOpenAI.__init__", return_value=None)
    def test_strips_temperature_for_gpt5_variant(self, mock_init):
        UnifiedChatOpenAI(model="gpt-5.2", temperature=0.5)
        call_kwargs = mock_init.call_args[1]
        assert "temperature" not in call_kwargs

    @patch("tradingagents.llm_clients.openai_client.ChatOpenAI.__init__", return_value=None)
    def test_no_error_when_temperature_not_provided_for_gpt5(self, mock_init):
        UnifiedChatOpenAI(model="gpt-5")
        call_kwargs = mock_init.call_args[1]
        assert "temperature" not in call_kwargs
