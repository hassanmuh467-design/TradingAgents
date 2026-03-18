"""Tests for tradingagents.llm_clients.anthropic_client."""

import pytest
from unittest.mock import patch, MagicMock

from tradingagents.llm_clients.anthropic_client import AnthropicClient


class TestAnthropicClient:
    """Tests for AnthropicClient."""

    def test_constructor_stores_model(self):
        client = AnthropicClient("claude-sonnet-4-5")
        assert client.model == "claude-sonnet-4-5"

    def test_constructor_stores_base_url(self):
        client = AnthropicClient("claude-sonnet-4-5", base_url="http://proxy:8080")
        assert client.base_url == "http://proxy:8080"

    def test_constructor_base_url_none_by_default(self):
        client = AnthropicClient("claude-sonnet-4-5")
        assert client.base_url is None

    def test_constructor_stores_kwargs(self):
        client = AnthropicClient(
            "claude-sonnet-4-5", timeout=30, max_retries=3, max_tokens=4096
        )
        assert client.kwargs["timeout"] == 30
        assert client.kwargs["max_retries"] == 3
        assert client.kwargs["max_tokens"] == 4096

    def test_constructor_warns_about_base_url(self, caplog):
        """AnthropicClient logs a warning when base_url is provided."""
        import logging

        with caplog.at_level(logging.WARNING):
            AnthropicClient("claude-sonnet-4-5", base_url="http://proxy:8080")
        assert "does not support 'base_url'" in caplog.text

    def test_validate_model_delegates_to_validators(self):
        with patch(
            "tradingagents.llm_clients.anthropic_client.validate_model"
        ) as mock_validate:
            mock_validate.return_value = True
            client = AnthropicClient("claude-sonnet-4-5")
            result = client.validate_model()
            mock_validate.assert_called_once_with("anthropic", "claude-sonnet-4-5")
            assert result is True

    def test_validate_model_returns_false_for_invalid(self):
        with patch(
            "tradingagents.llm_clients.anthropic_client.validate_model"
        ) as mock_validate:
            mock_validate.return_value = False
            client = AnthropicClient("claude-2")
            result = client.validate_model()
            assert result is False

    @patch("tradingagents.llm_clients.anthropic_client.ChatAnthropic")
    def test_get_llm_returns_chat_anthropic_instance(self, MockChatAnthropic):
        mock_instance = MagicMock()
        MockChatAnthropic.return_value = mock_instance
        client = AnthropicClient("claude-sonnet-4-5")
        result = client.get_llm()
        assert result is mock_instance
        MockChatAnthropic.assert_called_once()
        call_kwargs = MockChatAnthropic.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-5"

    @patch("tradingagents.llm_clients.anthropic_client.ChatAnthropic")
    def test_get_llm_forwards_timeout(self, MockChatAnthropic):
        MockChatAnthropic.return_value = MagicMock()
        client = AnthropicClient("claude-sonnet-4-5", timeout=60)
        client.get_llm()
        call_kwargs = MockChatAnthropic.call_args[1]
        assert call_kwargs["timeout"] == 60

    @patch("tradingagents.llm_clients.anthropic_client.ChatAnthropic")
    def test_get_llm_forwards_max_retries(self, MockChatAnthropic):
        MockChatAnthropic.return_value = MagicMock()
        client = AnthropicClient("claude-sonnet-4-5", max_retries=5)
        client.get_llm()
        call_kwargs = MockChatAnthropic.call_args[1]
        assert call_kwargs["max_retries"] == 5

    @patch("tradingagents.llm_clients.anthropic_client.ChatAnthropic")
    def test_get_llm_forwards_api_key(self, MockChatAnthropic):
        MockChatAnthropic.return_value = MagicMock()
        client = AnthropicClient("claude-sonnet-4-5", api_key="sk-test-key")
        client.get_llm()
        call_kwargs = MockChatAnthropic.call_args[1]
        assert call_kwargs["api_key"] == "sk-test-key"

    @patch("tradingagents.llm_clients.anthropic_client.ChatAnthropic")
    def test_get_llm_forwards_max_tokens(self, MockChatAnthropic):
        MockChatAnthropic.return_value = MagicMock()
        client = AnthropicClient("claude-sonnet-4-5", max_tokens=8192)
        client.get_llm()
        call_kwargs = MockChatAnthropic.call_args[1]
        assert call_kwargs["max_tokens"] == 8192

    @patch("tradingagents.llm_clients.anthropic_client.ChatAnthropic")
    def test_get_llm_forwards_callbacks(self, MockChatAnthropic):
        MockChatAnthropic.return_value = MagicMock()
        mock_callback = MagicMock()
        client = AnthropicClient("claude-sonnet-4-5", callbacks=[mock_callback])
        client.get_llm()
        call_kwargs = MockChatAnthropic.call_args[1]
        assert call_kwargs["callbacks"] == [mock_callback]

    @patch("tradingagents.llm_clients.anthropic_client.ChatAnthropic")
    def test_get_llm_does_not_forward_unknown_kwargs(self, MockChatAnthropic):
        MockChatAnthropic.return_value = MagicMock()
        client = AnthropicClient("claude-sonnet-4-5", unknown_param="value")
        client.get_llm()
        call_kwargs = MockChatAnthropic.call_args[1]
        assert "unknown_param" not in call_kwargs

    @patch("tradingagents.llm_clients.anthropic_client.ChatAnthropic")
    def test_get_llm_warns_on_invalid_model(self, MockChatAnthropic, caplog):
        import logging

        MockChatAnthropic.return_value = MagicMock()
        with caplog.at_level(logging.WARNING):
            client = AnthropicClient("claude-2")
            client.get_llm()
        assert "may not be supported" in caplog.text
