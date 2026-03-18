"""Tests for tradingagents.llm_clients.google_client."""

import pytest
from unittest.mock import patch, MagicMock

from tradingagents.llm_clients.google_client import GoogleClient, NormalizedChatGoogleGenerativeAI


class TestGoogleClient:
    """Tests for GoogleClient."""

    def test_constructor_stores_model(self):
        client = GoogleClient("gemini-2.5-flash")
        assert client.model == "gemini-2.5-flash"

    def test_constructor_stores_base_url(self):
        client = GoogleClient("gemini-2.5-flash", base_url="http://proxy:8080")
        assert client.base_url == "http://proxy:8080"

    def test_constructor_base_url_none_by_default(self):
        client = GoogleClient("gemini-2.5-flash")
        assert client.base_url is None

    def test_constructor_stores_kwargs(self):
        client = GoogleClient(
            "gemini-2.5-flash", timeout=30, thinking_level="high"
        )
        assert client.kwargs["timeout"] == 30
        assert client.kwargs["thinking_level"] == "high"

    def test_constructor_warns_about_base_url(self, caplog):
        """GoogleClient logs a warning when base_url is provided."""
        import logging

        with caplog.at_level(logging.WARNING):
            GoogleClient("gemini-2.5-flash", base_url="http://proxy:8080")
        assert "does not support 'base_url'" in caplog.text

    def test_validate_model_delegates_to_validators(self):
        with patch(
            "tradingagents.llm_clients.google_client.validate_model"
        ) as mock_validate:
            mock_validate.return_value = True
            client = GoogleClient("gemini-2.5-flash")
            result = client.validate_model()
            mock_validate.assert_called_once_with("google", "gemini-2.5-flash")
            assert result is True

    def test_validate_model_returns_false_for_invalid(self):
        with patch(
            "tradingagents.llm_clients.google_client.validate_model"
        ) as mock_validate:
            mock_validate.return_value = False
            client = GoogleClient("gemini-1.0-pro")
            result = client.validate_model()
            assert result is False

    @patch("tradingagents.llm_clients.google_client.NormalizedChatGoogleGenerativeAI")
    def test_get_llm_returns_normalized_instance(self, MockNormalized):
        mock_instance = MagicMock()
        MockNormalized.return_value = mock_instance
        client = GoogleClient("gemini-2.5-flash")
        result = client.get_llm()
        assert result is mock_instance
        MockNormalized.assert_called_once()
        call_kwargs = MockNormalized.call_args[1]
        assert call_kwargs["model"] == "gemini-2.5-flash"

    @patch("tradingagents.llm_clients.google_client.NormalizedChatGoogleGenerativeAI")
    def test_get_llm_forwards_timeout(self, MockNormalized):
        MockNormalized.return_value = MagicMock()
        client = GoogleClient("gemini-2.5-flash", timeout=60)
        client.get_llm()
        call_kwargs = MockNormalized.call_args[1]
        assert call_kwargs["timeout"] == 60

    @patch("tradingagents.llm_clients.google_client.NormalizedChatGoogleGenerativeAI")
    def test_get_llm_maps_api_key_to_google_api_key(self, MockNormalized):
        MockNormalized.return_value = MagicMock()
        client = GoogleClient("gemini-2.5-flash", api_key="test-key")
        client.get_llm()
        call_kwargs = MockNormalized.call_args[1]
        assert call_kwargs["google_api_key"] == "test-key"

    @patch("tradingagents.llm_clients.google_client.NormalizedChatGoogleGenerativeAI")
    def test_get_llm_prefers_google_api_key_over_api_key(self, MockNormalized):
        MockNormalized.return_value = MagicMock()
        client = GoogleClient(
            "gemini-2.5-flash",
            api_key="generic-key",
            google_api_key="specific-key",
        )
        client.get_llm()
        call_kwargs = MockNormalized.call_args[1]
        assert call_kwargs["google_api_key"] == "specific-key"

    @patch("tradingagents.llm_clients.google_client.NormalizedChatGoogleGenerativeAI")
    def test_get_llm_thinking_level_for_gemini3_model(self, MockNormalized):
        """Gemini 3 models should get thinking_level passed through."""
        MockNormalized.return_value = MagicMock()
        client = GoogleClient("gemini-3-flash-preview", thinking_level="high")
        client.get_llm()
        call_kwargs = MockNormalized.call_args[1]
        assert call_kwargs["thinking_level"] == "high"
        assert "thinking_budget" not in call_kwargs

    @patch("tradingagents.llm_clients.google_client.NormalizedChatGoogleGenerativeAI")
    def test_get_llm_thinking_level_minimal_remapped_for_gemini3_pro(self, MockNormalized):
        """Gemini 3 Pro does not support 'minimal'; should be remapped to 'low'."""
        MockNormalized.return_value = MagicMock()
        client = GoogleClient("gemini-3.1-pro-preview", thinking_level="minimal")
        client.get_llm()
        call_kwargs = MockNormalized.call_args[1]
        assert call_kwargs["thinking_level"] == "low"

    @patch("tradingagents.llm_clients.google_client.NormalizedChatGoogleGenerativeAI")
    def test_get_llm_thinking_level_for_gemini25_model_high(self, MockNormalized):
        """Gemini 2.5 models should map thinking_level='high' to thinking_budget=-1."""
        MockNormalized.return_value = MagicMock()
        client = GoogleClient("gemini-2.5-pro", thinking_level="high")
        client.get_llm()
        call_kwargs = MockNormalized.call_args[1]
        assert call_kwargs["thinking_budget"] == -1
        assert "thinking_level" not in call_kwargs

    @patch("tradingagents.llm_clients.google_client.NormalizedChatGoogleGenerativeAI")
    def test_get_llm_thinking_level_for_gemini25_model_low(self, MockNormalized):
        """Gemini 2.5 models should map non-'high' thinking_level to thinking_budget=0."""
        MockNormalized.return_value = MagicMock()
        client = GoogleClient("gemini-2.5-flash", thinking_level="low")
        client.get_llm()
        call_kwargs = MockNormalized.call_args[1]
        assert call_kwargs["thinking_budget"] == 0
        assert "thinking_level" not in call_kwargs

    @patch("tradingagents.llm_clients.google_client.NormalizedChatGoogleGenerativeAI")
    def test_get_llm_no_thinking_params_when_not_set(self, MockNormalized):
        """When thinking_level is not set, neither thinking_level nor thinking_budget should appear."""
        MockNormalized.return_value = MagicMock()
        client = GoogleClient("gemini-2.5-flash")
        client.get_llm()
        call_kwargs = MockNormalized.call_args[1]
        assert "thinking_level" not in call_kwargs
        assert "thinking_budget" not in call_kwargs

    @patch("tradingagents.llm_clients.google_client.NormalizedChatGoogleGenerativeAI")
    def test_get_llm_forwards_callbacks(self, MockNormalized):
        MockNormalized.return_value = MagicMock()
        mock_callback = MagicMock()
        client = GoogleClient("gemini-2.5-flash", callbacks=[mock_callback])
        client.get_llm()
        call_kwargs = MockNormalized.call_args[1]
        assert call_kwargs["callbacks"] == [mock_callback]

    @patch("tradingagents.llm_clients.google_client.NormalizedChatGoogleGenerativeAI")
    def test_get_llm_does_not_forward_unknown_kwargs(self, MockNormalized):
        MockNormalized.return_value = MagicMock()
        client = GoogleClient("gemini-2.5-flash", unknown_param="value")
        client.get_llm()
        call_kwargs = MockNormalized.call_args[1]
        assert "unknown_param" not in call_kwargs


class TestNormalizedChatGoogleGenerativeAI:
    """Tests for NormalizedChatGoogleGenerativeAI content normalization."""

    def test_normalize_content_converts_list_to_string(self):
        """List content like [{'type': 'text', 'text': '...'}] should become a string."""
        normalizer = NormalizedChatGoogleGenerativeAI.__new__(
            NormalizedChatGoogleGenerativeAI
        )
        mock_response = MagicMock()
        mock_response.content = [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": "World"},
        ]
        result = normalizer._normalize_content(mock_response)
        assert result.content == "Hello\nWorld"

    def test_normalize_content_handles_string_items_in_list(self):
        normalizer = NormalizedChatGoogleGenerativeAI.__new__(
            NormalizedChatGoogleGenerativeAI
        )
        mock_response = MagicMock()
        mock_response.content = ["Hello", "World"]
        result = normalizer._normalize_content(mock_response)
        assert result.content == "Hello\nWorld"

    def test_normalize_content_passes_through_string_content(self):
        normalizer = NormalizedChatGoogleGenerativeAI.__new__(
            NormalizedChatGoogleGenerativeAI
        )
        mock_response = MagicMock()
        mock_response.content = "Already a string"
        result = normalizer._normalize_content(mock_response)
        assert result.content == "Already a string"

    def test_normalize_content_handles_empty_list(self):
        normalizer = NormalizedChatGoogleGenerativeAI.__new__(
            NormalizedChatGoogleGenerativeAI
        )
        mock_response = MagicMock()
        mock_response.content = []
        result = normalizer._normalize_content(mock_response)
        assert result.content == ""

    def test_normalize_content_skips_non_text_items(self):
        normalizer = NormalizedChatGoogleGenerativeAI.__new__(
            NormalizedChatGoogleGenerativeAI
        )
        mock_response = MagicMock()
        mock_response.content = [
            {"type": "text", "text": "Hello"},
            {"type": "image", "url": "http://example.com/img.png"},
            {"type": "text", "text": "World"},
        ]
        result = normalizer._normalize_content(mock_response)
        assert result.content == "Hello\nWorld"

    @patch(
        "tradingagents.llm_clients.google_client.ChatGoogleGenerativeAI.invoke"
    )
    @patch(
        "tradingagents.llm_clients.google_client.ChatGoogleGenerativeAI.__init__",
        return_value=None,
    )
    def test_invoke_normalizes_list_content(self, mock_init, mock_super_invoke):
        """invoke() should normalize list content from the parent class."""
        mock_response = MagicMock()
        mock_response.content = [{"type": "text", "text": "BUY"}]
        mock_super_invoke.return_value = mock_response

        instance = NormalizedChatGoogleGenerativeAI(model="gemini-2.5-flash")
        result = instance.invoke("test input")
        assert result.content == "BUY"
