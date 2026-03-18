"""Tests for tradingagents.graph.signal_processing.SignalProcessor."""

import pytest
from unittest.mock import MagicMock

from tradingagents.graph.signal_processing import SignalProcessor


@pytest.fixture
def mock_llm():
    """Create a mock LLM for signal processing tests."""
    llm = MagicMock()
    response = MagicMock()
    response.content = "BUY"
    llm.invoke.return_value = response
    return llm


@pytest.fixture
def processor(mock_llm):
    """Create a SignalProcessor with a mock LLM."""
    return SignalProcessor(mock_llm)


class TestSignalProcessor:
    def test_process_signal_returns_llm_content(self, processor):
        result = processor.process_signal("Buy 100 shares of AAPL")
        assert result == "BUY"

    def test_process_signal_calls_llm_invoke(self, processor, mock_llm):
        processor.process_signal("Some signal text")
        mock_llm.invoke.assert_called_once()

    def test_process_signal_passes_correct_system_prompt(self, processor, mock_llm):
        processor.process_signal("Test signal")
        call_args = mock_llm.invoke.call_args[0][0]
        # call_args should be a list of tuples: [("system", ...), ("human", ...)]
        assert len(call_args) == 2
        assert call_args[0][0] == "system"
        assert "extract the investment decision" in call_args[0][1].lower()

    def test_process_signal_passes_full_signal_as_human_message(self, processor, mock_llm):
        processor.process_signal("Detailed analysis: recommend SELL")
        call_args = mock_llm.invoke.call_args[0][0]
        assert call_args[1][0] == "human"
        assert call_args[1][1] == "Detailed analysis: recommend SELL"

    def test_process_signal_returns_sell(self, mock_llm):
        mock_llm.invoke.return_value.content = "SELL"
        processor = SignalProcessor(mock_llm)
        result = processor.process_signal("Bearish outlook, sell everything")
        assert result == "SELL"

    def test_process_signal_returns_hold(self, mock_llm):
        mock_llm.invoke.return_value.content = "HOLD"
        processor = SignalProcessor(mock_llm)
        result = processor.process_signal("Uncertain market conditions")
        assert result == "HOLD"

    def test_stores_llm_reference(self, mock_llm):
        processor = SignalProcessor(mock_llm)
        assert processor.quick_thinking_llm is mock_llm
