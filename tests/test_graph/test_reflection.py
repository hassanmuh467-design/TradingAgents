"""Tests for tradingagents.graph.reflection.Reflector."""

import pytest
from unittest.mock import MagicMock, patch, call

from tradingagents.graph.reflection import Reflector


@pytest.fixture
def mock_llm():
    """Create a mock LLM for reflection tests."""
    llm = MagicMock()
    response = MagicMock()
    response.content = "Reflection: The decision was correct based on strong fundamentals."
    llm.invoke.return_value = response
    return llm


@pytest.fixture
def reflector(mock_llm):
    """Create a Reflector with a mock LLM."""
    return Reflector(mock_llm)


@pytest.fixture
def sample_state():
    """Create a sample state for reflection tests."""
    return {
        "market_report": "AAPL shows bullish technical patterns.",
        "sentiment_report": "Social media sentiment is positive.",
        "news_report": "Apple announced record earnings.",
        "fundamentals_report": "P/E ratio is 28, strong cash flow.",
        "investment_debate_state": {
            "bull_history": "Bull argued strong growth prospects.",
            "bear_history": "Bear argued overvaluation concerns.",
            "judge_decision": "BUY based on growth outlook.",
        },
        "risk_debate_state": {
            "judge_decision": "Moderate risk with stop-loss at 5%.",
        },
        "trader_investment_plan": "Buy 100 shares at market price.",
    }


class TestExtractCurrentSituation:
    def test_combines_all_reports(self, reflector, sample_state):
        result = reflector._extract_current_situation(sample_state)
        assert "AAPL shows bullish technical patterns." in result
        assert "Social media sentiment is positive." in result
        assert "Apple announced record earnings." in result
        assert "P/E ratio is 28, strong cash flow." in result

    def test_reports_separated_by_newlines(self, reflector, sample_state):
        result = reflector._extract_current_situation(sample_state)
        # Each report should be separated by double newlines
        assert "\n\n" in result

    def test_handles_empty_reports(self, reflector):
        state = {
            "market_report": "",
            "sentiment_report": "",
            "news_report": "",
            "fundamentals_report": "",
        }
        result = reflector._extract_current_situation(state)
        # Should still return a string (with separators)
        assert isinstance(result, str)


class TestReflectOnComponent:
    def test_calls_llm_with_system_and_human_messages(self, reflector, mock_llm):
        reflector._reflect_on_component("TEST", "some report", "some situation", 0.05)
        mock_llm.invoke.assert_called_once()
        messages = mock_llm.invoke.call_args[0][0]
        assert len(messages) == 2
        assert messages[0][0] == "system"
        assert messages[1][0] == "human"

    def test_human_message_contains_returns(self, reflector, mock_llm):
        reflector._reflect_on_component("BULL", "report text", "situation text", 0.10)
        messages = mock_llm.invoke.call_args[0][0]
        human_msg = messages[1][1]
        assert "0.1" in human_msg

    def test_human_message_contains_report(self, reflector, mock_llm):
        reflector._reflect_on_component("BEAR", "bearish analysis", "market data", -0.05)
        messages = mock_llm.invoke.call_args[0][0]
        human_msg = messages[1][1]
        assert "bearish analysis" in human_msg

    def test_human_message_contains_situation(self, reflector, mock_llm):
        reflector._reflect_on_component("TRADER", "trade plan", "current market", 0.02)
        messages = mock_llm.invoke.call_args[0][0]
        human_msg = messages[1][1]
        assert "current market" in human_msg

    def test_returns_llm_content(self, reflector):
        result = reflector._reflect_on_component("TEST", "report", "situation", 0.01)
        assert result == "Reflection: The decision was correct based on strong fundamentals."


class TestReflectBullResearcher:
    def test_updates_bull_memory(self, reflector, sample_state, mock_llm):
        mock_memory = MagicMock()
        reflector.reflect_bull_researcher(sample_state, 0.05, mock_memory)
        mock_memory.add_situations.assert_called_once()
        # Should pass a list of (situation, result) tuples
        args = mock_memory.add_situations.call_args[0][0]
        assert len(args) == 1
        situation, recommendation = args[0]
        assert isinstance(situation, str)
        assert isinstance(recommendation, str)

    def test_calls_llm(self, reflector, sample_state, mock_llm):
        mock_memory = MagicMock()
        reflector.reflect_bull_researcher(sample_state, 0.05, mock_memory)
        mock_llm.invoke.assert_called_once()


class TestReflectBearResearcher:
    def test_updates_bear_memory(self, reflector, sample_state, mock_llm):
        mock_memory = MagicMock()
        reflector.reflect_bear_researcher(sample_state, -0.03, mock_memory)
        mock_memory.add_situations.assert_called_once()

    def test_calls_llm(self, reflector, sample_state, mock_llm):
        mock_memory = MagicMock()
        reflector.reflect_bear_researcher(sample_state, -0.03, mock_memory)
        mock_llm.invoke.assert_called_once()


class TestReflectTrader:
    def test_updates_trader_memory(self, reflector, sample_state, mock_llm):
        mock_memory = MagicMock()
        reflector.reflect_trader(sample_state, 0.08, mock_memory)
        mock_memory.add_situations.assert_called_once()

    def test_uses_trader_investment_plan(self, reflector, sample_state, mock_llm):
        mock_memory = MagicMock()
        reflector.reflect_trader(sample_state, 0.08, mock_memory)
        messages = mock_llm.invoke.call_args[0][0]
        human_msg = messages[1][1]
        assert "Buy 100 shares at market price." in human_msg


class TestReflectInvestJudge:
    def test_updates_invest_judge_memory(self, reflector, sample_state, mock_llm):
        mock_memory = MagicMock()
        reflector.reflect_invest_judge(sample_state, 0.05, mock_memory)
        mock_memory.add_situations.assert_called_once()

    def test_uses_judge_decision(self, reflector, sample_state, mock_llm):
        mock_memory = MagicMock()
        reflector.reflect_invest_judge(sample_state, 0.05, mock_memory)
        messages = mock_llm.invoke.call_args[0][0]
        human_msg = messages[1][1]
        assert "BUY based on growth outlook." in human_msg


class TestReflectRiskManager:
    def test_updates_risk_manager_memory(self, reflector, sample_state, mock_llm):
        mock_memory = MagicMock()
        reflector.reflect_risk_manager(sample_state, -0.02, mock_memory)
        mock_memory.add_situations.assert_called_once()

    def test_uses_risk_judge_decision(self, reflector, sample_state, mock_llm):
        mock_memory = MagicMock()
        reflector.reflect_risk_manager(sample_state, -0.02, mock_memory)
        messages = mock_llm.invoke.call_args[0][0]
        human_msg = messages[1][1]
        assert "Moderate risk with stop-loss at 5%." in human_msg


class TestReflectorInit:
    def test_stores_llm(self, mock_llm):
        reflector = Reflector(mock_llm)
        assert reflector.quick_thinking_llm is mock_llm

    def test_has_reflection_system_prompt(self, mock_llm):
        reflector = Reflector(mock_llm)
        assert isinstance(reflector.reflection_system_prompt, str)
        assert len(reflector.reflection_system_prompt) > 0
        assert "financial" in reflector.reflection_system_prompt.lower()
