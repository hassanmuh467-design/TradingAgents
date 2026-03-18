"""Tests for tradingagents.graph.conditional_logic.ConditionalLogic."""

import pytest
from unittest.mock import MagicMock

from tradingagents.graph.conditional_logic import ConditionalLogic


@pytest.fixture
def logic():
    """Create a ConditionalLogic instance with default settings."""
    return ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1)


@pytest.fixture
def logic_multi_round():
    """Create a ConditionalLogic with multiple debate rounds."""
    return ConditionalLogic(max_debate_rounds=3, max_risk_discuss_rounds=2)


def _make_message(has_tool_calls=False):
    """Create a mock message with or without tool_calls."""
    msg = MagicMock()
    msg.tool_calls = [MagicMock()] if has_tool_calls else []
    return msg


def _make_state_with_messages(has_tool_calls=False):
    """Create a minimal AgentState-like dict for analyst continue checks."""
    return {"messages": [_make_message(has_tool_calls)]}


# --- should_continue_market ---


class TestShouldContinueMarket:
    def test_returns_tools_market_when_tool_calls_present(self, logic):
        state = _make_state_with_messages(has_tool_calls=True)
        assert logic.should_continue_market(state) == "tools_market"

    def test_returns_msg_clear_market_otherwise(self, logic):
        state = _make_state_with_messages(has_tool_calls=False)
        assert logic.should_continue_market(state) == "Msg Clear Market"


# --- should_continue_social ---


class TestShouldContinueSocial:
    def test_returns_tools_social_when_tool_calls_present(self, logic):
        state = _make_state_with_messages(has_tool_calls=True)
        assert logic.should_continue_social(state) == "tools_social"

    def test_returns_msg_clear_social_otherwise(self, logic):
        state = _make_state_with_messages(has_tool_calls=False)
        assert logic.should_continue_social(state) == "Msg Clear Social"


# --- should_continue_news ---


class TestShouldContinueNews:
    def test_returns_tools_news_when_tool_calls_present(self, logic):
        state = _make_state_with_messages(has_tool_calls=True)
        assert logic.should_continue_news(state) == "tools_news"

    def test_returns_msg_clear_news_otherwise(self, logic):
        state = _make_state_with_messages(has_tool_calls=False)
        assert logic.should_continue_news(state) == "Msg Clear News"


# --- should_continue_fundamentals ---


class TestShouldContinueFundamentals:
    def test_returns_tools_fundamentals_when_tool_calls_present(self, logic):
        state = _make_state_with_messages(has_tool_calls=True)
        assert logic.should_continue_fundamentals(state) == "tools_fundamentals"

    def test_returns_msg_clear_fundamentals_otherwise(self, logic):
        state = _make_state_with_messages(has_tool_calls=False)
        assert logic.should_continue_fundamentals(state) == "Msg Clear Fundamentals"


# --- should_continue_debate ---


class TestShouldContinueDebate:
    def test_returns_research_manager_when_count_at_limit(self, logic):
        """count >= 2 * max_debate_rounds => 'Research Manager'."""
        state = {
            "investment_debate_state": {
                "count": 2,  # 2 >= 2 * 1
                "current_response": "Bull: something",
            }
        }
        assert logic.should_continue_debate(state) == "Research Manager"

    def test_returns_research_manager_when_count_exceeds_limit(self, logic):
        state = {
            "investment_debate_state": {
                "count": 5,
                "current_response": "Bull: something",
            }
        }
        assert logic.should_continue_debate(state) == "Research Manager"

    def test_returns_bear_researcher_when_bull_spoke(self, logic):
        """When current_response starts with 'Bull', route to Bear Researcher."""
        state = {
            "investment_debate_state": {
                "count": 0,
                "current_response": "Bull: The company shows strong growth.",
            }
        }
        assert logic.should_continue_debate(state) == "Bear Researcher"

    def test_returns_bull_researcher_when_bear_spoke(self, logic):
        """When current_response starts with 'Bear', route to Bull Researcher."""
        state = {
            "investment_debate_state": {
                "count": 0,
                "current_response": "Bear: The company is overvalued.",
            }
        }
        assert logic.should_continue_debate(state) == "Bull Researcher"

    def test_returns_bull_researcher_for_empty_response(self, logic):
        """Default case: route to Bull Researcher."""
        state = {
            "investment_debate_state": {
                "count": 0,
                "current_response": "",
            }
        }
        assert logic.should_continue_debate(state) == "Bull Researcher"

    def test_multi_round_debate_continues(self, logic_multi_round):
        """With max_debate_rounds=3, count must reach 6 to stop."""
        state = {
            "investment_debate_state": {
                "count": 4,
                "current_response": "Bull: More arguments.",
            }
        }
        assert logic_multi_round.should_continue_debate(state) == "Bear Researcher"

    def test_multi_round_debate_stops_at_limit(self, logic_multi_round):
        state = {
            "investment_debate_state": {
                "count": 6,  # 6 >= 2 * 3
                "current_response": "Bull: Final argument.",
            }
        }
        assert logic_multi_round.should_continue_debate(state) == "Research Manager"


# --- should_continue_risk_analysis ---


class TestShouldContinueRiskAnalysis:
    def test_returns_risk_judge_when_count_at_limit(self, logic):
        """count >= 3 * max_risk_discuss_rounds => 'Risk Judge'."""
        state = {
            "risk_debate_state": {
                "count": 3,  # 3 >= 3 * 1
                "latest_speaker": "Aggressive Analyst",
            }
        }
        assert logic.should_continue_risk_analysis(state) == "Risk Judge"

    def test_returns_risk_judge_when_count_exceeds_limit(self, logic):
        state = {
            "risk_debate_state": {
                "count": 10,
                "latest_speaker": "Neutral Analyst",
            }
        }
        assert logic.should_continue_risk_analysis(state) == "Risk Judge"

    def test_routes_to_conservative_after_aggressive(self, logic):
        """When latest_speaker starts with 'Aggressive', route to Conservative."""
        state = {
            "risk_debate_state": {
                "count": 0,
                "latest_speaker": "Aggressive Analyst",
            }
        }
        assert logic.should_continue_risk_analysis(state) == "Conservative Analyst"

    def test_routes_to_neutral_after_conservative(self, logic):
        """When latest_speaker starts with 'Conservative', route to Neutral."""
        state = {
            "risk_debate_state": {
                "count": 0,
                "latest_speaker": "Conservative Analyst",
            }
        }
        assert logic.should_continue_risk_analysis(state) == "Neutral Analyst"

    def test_routes_to_aggressive_by_default(self, logic):
        """Default case (e.g. 'Neutral' or empty) routes to Aggressive Analyst."""
        state = {
            "risk_debate_state": {
                "count": 0,
                "latest_speaker": "Neutral Analyst",
            }
        }
        assert logic.should_continue_risk_analysis(state) == "Aggressive Analyst"

    def test_routes_to_aggressive_for_empty_speaker(self, logic):
        state = {
            "risk_debate_state": {
                "count": 0,
                "latest_speaker": "",
            }
        }
        assert logic.should_continue_risk_analysis(state) == "Aggressive Analyst"

    def test_multi_round_risk_continues(self, logic_multi_round):
        """With max_risk_discuss_rounds=2, count must reach 6 to stop."""
        state = {
            "risk_debate_state": {
                "count": 4,
                "latest_speaker": "Aggressive Analyst",
            }
        }
        assert logic_multi_round.should_continue_risk_analysis(state) == "Conservative Analyst"

    def test_multi_round_risk_stops_at_limit(self, logic_multi_round):
        state = {
            "risk_debate_state": {
                "count": 6,  # 6 >= 3 * 2
                "latest_speaker": "Aggressive Analyst",
            }
        }
        assert logic_multi_round.should_continue_risk_analysis(state) == "Risk Judge"
