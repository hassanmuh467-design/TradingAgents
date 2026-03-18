"""Tests for tradingagents.agents.utils.agent_states state TypedDicts."""

import pytest

from tradingagents.agents.utils.agent_states import (
    InvestDebateState,
    RiskDebateState,
    AgentState,
)


class TestInvestDebateState:
    """Tests for InvestDebateState TypedDict."""

    def test_can_create_with_all_fields(self):
        state = InvestDebateState(
            bull_history="Bull argues strong growth.",
            bear_history="Bear argues overvaluation.",
            history="Round 1 debate.",
            current_response="Bull: Strong fundamentals.",
            judge_decision="BUY",
            count=2,
        )
        assert state["bull_history"] == "Bull argues strong growth."
        assert state["bear_history"] == "Bear argues overvaluation."
        assert state["history"] == "Round 1 debate."
        assert state["current_response"] == "Bull: Strong fundamentals."
        assert state["judge_decision"] == "BUY"
        assert state["count"] == 2

    def test_is_dict_compatible(self):
        state = InvestDebateState(
            bull_history="",
            bear_history="",
            history="",
            current_response="",
            judge_decision="",
            count=0,
        )
        assert isinstance(state, dict)

    def test_can_access_keys(self):
        state = InvestDebateState(
            bull_history="test",
            bear_history="test",
            history="test",
            current_response="test",
            judge_decision="test",
            count=1,
        )
        expected_keys = {
            "bull_history",
            "bear_history",
            "history",
            "current_response",
            "judge_decision",
            "count",
        }
        assert set(state.keys()) == expected_keys

    def test_can_be_created_from_dict(self):
        """TypedDicts can be created from plain dicts."""
        raw = {
            "bull_history": "test",
            "bear_history": "test",
            "history": "test",
            "current_response": "test",
            "judge_decision": "test",
            "count": 0,
        }
        state = InvestDebateState(raw)
        assert state["count"] == 0


class TestRiskDebateState:
    """Tests for RiskDebateState TypedDict."""

    def test_can_create_with_all_fields(self):
        state = RiskDebateState(
            aggressive_history="Go all in.",
            conservative_history="Limit exposure.",
            neutral_history="Balanced allocation.",
            history="Risk discussion.",
            latest_speaker="Aggressive Analyst",
            current_aggressive_response="Full position.",
            current_conservative_response="Half position.",
            current_neutral_response="Moderate position.",
            judge_decision="Moderate risk.",
            count=3,
        )
        assert state["aggressive_history"] == "Go all in."
        assert state["conservative_history"] == "Limit exposure."
        assert state["neutral_history"] == "Balanced allocation."
        assert state["latest_speaker"] == "Aggressive Analyst"
        assert state["count"] == 3

    def test_is_dict_compatible(self):
        state = RiskDebateState(
            aggressive_history="",
            conservative_history="",
            neutral_history="",
            history="",
            latest_speaker="",
            current_aggressive_response="",
            current_conservative_response="",
            current_neutral_response="",
            judge_decision="",
            count=0,
        )
        assert isinstance(state, dict)

    def test_has_all_expected_keys(self):
        state = RiskDebateState(
            aggressive_history="",
            conservative_history="",
            neutral_history="",
            history="",
            latest_speaker="",
            current_aggressive_response="",
            current_conservative_response="",
            current_neutral_response="",
            judge_decision="",
            count=0,
        )
        expected_keys = {
            "aggressive_history",
            "conservative_history",
            "neutral_history",
            "history",
            "latest_speaker",
            "current_aggressive_response",
            "current_conservative_response",
            "current_neutral_response",
            "judge_decision",
            "count",
        }
        assert set(state.keys()) == expected_keys


class TestAgentState:
    """Tests for AgentState (inherits from MessagesState)."""

    def test_agent_state_inherits_from_messages_state(self):
        """AgentState should be a subclass of MessagesState."""
        from langgraph.graph import MessagesState

        assert issubclass(AgentState, MessagesState)

    def test_agent_state_has_expected_annotations(self):
        """AgentState should declare the expected fields via annotations."""
        annotations = AgentState.__annotations__
        expected_fields = [
            "company_of_interest",
            "trade_date",
            "sender",
            "market_report",
            "sentiment_report",
            "news_report",
            "fundamentals_report",
            "investment_debate_state",
            "investment_plan",
            "trader_investment_plan",
            "risk_debate_state",
            "final_trade_decision",
        ]
        for field in expected_fields:
            assert field in annotations, f"Missing annotation: {field}"
