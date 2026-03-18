"""Tests for tradingagents.graph.propagation.Propagator."""

import pytest
from unittest.mock import MagicMock

from tradingagents.graph.propagation import Propagator


class TestPropagatorCreateInitialState:
    """Tests for Propagator.create_initial_state."""

    def test_returns_dict(self):
        propagator = Propagator()
        state = propagator.create_initial_state("AAPL", "2025-01-15")
        assert isinstance(state, dict)

    def test_messages_contains_company_name(self):
        propagator = Propagator()
        state = propagator.create_initial_state("AAPL", "2025-01-15")
        assert state["messages"] == [("human", "AAPL")]

    def test_company_of_interest_set(self):
        propagator = Propagator()
        state = propagator.create_initial_state("TSLA", "2025-03-01")
        assert state["company_of_interest"] == "TSLA"

    def test_trade_date_is_string(self):
        propagator = Propagator()
        state = propagator.create_initial_state("AAPL", "2025-01-15")
        assert state["trade_date"] == "2025-01-15"
        assert isinstance(state["trade_date"], str)

    def test_trade_date_converts_non_string(self):
        """trade_date is wrapped in str() so non-string input is fine."""
        propagator = Propagator()
        from datetime import date

        state = propagator.create_initial_state("AAPL", date(2025, 1, 15))
        assert state["trade_date"] == "2025-01-15"

    def test_investment_debate_state_initialized(self):
        propagator = Propagator()
        state = propagator.create_initial_state("AAPL", "2025-01-15")
        ids = state["investment_debate_state"]
        assert ids["bull_history"] == ""
        assert ids["bear_history"] == ""
        assert ids["history"] == ""
        assert ids["current_response"] == ""
        assert ids["judge_decision"] == ""
        assert ids["count"] == 0

    def test_risk_debate_state_initialized(self):
        propagator = Propagator()
        state = propagator.create_initial_state("AAPL", "2025-01-15")
        rds = state["risk_debate_state"]
        assert rds["aggressive_history"] == ""
        assert rds["conservative_history"] == ""
        assert rds["neutral_history"] == ""
        assert rds["history"] == ""
        assert rds["latest_speaker"] == ""
        assert rds["current_aggressive_response"] == ""
        assert rds["current_conservative_response"] == ""
        assert rds["current_neutral_response"] == ""
        assert rds["judge_decision"] == ""
        assert rds["count"] == 0

    def test_reports_initialized_empty(self):
        propagator = Propagator()
        state = propagator.create_initial_state("AAPL", "2025-01-15")
        assert state["market_report"] == ""
        assert state["fundamentals_report"] == ""
        assert state["sentiment_report"] == ""
        assert state["news_report"] == ""

    def test_all_required_keys_present(self):
        propagator = Propagator()
        state = propagator.create_initial_state("AAPL", "2025-01-15")
        required_keys = [
            "messages",
            "company_of_interest",
            "trade_date",
            "investment_debate_state",
            "risk_debate_state",
            "market_report",
            "fundamentals_report",
            "sentiment_report",
            "news_report",
        ]
        for key in required_keys:
            assert key in state, f"Missing key: {key}"


class TestPropagatorGetGraphArgs:
    """Tests for Propagator.get_graph_args."""

    def test_returns_dict_with_stream_mode_and_config(self):
        propagator = Propagator()
        args = propagator.get_graph_args()
        assert "stream_mode" in args
        assert "config" in args
        assert args["stream_mode"] == "values"

    def test_config_has_recursion_limit(self):
        propagator = Propagator(max_recur_limit=100)
        args = propagator.get_graph_args()
        assert args["config"]["recursion_limit"] == 100

    def test_custom_recursion_limit(self):
        propagator = Propagator(max_recur_limit=200)
        args = propagator.get_graph_args()
        assert args["config"]["recursion_limit"] == 200

    def test_no_callbacks_by_default(self):
        propagator = Propagator()
        args = propagator.get_graph_args()
        assert "callbacks" not in args["config"]

    def test_callbacks_included_when_provided(self):
        propagator = Propagator()
        mock_cb = MagicMock()
        args = propagator.get_graph_args(callbacks=[mock_cb])
        assert args["config"]["callbacks"] == [mock_cb]

    def test_empty_callbacks_not_included(self):
        propagator = Propagator()
        args = propagator.get_graph_args(callbacks=[])
        assert "callbacks" not in args["config"]

    def test_none_callbacks_not_included(self):
        propagator = Propagator()
        args = propagator.get_graph_args(callbacks=None)
        assert "callbacks" not in args["config"]
