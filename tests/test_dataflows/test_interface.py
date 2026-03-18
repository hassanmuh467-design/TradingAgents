"""Tests for tradingagents.dataflows.interface module."""

import pytest
from unittest.mock import patch, MagicMock

from tradingagents.dataflows.interface import (
    get_category_for_method,
    get_vendor,
    route_to_vendor,
    TOOLS_CATEGORIES,
    VENDOR_METHODS,
    VENDOR_LIST,
)
from tradingagents.dataflows.alpha_vantage_common import AlphaVantageRateLimitError


class TestGetCategoryForMethod:
    """Tests for get_category_for_method."""

    def test_get_stock_data_is_core_stock_apis(self):
        assert get_category_for_method("get_stock_data") == "core_stock_apis"

    def test_get_indicators_is_technical_indicators(self):
        assert get_category_for_method("get_indicators") == "technical_indicators"

    def test_get_fundamentals_is_fundamental_data(self):
        assert get_category_for_method("get_fundamentals") == "fundamental_data"

    def test_get_balance_sheet_is_fundamental_data(self):
        assert get_category_for_method("get_balance_sheet") == "fundamental_data"

    def test_get_cashflow_is_fundamental_data(self):
        assert get_category_for_method("get_cashflow") == "fundamental_data"

    def test_get_income_statement_is_fundamental_data(self):
        assert get_category_for_method("get_income_statement") == "fundamental_data"

    def test_get_news_is_news_data(self):
        assert get_category_for_method("get_news") == "news_data"

    def test_get_global_news_is_news_data(self):
        assert get_category_for_method("get_global_news") == "news_data"

    def test_get_insider_transactions_is_news_data(self):
        assert get_category_for_method("get_insider_transactions") == "news_data"

    def test_unknown_method_raises_value_error(self):
        with pytest.raises(ValueError, match="not found in any category"):
            get_category_for_method("nonexistent_method")

    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError):
            get_category_for_method("")


class TestGetVendor:
    """Tests for get_vendor."""

    @patch("tradingagents.dataflows.interface.get_config")
    def test_respects_tool_level_override(self, mock_get_config):
        mock_get_config.return_value = {
            "data_vendors": {"core_stock_apis": "yfinance"},
            "tool_vendors": {"get_stock_data": "alpha_vantage"},
        }
        result = get_vendor("core_stock_apis", method="get_stock_data")
        assert result == "alpha_vantage"

    @patch("tradingagents.dataflows.interface.get_config")
    def test_falls_back_to_category_level(self, mock_get_config):
        mock_get_config.return_value = {
            "data_vendors": {"core_stock_apis": "yfinance"},
            "tool_vendors": {},
        }
        result = get_vendor("core_stock_apis", method="get_stock_data")
        assert result == "yfinance"

    @patch("tradingagents.dataflows.interface.get_config")
    def test_falls_back_to_category_when_no_method(self, mock_get_config):
        mock_get_config.return_value = {
            "data_vendors": {"news_data": "alpha_vantage"},
            "tool_vendors": {},
        }
        result = get_vendor("news_data")
        assert result == "alpha_vantage"

    @patch("tradingagents.dataflows.interface.get_config")
    def test_returns_default_when_category_not_found(self, mock_get_config):
        mock_get_config.return_value = {
            "data_vendors": {},
            "tool_vendors": {},
        }
        result = get_vendor("nonexistent_category")
        assert result == "default"

    @patch("tradingagents.dataflows.interface.get_config")
    def test_tool_override_takes_precedence(self, mock_get_config):
        mock_get_config.return_value = {
            "data_vendors": {"fundamental_data": "yfinance"},
            "tool_vendors": {"get_fundamentals": "alpha_vantage"},
        }
        result = get_vendor("fundamental_data", method="get_fundamentals")
        assert result == "alpha_vantage"


class TestRouteToVendor:
    """Tests for route_to_vendor."""

    @patch("tradingagents.dataflows.interface.get_config")
    def test_calls_correct_vendor_implementation(self, mock_get_config):
        mock_get_config.return_value = {
            "data_vendors": {"core_stock_apis": "yfinance"},
            "tool_vendors": {},
        }
        with patch.dict(
            "tradingagents.dataflows.interface.VENDOR_METHODS",
            {
                "get_stock_data": {
                    "yfinance": MagicMock(return_value="yfinance_result"),
                    "alpha_vantage": MagicMock(return_value="av_result"),
                }
            },
        ):
            result = route_to_vendor("get_stock_data", "AAPL", "2025-01-15")
            assert result == "yfinance_result"

    @patch("tradingagents.dataflows.interface.get_config")
    def test_falls_back_on_alpha_vantage_rate_limit(self, mock_get_config):
        mock_get_config.return_value = {
            "data_vendors": {"core_stock_apis": "alpha_vantage"},
            "tool_vendors": {},
        }
        av_mock = MagicMock(side_effect=AlphaVantageRateLimitError("Rate limited"))
        yf_mock = MagicMock(return_value="yfinance_fallback")

        with patch.dict(
            "tradingagents.dataflows.interface.VENDOR_METHODS",
            {
                "get_stock_data": {
                    "alpha_vantage": av_mock,
                    "yfinance": yf_mock,
                }
            },
        ):
            result = route_to_vendor("get_stock_data", "AAPL", "2025-01-15")
            assert result == "yfinance_fallback"
            av_mock.assert_called_once()
            yf_mock.assert_called_once()

    @patch("tradingagents.dataflows.interface.get_config")
    def test_raises_runtime_error_when_all_vendors_fail(self, mock_get_config):
        mock_get_config.return_value = {
            "data_vendors": {"core_stock_apis": "alpha_vantage"},
            "tool_vendors": {},
        }
        av_mock = MagicMock(side_effect=AlphaVantageRateLimitError("Rate limited"))
        yf_mock = MagicMock(side_effect=AlphaVantageRateLimitError("Also rate limited"))

        with patch.dict(
            "tradingagents.dataflows.interface.VENDOR_METHODS",
            {
                "get_stock_data": {
                    "alpha_vantage": av_mock,
                    "yfinance": yf_mock,
                }
            },
        ):
            with pytest.raises(RuntimeError, match="No available vendor"):
                route_to_vendor("get_stock_data", "AAPL", "2025-01-15")

    def test_unsupported_method_raises_value_error(self):
        with pytest.raises(ValueError, match="not found in any category"):
            route_to_vendor("nonexistent_method")

    @patch("tradingagents.dataflows.interface.get_config")
    def test_non_rate_limit_errors_propagate(self, mock_get_config):
        """Non-AlphaVantageRateLimitError exceptions should not trigger fallback."""
        mock_get_config.return_value = {
            "data_vendors": {"core_stock_apis": "alpha_vantage"},
            "tool_vendors": {},
        }
        av_mock = MagicMock(side_effect=ConnectionError("Network error"))

        with patch.dict(
            "tradingagents.dataflows.interface.VENDOR_METHODS",
            {
                "get_stock_data": {
                    "alpha_vantage": av_mock,
                    "yfinance": MagicMock(return_value="fallback"),
                }
            },
        ):
            with pytest.raises(ConnectionError, match="Network error"):
                route_to_vendor("get_stock_data", "AAPL", "2025-01-15")


class TestVendorMethodsCompleteness:
    """Test that VENDOR_METHODS covers all tools in TOOLS_CATEGORIES."""

    def test_all_tools_have_vendor_methods(self):
        """Every tool listed in TOOLS_CATEGORIES should have an entry in VENDOR_METHODS."""
        all_tools = []
        for category, info in TOOLS_CATEGORIES.items():
            all_tools.extend(info["tools"])

        for tool in all_tools:
            assert tool in VENDOR_METHODS, (
                f"Tool '{tool}' from TOOLS_CATEGORIES is missing from VENDOR_METHODS"
            )

    def test_vendor_methods_have_at_least_one_vendor(self):
        """Each entry in VENDOR_METHODS should have at least one vendor implementation."""
        for method, vendors in VENDOR_METHODS.items():
            assert len(vendors) > 0, (
                f"VENDOR_METHODS['{method}'] has no vendor implementations"
            )

    def test_vendor_list_has_expected_vendors(self):
        assert "yfinance" in VENDOR_LIST
        assert "alpha_vantage" in VENDOR_LIST

    def test_tools_categories_has_expected_categories(self):
        expected = {"core_stock_apis", "technical_indicators", "fundamental_data", "news_data"}
        assert set(TOOLS_CATEGORIES.keys()) == expected
