"""Tests for tradingagents.graph.trading_graph.TradingAgentsGraph."""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock

from tradingagents.default_config import DEFAULT_CONFIG


class TestTradingAgentsGraphInit:
    """Test TradingAgentsGraph constructor behavior."""

    @patch("tradingagents.graph.trading_graph.GraphSetup")
    @patch("tradingagents.graph.trading_graph.create_llm_client")
    @patch("tradingagents.graph.trading_graph.set_config")
    @patch("tradingagents.graph.trading_graph.os.makedirs")
    def test_constructor_with_default_config(
        self, mock_makedirs, mock_set_config, mock_create_client, mock_graph_setup
    ):
        """Constructor should use DEFAULT_CONFIG when no config provided."""
        mock_client = MagicMock()
        mock_client.get_llm.return_value = MagicMock()
        mock_create_client.return_value = mock_client
        mock_graph_setup_instance = MagicMock()
        mock_graph_setup_instance.setup_graph.return_value = MagicMock()
        mock_graph_setup.return_value = mock_graph_setup_instance

        from tradingagents.graph.trading_graph import TradingAgentsGraph

        graph = TradingAgentsGraph()
        assert graph.config is DEFAULT_CONFIG

    @patch("tradingagents.graph.trading_graph.GraphSetup")
    @patch("tradingagents.graph.trading_graph.create_llm_client")
    @patch("tradingagents.graph.trading_graph.set_config")
    @patch("tradingagents.graph.trading_graph.os.makedirs")
    def test_constructor_with_custom_config(
        self, mock_makedirs, mock_set_config, mock_create_client, mock_graph_setup
    ):
        """Constructor should use custom config when provided."""
        mock_client = MagicMock()
        mock_client.get_llm.return_value = MagicMock()
        mock_create_client.return_value = mock_client
        mock_graph_setup_instance = MagicMock()
        mock_graph_setup_instance.setup_graph.return_value = MagicMock()
        mock_graph_setup.return_value = mock_graph_setup_instance

        from tradingagents.graph.trading_graph import TradingAgentsGraph

        custom_config = DEFAULT_CONFIG.copy()
        custom_config["llm_provider"] = "anthropic"
        custom_config["deep_think_llm"] = "claude-sonnet-4-5"
        custom_config["quick_think_llm"] = "claude-haiku-4-5"

        graph = TradingAgentsGraph(config=custom_config)
        assert graph.config["llm_provider"] == "anthropic"
        assert graph.config["deep_think_llm"] == "claude-sonnet-4-5"

    @patch("tradingagents.graph.trading_graph.GraphSetup")
    @patch("tradingagents.graph.trading_graph.create_llm_client")
    @patch("tradingagents.graph.trading_graph.set_config")
    @patch("tradingagents.graph.trading_graph.os.makedirs")
    def test_constructor_calls_set_config(
        self, mock_makedirs, mock_set_config, mock_create_client, mock_graph_setup
    ):
        mock_client = MagicMock()
        mock_client.get_llm.return_value = MagicMock()
        mock_create_client.return_value = mock_client
        mock_graph_setup_instance = MagicMock()
        mock_graph_setup_instance.setup_graph.return_value = MagicMock()
        mock_graph_setup.return_value = mock_graph_setup_instance

        from tradingagents.graph.trading_graph import TradingAgentsGraph

        TradingAgentsGraph()
        mock_set_config.assert_called_once()

    @patch("tradingagents.graph.trading_graph.GraphSetup")
    @patch("tradingagents.graph.trading_graph.create_llm_client")
    @patch("tradingagents.graph.trading_graph.set_config")
    @patch("tradingagents.graph.trading_graph.os.makedirs")
    def test_constructor_creates_two_llm_clients(
        self, mock_makedirs, mock_set_config, mock_create_client, mock_graph_setup
    ):
        mock_client = MagicMock()
        mock_client.get_llm.return_value = MagicMock()
        mock_create_client.return_value = mock_client
        mock_graph_setup_instance = MagicMock()
        mock_graph_setup_instance.setup_graph.return_value = MagicMock()
        mock_graph_setup.return_value = mock_graph_setup_instance

        from tradingagents.graph.trading_graph import TradingAgentsGraph

        TradingAgentsGraph()
        assert mock_create_client.call_count == 2  # deep + quick

    @patch("tradingagents.graph.trading_graph.GraphSetup")
    @patch("tradingagents.graph.trading_graph.create_llm_client")
    @patch("tradingagents.graph.trading_graph.set_config")
    @patch("tradingagents.graph.trading_graph.os.makedirs")
    def test_constructor_stores_callbacks(
        self, mock_makedirs, mock_set_config, mock_create_client, mock_graph_setup
    ):
        mock_client = MagicMock()
        mock_client.get_llm.return_value = MagicMock()
        mock_create_client.return_value = mock_client
        mock_graph_setup_instance = MagicMock()
        mock_graph_setup_instance.setup_graph.return_value = MagicMock()
        mock_graph_setup.return_value = mock_graph_setup_instance

        from tradingagents.graph.trading_graph import TradingAgentsGraph

        mock_cb = MagicMock()
        graph = TradingAgentsGraph(callbacks=[mock_cb])
        assert graph.callbacks == [mock_cb]


class TestGetProviderKwargs:
    """Test _get_provider_kwargs method."""

    @patch("tradingagents.graph.trading_graph.GraphSetup")
    @patch("tradingagents.graph.trading_graph.create_llm_client")
    @patch("tradingagents.graph.trading_graph.set_config")
    @patch("tradingagents.graph.trading_graph.os.makedirs")
    def test_google_provider_includes_thinking_level(
        self, mock_makedirs, mock_set_config, mock_create_client, mock_graph_setup
    ):
        mock_client = MagicMock()
        mock_client.get_llm.return_value = MagicMock()
        mock_create_client.return_value = mock_client
        mock_graph_setup_instance = MagicMock()
        mock_graph_setup_instance.setup_graph.return_value = MagicMock()
        mock_graph_setup.return_value = mock_graph_setup_instance

        from tradingagents.graph.trading_graph import TradingAgentsGraph

        config = DEFAULT_CONFIG.copy()
        config["llm_provider"] = "google"
        config["google_thinking_level"] = "high"
        config["deep_think_llm"] = "gemini-2.5-pro"
        config["quick_think_llm"] = "gemini-2.5-flash"

        graph = TradingAgentsGraph(config=config)
        kwargs = graph._get_provider_kwargs()
        assert kwargs["thinking_level"] == "high"

    @patch("tradingagents.graph.trading_graph.GraphSetup")
    @patch("tradingagents.graph.trading_graph.create_llm_client")
    @patch("tradingagents.graph.trading_graph.set_config")
    @patch("tradingagents.graph.trading_graph.os.makedirs")
    def test_openai_provider_includes_reasoning_effort(
        self, mock_makedirs, mock_set_config, mock_create_client, mock_graph_setup
    ):
        mock_client = MagicMock()
        mock_client.get_llm.return_value = MagicMock()
        mock_create_client.return_value = mock_client
        mock_graph_setup_instance = MagicMock()
        mock_graph_setup_instance.setup_graph.return_value = MagicMock()
        mock_graph_setup.return_value = mock_graph_setup_instance

        from tradingagents.graph.trading_graph import TradingAgentsGraph

        config = DEFAULT_CONFIG.copy()
        config["llm_provider"] = "openai"
        config["openai_reasoning_effort"] = "high"

        graph = TradingAgentsGraph(config=config)
        kwargs = graph._get_provider_kwargs()
        assert kwargs["reasoning_effort"] == "high"

    @patch("tradingagents.graph.trading_graph.GraphSetup")
    @patch("tradingagents.graph.trading_graph.create_llm_client")
    @patch("tradingagents.graph.trading_graph.set_config")
    @patch("tradingagents.graph.trading_graph.os.makedirs")
    def test_provider_kwargs_empty_when_no_special_config(
        self, mock_makedirs, mock_set_config, mock_create_client, mock_graph_setup
    ):
        mock_client = MagicMock()
        mock_client.get_llm.return_value = MagicMock()
        mock_create_client.return_value = mock_client
        mock_graph_setup_instance = MagicMock()
        mock_graph_setup_instance.setup_graph.return_value = MagicMock()
        mock_graph_setup.return_value = mock_graph_setup_instance

        from tradingagents.graph.trading_graph import TradingAgentsGraph

        config = DEFAULT_CONFIG.copy()
        config["llm_provider"] = "openai"
        config["openai_reasoning_effort"] = None
        config["google_thinking_level"] = None

        graph = TradingAgentsGraph(config=config)
        kwargs = graph._get_provider_kwargs()
        assert kwargs == {}


class TestCreateToolNodes:
    """Test _create_tool_nodes method."""

    @patch("tradingagents.graph.trading_graph.GraphSetup")
    @patch("tradingagents.graph.trading_graph.create_llm_client")
    @patch("tradingagents.graph.trading_graph.set_config")
    @patch("tradingagents.graph.trading_graph.os.makedirs")
    def test_returns_dict_with_expected_keys(
        self, mock_makedirs, mock_set_config, mock_create_client, mock_graph_setup
    ):
        mock_client = MagicMock()
        mock_client.get_llm.return_value = MagicMock()
        mock_create_client.return_value = mock_client
        mock_graph_setup_instance = MagicMock()
        mock_graph_setup_instance.setup_graph.return_value = MagicMock()
        mock_graph_setup.return_value = mock_graph_setup_instance

        from tradingagents.graph.trading_graph import TradingAgentsGraph

        graph = TradingAgentsGraph()
        tool_nodes = graph.tool_nodes
        assert "market" in tool_nodes
        assert "social" in tool_nodes
        assert "news" in tool_nodes
        assert "fundamentals" in tool_nodes
