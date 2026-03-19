from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Create a custom config using Claude (Anthropic) as the LLM provider
config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "anthropic"
config["deep_think_llm"] = "claude-sonnet-4-6"  # Best balance of speed and intelligence
config["quick_think_llm"] = "claude-haiku-4-5"  # Fast and cost-effective
config["max_debate_rounds"] = 1
config["max_risk_discuss_rounds"] = 1

# Configure data vendors (yfinance is free, no extra API keys needed)
config["data_vendors"] = {
    "core_stock_apis": "yfinance",
    "technical_indicators": "yfinance",
    "fundamental_data": "yfinance",
    "news_data": "yfinance",
}

# Initialize with custom config
ta = TradingAgentsGraph(debug=True, config=config)

# Run analysis on NVDA
_, decision = ta.propagate("NVDA", "2025-03-19")
print(decision)

# Memorize mistakes and reflect
# ta.reflect_and_remember(1000) # parameter is the position returns
