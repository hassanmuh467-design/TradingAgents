"""Shared fixtures for the TradingAgents test suite."""

import sys
from unittest.mock import MagicMock, patch

# Mock all heavy dependencies before they're imported
MOCK_MODULES = [
    'langchain_openai',
    'langchain_anthropic',
    'langchain_google_genai',
    'langchain_core',
    'langchain_core.messages',
    'langchain_core.tools',
    'langchain_core.prompts',
    'langchain_experimental',
    'langgraph',
    'langgraph.prebuilt',
    'langgraph.graph',
    'yfinance',
    'stockstats',
    'pandas',
    'numpy',
    'redis',
    'backtrader',
    'rank_bm25',
    'parsel',
    'tqdm',
    'rich',
    'typer',
    'questionary',
    'typing_extensions',
    'dateutil',
    'dateutil.relativedelta',
]

for mod_name in MOCK_MODULES:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

# typing_extensions needs TypedDict to work like the real one
import typing
sys.modules['typing_extensions'].TypedDict = typing.TypedDict
sys.modules['typing_extensions'].Annotated = typing.Annotated

# Now set up specific mock attributes that code relies on

# langchain_openai.ChatOpenAI needs to be a real class for inheritance
_ChatOpenAI_meta = type('ChatOpenAI', (), {
    '__init__': lambda self, **kwargs: None,
    'invoke': lambda self, *args, **kwargs: MagicMock(content="mocked"),
})
sys.modules['langchain_openai'].ChatOpenAI = _ChatOpenAI_meta

# langchain_anthropic.ChatAnthropic needs to be a class
sys.modules['langchain_anthropic'].ChatAnthropic = type('ChatAnthropic', (), {
    '__init__': lambda self, **kwargs: None,
})

# langchain_google_genai.ChatGoogleGenerativeAI needs to be a real class for inheritance
_ChatGoogleGenerativeAI_meta = type('ChatGoogleGenerativeAI', (), {
    '__init__': lambda self, **kwargs: None,
    'invoke': lambda self, *args, **kwargs: MagicMock(content="mocked"),
})
sys.modules['langchain_google_genai'].ChatGoogleGenerativeAI = _ChatGoogleGenerativeAI_meta

# langgraph needs MessagesState and StateGraph
sys.modules['langgraph.graph'].MessagesState = type('MessagesState', (dict,), {})
sys.modules['langgraph.graph'].StateGraph = MagicMock()
sys.modules['langgraph.graph'].END = 'END'
sys.modules['langgraph.graph'].START = 'START'
sys.modules['langgraph.prebuilt'].ToolNode = MagicMock()

# langchain_core.messages needs specific message classes
sys.modules['langchain_core.messages'].HumanMessage = type('HumanMessage', (), {
    '__init__': lambda self, **kwargs: None,
})
sys.modules['langchain_core.messages'].RemoveMessage = type('RemoveMessage', (), {
    '__init__': lambda self, **kwargs: None,
})
sys.modules['langchain_core.messages'].AIMessage = type('AIMessage', (), {
    '__init__': lambda self, **kwargs: None,
})

# langchain_core.tools needs the @tool decorator
sys.modules['langchain_core.tools'].tool = lambda f: f  # decorator that returns the function as-is

# langchain_core.prompts needs ChatPromptTemplate and MessagesPlaceholder
sys.modules['langchain_core.prompts'].ChatPromptTemplate = MagicMock()
sys.modules['langchain_core.prompts'].MessagesPlaceholder = MagicMock()

# rank_bm25 needs BM25Okapi
sys.modules['rank_bm25'].BM25Okapi = MagicMock()

# stockstats needs wrap
sys.modules['stockstats'].wrap = MagicMock()

# dateutil.relativedelta needs relativedelta
sys.modules['dateutil.relativedelta'].relativedelta = MagicMock()

# yfinance needs Ticker
sys.modules['yfinance'].Ticker = MagicMock()
sys.modules['yfinance'].download = MagicMock()

# pandas needs DataFrame and basic functionality
_mock_pandas = sys.modules['pandas']
_mock_pandas.DataFrame = MagicMock()
_mock_pandas.read_csv = MagicMock()
_mock_pandas.to_datetime = MagicMock()

# numpy basics
_mock_numpy = sys.modules['numpy']
_mock_numpy.array = MagicMock()

# ---- Now safe to import tradingagents modules ----

import os
import pytest

from tradingagents.default_config import DEFAULT_CONFIG


@pytest.fixture
def mock_llm():
    """A mock LLM (ChatOpenAI-like) that returns predictable responses."""
    llm = MagicMock()
    response = MagicMock()
    response.content = "BUY"
    response.tool_calls = []
    llm.invoke.return_value = response
    llm.bind_tools.return_value = llm
    return llm


@pytest.fixture
def sample_invest_debate_state():
    """Sample InvestDebateState dictionary."""
    return {
        "bull_history": "Bull argues strong earnings growth and market momentum.",
        "bear_history": "Bear argues overvaluation and rising interest rates.",
        "history": "Round 1 debate between bull and bear researchers.",
        "current_response": "Bull: The company shows strong fundamentals.",
        "judge_decision": "BUY based on strong earnings outlook.",
        "count": 2,
    }


@pytest.fixture
def sample_risk_debate_state():
    """Sample RiskDebateState dictionary."""
    return {
        "aggressive_history": "Aggressive analyst recommends full position.",
        "conservative_history": "Conservative analyst recommends half position.",
        "neutral_history": "Neutral analyst recommends moderate position.",
        "history": "Risk discussion round 1.",
        "latest_speaker": "Aggressive Analyst",
        "current_aggressive_response": "Go all in on this opportunity.",
        "current_conservative_response": "Limit exposure to 25% of portfolio.",
        "current_neutral_response": "A balanced 50% allocation is prudent.",
        "judge_decision": "Moderate position with stop-loss at 5%.",
        "count": 3,
    }


@pytest.fixture
def sample_agent_state(sample_invest_debate_state, sample_risk_debate_state):
    """A full AgentState dict with sample data for all fields."""
    return {
        "messages": [("human", "AAPL")],
        "company_of_interest": "AAPL",
        "trade_date": "2025-01-15",
        "sender": "Market Analyst",
        "market_report": "AAPL shows bullish technical patterns with RSI at 65.",
        "sentiment_report": "Social media sentiment is predominantly positive for AAPL.",
        "news_report": "Apple announced record quarterly earnings beating estimates.",
        "fundamentals_report": "P/E ratio of 28, strong cash flow of $100B.",
        "investment_debate_state": sample_invest_debate_state,
        "investment_plan": "Buy AAPL with a target price of $200.",
        "trader_investment_plan": "Execute market buy order for 100 shares of AAPL.",
        "risk_debate_state": sample_risk_debate_state,
        "final_trade_decision": "BUY 100 shares of AAPL at market price with stop-loss at $175.",
    }


@pytest.fixture
def sample_config():
    """A test config dict based on DEFAULT_CONFIG with test-friendly overrides."""
    config = DEFAULT_CONFIG.copy()
    config.update({
        "llm_provider": "openai",
        "deep_think_llm": "gpt-5-mini",
        "quick_think_llm": "gpt-5-mini",
        "backend_url": "https://api.openai.com/v1",
        "max_debate_rounds": 1,
        "max_risk_discuss_rounds": 1,
        "max_recur_limit": 50,
        "google_thinking_level": None,
        "openai_reasoning_effort": None,
        "data_vendors": {
            "core_stock_apis": "yfinance",
            "technical_indicators": "yfinance",
            "fundamental_data": "yfinance",
            "news_data": "yfinance",
        },
        "tool_vendors": {},
    })
    return config
