# TradingAgents Architecture

This document describes the internal architecture of the TradingAgents framework -- a multi-agent LLM system for financial trading decisions.

## System Overview

TradingAgents models a financial research team as a directed graph of specialized LLM agents. Each agent performs a distinct role (analysis, debate, decision-making, risk assessment) and the system flows through them in a defined order, producing a final BUY/SELL/HOLD signal.

### Agent Flow

```
                          +-------------------+
                          |       START       |
                          +-------------------+
                                  |
                  +---------------+---------------+
                  |               |               |
                  v               v               v
          +-----------+   +-----------+   +-----------+   +----------------+
          |  Market   |   |  Social   |   |   News    |   | Fundamentals   |
          |  Analyst  |-->|  Analyst  |-->|  Analyst  |-->|    Analyst     |
          +-----------+   +-----------+   +-----------+   +----------------+
                  |               |               |               |
                  |   (each loops with tools      |               |
                  |    until analysis complete)    |               |
                                                                  |
                                                                  v
                                                      +-----------------------+
                                                      |    Bull Researcher    |<--+
                                                      +-----------------------+   |
                                                                  |               |
                                                                  v               |
                                                      +-----------------------+   |
                                                      |    Bear Researcher    |---+
                                                      +-----------------------+
                                                         (debate rounds)
                                                                  |
                                                                  v
                                                      +-----------------------+
                                                      |   Research Manager    |
                                                      |   (Investment Judge)  |
                                                      +-----------------------+
                                                                  |
                                                                  v
                                                      +-----------------------+
                                                      |        Trader        |
                                                      +-----------------------+
                                                                  |
                                                                  v
                                                      +-----------------------+
                                                      | Aggressive Analyst    |<--+
                                                      +-----------------------+   |
                                                                  |               |
                                                                  v               |
                                                      +-----------------------+   |
                                                      | Conservative Analyst  |   |
                                                      +-----------------------+   |
                                                                  |               |
                                                                  v               |
                                                      +-----------------------+   |
                                                      |   Neutral Analyst     |---+
                                                      +-----------------------+
                                                         (risk debate rounds)
                                                                  |
                                                                  v
                                                      +-----------------------+
                                                      |      Risk Judge       |
                                                      +-----------------------+
                                                                  |
                                                                  v
                                                      +-----------------------+
                                                      |         END          |
                                                      +-----------------------+
```

Condensed linear view:

```
START -> Analysts (Market, Social, News, Fundamentals)
      -> Bull/Bear Debate
      -> Research Manager
      -> Trader
      -> Risk Debate (Aggressive, Conservative, Neutral)
      -> Risk Judge
      -> END
```

## Component Descriptions

### Analysts

Analysts are the first stage of the pipeline. They use LLM tool-calling to fetch real market data and produce structured reports. Each analyst runs sequentially (configurable subset via `selected_analysts`), calling tools in a loop until the LLM decides it has enough information.

| Analyst | Role | Tools Used | Output Field |
|---------|------|-----------|--------------|
| **Market Analyst** | Analyzes price action, volume, and technical indicators | `get_stock_data`, `get_indicators` | `market_report` |
| **Social Media Analyst** | Gauges social sentiment from news headlines | `get_news` | `sentiment_report` |
| **News Analyst** | Researches current events, insider activity | `get_news`, `get_global_news`, `get_insider_transactions` | `news_report` |
| **Fundamentals Analyst** | Reviews financials -- balance sheet, cash flow, income | `get_fundamentals`, `get_balance_sheet`, `get_cashflow`, `get_income_statement` | `fundamentals_report` |

**Key design decision**: Analysts use the "quick thinking" LLM (`quick_think_llm`) because they need to make many tool calls rapidly. The heavier reasoning is reserved for judges.

After each analyst completes, a `Msg Clear` node removes tool-call messages from the shared state to keep the context window clean for subsequent agents.

### Investment Debate (Bull/Bear Researchers)

Once all analyst reports are in, a structured debate begins:

- **Bull Researcher**: Argues the bullish case for the stock, drawing on analyst reports and its own memory of past situations.
- **Bear Researcher**: Argues the bearish case.

They alternate for `max_debate_rounds` rounds (default: 1, meaning 2 total responses -- one bull, one bear). The debate history is tracked in `InvestDebateState`.

**Key design decision**: Memory-augmented debate. Each researcher queries `FinancialSituationMemory` with the current market situation to retrieve lessons from past trades, grounding the debate in experience rather than pure LLM reasoning.

### Research Manager (Investment Judge)

The Research Manager receives the full debate history and all analyst reports, then produces a synthesized investment recommendation. This agent uses the "deep thinking" LLM (`deep_think_llm`) for more thorough reasoning.

It also queries its own memory for past judging experiences, allowing it to learn from previous correct and incorrect calls.

### Trader

The Trader takes the Research Manager's recommendation and formulates a concrete investment plan with specific allocation details and trade parameters.

### Risk Debate (Aggressive, Conservative, Neutral)

The Trader's plan enters a three-way risk assessment debate:

- **Aggressive Analyst**: Argues for higher risk tolerance, larger positions.
- **Conservative Analyst**: Argues for caution, smaller positions, hedging.
- **Neutral Analyst**: Provides balanced perspective.

They cycle through `max_risk_discuss_rounds` rounds (default: 1, meaning 3 total responses). The order is: Aggressive -> Conservative -> Neutral -> (repeat).

### Risk Judge

The Risk Judge receives the full risk debate and produces the `final_trade_decision`. This agent uses the deep thinking LLM and has its own memory. Its output is the terminal state of the graph.

## State Management

The framework uses LangGraph's `StateGraph` with three TypedDict state classes.

### `AgentState` (Primary Graph State)

Extends LangGraph's `MessagesState` and is the top-level state flowing through the entire graph.

```python
class AgentState(MessagesState):
    company_of_interest: str    # Ticker symbol (e.g., "AAPL")
    trade_date: str             # Trading date
    sender: str                 # Current agent identifier

    # Analyst reports (populated sequentially)
    market_report: str
    sentiment_report: str
    news_report: str
    fundamentals_report: str

    # Debate substates (nested TypedDicts)
    investment_debate_state: InvestDebateState
    risk_debate_state: RiskDebateState

    # Decision outputs
    investment_plan: str
    trader_investment_plan: str
    final_trade_decision: str
```

### `InvestDebateState`

Tracks the bull/bear investment debate.

| Field | Type | Purpose |
|-------|------|---------|
| `bull_history` | `str` | Full history of bull researcher's arguments |
| `bear_history` | `str` | Full history of bear researcher's arguments |
| `history` | `str` | Combined debate transcript |
| `current_response` | `str` | Latest response (prefixed with "Bull" or "Bear" for routing) |
| `judge_decision` | `str` | Research Manager's final decision |
| `count` | `int` | Number of debate turns completed |

### `RiskDebateState`

Tracks the three-way risk assessment debate.

| Field | Type | Purpose |
|-------|------|---------|
| `aggressive_history` | `str` | Aggressive analyst's argument history |
| `conservative_history` | `str` | Conservative analyst's argument history |
| `neutral_history` | `str` | Neutral analyst's argument history |
| `history` | `str` | Combined debate transcript |
| `latest_speaker` | `str` | Identifies last speaker for round-robin routing |
| `current_*_response` | `str` | Latest response from each debater |
| `judge_decision` | `str` | Risk Judge's final decision |
| `count` | `int` | Number of debate turns completed |

## Memory System

### BM25-Based Retrieval

The memory system uses BM25 (Best Matching 25), a classical information retrieval algorithm, to store and retrieve past trading experiences. This is implemented in `FinancialSituationMemory`.

**How it works:**

1. After each trade, the `Reflector` asks the LLM to analyze what went right/wrong.
2. The reflection is stored as a (situation, recommendation) pair in each agent's memory.
3. On the next trade, the current market situation is used as a query against stored situations.
4. BM25 ranks stored situations by lexical similarity and returns the top matches.
5. Matched recommendations are injected into the agent's prompt as context.

**Why BM25 instead of embeddings:**

- No API calls needed -- runs entirely in-process with `rank_bm25` library.
- No token limits or rate limits on retrieval.
- Works offline with any LLM provider (including Ollama).
- Zero additional cost.
- Sufficient for matching financial situations that share domain-specific vocabulary.

**Memory instances**: The framework maintains five separate memory stores, one for each memory-augmented agent:

| Memory Instance | Agent |
|----------------|-------|
| `bull_memory` | Bull Researcher |
| `bear_memory` | Bear Researcher |
| `trader_memory` | Trader |
| `invest_judge_memory` | Research Manager |
| `risk_manager_memory` | Risk Judge |

### Reflection Loop

Memory is populated through the `reflect_and_remember()` method on `TradingAgentsGraph`. After a trade's outcome is known, call this with the returns/losses:

```
propagate() -> get outcome -> reflect_and_remember(returns) -> memories updated
     ^                                                              |
     |______________________________________________________________|
                    (next propagate benefits from memory)
```

## Data Vendor Routing

The framework abstracts data sources behind a routing layer (`tradingagents/dataflows/interface.py`) that supports multiple vendors with automatic fallback.

### Configuration Levels

**Category-level** (default for all tools in a category):

```python
"data_vendors": {
    "core_stock_apis": "yfinance",
    "technical_indicators": "yfinance",
    "fundamental_data": "yfinance",
    "news_data": "yfinance",
}
```

**Tool-level** (overrides category default for a specific tool):

```python
"tool_vendors": {
    "get_stock_data": "alpha_vantage",  # Override just this tool
}
```

Tool-level configuration takes precedence over category-level.

### Supported Vendors

| Vendor | API Key Required | Notes |
|--------|-----------------|-------|
| `yfinance` | No | Free, no rate limits. Default for all categories. |
| `alpha_vantage` | Yes (`ALPHA_VANTAGE_API_KEY`) | Higher quality data, but rate-limited (5 calls/min on free tier). |

### Fallback Chain

When a vendor fails due to rate limiting (`AlphaVantageRateLimitError`), the router automatically falls back to the next available vendor. The fallback chain is:

1. Configured primary vendor(s)
2. All remaining vendors that implement the method

Only rate limit errors trigger fallback. Other errors propagate normally.

### Tool Categories

| Category | Tools | Description |
|----------|-------|-------------|
| `core_stock_apis` | `get_stock_data` | OHLCV price data |
| `technical_indicators` | `get_indicators` | Technical analysis indicators (RSI, MACD, etc.) |
| `fundamental_data` | `get_fundamentals`, `get_balance_sheet`, `get_cashflow`, `get_income_statement` | Company financial statements |
| `news_data` | `get_news`, `get_global_news`, `get_insider_transactions` | News, global events, insider activity |

## LLM Client Abstraction

### Factory Pattern

The `create_llm_client()` factory function in `tradingagents/llm_clients/factory.py` creates provider-specific clients from a unified interface.

```
create_llm_client(provider, model, base_url, **kwargs)
        |
        +-- "openai"      --> OpenAIClient   (ChatOpenAI / UnifiedChatOpenAI)
        +-- "ollama"       --> OpenAIClient   (base_url = localhost:11434)
        +-- "openrouter"   --> OpenAIClient   (base_url = openrouter.ai)
        +-- "xai"          --> OpenAIClient   (base_url = api.x.ai)
        +-- "anthropic"    --> AnthropicClient (ChatAnthropic)
        +-- "google"       --> GoogleClient    (NormalizedChatGoogleGenerativeAI)
```

### Provider Support

| Provider | Client Class | LangChain Class | API Key Env Var |
|----------|-------------|-----------------|-----------------|
| OpenAI | `OpenAIClient` | `UnifiedChatOpenAI` | `OPENAI_API_KEY` |
| Anthropic | `AnthropicClient` | `ChatAnthropic` | `ANTHROPIC_API_KEY` |
| Google | `GoogleClient` | `NormalizedChatGoogleGenerativeAI` | `GOOGLE_API_KEY` |
| xAI | `OpenAIClient` | `UnifiedChatOpenAI` | `XAI_API_KEY` |
| Ollama | `OpenAIClient` | `UnifiedChatOpenAI` | None (uses "ollama") |
| OpenRouter | `OpenAIClient` | `UnifiedChatOpenAI` | `OPENROUTER_API_KEY` |

### Provider-Specific Behaviors

- **OpenAI / GPT-5**: `UnifiedChatOpenAI` automatically strips `temperature` and `top_p` parameters for GPT-5 family models, which use native reasoning and reject these parameters.
- **Google / Gemini**: `NormalizedChatGoogleGenerativeAI` normalizes list-format content responses (common in Gemini 3) to plain strings. Thinking level configuration differs between Gemini 3 (uses `thinking_level`) and Gemini 2.5 (uses `thinking_budget`).
- **Anthropic**: Does not support `base_url` -- the parameter is ignored with a warning.

### Two-Tier LLM Usage

The framework uses two LLM instances:

| LLM | Config Key | Default | Used By |
|-----|-----------|---------|---------|
| **Deep thinking** | `deep_think_llm` | `gpt-5.2` | Research Manager, Risk Judge |
| **Quick thinking** | `quick_think_llm` | `gpt-5-mini` | All analysts, researchers, trader, debaters, reflector, signal processor |

This allows cost optimization: expensive reasoning models for critical decisions, cheaper models for data gathering and routine tasks.

## Graph Flow Control

### LangGraph StateGraph

The graph is built using LangGraph's `StateGraph` with `AgentState` as the state schema. Nodes are added dynamically based on `selected_analysts`.

### Conditional Logic

The `ConditionalLogic` class controls branching at three points:

**1. Analyst Tool Loops**

Each analyst has a conditional edge: if the LLM's last message contains `tool_calls`, route back to the tool node; otherwise, proceed to the message-clear node. This lets analysts make as many tool calls as needed.

```
Analyst -> (has tool_calls?) -> Yes -> ToolNode -> Analyst
                             -> No  -> Msg Clear -> Next Stage
```

**2. Investment Debate Rounds**

After each bull/bear response, the system checks:
- If `count >= 2 * max_debate_rounds`: route to Research Manager (debate complete).
- If last response started with "Bull": route to Bear Researcher.
- Otherwise: route to Bull Researcher.

**3. Risk Debate Rounds**

After each risk analyst response, the system checks:
- If `count >= 3 * max_risk_discuss_rounds`: route to Risk Judge (debate complete).
- Round-robin routing: Aggressive -> Conservative -> Neutral -> Aggressive -> ...

### Graph Compilation

The graph is compiled once during `TradingAgentsGraph.__init__()` via `GraphSetup.setup_graph()`. The compiled graph is reusable across multiple `propagate()` calls with different tickers and dates.

## Directory Structure

```
tradingagents/
    __init__.py
    default_config.py           # DEFAULT_CONFIG dictionary
    logging_config.py           # Logging setup utilities
    agents/
        __init__.py             # Re-exports all agent factory functions
        analysts/
            market_analyst.py
            social_media_analyst.py
            news_analyst.py
            fundamentals_analyst.py
        researchers/
            bull_researcher.py
            bear_researcher.py
        managers/
            research_manager.py  # Investment judge
            risk_manager.py      # Risk judge
        risk_mgmt/
            aggressive_debator.py
            conservative_debator.py
            neutral_debator.py
        trader/
            trader.py
        utils/
            agent_states.py      # AgentState, InvestDebateState, RiskDebateState
            agent_utils.py       # Tool wrapper functions, message utilities
            memory.py            # FinancialSituationMemory (BM25)
    dataflows/
        __init__.py
        config.py               # Runtime config management
        interface.py            # Vendor routing, fallback logic
        y_finance.py            # yfinance implementations
        yfinance_news.py        # yfinance news implementations
        alpha_vantage.py        # Alpha Vantage facade
        alpha_vantage_*.py      # Alpha Vantage per-category modules
    graph/
        __init__.py
        trading_graph.py        # TradingAgentsGraph (main entry point)
        setup.py                # GraphSetup (node/edge wiring)
        conditional_logic.py    # Branching conditions
        propagation.py          # State initialization
        reflection.py           # Post-trade reflection
        signal_processing.py    # BUY/SELL/HOLD extraction
    llm_clients/
        __init__.py             # Re-exports create_llm_client
        factory.py              # create_llm_client() factory
        base_client.py          # BaseLLMClient ABC
        openai_client.py        # OpenAI/Ollama/OpenRouter/xAI
        anthropic_client.py     # Anthropic Claude
        google_client.py        # Google Gemini
        validators.py           # Model name validation
```
