# TradingAgents API Reference

## TradingAgentsGraph

The main entry point for the framework. Located in `tradingagents/graph/trading_graph.py`.

### Constructor

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph

graph = TradingAgentsGraph(
    selected_analysts=["market", "social", "news", "fundamentals"],
    debug=False,
    config=None,
    callbacks=None,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `selected_analysts` | `list[str]` | `["market", "social", "news", "fundamentals"]` | Which analyst agents to include in the pipeline. Any subset of the four types is valid. At least one is required. |
| `debug` | `bool` | `False` | When `True`, uses `graph.stream()` instead of `graph.invoke()` and pretty-prints each chunk for tracing. |
| `config` | `dict` | `None` | Configuration dictionary. If `None`, uses `DEFAULT_CONFIG`. See [Configuration Options](#configuration-options). |
| `callbacks` | `list` | `None` | LangChain callback handlers passed to LLM constructors (e.g., for tracking token usage or tool invocations). |

### Methods

#### `propagate(company_name, trade_date)`

Run the full agent pipeline for a given company and date.

```python
final_state, signal = graph.propagate("AAPL", "2025-01-15")
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `company_name` | `str` | Ticker symbol (e.g., `"AAPL"`, `"GOOGL"`) |
| `trade_date` | `str` or `date` | The date to analyze. Converted to string internally. |

**Returns:** `Tuple[dict, str]`

- `final_state` -- The complete `AgentState` dictionary containing all reports, debate histories, and decisions.
- `signal` -- Extracted decision string: `"BUY"`, `"SELL"`, or `"HOLD"`.

**Side effects:**
- Logs the full state to `eval_results/{ticker}/TradingAgentsStrategy_logs/full_states_log_{date}.json`.
- Stores the state internally for use by `reflect_and_remember()`.

#### `reflect_and_remember(returns_losses)`

After learning the trade outcome, reflect on the decision and update all agent memories.

```python
graph.reflect_and_remember("5.2% return over 3 days")
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `returns_losses` | `str` or any | Description of the trade outcome. Passed to the reflection LLM for analysis. |

**Side effects:**
- Updates all five memory stores (bull, bear, trader, invest_judge, risk_manager) with new (situation, recommendation) pairs derived from the reflection.

Must be called after `propagate()` -- it uses the internally stored state from the most recent run.

#### `process_signal(full_signal)`

Extract a BUY/SELL/HOLD decision from a full-text trading signal. This is called automatically by `propagate()` but can also be used standalone.

```python
signal = graph.process_signal("Based on our analysis, we recommend buying AAPL...")
# Returns: "BUY"
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `full_signal` | `str` | Full text of a trading decision/report. |

**Returns:** `str` -- One of `"BUY"`, `"SELL"`, or `"HOLD"`.

---

## Configuration Options

The `DEFAULT_CONFIG` dictionary is defined in `tradingagents/default_config.py`. Pass a modified copy to the `TradingAgentsGraph` constructor.

```python
from tradingagents.default_config import DEFAULT_CONFIG

my_config = DEFAULT_CONFIG.copy()
my_config["llm_provider"] = "anthropic"
my_config["deep_think_llm"] = "claude-opus-4-6"
my_config["quick_think_llm"] = "claude-sonnet-4-6"

graph = TradingAgentsGraph(config=my_config)
```

### All Configuration Keys

#### Directory Settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `project_dir` | `str` | Auto-detected | Root directory of the `tradingagents` package. Used for locating data cache. |
| `results_dir` | `str` | `"./results"` | Directory for evaluation results. Can be overridden via `TRADINGAGENTS_RESULTS_DIR` env var. |
| `data_cache_dir` | `str` | `"{project_dir}/dataflows/data_cache"` | Cache directory for downloaded market data. |

#### LLM Settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `llm_provider` | `str` | `"openai"` | LLM provider. Options: `"openai"`, `"anthropic"`, `"google"`, `"xai"`, `"ollama"`, `"openrouter"`. |
| `deep_think_llm` | `str` | `"gpt-5.2"` | Model for Research Manager and Risk Judge (critical decisions). |
| `quick_think_llm` | `str` | `"gpt-5-mini"` | Model for analysts, researchers, trader, debaters, reflector, and signal processor. |
| `backend_url` | `str` | `"https://api.openai.com/v1"` | Base URL for the LLM API. Overridden automatically for xai, ollama, and openrouter. |

#### Provider-Specific Thinking Configuration

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `google_thinking_level` | `str` or `None` | `None` | Thinking level for Gemini models. Gemini 3 Flash: `"minimal"`, `"low"`, `"medium"`, `"high"`. Gemini 3 Pro: `"low"`, `"high"`. Gemini 2.5: mapped to `thinking_budget`. |
| `openai_reasoning_effort` | `str` or `None` | `None` | Reasoning effort for OpenAI models. Options: `"low"`, `"medium"`, `"high"`. |

#### Debate and Discussion Settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `max_debate_rounds` | `int` | `1` | Number of bull/bear debate rounds. Each round = 2 responses (one bull, one bear). Total responses = `2 * max_debate_rounds`. |
| `max_risk_discuss_rounds` | `int` | `1` | Number of risk debate rounds. Each round = 3 responses (aggressive, conservative, neutral). Total responses = `3 * max_risk_discuss_rounds`. |
| `max_recur_limit` | `int` | `100` | LangGraph recursion limit. Increase if using many debate rounds or analysts with heavy tool usage. |

#### Data Vendor Configuration

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `data_vendors` | `dict` | See below | Category-level vendor selection. |
| `tool_vendors` | `dict` | `{}` | Tool-level vendor overrides. Takes precedence over `data_vendors`. |

Default `data_vendors`:

```python
{
    "core_stock_apis": "yfinance",
    "technical_indicators": "yfinance",
    "fundamental_data": "yfinance",
    "news_data": "yfinance",
}
```

---

## LLM Provider Setup

### OpenAI (Default)

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."

config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "openai"
config["deep_think_llm"] = "gpt-5.2"
config["quick_think_llm"] = "gpt-5-mini"
# Optional: config["openai_reasoning_effort"] = "high"

graph = TradingAgentsGraph(config=config)
```

Supported models: `gpt-5.4-pro`, `gpt-5.4`, `gpt-5.2`, `gpt-5.1`, `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`.

### Anthropic

```python
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."

config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "anthropic"
config["deep_think_llm"] = "claude-opus-4-6"
config["quick_think_llm"] = "claude-sonnet-4-6"

graph = TradingAgentsGraph(config=config)
```

Supported models: `claude-opus-4-6`, `claude-sonnet-4-6`, `claude-opus-4-5`, `claude-sonnet-4-5`, `claude-haiku-4-5`.

Note: Anthropic does not support custom `base_url`. The parameter is ignored.

### Google (Gemini)

```python
os.environ["GOOGLE_API_KEY"] = "AI..."

config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "google"
config["deep_think_llm"] = "gemini-2.5-pro"
config["quick_think_llm"] = "gemini-2.5-flash"
# Optional: config["google_thinking_level"] = "high"

graph = TradingAgentsGraph(config=config)
```

Supported models: `gemini-3.1-pro-preview`, `gemini-3.1-flash-lite-preview`, `gemini-3-flash-preview`, `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`.

### xAI (Grok)

```python
os.environ["XAI_API_KEY"] = "xai-..."

config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "xai"
config["deep_think_llm"] = "grok-4-fast-reasoning"
config["quick_think_llm"] = "grok-4-fast-non-reasoning"

graph = TradingAgentsGraph(config=config)
```

Supported models: `grok-4-1-fast-reasoning`, `grok-4-1-fast-non-reasoning`, `grok-4-0709`, `grok-4-fast-reasoning`, `grok-4-fast-non-reasoning`.

The xAI provider uses the OpenAI-compatible client with `base_url` set to `https://api.x.ai/v1`.

### Ollama (Local)

```python
# No API key needed -- Ollama runs locally
config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "ollama"
config["deep_think_llm"] = "llama3.1:70b"
config["quick_think_llm"] = "llama3.1:8b"

graph = TradingAgentsGraph(config=config)
```

Ollama uses the OpenAI-compatible client with `base_url` set to `http://localhost:11434/v1` and `api_key` set to `"ollama"`. Any model name is accepted (no validation).

Make sure your Ollama server is running before creating the graph.

### OpenRouter

```python
os.environ["OPENROUTER_API_KEY"] = "sk-or-..."

config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "openrouter"
config["deep_think_llm"] = "anthropic/claude-sonnet-4-6"
config["quick_think_llm"] = "anthropic/claude-haiku-4-5"

graph = TradingAgentsGraph(config=config)
```

OpenRouter uses the OpenAI-compatible client with `base_url` set to `https://openrouter.ai/api/v1`. Any model name is accepted (no validation). Use OpenRouter's model naming format (e.g., `provider/model-name`).

---

## Data Vendor Configuration

### Switching Vendors

Change all data sources to Alpha Vantage:

```python
config = DEFAULT_CONFIG.copy()
config["data_vendors"] = {
    "core_stock_apis": "alpha_vantage",
    "technical_indicators": "alpha_vantage",
    "fundamental_data": "alpha_vantage",
    "news_data": "alpha_vantage",
}

# Alpha Vantage requires an API key
os.environ["ALPHA_VANTAGE_API_KEY"] = "your-key"

graph = TradingAgentsGraph(config=config)
```

### Override at Tool Level

Use Alpha Vantage for stock data only, yfinance for everything else:

```python
config = DEFAULT_CONFIG.copy()
config["data_vendors"] = {
    "core_stock_apis": "yfinance",       # Category default
    "technical_indicators": "yfinance",
    "fundamental_data": "yfinance",
    "news_data": "yfinance",
}
config["tool_vendors"] = {
    "get_stock_data": "alpha_vantage",   # Overrides core_stock_apis for this tool
}

graph = TradingAgentsGraph(config=config)
```

### Available Tools by Category

| Category | Tool Names |
|----------|-----------|
| `core_stock_apis` | `get_stock_data` |
| `technical_indicators` | `get_indicators` |
| `fundamental_data` | `get_fundamentals`, `get_balance_sheet`, `get_cashflow`, `get_income_statement` |
| `news_data` | `get_news`, `get_global_news`, `get_insider_transactions` |

---

## Memory System API

### FinancialSituationMemory

Located in `tradingagents/agents/utils/memory.py`.

```python
from tradingagents.agents.utils.memory import FinancialSituationMemory
```

#### Constructor

```python
memory = FinancialSituationMemory(name="my_memory", config=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | Required | Identifier for this memory instance (used in log messages). |
| `config` | `dict` | `None` | Kept for API compatibility. Not used by BM25 implementation. |

#### `add_situations(situations_and_advice)`

Store financial situations and their corresponding recommendations.

```python
memory.add_situations([
    ("High inflation with rising rates", "Consider defensive sectors"),
    ("Tech sector volatile with institutional selling", "Reduce growth exposure"),
])
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `situations_and_advice` | `list[tuple[str, str]]` | List of `(situation_description, recommendation)` pairs. |

Rebuilds the BM25 index after adding.

#### `get_memories(current_situation, n_matches=1)`

Retrieve matching recommendations for a given situation.

```python
results = memory.get_memories(
    "Market showing increased tech volatility",
    n_matches=2
)
for r in results:
    print(r["matched_situation"])
    print(r["recommendation"])
    print(r["similarity_score"])  # 0.0 to 1.0 (normalized)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `current_situation` | `str` | Required | The current market situation to match against stored situations. |
| `n_matches` | `int` | `1` | Number of top matches to return. |

**Returns:** `list[dict]` with keys:
- `matched_situation` (str) -- The stored situation that matched.
- `recommendation` (str) -- The recommendation associated with the matched situation.
- `similarity_score` (float) -- Normalized BM25 score between 0 and 1.

Returns an empty list if no documents are stored.

#### `clear()`

Remove all stored memories and reset the BM25 index.

```python
memory.clear()
```

---

## Signal Processing

The `SignalProcessor` (used internally by `TradingAgentsGraph.process_signal()`) extracts a single-word decision from a full trading report.

It sends the full signal text to the quick thinking LLM with a system prompt instructing it to extract only `BUY`, `SELL`, or `HOLD`.

---

## Logging

The framework uses Python's standard `logging` module. Configure it before creating a graph:

```python
from tradingagents import setup_logging
import logging

# Basic setup -- logs to stderr at INFO level
setup_logging()

# Custom setup -- logs to file at DEBUG level
setup_logging(
    level=logging.DEBUG,
    log_file="tradingagents.log",
    format_string="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
```

All framework loggers are prefixed with `tradingagents.` (e.g., `tradingagents.trading_graph`, `tradingagents.memory`, `tradingagents.dataflows.interface`).

Third-party loggers (`httpx`, `httpcore`, `urllib3`, `openai`, `anthropic`) are automatically set to WARNING level to reduce noise.

---

## Complete Example

```python
import os
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents import setup_logging

# Configure logging
setup_logging()

# Set API key
os.environ["OPENAI_API_KEY"] = "sk-..."

# Configure
config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "openai"
config["deep_think_llm"] = "gpt-5-mini"
config["quick_think_llm"] = "gpt-5-mini"
config["max_debate_rounds"] = 2
config["max_risk_discuss_rounds"] = 1

# Create graph with a subset of analysts
graph = TradingAgentsGraph(
    selected_analysts=["market", "news", "fundamentals"],
    config=config,
)

# Run analysis
final_state, signal = graph.propagate("NVDA", "2025-06-15")

print(f"Signal: {signal}")
print(f"Full decision: {final_state['final_trade_decision']}")

# After learning the outcome...
graph.reflect_and_remember("Stock rose 3.5% over the next week")

# Next analysis benefits from updated memory
final_state2, signal2 = graph.propagate("NVDA", "2025-06-22")
```
