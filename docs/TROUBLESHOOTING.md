# TradingAgents Troubleshooting Guide

## Common Errors and Solutions

### API Key Issues

#### `openai.AuthenticationError: Incorrect API key provided`

**Cause**: The `OPENAI_API_KEY` environment variable is missing or invalid.

**Solution**:
```bash
export OPENAI_API_KEY="sk-..."
```

Verify it is set:
```bash
python -c "import os; print(os.environ.get('OPENAI_API_KEY', 'NOT SET')[:10])"
```

#### `anthropic.AuthenticationError` or `google.auth` errors

Each provider reads its own environment variable:

| Provider | Environment Variable |
|----------|---------------------|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Google | `GOOGLE_API_KEY` |
| xAI | `XAI_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |
| Ollama | None (no auth required) |

Make sure you have set the correct variable for your configured `llm_provider`.

#### `Alpha Vantage: Invalid API Key`

If using Alpha Vantage as a data vendor:
```bash
export ALPHA_VANTAGE_API_KEY="your-key"
```

The free tier allows 5 API calls per minute and 500 per day. If you hit the limit, the framework will automatically fall back to yfinance (see [Data Vendor Fallback Behavior](#data-vendor-fallback-behavior)).

### Rate Limiting

#### `openai.RateLimitError: Rate limit reached`

**Cause**: Too many requests or tokens per minute for your OpenAI tier.

**Solutions**:
1. Reduce `max_debate_rounds` and `max_risk_discuss_rounds` to lower total LLM calls.
2. Use a smaller model for `quick_think_llm` (e.g., `gpt-5-nano`).
3. Reduce the number of analysts: `selected_analysts=["market", "fundamentals"]`.
4. Upgrade your OpenAI API tier for higher rate limits.

#### `AlphaVantageRateLimitError`

**Cause**: Alpha Vantage free tier limits (5 calls/min).

**Behavior**: The framework handles this automatically. When a rate limit is hit, the vendor router falls back to the next available vendor (typically yfinance). No action needed unless you need Alpha Vantage-exclusive data.

**Solution** (if fallback is not acceptable): Get a premium Alpha Vantage API key, or add delays between `propagate()` calls.

### Model Not Found

#### `ValueError: Unsupported LLM provider: <name>`

**Cause**: The `llm_provider` in your config is not one of the six supported values.

**Solution**: Use one of: `"openai"`, `"anthropic"`, `"google"`, `"xai"`, `"ollama"`, `"openrouter"`.

#### Warning: `Model '<name>' may not be supported by <provider>. Proceeding anyway.`

**Cause**: The model name does not appear in the validators' known-model list. This is a warning, not an error -- the framework will still attempt to use the model.

**Solution**: Check `tradingagents/llm_clients/validators.py` for the list of validated model names. For Ollama and OpenRouter, any model name is accepted without validation.

### Recursion Limit

#### `RecursionError` or `langgraph.errors.GraphRecursionError`

**Cause**: The graph exceeded the `max_recur_limit` (default: 100). This can happen when analysts make many tool calls or debate rounds are set high.

**Solution**: Increase the limit in your config:
```python
config["max_recur_limit"] = 200
```

### No Analysts Selected

#### `ValueError: Trading Agents Graph Setup Error: no analysts selected!`

**Cause**: An empty list was passed to `selected_analysts`.

**Solution**: Provide at least one analyst type:
```python
graph = TradingAgentsGraph(selected_analysts=["market"])
```

---

## Provider-Specific Issues

### OpenAI / GPT-5: Temperature Stripping

**Issue**: GPT-5 family models use native reasoning and reject `temperature` and `top_p` parameters (unless `reasoning.effort` is set to `"none"`). LangChain defaults `temperature=0.7`, which would cause API errors.

**How it is handled**: The `UnifiedChatOpenAI` subclass automatically strips `temperature` and `top_p` for any model containing `"gpt-5"` in its name. No user action needed.

**If you see temperature-related errors**: Verify you are not manually constructing a `ChatOpenAI` instance with temperature set. Always use the framework's `create_llm_client()` factory or the `TradingAgentsGraph` class.

### Google / Gemini: Content Normalization

**Issue**: Gemini 3 models return message content as a list of dicts (`[{"type": "text", "text": "..."}]`) instead of a plain string. Downstream agents that expect `str` content will fail.

**How it is handled**: The `NormalizedChatGoogleGenerativeAI` subclass intercepts the `invoke()` response and normalizes list content to a joined string. No user action needed.

**If you see `TypeError` on response content**: Make sure you are using the framework's Google client, not constructing `ChatGoogleGenerativeAI` directly.

### Google / Gemini: Thinking Configuration

The thinking config API differs between Gemini generations:

| Model Family | Parameter | Values |
|-------------|-----------|--------|
| Gemini 3 Flash | `thinking_level` | `"minimal"`, `"low"`, `"medium"`, `"high"` |
| Gemini 3 Pro | `thinking_level` | `"low"`, `"high"` (note: `"minimal"` is auto-mapped to `"low"`) |
| Gemini 2.5 | `thinking_budget` | `-1` (dynamic/high), `0` (disabled) |

Set via config:
```python
config["google_thinking_level"] = "high"
```

The `GoogleClient` handles the mapping automatically based on the model name.

### Anthropic: No base_url Support

**Issue**: Passing `base_url` to the Anthropic provider has no effect.

**How it is handled**: The `AnthropicClient` logs a warning and ignores the parameter. Anthropic's API endpoint is fixed.

### xAI / Ollama / OpenRouter: Base URL Handling

These providers override the `backend_url` config automatically:

| Provider | Forced base_url |
|----------|----------------|
| xAI | `https://api.x.ai/v1` |
| Ollama | `http://localhost:11434/v1` |
| OpenRouter | `https://openrouter.ai/api/v1` |

Setting `backend_url` in the config has no effect for these providers.

### Ollama: Connection Refused

**Issue**: `ConnectionError: Connection refused` when using the Ollama provider.

**Solution**: Make sure Ollama is running:
```bash
ollama serve
```

Verify it is accessible:
```bash
curl http://localhost:11434/v1/models
```

Also ensure the model you specified is pulled:
```bash
ollama pull llama3.1:8b
```

---

## Data Vendor Fallback Behavior

The vendor routing system in `tradingagents/dataflows/interface.py` implements automatic fallback:

1. The configured vendor (from `tool_vendors` or `data_vendors`) is tried first.
2. If it raises `AlphaVantageRateLimitError`, the next available vendor is tried.
3. The fallback chain includes all vendors that implement the requested method.
4. If all vendors fail, a `RuntimeError` is raised.

**Important**: Only `AlphaVantageRateLimitError` triggers fallback. Other exceptions (network errors, invalid data, etc.) are raised immediately without trying alternative vendors.

### yfinance Data Issues

**Issue**: Empty or stale data returned.

**Solutions**:
- Verify the ticker symbol is valid and listed on a supported exchange.
- yfinance may have delays for real-time data; use slightly older dates.
- Check your internet connection.
- Try using Alpha Vantage as an alternative: `config["data_vendors"]["core_stock_apis"] = "alpha_vantage"`.

### Checking Which Vendor Was Used

Enable DEBUG logging to see routing decisions:
```python
from tradingagents import setup_logging
import logging
setup_logging(level=logging.DEBUG)
```

You will see log messages like:
```
INFO: Routing method=get_stock_data to vendor(s)=['alpha_vantage'] (category=core_stock_apis)
WARNING: AlphaVantage rate limit hit for method=get_stock_data, falling back to next vendor
INFO: Vendor yfinance succeeded for method=get_stock_data
```

---

## Memory System Issues

### Empty Memory Results

**Symptom**: `get_memories()` returns an empty list.

**Cause**: No situations have been added yet. Memory is populated only after calling `reflect_and_remember()`.

**Solution**: The first `propagate()` call always runs without memory context. Memory accumulates over successive runs with reflection:

```python
state1, signal1 = graph.propagate("AAPL", "2025-01-15")
# ... observe outcome ...
graph.reflect_and_remember("3% gain")  # Now memory has data

state2, signal2 = graph.propagate("AAPL", "2025-01-22")  # Uses memory
```

### Memory Not Persisting Between Sessions

**By design**: The BM25-based memory is in-process only. It does not persist to disk. When the Python process exits, all memory is lost.

**Workaround**: If you need persistence, serialize the memory's internal lists:
```python
import json

# Save
data = {
    "documents": memory.documents,
    "recommendations": memory.recommendations,
}
with open("memory_backup.json", "w") as f:
    json.dump(data, f)

# Restore
with open("memory_backup.json") as f:
    data = json.load(f)
memory.documents = data["documents"]
memory.recommendations = data["recommendations"]
memory._rebuild_index()
```

### Memory Growing Too Large

If you run many reflection cycles, memory can grow unbounded.

**Solution**: Periodically clear and re-seed memory:
```python
memory.clear()
```

Or manually trim the oldest entries:
```python
memory.documents = memory.documents[-100:]  # Keep last 100
memory.recommendations = memory.recommendations[-100:]
memory._rebuild_index()
```

---

## Debug Mode

Enable debug mode to trace the full agent execution:

```python
graph = TradingAgentsGraph(debug=True, config=config)
final_state, signal = graph.propagate("AAPL", "2025-01-15")
```

In debug mode:
- The graph uses `stream()` instead of `invoke()`, yielding intermediate states.
- Each chunk's last message is pretty-printed to stdout.
- All chunks are collected in a trace list.

This is useful for:
- Seeing which tools each analyst calls.
- Reading the full debate transcript as it unfolds.
- Identifying where the pipeline produces unexpected results.

---

## Logging

### Setup

```python
from tradingagents import setup_logging
import logging

# Minimal -- INFO level to stderr
setup_logging()

# Verbose -- DEBUG level to file
setup_logging(level=logging.DEBUG, log_file="trading.log")
```

### Logger Names

All framework loggers use the `tradingagents.*` namespace:

| Logger Name | Component |
|-------------|-----------|
| `tradingagents.trading_graph` | Main graph orchestration |
| `tradingagents.memory` | Memory add/retrieve operations |
| `tradingagents.reflection` | Post-trade reflection |
| `tradingagents.signal_processing` | BUY/SELL/HOLD extraction |
| `tradingagents.dataflows.interface` | Vendor routing decisions |
| `tradingagents.dataflows.config` | Config initialization/updates |

### Checking Logs

With file logging enabled:
```bash
# Follow logs in real time
tail -f trading.log

# Search for routing decisions
grep "Routing method" trading.log

# Search for vendor fallbacks
grep "rate limit" trading.log

# Search for memory operations
grep "memory" trading.log
```

### Reducing Log Noise

Third-party loggers (`httpx`, `httpcore`, `urllib3`, `openai`, `anthropic`) are automatically set to WARNING level by `setup_logging()`. If you still see too much output, set them to ERROR:

```python
import logging
logging.getLogger("httpx").setLevel(logging.ERROR)
```

---

## Getting Help

- Check the [GitHub Issues](https://github.com/TradingAgents-AI/TradingAgents/issues)
- Review [docs/ARCHITECTURE.md](./ARCHITECTURE.md) for system design understanding
- Review [docs/API.md](./API.md) for API reference
