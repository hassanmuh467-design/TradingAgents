# Contributing to TradingAgents

Thank you for your interest in contributing to TradingAgents! This guide covers development setup, code standards, and how to extend the framework.

## Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/TradingAgents-AI/TradingAgents.git
cd TradingAgents
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows
```

Python 3.10 or later is required.

### 3. Install with Dev Dependencies

```bash
pip install -e ".[dev]"
```

This installs the package in editable mode along with development tools:
- `pytest` -- test runner
- `pytest-cov` -- coverage reporting
- `ruff` -- linter and formatter

### 4. Set Up Environment Variables

Set the API key for your chosen LLM provider:

```bash
export OPENAI_API_KEY="sk-..."
# or
export ANTHROPIC_API_KEY="sk-ant-..."
# or
export GOOGLE_API_KEY="AI..."
```

If you plan to use Alpha Vantage for data:
```bash
export ALPHA_VANTAGE_API_KEY="your-key"
```

---

## Running Tests

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run a specific test module
pytest tests/test_llm_clients/ -v
pytest tests/test_graph/test_conditional_logic.py -v

# Run tests matching a keyword
pytest tests/ -v -k "test_factory"

# Run with coverage report
pytest tests/ --cov=tradingagents --cov-report=term-missing

# Run with coverage and generate HTML report
pytest tests/ --cov=tradingagents --cov-report=html
```

All tests must pass before submitting a PR. The test suite uses mocks for LLM calls and external APIs so no API keys are needed to run tests.

---

## Code Style

### Linting with Ruff

The project uses [ruff](https://docs.astral.sh/ruff/) for linting, configured in `pyproject.toml`:

```bash
# Check for lint issues
ruff check .

# Auto-fix issues where possible
ruff check --fix .
```

Configuration (from `pyproject.toml`):
- Target: Python 3.10
- Line length: 100 characters
- Enabled rules: `E` (pycodestyle errors), `F` (pyflakes), `I` (import sorting), `W` (warnings)
- `E501` (line too long) is ignored -- the 100-char limit is advisory

### Type Hints

Type hints are encouraged for all public APIs. Internal helper functions can omit them if the types are obvious. Use `typing` module imports:

```python
from typing import Dict, Any, List, Optional, Tuple

def my_function(ticker: str, config: Dict[str, Any]) -> Tuple[str, float]:
    ...
```

### Docstrings

Use triple-quoted docstrings for all public classes and functions. Follow the Google style with `Args:` and `Returns:` sections:

```python
def create_my_analyst(llm):
    """Create an analyst node for custom data analysis.

    Args:
        llm: LangChain chat model instance.

    Returns:
        A callable node function for use in the LangGraph StateGraph.
    """
```

### Imports

Organize imports in three groups separated by blank lines: standard library, third-party, local. Ruff's `I` rule enforces this automatically.

---

## Pull Request Process

1. **Fork** the repository and create a feature branch from `main`:
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Make changes** with clear, descriptive commits. Each commit should represent a logical unit of work.

3. **Add tests** for new functionality. Place tests in the corresponding `tests/` subdirectory:
   - `tests/test_llm_clients/` for LLM client changes
   - `tests/test_graph/` for graph logic changes
   - `tests/test_agents/` for agent changes
   - `tests/test_dataflows/` for data vendor changes

4. **Ensure all tests pass**:
   ```bash
   pytest tests/ -v
   ```

5. **Ensure linting passes**:
   ```bash
   ruff check .
   ```

6. **Submit a PR** with a clear description including:
   - What the change does and why
   - Any breaking changes
   - How to test the change

---

## Architecture Overview

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full system design document.

The key components at a glance:

| Directory | Purpose |
|-----------|---------|
| `tradingagents/agents/analysts/` | Data-gathering analyst agents (market, social, news, fundamentals) |
| `tradingagents/agents/researchers/` | Bull/Bear investment debate agents |
| `tradingagents/agents/managers/` | Decision-making judges (Research Manager, Risk Manager) |
| `tradingagents/agents/risk_mgmt/` | Risk debate agents (aggressive, conservative, neutral) |
| `tradingagents/agents/trader/` | Trader agent that formulates investment plans |
| `tradingagents/agents/utils/` | Shared utilities: state definitions, memory, tool wrappers |
| `tradingagents/graph/` | LangGraph orchestration: graph setup, conditional logic, propagation, reflection, signal processing |
| `tradingagents/dataflows/` | Data vendor abstraction layer with routing and fallback |
| `tradingagents/llm_clients/` | LLM provider abstraction with factory pattern |

The pipeline flow is:

```
Analysts -> Bull/Bear Debate -> Research Manager -> Trader -> Risk Debate -> Risk Judge
```

---

## Adding New Analyst Types

To add a new analyst (e.g., a "technical patterns" analyst):

### Step 1: Create the Analyst Module

Create `tradingagents/agents/analysts/my_analyst.py`:

```python
from tradingagents.agents.utils.agent_states import AgentState


def create_my_analyst(llm):
    """Create a custom analyst node.

    Args:
        llm: LangChain chat model with tool-calling support.

    Returns:
        Callable node for the LangGraph StateGraph.
    """
    # Bind tools the analyst will use
    tools = [...]  # LangChain tool functions
    llm_with_tools = llm.bind_tools(tools)

    def my_analyst_node(state: AgentState):
        # Build prompt with state context
        company = state["company_of_interest"]
        trade_date = state["trade_date"]

        messages = [
            ("system", f"You are a ... analyst. Analyze {company} as of {trade_date}."),
            ("human", company),
        ]

        response = llm_with_tools.invoke(state["messages"])

        return {
            "messages": [response],
            "my_custom_report": response.content,
            "sender": "My Analyst",
        }

    return my_analyst_node
```

### Step 2: Add State Field

In `tradingagents/agents/utils/agent_states.py`, add a field to `AgentState`:

```python
class AgentState(MessagesState):
    # ... existing fields ...
    my_custom_report: Annotated[str, "Report from the Custom Analyst"]
```

### Step 3: Export the Factory Function

In `tradingagents/agents/__init__.py`, add:

```python
from .analysts.my_analyst import create_my_analyst
```

And add `"create_my_analyst"` to the `__all__` list.

### Step 4: Register Tools and Conditional Logic

In `tradingagents/graph/trading_graph.py`, add tool nodes in `_create_tool_nodes()`:

```python
"my_custom": ToolNode([my_tool_1, my_tool_2]),
```

In `tradingagents/graph/conditional_logic.py`, add:

```python
def should_continue_my_custom(self, state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools_my_custom"
    return "Msg Clear My_custom"
```

### Step 5: Wire Into the Graph

In `tradingagents/graph/setup.py`, add the analyst to the `setup_graph()` method following the pattern of existing analysts (add node, delete node, tool node, conditional edges).

### Step 6: Add Tests

Create `tests/test_agents/test_my_analyst.py` with unit tests for the new analyst.

---

## Adding New LLM Providers

### Step 1: Create a Client Class

Create `tradingagents/llm_clients/my_provider_client.py`:

```python
import logging
from typing import Any, Optional

from .base_client import BaseLLMClient
from .validators import validate_model

logger = logging.getLogger(__name__)


class MyProviderClient(BaseLLMClient):
    """Client for MyProvider models."""

    def get_llm(self) -> Any:
        """Return configured LangChain chat model instance."""
        if not self.validate_model():
            logger.warning(
                f"Model '{self.model}' may not be supported. Proceeding anyway."
            )

        # Use an existing LangChain chat class or create a custom one.
        # The returned object must support .invoke() and .bind_tools().
        llm_kwargs = {"model": self.model}

        # Forward relevant kwargs
        for key in ("timeout", "max_retries", "api_key", "callbacks"):
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        return SomeLangChainChatClass(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for this provider."""
        return validate_model("my_provider", self.model)
```

**Requirements for the returned LLM object**:
- Must implement `.invoke(messages)` returning a message with `.content` attribute.
- Must implement `.bind_tools(tools)` for analyst tool-calling.
- Should be a LangChain `BaseChatModel` subclass for full compatibility.

### Step 2: Register in the Factory

In `tradingagents/llm_clients/factory.py`:

```python
from .my_provider_client import MyProviderClient

def create_llm_client(provider, model, base_url=None, **kwargs):
    # ... existing providers ...

    if provider_lower == "my_provider":
        return MyProviderClient(model, base_url, **kwargs)

    raise ValueError(f"Unsupported LLM provider: {provider}")
```

### Step 3: Add Validated Models

In `tradingagents/llm_clients/validators.py`, add to `VALID_MODELS`:

```python
VALID_MODELS = {
    # ... existing ...
    "my_provider": [
        "model-a",
        "model-b",
    ],
}
```

### Step 4: Add Tests

Create `tests/test_llm_clients/test_my_provider_client.py`. Follow the pattern in existing test files (mock the LangChain class, test `get_llm()` and `validate_model()`).

---

## Adding New Data Vendors

### Step 1: Create Vendor Implementation

Create `tradingagents/dataflows/my_vendor.py` implementing the tool functions:

```python
def get_stock_data(ticker: str, start_date: str, end_date: str) -> str:
    """Fetch OHLCV data from MyVendor."""
    # Return formatted string of stock data
    ...

def get_indicators(ticker: str, start_date: str, end_date: str) -> str:
    """Fetch technical indicators from MyVendor."""
    ...

def get_fundamentals(ticker: str) -> str:
    """Fetch company fundamentals from MyVendor."""
    ...

def get_balance_sheet(ticker: str) -> str:
    ...

def get_cashflow(ticker: str) -> str:
    ...

def get_income_statement(ticker: str) -> str:
    ...

def get_news(ticker: str) -> str:
    ...

def get_global_news() -> str:
    ...

def get_insider_transactions(ticker: str) -> str:
    ...
```

Each function should return a formatted string suitable for LLM consumption. Match the signatures of existing vendor implementations.

### Step 2: Register in the Interface

In `tradingagents/dataflows/interface.py`:

1. Import your implementations:
   ```python
   from .my_vendor import (
       get_stock_data as get_my_vendor_stock,
       get_indicators as get_my_vendor_indicators,
       # ... etc
   )
   ```

2. Add to `VENDOR_LIST`:
   ```python
   VENDOR_LIST = ["yfinance", "alpha_vantage", "my_vendor"]
   ```

3. Add to each entry in `VENDOR_METHODS`:
   ```python
   "get_stock_data": {
       "alpha_vantage": get_alpha_vantage_stock,
       "yfinance": get_YFin_data_online,
       "my_vendor": get_my_vendor_stock,  # Add here
   },
   ```

### Step 3: Update Default Config

In `tradingagents/default_config.py`, add your vendor as an option comment:

```python
"data_vendors": {
    "core_stock_apis": "yfinance",       # Options: alpha_vantage, yfinance, my_vendor
    ...
},
```

### Step 4: Handle Rate Limiting (if applicable)

If your vendor has rate limits, create a custom exception and handle it in `route_to_vendor()` alongside `AlphaVantageRateLimitError` so that rate limit failures trigger fallback to the next vendor.

### Step 5: Add Tests

Create `tests/test_dataflows/test_my_vendor.py` with tests for each tool function. Use mocks for external API calls.

---

## License

By contributing, you agree that your contributions will be licensed under the project's existing license.
