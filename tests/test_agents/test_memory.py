"""Tests for tradingagents.agents.utils.memory.FinancialSituationMemory."""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from tradingagents.agents.utils.memory import FinancialSituationMemory


@pytest.fixture
def memory():
    """Create a fresh FinancialSituationMemory instance."""
    return FinancialSituationMemory("test_memory")


@pytest.fixture
def populated_memory():
    """Create a memory with sample data pre-loaded, with BM25 properly mocked."""
    mem = FinancialSituationMemory("test_memory")
    data = [
        (
            "High inflation rate with rising interest rates and declining consumer spending",
            "Consider defensive sectors like consumer staples and utilities.",
        ),
        (
            "Tech sector showing high volatility with increasing institutional selling pressure",
            "Reduce exposure to high-growth tech stocks.",
        ),
        (
            "Strong dollar affecting emerging markets with increasing forex volatility",
            "Hedge currency exposure in international positions.",
        ),
        (
            "Market showing signs of sector rotation with rising yields",
            "Rebalance portfolio to maintain target allocations.",
        ),
    ]

    # Mock BM25Okapi to return predictable scores
    mock_bm25 = MagicMock()
    with patch('tradingagents.agents.utils.memory.BM25Okapi', return_value=mock_bm25):
        mem.add_situations(data)
    # Replace the bm25 instance with our controllable mock
    mem.bm25 = mock_bm25
    return mem


class TestConstructor:
    def test_initializes_with_name(self):
        mem = FinancialSituationMemory("bull_memory")
        assert mem.name == "bull_memory"

    def test_initializes_empty_documents(self):
        mem = FinancialSituationMemory("test")
        assert mem.documents == []

    def test_initializes_empty_recommendations(self):
        mem = FinancialSituationMemory("test")
        assert mem.recommendations == []

    def test_initializes_bm25_none(self):
        mem = FinancialSituationMemory("test")
        assert mem.bm25 is None

    def test_accepts_config_parameter(self):
        """Config parameter exists for API compatibility."""
        mem = FinancialSituationMemory("test", config={"key": "value"})
        assert mem.name == "test"


class TestAddSituations:
    def test_adds_documents(self, memory):
        memory.add_situations([("situation1", "advice1")])
        assert len(memory.documents) == 1
        assert memory.documents[0] == "situation1"

    def test_adds_recommendations(self, memory):
        memory.add_situations([("situation1", "advice1")])
        assert len(memory.recommendations) == 1
        assert memory.recommendations[0] == "advice1"

    def test_adds_multiple_entries(self, memory):
        memory.add_situations([
            ("situation1", "advice1"),
            ("situation2", "advice2"),
        ])
        assert len(memory.documents) == 2
        assert len(memory.recommendations) == 2

    def test_rebuilds_bm25_index(self, memory):
        assert memory.bm25 is None
        memory.add_situations([("some text", "some advice")])
        assert memory.bm25 is not None

    def test_accumulates_across_calls(self, memory):
        memory.add_situations([("situation1", "advice1")])
        memory.add_situations([("situation2", "advice2")])
        assert len(memory.documents) == 2
        assert len(memory.recommendations) == 2


class TestTokenize:
    def test_returns_lowercase_tokens(self, memory):
        tokens = memory._tokenize("Hello World TEST")
        assert all(t == t.lower() for t in tokens)

    def test_splits_on_whitespace(self, memory):
        tokens = memory._tokenize("hello world")
        assert tokens == ["hello", "world"]

    def test_handles_punctuation(self, memory):
        tokens = memory._tokenize("hello, world! test.")
        assert tokens == ["hello", "world", "test"]

    def test_handles_empty_string(self, memory):
        tokens = memory._tokenize("")
        assert tokens == []

    def test_handles_numbers(self, memory):
        tokens = memory._tokenize("price is 100 dollars")
        assert "100" in tokens


class TestGetMemories:
    def test_returns_empty_list_when_no_memories(self, memory):
        result = memory.get_memories("some query")
        assert result == []

    def test_returns_correct_matches_after_adding(self, populated_memory):
        # Configure mock to return scores where index 1 (tech sector) is highest
        populated_memory.bm25.get_scores.return_value = [1.0, 5.0, 0.5, 0.2]
        result = populated_memory.get_memories(
            "tech sector volatility and institutional selling"
        )
        assert len(result) == 1
        assert "tech" in result[0]["matched_situation"].lower()

    def test_returns_dict_with_expected_keys(self, populated_memory):
        # Configure mock to return scores where index 0 (inflation) is highest
        populated_memory.bm25.get_scores.return_value = [5.0, 1.0, 0.5, 0.2]
        result = populated_memory.get_memories("inflation and interest rates")
        assert len(result) == 1
        entry = result[0]
        assert "matched_situation" in entry
        assert "recommendation" in entry
        assert "similarity_score" in entry

    def test_n_matches_returns_requested_number(self, populated_memory):
        populated_memory.bm25.get_scores.return_value = [3.0, 2.0, 1.0, 0.5]
        result = populated_memory.get_memories(
            "market conditions", n_matches=3
        )
        assert len(result) == 3

    def test_n_matches_capped_at_total_documents(self, populated_memory):
        populated_memory.bm25.get_scores.return_value = [3.0, 2.0, 1.0, 0.5]
        result = populated_memory.get_memories(
            "market conditions", n_matches=100
        )
        assert len(result) == 4  # Only 4 documents exist

    def test_similarity_scores_between_0_and_1(self, populated_memory):
        populated_memory.bm25.get_scores.return_value = [4.0, 3.0, 2.0, 1.0]
        result = populated_memory.get_memories(
            "tech sector with high volatility", n_matches=4
        )
        for entry in result:
            assert 0.0 <= entry["similarity_score"] <= 1.0

    def test_best_match_has_highest_score(self, populated_memory):
        populated_memory.bm25.get_scores.return_value = [4.0, 3.0, 2.0, 1.0]
        result = populated_memory.get_memories(
            "inflation and rising interest rates", n_matches=4
        )
        scores = [r["similarity_score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_recommendation_matches_document(self, populated_memory):
        # Index 2 is "Strong dollar affecting emerging markets..."
        # with recommendation "Hedge currency exposure..."
        populated_memory.bm25.get_scores.return_value = [0.1, 0.2, 5.0, 0.3]
        result = populated_memory.get_memories(
            "emerging markets forex dollar", n_matches=1
        )
        assert len(result) == 1
        assert "Hedge currency" in result[0]["recommendation"]


class TestClear:
    def test_removes_all_documents(self, populated_memory):
        populated_memory.clear()
        assert populated_memory.documents == []

    def test_removes_all_recommendations(self, populated_memory):
        populated_memory.clear()
        assert populated_memory.recommendations == []

    def test_resets_bm25(self, populated_memory):
        populated_memory.clear()
        assert populated_memory.bm25 is None

    def test_get_memories_returns_empty_after_clear(self, populated_memory):
        populated_memory.clear()
        result = populated_memory.get_memories("test query")
        assert result == []

    def test_can_add_after_clear(self, populated_memory):
        populated_memory.clear()
        populated_memory.add_situations([("new situation", "new advice")])
        assert len(populated_memory.documents) == 1
        # Configure the mock for the new bm25 instance
        populated_memory.bm25.get_scores.return_value = [5.0]
        result = populated_memory.get_memories("new situation")
        assert len(result) == 1
