"""Tests for tradingagents.dataflows.config module."""

import pytest
from unittest.mock import patch
import importlib

import tradingagents.dataflows.config as config_module
from tradingagents.default_config import DEFAULT_CONFIG


@pytest.fixture(autouse=True)
def reset_config():
    """Reset the module-level _config before each test."""
    config_module._config = None
    config_module.initialize_config()
    yield
    config_module._config = None


class TestInitializeConfig:
    def test_sets_default_config(self):
        config_module._config = None
        config_module.initialize_config()
        assert config_module._config is not None
        assert config_module._config["llm_provider"] == DEFAULT_CONFIG["llm_provider"]

    def test_does_not_overwrite_existing_config(self):
        config_module._config = {"custom": "value"}
        config_module.initialize_config()
        assert config_module._config == {"custom": "value"}

    def test_idempotent_when_already_initialized(self):
        config_module.initialize_config()
        first = config_module._config
        config_module.initialize_config()
        assert config_module._config is first


class TestSetConfig:
    def test_updates_config_values(self):
        config_module.set_config({"llm_provider": "anthropic"})
        assert config_module._config["llm_provider"] == "anthropic"

    def test_preserves_existing_values_when_updating(self):
        original_provider = config_module._config["llm_provider"]
        config_module.set_config({"max_debate_rounds": 5})
        assert config_module._config["llm_provider"] == original_provider
        assert config_module._config["max_debate_rounds"] == 5

    def test_initializes_config_if_none(self):
        config_module._config = None
        config_module.set_config({"llm_provider": "google"})
        # Should have initialized from DEFAULT_CONFIG first, then updated
        assert config_module._config["llm_provider"] == "google"
        # Should also have other default keys
        assert "max_debate_rounds" in config_module._config

    def test_can_add_new_keys(self):
        config_module.set_config({"new_custom_key": "custom_value"})
        assert config_module._config["new_custom_key"] == "custom_value"

    def test_overwrites_nested_dicts(self):
        """dict.update replaces top-level keys, not deep merge."""
        new_vendors = {"core_stock_apis": "alpha_vantage"}
        config_module.set_config({"data_vendors": new_vendors})
        assert config_module._config["data_vendors"] == new_vendors


class TestGetConfig:
    def test_returns_dict(self):
        result = config_module.get_config()
        assert isinstance(result, dict)

    def test_returns_copy_not_reference(self):
        result = config_module.get_config()
        result["llm_provider"] = "mutated"
        # Original should be unchanged
        assert config_module._config["llm_provider"] != "mutated"

    def test_returns_all_default_keys(self):
        result = config_module.get_config()
        for key in DEFAULT_CONFIG:
            assert key in result

    def test_initializes_if_needed(self):
        config_module._config = None
        result = config_module.get_config()
        assert isinstance(result, dict)
        assert "llm_provider" in result

    def test_reflects_set_config_changes(self):
        config_module.set_config({"llm_provider": "google"})
        result = config_module.get_config()
        assert result["llm_provider"] == "google"
