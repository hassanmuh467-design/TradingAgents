"""Tests for tradingagents.llm_clients.validators."""

import pytest

from tradingagents.llm_clients.validators import validate_model, VALID_MODELS


class TestValidateModel:
    """Tests for the validate_model function."""

    # --- OpenAI models ---

    def test_valid_openai_model(self):
        assert validate_model("openai", "gpt-5-mini") is True

    def test_valid_openai_gpt5_model(self):
        assert validate_model("openai", "gpt-5") is True

    def test_valid_openai_gpt41_model(self):
        assert validate_model("openai", "gpt-4.1") is True

    def test_invalid_openai_model(self):
        assert validate_model("openai", "gpt-3.5-turbo") is False

    def test_invalid_openai_nonexistent_model(self):
        assert validate_model("openai", "not-a-real-model") is False

    # --- Anthropic models ---

    def test_valid_anthropic_model(self):
        assert validate_model("anthropic", "claude-sonnet-4-5") is True

    def test_valid_anthropic_opus_model(self):
        assert validate_model("anthropic", "claude-opus-4-6") is True

    def test_invalid_anthropic_model(self):
        assert validate_model("anthropic", "claude-2") is False

    # --- Google models ---

    def test_valid_google_gemini_model(self):
        assert validate_model("google", "gemini-2.5-flash") is True

    def test_valid_google_gemini3_model(self):
        assert validate_model("google", "gemini-3-flash-preview") is True

    def test_invalid_google_model(self):
        assert validate_model("google", "gemini-1.0-pro") is False

    # --- xAI models ---

    def test_valid_xai_model(self):
        assert validate_model("xai", "grok-4-fast-reasoning") is True

    def test_invalid_xai_model(self):
        assert validate_model("xai", "grok-1") is False

    # --- Ollama: any model accepted ---

    def test_ollama_always_valid(self):
        assert validate_model("ollama", "llama3") is True

    def test_ollama_arbitrary_model_valid(self):
        assert validate_model("ollama", "any-arbitrary-model-name") is True

    # --- OpenRouter: any model accepted ---

    def test_openrouter_always_valid(self):
        assert validate_model("openrouter", "meta-llama/llama-3-70b") is True

    def test_openrouter_arbitrary_model_valid(self):
        assert validate_model("openrouter", "xyz/custom-model") is True

    # --- Unknown provider returns True ---

    def test_unknown_provider_returns_true(self):
        assert validate_model("unknown_provider", "any-model") is True

    # --- Case sensitivity ---

    def test_provider_is_case_insensitive(self):
        assert validate_model("OpenAI", "gpt-5-mini") is True
        assert validate_model("OLLAMA", "anything") is True

    # --- All valid models in VALID_MODELS pass ---

    @pytest.mark.parametrize("provider", ["openai", "anthropic", "google", "xai"])
    def test_all_listed_models_are_valid(self, provider):
        for model in VALID_MODELS[provider]:
            assert validate_model(provider, model) is True, (
                f"{model} should be valid for {provider}"
            )
