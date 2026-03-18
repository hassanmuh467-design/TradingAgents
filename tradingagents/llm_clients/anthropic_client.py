import logging
from typing import Any, Optional

from langchain_anthropic import ChatAnthropic

from .base_client import BaseLLMClient
from .validators import validate_model

logger = logging.getLogger(__name__)


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic Claude models."""

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        if base_url is not None:
            logger.warning("AnthropicClient does not support 'base_url'. The parameter will be ignored.")
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return configured ChatAnthropic instance."""
        if not self.validate_model():
            logger.warning(f"Model '{self.model}' may not be supported by Anthropic. Proceeding anyway.")

        llm_kwargs = {"model": self.model}

        for key in ("timeout", "max_retries", "api_key", "max_tokens", "callbacks", "http_client", "http_async_client"):
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        return ChatAnthropic(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for Anthropic."""
        return validate_model("anthropic", self.model)
