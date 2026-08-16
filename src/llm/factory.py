"""Factory for creating configured LLM provider instances."""

import logging
from src.config import settings
from src.llm.antigravity_llm import AntigravityLLM
from src.llm.base import LLMProvider
from src.llm.mock_llm import MockLLM
from src.llm.ollama_llm import OllamaLLM

logger = logging.getLogger(__name__)


def get_llm_provider(
    provider_name: str | None = None,
    **kwargs,
) -> LLMProvider:
    """Create and return an LLMProvider instance based on configuration.

    Args:
        provider_name: Explicit provider name ('antigravity', 'ollama', 'mock').
                       Defaults to settings.LLM_PROVIDER.
        **kwargs: Additional parameters passed to provider constructor.

    Returns:
        Configured LLMProvider instance.
    """
    selected = (provider_name or getattr(settings, "LLM_PROVIDER", "antigravity")).lower().strip()

    if selected == "ollama":
        base_url = kwargs.get("base_url", getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434"))
        model = kwargs.get("model", getattr(settings, "OLLAMA_MODEL", "llama3.2"))
        logger.info("Initializing Ollama LLM provider (model=%s, url=%s)", model, base_url)
        return OllamaLLM(base_url=base_url, model=model)

    elif selected == "mock":
        logger.info("Initializing Mock LLM provider")
        return MockLLM(**kwargs)

    else:
        # Default: Antigravity CLI
        binary_path = kwargs.get("binary_path", getattr(settings, "AGY_PATH", None))
        model = kwargs.get("model", getattr(settings, "LLM_MODEL", None))
        effort = kwargs.get("effort", getattr(settings, "LLM_EFFORT", None))
        try:
            return AntigravityLLM(
                binary_path=binary_path,
                model=model,
                effort=effort,
            )
        except Exception as err:
            logger.warning("Antigravity CLI unavailable (%s); falling back to MockLLM.", err)
            return MockLLM()


__all__ = ["get_llm_provider"]
