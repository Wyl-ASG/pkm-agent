"""Factory for creating configured LLM provider instances."""

import logging
from typing import TypeVar

from pydantic import BaseModel

from src.config import settings
from src.llm.antigravity_llm import AntigravityLLM
from src.llm.base import LLMProvider
from src.llm.mock_llm import MockLLM
from src.llm.ollama_llm import OllamaLLM

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMUnavailableError(RuntimeError):
    """Raised when the configured LLM provider cannot be used."""


def get_llm_provider(
    provider_name: str | None = None,
    **kwargs,
) -> LLMProvider:
    """Create and return an LLMProvider instance based on configuration.

    The mock provider is only ever used when selected explicitly, so a broken
    production setup can never answer with fake output.

    Args:
        provider_name: Explicit provider name ('antigravity', 'ollama', 'mock').
                       Defaults to settings.LLM_PROVIDER.
        **kwargs: Additional parameters passed to provider constructor.

    Returns:
        Configured LLMProvider instance.

    Raises:
        LLMUnavailableError: If the provider is unknown or cannot be initialized.
    """
    selected = (provider_name or getattr(settings, "LLM_PROVIDER", "antigravity")).lower().strip()

    if selected == "ollama":
        base_url = kwargs.get("base_url", getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434"))
        model = kwargs.get("model", getattr(settings, "OLLAMA_MODEL", "llama3.2"))
        logger.info("Initializing Ollama LLM provider (model=%s, url=%s)", model, base_url)
        think = kwargs.get("think", getattr(settings, "OLLAMA_THINK", False))
        temperature = kwargs.get("temperature", getattr(settings, "OLLAMA_TEMPERATURE", 0.0))
        return OllamaLLM(base_url=base_url, model=model, think=think, temperature=temperature)

    if selected == "mock":
        logger.warning("Using MockLLM: all LLM output is fake. This is only for tests and development.")
        return MockLLM(**kwargs)

    if selected == "antigravity":
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
            raise LLMUnavailableError(
                f"LLM_PROVIDER=antigravity is configured but unavailable: {err}"
            ) from err

    raise LLMUnavailableError(
        f"Unknown LLM_PROVIDER '{selected}'. Use 'antigravity', 'ollama', or 'mock'."
    )


class LazyLLMProvider(LLMProvider):
    """Create the configured provider on first use so importing the app has no side effects."""

    def __init__(self) -> None:
        self._provider: LLMProvider | None = None

    def resolve(self) -> LLMProvider:
        """Return the real provider, raising LLMUnavailableError if it cannot be used."""
        if self._provider is None:
            self._provider = get_llm_provider()
        return self._provider

    async def generate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """Generate text with the configured provider."""
        return await self.resolve().generate_text(prompt, system_prompt=system_prompt)

    async def generate_json(
        self,
        prompt: str,
        schema_model: type[T],
        system_prompt: str | None = None,
    ) -> T:
        """Generate structured output with the configured provider."""
        return await self.resolve().generate_json(prompt, schema_model, system_prompt=system_prompt)


__all__ = ["get_llm_provider", "LazyLLMProvider", "LLMUnavailableError"]
