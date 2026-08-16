"""Abstract Base Class for LLM Providers in the PKM system."""

from abc import ABC, abstractmethod
import asyncio
from typing import Any, TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMProvider(ABC):
    """Abstract interface defining required LLM generation capabilities."""

    @abstractmethod
    async def generate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """Generate unstructured text response from the model.

        Args:
            prompt: Main prompt or task instructions.
            system_prompt: Optional system prompt to prepend.

        Returns:
            Generated text string response.
        """
        pass

    @abstractmethod
    async def generate_json(
        self,
        prompt: str,
        schema_model: type[T],
        system_prompt: str | None = None,
    ) -> T:
        """Generate structured output validated against a Pydantic v2 schema model.

        Args:
            prompt: Main prompt or task instructions.
            schema_model: Pydantic v2 BaseModel class.
            system_prompt: Optional system prompt to prepend.

        Returns:
            Validated Pydantic model instance of type T.
        """
        pass

    def generate_text_sync(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """Synchronously generate text response, safe inside or outside running event loops."""
        try:
            asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(lambda: asyncio.run(self.generate_text(prompt, system_prompt))).result()
        except RuntimeError:
            return asyncio.run(self.generate_text(prompt, system_prompt))

    def generate_json_sync(
        self,
        prompt: str,
        schema_model: type[T],
        system_prompt: str | None = None,
    ) -> T:
        """Synchronously generate structured JSON output, safe inside or outside running event loops."""
        try:
            asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(lambda: asyncio.run(self.generate_json(prompt, schema_model, system_prompt))).result()
        except RuntimeError:
            return asyncio.run(self.generate_json(prompt, schema_model, system_prompt))
