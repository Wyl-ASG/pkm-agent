"""Mock LLM provider for unit testing and offline development."""

import json
from typing import Any, Callable, TypeVar
from pydantic import BaseModel

from src.llm.base import LLMProvider

T = TypeVar("T", bound=BaseModel)


class MockLLM(LLMProvider):
    """Deterministic Mock LLM provider for tests."""

    def __init__(
        self,
        default_text_response: str = "Mock answer synthesized from context.",
        default_json_response: dict[str, Any] | None = None,
        custom_handler: Callable[[str, str | None], Any] | None = None,
    ) -> None:
        """Initialize MockLLM."""
        self.default_text_response = default_text_response
        self.default_json_response = default_json_response or {}
        self.custom_handler = custom_handler
        self.recorded_prompts: list[tuple[str, str | None]] = []

    async def generate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """Return deterministic text response."""
        self.recorded_prompts.append((prompt, system_prompt))
        if self.custom_handler:
            res = self.custom_handler(prompt, system_prompt)
            if isinstance(res, str):
                return res
        return self.default_text_response

    async def generate_json(
        self,
        prompt: str,
        schema_model: type[T],
        system_prompt: str | None = None,
    ) -> T:
        """Return validated model from default_json_response or custom_handler."""
        self.recorded_prompts.append((prompt, system_prompt))
        if self.custom_handler:
            res = self.custom_handler(prompt, system_prompt)
            if isinstance(res, schema_model):
                return res
            if isinstance(res, dict):
                return schema_model.model_validate(res)
            if isinstance(res, str):
                return schema_model.model_validate_json(res)

        if self.default_json_response:
            return schema_model.model_validate(self.default_json_response)

        # Generate minimal valid dummy model instance
        return schema_model.model_validate({})


__all__ = ["MockLLM"]
