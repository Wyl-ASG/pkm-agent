"""Ollama LLM provider implementation for local self-hosted inference."""

import json
import logging
from typing import Any, TypeVar
import httpx
from pydantic import BaseModel

from src.llm.base import LLMProvider

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class OllamaLLM(LLMProvider):
    """LLM provider connecting to local Ollama server."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2",
        timeout: float = 120.0,
    ) -> None:
        """Initialize Ollama LLM provider.

        Args:
            base_url: Base URL of running Ollama server.
            model: Model tag (e.g. 'llama3.2', 'mistral', 'qwen2.5').
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    async def generate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """Generate text from Ollama server."""
        url = f"{self.base_url}/api/chat"
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "").strip()

    async def generate_json(
        self,
        prompt: str,
        schema_model: type[T],
        system_prompt: str | None = None,
    ) -> T:
        """Generate structured JSON conforming to schema_model via Ollama format parameter."""
        url = f"{self.base_url}/api/chat"
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        schema_json = schema_model.model_json_schema()
        payload = {
            "model": self.model,
            "messages": messages,
            "format": schema_json,
            "stream": False,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content", "")
            return schema_model.model_validate_json(content)


__all__ = ["OllamaLLM"]
