"""OpenRouter LLM provider using its OpenAI-compatible chat completions API."""

import json
import logging
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from src.llm.base import LLMProvider

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterError(RuntimeError):
    """Raised when OpenRouter rejects a request or returns no usable answer."""


def _extract_json(text: str) -> str:
    """Return the JSON object in a reply, ignoring code fences or text around it."""
    cleaned = text.strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned


class OpenRouterLLM(LLMProvider):
    """LLM provider calling OpenRouter over HTTPS."""

    def __init__(
        self,
        api_key: str,
        model: str,
        fallback_models: list[str] | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 120.0,
        temperature: float = 0.0,
        data_collection: str = "deny",
        reasoning: bool = False,
    ) -> None:
        """Initialize the OpenRouter provider.

        Args:
            api_key: OpenRouter API key.
            model: Model ID, e.g. 'nvidia/nemotron-3.5-super-vl:free'.
            fallback_models: Models OpenRouter may use if the main one is unavailable.
            base_url: API base URL.
            timeout: Request timeout in seconds.
            temperature: Sampling temperature. 0 gives the most repeatable answers.
            data_collection: 'deny' routes only to providers that do not store or train on prompts.
            reasoning: Whether reasoning models should think before answering. Off keeps captures fast.
        """
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set.")
        if not model:
            raise ValueError("OPENROUTER_MODEL is not set.")
        if data_collection not in ("deny", "allow"):
            raise ValueError("OPENROUTER_DATA_COLLECTION must be 'deny' or 'allow'.")
        self._api_key = api_key
        self.model = model
        self.fallback_models = [m for m in (fallback_models or []) if m and m != model]
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.data_collection = data_collection
        self.reasoning = reasoning

    def _payload(
        self,
        prompt: str,
        system_prompt: str | None,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        provider: dict[str, Any] = {"data_collection": self.data_collection}
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "reasoning": {"enabled": self.reasoning},
            "provider": provider,
        }
        if self.fallback_models:
            payload["models"] = [self.model, *self.fallback_models]
        if response_format:
            payload["response_format"] = response_format
            # Only route to endpoints that actually enforce the schema.
            provider["require_parameters"] = True
        return payload

    async def _complete(self, payload: dict[str, Any]) -> str:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "X-Title": "pkm-agent",
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)

        try:
            data = resp.json()
        except ValueError:
            data = {}
        error = data.get("error") if isinstance(data, dict) else None
        if resp.status_code >= 400 or error:
            message = error.get("message") if isinstance(error, dict) else None
            raise OpenRouterError(
                f"OpenRouter returned HTTP {resp.status_code}: {message or 'no error message'}"
            )

        choices = data.get("choices") or []
        if not choices:
            raise OpenRouterError("OpenRouter returned no choices.")
        choice = choices[0]
        if choice.get("error"):
            raise OpenRouterError(f"OpenRouter model error: {choice['error'].get('message', choice['error'])}")
        used_model = data.get("model")
        if used_model and used_model != self.model:
            logger.info("OpenRouter answered with fallback model %s", used_model)
        return (choice.get("message", {}).get("content") or "").strip()

    async def generate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """Generate plain text."""
        return await self._complete(self._payload(prompt, system_prompt))

    async def generate_json(
        self,
        prompt: str,
        schema_model: type[T],
        system_prompt: str | None = None,
    ) -> T:
        """Generate output validated against schema_model, using structured outputs when available."""
        schema = schema_model.model_json_schema()
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": schema_model.__name__, "strict": False, "schema": schema},
        }
        try:
            content = await self._complete(self._payload(prompt, system_prompt, response_format))
            return schema_model.model_validate_json(_extract_json(content))
        except (OpenRouterError, ValidationError, ValueError) as err:
            logger.warning("Structured output failed for %s (%s); retrying with schema in prompt.", schema_model.__name__, err)

        fallback_prompt = (
            f"{prompt}\n\nRespond ONLY with one JSON object that matches this JSON schema, "
            f"with no other text:\n{json.dumps(schema)}"
        )
        content = await self._complete(self._payload(fallback_prompt, system_prompt))
        return schema_model.model_validate_json(_extract_json(content))


__all__ = ["OpenRouterLLM", "OpenRouterError"]
