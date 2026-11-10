"""OpenRouter provider: request shape, privacy routing, structured output, and failures."""

from unittest.mock import patch

import pytest
from pydantic import BaseModel

from src.config import settings
from src.llm.factory import LLMUnavailableError, get_llm_provider
from src.llm.openrouter_llm import OpenRouterError, OpenRouterLLM


class Answer(BaseModel):
    category: str
    confidence: float


class _Response:
    def __init__(self, status_code: int, body: object) -> None:
        self.status_code = status_code
        self._body = body

    def json(self) -> object:
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


def _client(responses: list[_Response], calls: list[dict]):
    class _FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc) -> None:
            return None

        async def post(self, url, json=None, headers=None):
            calls.append({"url": url, "json": json, "headers": headers})
            return responses.pop(0)

    return _FakeClient


def _reply(content: str, model: str = "test/model:free") -> _Response:
    return _Response(200, {"model": model, "choices": [{"message": {"content": content}}]})


def _llm(**kwargs) -> OpenRouterLLM:
    return OpenRouterLLM(api_key="sk-test-key", model="test/model:free", **kwargs)


@pytest.mark.asyncio
async def test_text_request_denies_data_collection_and_disables_reasoning_by_default():
    calls: list[dict] = []
    with patch("src.llm.openrouter_llm.httpx.AsyncClient", _client([_reply(" Hello ")], calls)):
        answer = await _llm(fallback_models=["other/model:free"]).generate_text("hi", system_prompt="be brief")

    assert answer == "Hello"
    sent = calls[0]
    assert sent["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert sent["headers"]["Authorization"] == "Bearer sk-test-key"
    body = sent["json"]
    assert body["model"] == "test/model:free"
    assert body["models"] == ["test/model:free", "other/model:free"]
    assert body["messages"] == [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]
    assert body["provider"] == {"data_collection": "deny"}
    assert body["reasoning"] == {"enabled": False}
    assert body["temperature"] == 0.0
    assert "response_format" not in body


@pytest.mark.asyncio
async def test_json_uses_schema_and_requires_endpoints_that_enforce_it():
    calls: list[dict] = []
    reply = _reply('```json\n{"category": "task", "confidence": 0.9}\n```')
    with patch("src.llm.openrouter_llm.httpx.AsyncClient", _client([reply], calls)):
        result = await _llm().generate_json("parse this", Answer)

    assert result == Answer(category="task", confidence=0.9)
    body = calls[0]["json"]
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"]["title"] == "Answer"
    assert body["provider"] == {"data_collection": "deny", "require_parameters": True}


@pytest.mark.asyncio
async def test_json_retries_with_schema_in_prompt_when_structured_output_is_unavailable():
    calls: list[dict] = []
    responses = [
        _Response(404, {"error": {"message": "No endpoints found that support response_format"}}),
        _reply('Sure: {"category": "log", "confidence": 0.2}'),
    ]
    with patch("src.llm.openrouter_llm.httpx.AsyncClient", _client(responses, calls)):
        result = await _llm().generate_json("parse this", Answer)

    assert result.category == "log"
    assert "response_format" not in calls[1]["json"]
    assert "JSON schema" in calls[1]["json"]["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_errors_name_the_status_but_never_the_key():
    calls: list[dict] = []
    body = {"error": {"message": "Rate limit exceeded: free-models-per-day"}}
    with patch("src.llm.openrouter_llm.httpx.AsyncClient", _client([_Response(429, body)], calls)):
        with pytest.raises(OpenRouterError) as err:
            await _llm().generate_text("hi")

    assert "429" in str(err.value) and "free-models-per-day" in str(err.value)
    assert "sk-test-key" not in str(err.value)


@pytest.mark.asyncio
async def test_error_inside_a_successful_response_is_raised():
    calls: list[dict] = []
    body = {"choices": [{"error": {"message": "provider overloaded"}, "message": {"content": ""}}]}
    with patch("src.llm.openrouter_llm.httpx.AsyncClient", _client([_Response(200, body)], calls)):
        with pytest.raises(OpenRouterError, match="provider overloaded"):
            await _llm().generate_text("hi")


def test_invalid_settings_are_rejected():
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterLLM(api_key="", model="m")
    with pytest.raises(ValueError, match="DATA_COLLECTION"):
        OpenRouterLLM(api_key="k", model="m", data_collection="maybe")


def test_factory_refuses_to_start_without_a_key(monkeypatch):
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "")
    with pytest.raises(LLMUnavailableError, match="OPENROUTER_API_KEY"):
        get_llm_provider("openrouter")


def test_factory_reads_settings(monkeypatch):
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setattr(settings, "OPENROUTER_MODEL", "a/main:free")
    monkeypatch.setattr(settings, "OPENROUTER_FALLBACK_MODELS", "b/one:free, ,a/main:free,c/two:free")
    monkeypatch.setattr(settings, "OPENROUTER_DATA_COLLECTION", "allow")
    provider = get_llm_provider("openrouter")

    assert isinstance(provider, OpenRouterLLM)
    assert provider.model == "a/main:free"
    assert provider.fallback_models == ["b/one:free", "c/two:free"]
    assert provider.data_collection == "allow"
