"""Tests for how captures are filed: sections, atomic note confidence, wording, and block IDs."""

import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from src.agents.models import InterstitialEntry
from src.agents.parser import EntryParserAgent, enforce_task_syntax
from src.llm.mock_llm import MockLLM
from src.llm.ollama_llm import OllamaLLM
from src.vault.md_writer import resolve_header


async def _parse(raw_text: str, **llm_fields) -> InterstitialEntry:
    response = {"timestamp": "2026-08-30 20:00", "category": "log", "content": raw_text, **llm_fields}
    return await EntryParserAgent(llm=MockLLM(default_json_response=response)).parse(raw_text)


# Categories -----------------------------------------------------------------


def test_schema_offers_only_the_section_categories():
    schema = InterstitialEntry.model_json_schema()
    assert schema["properties"]["category"]["enum"] == ["task", "discovery", "log", "inbox"]


@pytest.mark.parametrize(
    ("llm_category", "expected"),
    [
        ("Discovery", "discovery"),
        ("learning", "discovery"),
        ("fact", "discovery"),
        ("todo", "task"),
        ("work", "log"),
        ("journal", "log"),
        ("observation", "log"),
        ("general", "log"),
        ("something-unexpected", "log"),
        ("capture", "inbox"),
    ],
)
def test_category_variants_map_to_a_real_section(llm_category, expected):
    entry = InterstitialEntry(timestamp="2026-08-30 20:00", category=llm_category, content="x")
    assert entry.category == expected


@pytest.mark.parametrize("category", ["task", "discovery", "log", "inbox"])
def test_every_category_has_its_own_header(category):
    assert resolve_header(category) != ""


@pytest.mark.asyncio
async def test_meta_chatter_falls_back_to_a_valid_category():
    entry = await _parse("Plain thought", category="discovery", content="Awaiting user instructions")
    assert entry.category == "log"
    assert entry.content == "Plain thought"


# Atomic note confidence -----------------------------------------------------


def test_missing_confidence_is_not_treated_as_confident():
    entry = InterstitialEntry(timestamp="2026-08-30 20:00", content="x")
    assert entry.atomic_note_confidence == 0.0


def test_out_of_range_confidence_is_rejected():
    with pytest.raises(ValidationError):
        InterstitialEntry(timestamp="2026-08-30 20:00", content="x", atomic_note_confidence=85)


@pytest.mark.asyncio
async def test_atomic_flag_without_confidence_creates_no_note():
    import src.main as app_main

    entry = InterstitialEntry(
        timestamp="2026-08-30 20:00",
        content="3/4 of users prefer dark mode",
        requires_atomic_note=True,
        atomic_note_title="No Confidence Note",
        atomic_note_content="Generated text.",
    )
    with (
        patch.object(app_main.parser_agent, "parse", new=AsyncMock(return_value=entry)),
        patch("src.main.send_telegram_message", new_callable=AsyncMock) as sent,
        patch("src.main._async_background_post_ingestion", new_callable=AsyncMock),
    ):
        await app_main.process_telegram_ingestion(1, entry.content)

    assert not (Path(os.environ["VAULT_PATH"]) / "Notes" / "No Confidence Note.md").exists()
    assert sent.call_args.kwargs["reply_markup"] is None


# Wording --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw_text", "llm_content"),
    [
        ("Pwn college sandboxing module", "- [ ] Pwn [[Sandboxing Module]] [[College]]"),
        ("I need to do my bgs by next friday", "- [ ] [[bgs]]"),
        ("Review the design doc", "- [ ] Review the design doc [[Design Doc]]"),
        ("Upload my photo by this Friday", "- [ ] Upload my photo"),
    ],
)
@pytest.mark.asyncio
async def test_task_with_changed_wording_keeps_the_original_text(raw_text, llm_content):
    entry = await _parse(raw_text, category="task", content=llm_content)
    assert entry.content.startswith(f"- [ ] {raw_text} ➕")


@pytest.mark.asyncio
async def test_note_with_changed_wording_keeps_the_original_text():
    raw_text = "It misses other forms of isolation"
    entry = await _parse(raw_text, category="discovery", content="Isolation is missing in chroot")
    assert entry.content == raw_text


# Block IDs ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_block_ids_invented_by_the_model_are_removed():
    entry = await _parse(
        "Finish the lab by Sunday",
        category="task",
        content="- [ ] Finish the lab by Sunday ^task-01 ➕ 2026-08-30",
        block_id="task-01",
    )
    assert "^" not in entry.content
    assert entry.block_id is None


@pytest.mark.asyncio
async def test_block_id_written_by_the_user_is_kept():
    entry = await _parse("Chroot is not a sandbox ^chroot-limits", category="discovery")
    assert entry.block_id == "chroot-limits"
    assert "^chroot-limits" not in entry.content


@pytest.mark.asyncio
async def test_caret_inside_a_formula_is_not_a_block_id():
    raw_text = "The cost grows as x^2 for each node"
    entry = await _parse(raw_text, category="discovery")
    assert entry.content == raw_text


def test_task_syntax_puts_block_id_last():
    formatted = enforce_task_syntax("Finish the lab ^lab-1", "2026-08-30 20:00", due_date="2026-09-06")
    assert formatted == "- [ ] Finish the lab ➕ 2026-08-30 📅 2026-09-06 ^lab-1"


# Ollama thinking ------------------------------------------------------------


class _FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"message": {"content": '{"timestamp": "2026-08-30 20:00", "content": "x"}'}}


class _FakeClient:
    payloads: list[dict] = []

    def __init__(self, timeout: float) -> None:
        self.timeout = timeout

    async def __aenter__(self) -> "_FakeClient":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    async def post(self, url: str, json: dict) -> _FakeResponse:
        _FakeClient.payloads.append(json)
        return _FakeResponse()


@pytest.mark.parametrize("think", [True, False])
@pytest.mark.asyncio
async def test_ollama_requests_carry_the_thinking_setting(think):
    _FakeClient.payloads = []
    llm = OllamaLLM(model="test-model", think=think)

    with patch("src.llm.ollama_llm.httpx.AsyncClient", _FakeClient):
        await llm.generate_text("hello")
        await llm.generate_json("hello", InterstitialEntry)

    assert [payload["think"] for payload in _FakeClient.payloads] == [think, think]


@pytest.mark.asyncio
async def test_ollama_requests_carry_the_temperature():
    _FakeClient.payloads = []
    llm = OllamaLLM(model="test-model", temperature=0.0)

    with patch("src.llm.ollama_llm.httpx.AsyncClient", _FakeClient):
        await llm.generate_text("hello")
        await llm.generate_json("hello", InterstitialEntry)

    assert [payload["options"]["temperature"] for payload in _FakeClient.payloads] == [0.0, 0.0]


def test_temperature_is_zero_by_default():
    from src.llm.factory import get_llm_provider

    assert get_llm_provider("ollama").temperature == 0.0


def test_thinking_is_off_by_default():
    from src.llm.factory import get_llm_provider

    assert get_llm_provider("ollama").think is False
