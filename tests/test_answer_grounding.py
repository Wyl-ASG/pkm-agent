"""Tests that unrelated captures stay separate and answers list only the sources they used."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.agents.parser import normalize_obsidian_markdown
from src.agents.qa import QA_SYSTEM_PROMPT, KnowledgeBaseQAAgent
from src.graphrag.chunker import StructureAwareChunker
from src.graphrag.versioning import CURRENT_CHUNKER_VERSION
from src.llm.mock_llm import MockLLM

DAILY_NOTE = """---
created: 2026-11-09
type: daily-note
tags:
  - daily
---

# 📅 Daily Note: 2026-11-09

## 📥 Inbox (Quick Capture)

## ⏱️ Log (Interstitial)

- **[18:56]** I need to submit the report by next friday
## 🎯 Priorities & Tasks

- [ ] I need to renew the homelab VPN certificate by 20 Dec ➕ 2026-11-09 📅 2026-12-20 ^vpn-cert
## 🧠 Discoveries & Learning
- **[18:55]** It has been replaced by cgroups, namespace and sec comp
- **[19:15]** WireGuard uses UDP port 51820 by default
- **[19:15]** Decided to use WireGuard because the config is simpler
"""


def _chunks(tmp_path: Path, text: str, name: str = "Daily Notes/2026-11-09.md"):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return StructureAwareChunker().chunk_file(path, tmp_path)


# Chunking -------------------------------------------------------------------


def test_each_daily_capture_gets_its_own_chunk(tmp_path):
    contents = [chunk.content for chunk in _chunks(tmp_path, DAILY_NOTE)]

    assert contents == [
        "- **[18:56]** I need to submit the report by next friday",
        "- [ ] I need to renew the homelab VPN certificate by 20 Dec ➕ 2026-11-09 📅 2026-12-20 ^vpn-cert",
        "- **[18:55]** It has been replaced by cgroups, namespace and sec comp",
        "- **[19:15]** WireGuard uses UDP port 51820 by default",
        "- **[19:15]** Decided to use WireGuard because the config is simpler",
    ]


def test_unrelated_captures_never_share_a_chunk(tmp_path):
    for chunk in _chunks(tmp_path, DAILY_NOTE):
        assert not ("cgroups" in chunk.content and "WireGuard" in chunk.content)


def test_daily_capture_chunks_keep_their_section_and_details(tmp_path):
    chunks = _chunks(tmp_path, DAILY_NOTE)

    assert [c.chunk_index for c in chunks] == [0, 1, 2, 3, 4]
    assert chunks[0].heading == "⏱️ Log (Interstitial)"
    assert chunks[1].heading == "🎯 Priorities & Tasks"
    assert chunks[1].memory_type == "task"
    assert chunks[1].block_ids == ["^vpn-cert"]
    assert chunks[2].title == "2026-11-09"


def test_multi_line_capture_stays_together(tmp_path):
    text = "---\ntype: daily-note\n---\n\n## ⏱️ Log\n- **[10:00]** First line\nSecond line\n- **[10:01]** Next capture\n"
    contents = [chunk.content for chunk in _chunks(tmp_path, text)]
    assert contents == ["- **[10:00]** First line\nSecond line", "- **[10:01]** Next capture"]


def test_callout_capture_stays_together(tmp_path):
    text = "---\ntype: daily-note\n---\n\n## ⏱️ Log\n> [!WARNING]\n> - **[10:00]** Disk is full\n- **[10:01]** Next capture\n"
    contents = [chunk.content for chunk in _chunks(tmp_path, text)]
    assert contents == ["> [!WARNING]\n> - **[10:00]** Disk is full", "- **[10:01]** Next capture"]


def test_code_block_in_a_daily_note_is_not_split(tmp_path):
    text = "---\ntype: daily-note\n---\n\n## ⏱️ Log\n- **[10:00]** Example\n```yaml\n- one\n- two\n```\n- **[10:01]** Next\n"
    contents = [chunk.content for chunk in _chunks(tmp_path, text)]
    assert contents == ["- **[10:00]** Example\n```yaml\n- one\n- two\n```", "- **[10:01]** Next"]


def test_paragraphs_typed_into_a_daily_note_are_kept(tmp_path):
    text = "---\ntype: daily-note\n---\n\n## ⏱️ Log\nSome prose I typed.\n\n- **[10:01]** A capture\n"
    contents = [chunk.content for chunk in _chunks(tmp_path, text)]
    assert contents == ["Some prose I typed.", "- **[10:01]** A capture"]


def test_notes_in_daily_folder_without_frontmatter_are_split_too(tmp_path):
    text = "## Log\n- one thing\n- another thing\n"
    assert len(_chunks(tmp_path, text)) == 2


def test_regular_notes_still_chunk_by_section(tmp_path):
    text = "# Topic\n\nIntro text.\n\n## Details\n- point one\n- point two\n"
    chunks = _chunks(tmp_path, text, name="Notes/Topic.md")

    assert len(chunks) == 2
    assert "point one" in chunks[1].content and "point two" in chunks[1].content


def test_empty_sections_produce_no_chunks(tmp_path):
    text = "# Topic\n\n## Empty\n\n## Filled\nSome text.\n"
    chunks = _chunks(tmp_path, text, name="Notes/Topic.md")
    assert [chunk.heading for chunk in chunks] == ["Filled"]


def test_chunker_version_forces_existing_notes_to_be_reindexed():
    assert CURRENT_CHUNKER_VERSION != "2.0"


# Answer rules ---------------------------------------------------------------


@pytest.mark.parametrize(
    "expected",
    ["separate", "pronoun", "speculate", "SOURCES:"],
)
def test_prompt_contains_grounding_rules(expected):
    assert expected in QA_SYSTEM_PROMPT


def _agent(answer: str) -> KnowledgeBaseQAAgent:
    retriever = AsyncMock()
    retriever.search.return_value = [
        {"content": "One", "payload": {"title": "2026-11-09", "file_path": "d.md", "heading": "Discoveries"}},
        {"content": "Two", "payload": {"title": "Docker", "file_path": "Notes/Docker.md", "heading": "Setup"}},
        {"content": "Three", "payload": {"title": "Qdrant", "file_path": "Notes/Qdrant.md", "heading": "Indexes"}},
    ]
    return KnowledgeBaseQAAgent(retriever=retriever, llm=MockLLM(default_text_response=answer))


@pytest.mark.asyncio
async def test_only_sources_the_answer_used_are_listed():
    response = await _agent("Use port 51820.\n\nSOURCES: 1, 3").query("q")

    assert "SOURCES:" not in response.answer
    assert "[[2026-11-09]]" in response.answer
    assert "[[Qdrant]]" in response.answer
    assert "[[Docker]]" not in response.answer
    assert [c.note_title for c in response.citations] == ["2026-11-09", "Qdrant"]
    assert response.sources == ["[[2026-11-09]]", "[[Qdrant]]"]
    assert "**Sources:**" in response.answer


@pytest.mark.parametrize("line", ["Sources: [2]", "**SOURCES:** 2", "SOURCES: Source [2]", "sources: 2."])
@pytest.mark.asyncio
async def test_source_line_variations_are_understood(line):
    response = await _agent(f"Answer text.\n{line}").query("q")

    assert line not in response.answer
    assert [c.note_title for c in response.citations] == ["Docker"]


@pytest.mark.asyncio
async def test_source_numbers_that_do_not_exist_are_ignored():
    response = await _agent("Answer text.\nSOURCES: 2, 9").query("q")
    assert [c.note_title for c in response.citations] == ["Docker"]


@pytest.mark.asyncio
async def test_answer_using_no_sources_lists_none():
    response = await _agent("Answer text.\nSOURCES: none").query("q")

    assert "Sources" not in response.answer
    assert response.citations == []


@pytest.mark.asyncio
async def test_missing_source_line_is_labelled_as_unverified():
    response = await _agent("Answer text with no source line.").query("q")

    assert "**Sources:**" not in response.answer
    assert "**Search results, not verified as used:**" in response.answer
    assert len(response.citations) == 3


@pytest.mark.asyncio
async def test_context_is_presented_as_numbered_separate_sources():
    agent = _agent("Answer.\nSOURCES: 1")
    await agent.query("q")

    prompt, _ = agent.llm.recorded_prompts[0]
    assert "Source [1]" in prompt and "Source [2]" in prompt and "Source [3]" in prompt


def test_timestamp_in_an_answer_is_not_turned_into_a_link():
    assert normalize_obsidian_markdown("- **[18:55]** It was replaced") == "- **[18:55]** It was replaced"
