"""Unit tests for QA agent, block-level provenance formatting, and citations."""

import pytest
from src.agents.models import SourceCitation
from src.agents.qa import KnowledgeBaseQAAgent
from src.llm.mock_llm import MockLLM


def test_source_citation_format():
    citation_with_block = SourceCitation(
        note_title="Architecture",
        note_path="Notes/Architecture.md",
        heading="Database Decision",
        heading_path=["Architecture", "Database Decision"],
        block_id="^db-decision-01",
        snippet="Chose PostgreSQL.",
    )
    formatted = citation_with_block.format_citation()
    assert "[[Architecture]]" in formatted
    assert "→ ## Database Decision" in formatted
    assert "→ ^db-decision-01" in formatted


def test_single_bracket_wikilink_normalization():
    from src.agents.parser import normalize_obsidian_markdown

    sample = (
        "Here is an overview of [Microsoft Azure] and [Availability Zones|Availability Zones (AZs)].\n"
        "Services: [Azure Functions] and [Azure App Service|App Service].\n"
        "Framework: [Azure Well-Architected Framework|Well-Architected Framework Pillars].\n"
        "Sources: [2026-08-15] and [Google Cloud Platform Architecture and Key Services Overview]."
    )
    normalized = normalize_obsidian_markdown(sample)

    assert "[[Microsoft Azure]]" in normalized
    assert "[[Availability Zones|Availability Zones (AZs)]]" in normalized
    assert "[[Azure Functions]]" in normalized
    assert "[[Azure App Service|App Service]]" in normalized
    assert "[[Azure Well-Architected Framework|Well-Architected Framework Pillars]]" in normalized
    assert "[[2026-08-15]]" in normalized
    assert "[[Google Cloud Platform Architecture and Key Services Overview]]" in normalized


def test_telegram_html_and_markdown_formatting():
    from src.telegram.formatters import convert_markdown_to_telegram_html, format_obsidian_for_telegram

    raw_text = (
        "Based on your notes, here is [Microsoft Azure]:\n\n"
        "---\n\n"
        "### 1. Core Infrastructure\n"
        " Regions: Global geographical areas.\n"
        " [Availability Zones|Availability Zones (AZs)]: Separate locations in [Microsoft Azure|Azure].\n\n"
        "**Sources:**\n"
        "• [[2026-08-15]] → ## Discoveries"
    )

    telegram_md = format_obsidian_for_telegram(raw_text)
    assert "🔹 *1. Core Infrastructure*" in telegram_md
    assert "───────────────────────" in telegram_md
    assert "[[Microsoft Azure]]" in telegram_md
    assert "[[Availability Zones|Availability Zones (AZs)]]" in telegram_md

    telegram_html = convert_markdown_to_telegram_html(raw_text)
    assert "<b>1. Core Infrastructure</b>" in telegram_html
    assert "───────────────────────" in telegram_html
    assert "[[Microsoft Azure]]" in telegram_html
    assert "[[Availability Zones|Availability Zones (AZs)]]" in telegram_html


@pytest.mark.asyncio
async def test_qa_agent_with_mock_llm():
    mock_llm = MockLLM(default_text_response="You chose [[PostgreSQL]] for high reliability.")
    agent = KnowledgeBaseQAAgent(llm=mock_llm)

    response = await agent.query("What database did I choose?", top_k=3)
    assert response.query == "What database did I choose?"
    assert "PostgreSQL" in response.answer
    assert isinstance(response.citations, list)
