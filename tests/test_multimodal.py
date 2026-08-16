"""Unit tests for multimodal ingestion pipeline abstractions."""

import pytest
from pathlib import Path
from src.agents.multimodal import IngestionItem, ModalityType, MultimodalIngestionPipeline


@pytest.mark.asyncio
async def test_multimodal_text_and_url(tmp_path):
    pipeline = MultimodalIngestionPipeline()

    # 1. Text processing
    text_item = IngestionItem(modality=ModalityType.TEXT, content="Logging task today")
    res_text = await pipeline.process(text_item)
    assert res_text == "Logging task today"

    # 2. URL processing
    url_item = IngestionItem(
        modality=ModalityType.URL,
        content="https://qdrant.tech",
        caption="Qdrant Docs",
        source_url="https://qdrant.tech",
    )
    res_url = await pipeline.process(url_item)
    assert "[Qdrant Docs](https://qdrant.tech)" in res_url

    # 3. Document processing
    doc_path = tmp_path / "imported.md"
    doc_path.write_text("# Imported Doc\nImportant details.", encoding="utf-8")
    doc_item = IngestionItem(modality=ModalityType.DOCUMENT, content="", file_path=doc_path)
    res_doc = await pipeline.process(doc_item)
    assert "Important details" in res_doc
