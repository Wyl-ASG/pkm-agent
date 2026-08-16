"""Unit tests for note lookup by prefix/title and Telegram callback query handling."""

import pytest
from pathlib import Path

from src.vault.md_writer import ObsidianVaultWriter


def test_find_note_by_title_or_prefix_exact_and_truncated(tmp_path: Path):
    """Test finding notes by exact match, truncated prefix, and case insensitivity."""
    vault_dir = tmp_path / "vault"
    notes_dir = vault_dir / "Notes"
    daily_dir = vault_dir / "Daily Notes"
    notes_dir.mkdir(parents=True)
    daily_dir.mkdir(parents=True)

    # Create atomic note with long title
    long_note = notes_dir / "AWS Infrastructure Fundamentals and Well-Architected Framework.md"
    long_note.write_text("""---
created: 2026-08-15
type: atomic-note
tags:
  - aws
  - architecture
---

# AWS Infrastructure Fundamentals and Well-Architected Framework

This note details AWS cloud architecture and well-architected pillars.
""", encoding="utf-8")

    # Create daily note
    daily_note = daily_dir / "2026-08-15.md"
    daily_note.write_text("# 📅 Daily Note: 2026-08-15\n\n## ⏱️ Log\n- [ ] Task 1\n", encoding="utf-8")

    writer = ObsidianVaultWriter(vault_path=vault_dir)

    # 1. Exact match
    found = writer.find_note_by_title_or_prefix("AWS Infrastructure Fundamentals and Well-Architected Framework")
    assert found is not None
    assert found.name == "AWS Infrastructure Fundamentals and Well-Architected Framework.md"

    # 2. Truncated prefix (as produced in Telegram callback_data)
    found_trunc = writer.find_note_by_title_or_prefix("AWS Infrastructure Fundamental")
    assert found_trunc is not None
    assert found_trunc.name == "AWS Infrastructure Fundamentals and Well-Architected Framework.md"

    # 3. Case-insensitive substring
    found_sub = writer.find_note_by_title_or_prefix("well-architected framework")
    assert found_sub is not None
    assert found_sub.name == "AWS Infrastructure Fundamentals and Well-Architected Framework.md"

    # 4. Daily note lookup
    found_daily = writer.find_note_by_title_or_prefix("2026-08-15")
    assert found_daily is not None
    assert found_daily.name == "2026-08-15.md"

    # 5. Non-existent note
    found_none = writer.find_note_by_title_or_prefix("NonExistentNote12345")
    assert found_none is None


@pytest.mark.asyncio
async def test_find_note_async(tmp_path: Path):
    """Test asynchronous note lookup."""
    vault_dir = tmp_path / "vault"
    notes_dir = vault_dir / "Notes"
    notes_dir.mkdir(parents=True)

    note = notes_dir / "Docker Networking Fundamentals.md"
    note.write_text("# Docker Networking Fundamentals\nContent here.", encoding="utf-8")

    writer = ObsidianVaultWriter(vault_path=vault_dir)
    found = await writer.find_note_by_title_or_prefix_async("Docker Networking")
    assert found is not None
    assert found.name == "Docker Networking Fundamentals.md"


def test_split_telegram_message():
    """Verify split_telegram_message correctly splits long texts without losing data."""
    from src.telegram.client import split_telegram_message

    # 1. Short text stays as single chunk
    short_text = "Short note content"
    chunks = split_telegram_message(short_text, max_chunk_size=100)
    assert chunks == ["Short note content"]

    # 2. Long text with sections splits along section breaks
    long_text = "Section 1 content\n\n## Section 2\n" + ("A" * 80) + "\n\n## Section 3\n" + ("B" * 80)
    chunks = split_telegram_message(long_text, max_chunk_size=100)
    assert len(chunks) >= 3
    assert chunks[0] == "Section 1 content"
    assert chunks[1].startswith("## Section 2")
    assert chunks[-1].startswith("## Section 3")
    for chunk in chunks:
        assert len(chunk) <= 100


def test_qdrant_concurrent_upserts(tmp_path: Path):
    """Verify QdrantVectorStore handles concurrent multi-threaded writes without SQLite transaction errors."""
    import concurrent.futures
    from src.graphrag.vector_db import QdrantVectorStore

    # Reset any existing singleton
    if QdrantVectorStore._instance is not None:
        QdrantVectorStore._instance.close()

    db_path = tmp_path / "qdrant_concurrent_test"
    store = QdrantVectorStore(
        collection_name="test_concurrency",
        storage_path=db_path,
    )
    store.ensure_collection(vector_size=4)

    def _worker(thread_idx: int):
        docs = [
            {
                "id": f"thread_{thread_idx}_doc_{i}",
                "vector": [0.1 * thread_idx, 0.2, 0.3, 0.4],
                "content": f"Document content from thread {thread_idx} item {i}",
                "metadata": {"thread": thread_idx, "idx": i},
            }
            for i in range(5)
        ]
        return store.upsert_documents(docs)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(_worker, i) for i in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert all(results)
        search_res = store.search_dense([0.1, 0.2, 0.3, 0.4], top_k=50)
        assert len(search_res) == 50
    finally:
        store.close()


