"""Unit tests for structure-aware Markdown chunking."""

import pytest
from pathlib import Path
from src.graphrag.chunker import (
    StructureAwareChunker,
    extract_frontmatter,
    extract_tags_from_text,
)


def test_extract_frontmatter():
    text = """---
created: 2026-08-15
type: atomic-note
tags:
  - architecture
  - distributed
memory_type: decision
---

# Architecture Notes
Body content here.
"""
    fm, body = extract_frontmatter(text)
    assert fm["created"] == "2026-08-15"
    assert fm["type"] == "atomic-note"
    assert fm["tags"] == ["architecture", "distributed"]
    assert fm["memory_type"] == "decision"
    assert "# Architecture Notes" in body


def test_extract_tags_from_text():
    text = "Studying #machine-learning and #graph_rag with #pkm/agent."
    tags = extract_tags_from_text(text)
    assert "#machine-learning" in tags
    assert "#graph_rag" in tags
    assert "#pkm/agent" in tags


def test_structure_aware_chunker(tmp_path):
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    note_path = vault_dir / "Distributed Systems.md"
    note_content = """---
created: 2026-08-15
tags:
  - architecture
---

# Distributed Systems

Overview of distributed principles.

## Consistency Models

Exploring strong consistency vs eventual consistency in [[CAP Theorem]].

- [ ] Review [[Raft]] consensus paper ➕ 2026-08-15 📅 2026-08-20 ^task-raft-01

> [!NOTE] Key Takeaway
> Always evaluate network partition tolerances.

```python
def consensus():
    return True
```

## Storage Engines

Comparing [[LSM-Tree]] vs [[B-Tree]] for high throughput writes.
"""
    note_path.write_text(note_content, encoding="utf-8")

    chunker = StructureAwareChunker()
    chunks = chunker.chunk_file(note_path, vault_dir)

    assert len(chunks) >= 3

    # Check first section chunk
    overview_chunk = chunks[0]
    assert overview_chunk.title == "Distributed Systems"
    assert "Overview of distributed principles" in overview_chunk.content

    # Check Consistency section chunk
    consistency_chunk = next(c for c in chunks if c.heading == "Consistency Models")
    assert "CAP Theorem" in consistency_chunk.wikilinks or "[[CAP Theorem]]" in consistency_chunk.content
    assert any("task-raft-01" in b for b in consistency_chunk.block_ids)
    assert consistency_chunk.heading_path == ["Distributed Systems", "Consistency Models"]
    assert "def consensus():" in consistency_chunk.content

    # Check Storage Engines chunk
    storage_chunk = next(c for c in chunks if c.heading == "Storage Engines")
    assert "LSM-Tree" in storage_chunk.wikilinks or "B-Tree" in storage_chunk.wikilinks
