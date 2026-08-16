"""Unit tests for Index Versioning and staleness detection."""

import json
import pytest
from pathlib import Path
from src.graphrag.versioning import (
    CURRENT_CHUNKER_VERSION,
    CURRENT_EMBEDDING_VERSION,
    CURRENT_PARSER_VERSION,
    FileIndexRecord,
    IndexVersionRegistry,
    compute_file_hash,
)


def test_file_index_record_staleness():
    rec = FileIndexRecord(
        path="Notes/DB.md",
        content_hash="hash123",
        embedding_model="all-MiniLM-L6-v2",
        embedding_version=CURRENT_EMBEDDING_VERSION,
        chunker_version=CURRENT_CHUNKER_VERSION,
        parser_version=CURRENT_PARSER_VERSION,
    )

    # 1. Same hash and model -> not stale
    assert not rec.is_stale(
        current_hash="hash123",
        current_embedding_model="all-MiniLM-L6-v2",
    )

    # 2. Changed content hash -> stale
    assert rec.is_stale(
        current_hash="hash456",
        current_embedding_model="all-MiniLM-L6-v2",
    )

    # 3. Changed embedding model -> stale
    assert rec.is_stale(
        current_hash="hash123",
        current_embedding_model="BAAI/bge-m3",
    )

    # 4. Changed chunker version -> stale
    assert rec.is_stale(
        current_hash="hash123",
        current_embedding_model="all-MiniLM-L6-v2",
        current_chunker_version="3.0",
    )


def test_index_version_registry_migration(tmp_path):
    registry_file = tmp_path / ".vault_hashes.json"

    # Write legacy flat dictionary
    legacy_data = {
        "Notes/Alpha.md": "hash_alpha_1",
        "Notes/Beta.md": "hash_beta_1",
    }
    registry_file.write_text(json.dumps(legacy_data), encoding="utf-8")

    registry = IndexVersionRegistry(registry_file)
    assert len(registry.records) == 2
    assert "Notes/Alpha.md" in registry.records
    assert registry.records["Notes/Alpha.md"].embedding_model == "legacy"

    # Check staleness against modern model
    assert registry.is_file_stale("Notes/Alpha.md", "hash_alpha_1", "all-MiniLM-L6-v2")

    # Update record
    registry.record_indexed(
        rel_path="Notes/Alpha.md",
        content_hash="hash_alpha_1",
        embedding_model="all-MiniLM-L6-v2",
        chunks_count=3,
    )
    registry.save()

    # Reload from disk
    reloaded = IndexVersionRegistry(registry_file)
    assert not reloaded.is_file_stale("Notes/Alpha.md", "hash_alpha_1", "all-MiniLM-L6-v2")
    assert reloaded.records["Notes/Alpha.md"].chunks_count == 3
