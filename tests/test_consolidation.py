"""Unit tests for knowledge consolidation and duplicate concept detection."""

import pytest
from pathlib import Path
from src.agents.consolidation import KnowledgeConsolidator
from src.graphrag.graph import VaultKnowledgeGraph


def test_find_potential_duplicate_notes(tmp_path):
    vault_dir = tmp_path / "vault"
    notes_dir = vault_dir / "Notes"
    notes_dir.mkdir(parents=True)

    (notes_dir / "Local AI.md").write_text("# Local AI\nOverview of local LLMs.", encoding="utf-8")
    (notes_dir / "Local AI Architecture.md").write_text("# Local AI Architecture\nArchitectural details.", encoding="utf-8")
    (notes_dir / "Cooking Recipes.md").write_text("# Cooking Recipes\nPasta recipes.", encoding="utf-8")

    graph = VaultKnowledgeGraph(vault_path=vault_dir)
    consolidator = KnowledgeConsolidator(vault_path=vault_dir, knowledge_graph=graph)

    graph.build_graph()
    duplicates = consolidator.find_potential_duplicate_notes()

    assert len(duplicates) >= 1
    pair = duplicates[0]
    assert ("Local AI" in pair["note_a"] or "Local AI" in pair["note_b"])
