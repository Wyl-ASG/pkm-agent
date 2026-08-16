"""Unit tests for knowledge graph extraction and neighborhood expansion."""

import pytest
from pathlib import Path
from src.graphrag.graph import VaultKnowledgeGraph


def test_vault_knowledge_graph(tmp_path):
    vault_dir = tmp_path / "vault"
    notes_dir = vault_dir / "Notes"
    daily_dir = vault_dir / "Daily Notes"
    notes_dir.mkdir(parents=True)
    daily_dir.mkdir(parents=True)

    # Create Note A: PostgreSQL
    (notes_dir / "PostgreSQL.md").write_text("""---
aliases:
  - postgres
  - psql
tags:
  - database
---

# PostgreSQL
Relational database connected to [[Database Architecture]] and [[Backend Services]].
""", encoding="utf-8")

    # Create Note B: Database Architecture
    (notes_dir / "Database Architecture.md").write_text("""---
tags:
  - architecture
---

# Database Architecture
Overview of storage patterns with [[PostgreSQL]] and [[Redis Cache]].
""", encoding="utf-8")

    # Create Note C: Backend Services
    (notes_dir / "Backend Services.md").write_text("""---
tags:
  - backend
---

# Backend Services
Microservices stack using [[FastAPI]].
""", encoding="utf-8")

    graph = VaultKnowledgeGraph(vault_path=vault_dir)
    graph.build_graph()

    assert len(graph.nodes) >= 3
    assert "PostgreSQL" in graph.nodes
    assert "Database Architecture" in graph.nodes

    # Check aliases
    assert graph.resolve_canonical_title("postgres") == "PostgreSQL"
    assert graph.resolve_canonical_title("psql") == "PostgreSQL"

    # Check neighbors (outgoing + backlinks)
    neighbors = graph.get_neighbors("PostgreSQL", max_hops=1, max_neighbors=5)
    assert "Database Architecture" in neighbors
    assert "Backend Services" in neighbors

    # Check backlinks on Database Architecture
    db_node = graph.nodes["Database Architecture"]
    assert "PostgreSQL" in db_node.outgoing_links or "PostgreSQL" in db_node.backlinks
