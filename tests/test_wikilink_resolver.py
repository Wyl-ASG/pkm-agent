"""Unit tests for safe WikiLink resolution and entity grounding."""

import pytest
from src.graphrag.resolver import WikiLinkResolver, normalize_term_stem, string_similarity


def test_stem_normalization():
    assert normalize_term_stem("Distributed Systems") == "distributed system"
    assert normalize_term_stem("Database Notes") == "database"
    assert normalize_term_stem("Categories") == "category"


def test_wikilink_resolver_exact_and_alias():
    notes = ["Distributed Systems", "Amazon Web Services", "PostgreSQL", "Machine Learning"]
    aliases = {
        "aws": "Amazon Web Services",
        "postgres": "PostgreSQL",
        "ml": "Machine Learning",
    }

    resolver = WikiLinkResolver(existing_notes=notes, alias_map=aliases, confidence_threshold=0.65)

    # 1. Exact match
    target, score = resolver.resolve("PostgreSQL")
    assert target == "PostgreSQL"
    assert score == 1.0

    # 2. Alias match
    target, score = resolver.resolve("AWS")
    assert target == "Amazon Web Services"
    assert score >= 0.95

    # 3. Singular vs Plural match
    target, score = resolver.resolve("Distributed System")
    assert target == "Distributed Systems"
    assert score >= 0.85

    # 4. Safe formatting
    formatted = resolver.format_wikilink_safe("AWS", display_text="AWS setup")
    assert formatted == "[[Amazon Web Services|AWS setup]]"

    # 5. Low-confidence unknown entity
    target, score = resolver.resolve("completely_unknown_concept_12345")
    assert target is None
    assert score < 0.65
