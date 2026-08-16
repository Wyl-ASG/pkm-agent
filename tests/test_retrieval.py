"""Unit tests for sparse BM25, dense retrieval, RRF, reranking, and temporal filtering."""

import pytest
from src.graphrag.reranker import CrossEncoderReranker
from src.graphrag.retriever import HybridRetriever, extract_temporal_filters
from src.graphrag.sparse import BM25Index, tokenize_for_bm25


def test_tokenize_for_bm25():
    text = "Setting up [[FastAPI]] with [[PostgreSQL]] database for #backend."
    tokens = tokenize_for_bm25(text)
    assert "fastapi" in tokens
    assert "postgresql" in tokens
    assert "database" in tokens


def test_bm25_index_search():
    index = BM25Index()
    docs = [
        {"id": "doc1", "content": "FastAPI is a modern web framework for Python."},
        {"id": "doc2", "content": "PostgreSQL is a powerful open-source relational database."},
        {"id": "doc3", "content": "Qdrant is a vector database for semantic search."},
    ]
    index.build_index(docs)

    results = index.search("relational database postgresql", top_k=2)
    assert len(results) > 0
    assert results[0]["id"] == "doc2"


def test_extract_temporal_filters():
    assert extract_temporal_filters("What did I work on on 2026-08-15?") == {"date": "2026-08-15"}
    assert extract_temporal_filters("Notes from 2026-05") == {"month": "2026-05"}
    assert extract_temporal_filters("What are my tasks for today?") is not None


def test_reciprocal_rank_fusion():
    retriever = HybridRetriever()

    dense_hits = [
        {"id": "docA", "score": 0.95, "payload": {"title": "Doc A"}},
        {"id": "docB", "score": 0.85, "payload": {"title": "Doc B"}},
    ]
    sparse_hits = [
        {"id": "docB", "score": 12.0, "payload": {"title": "Doc B"}},
        {"id": "docC", "score": 8.0, "payload": {"title": "Doc C"}},
    ]

    fused = retriever.reciprocal_rank_fusion(dense_hits, sparse_hits, k_constant=60.0)
    assert len(fused) == 3
    # docB was ranked in both dense and sparse, so it should have highest fused score
    assert fused[0]["id"] == "docB"


def test_reranker_fallback():
    # If reranker disabled, it should gracefully return top_k candidates without error
    reranker = CrossEncoderReranker(enabled=False)
    candidates = [
        {"id": "1", "content": "First item"},
        {"id": "2", "content": "Second item"},
    ]
    reranked = reranker.rerank("query text", candidates, top_k=1)
    assert len(reranked) == 1
    assert reranked[0]["id"] == "1"
