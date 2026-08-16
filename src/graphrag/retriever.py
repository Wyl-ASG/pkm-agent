"""Modern Multi-Stage Hybrid Retriever with BM25, Qdrant, Graph Expansion, and Reranking."""

import asyncio
from datetime import datetime, timedelta
import logging
import re
import time
from typing import Any

from src.config import settings
from src.graphrag.embedder import TextEmbedder
from src.graphrag.graph import VaultKnowledgeGraph
from src.graphrag.reranker import CrossEncoderReranker
from src.graphrag.sparse import BM25Index
from src.graphrag.vector_db import QdrantVectorStore, get_vector_store

logger = logging.getLogger(__name__)


def extract_temporal_filters(query: str) -> dict[str, str] | None:
    """Extract temporal intent or explicit dates from natural language query.

    Supports:
    - 'today', 'yesterday'
    - 'last week', 'this week', 'last month', 'this month'
    - specific dates like '2026-08', '2026-07-14', 'May 2026'
    """
    q_lower = query.lower().strip()
    now = datetime.now()

    # Exact date pattern YYYY-MM-DD
    date_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", query)
    if date_match:
        return {"date": date_match.group(1)}

    # Month pattern YYYY-MM
    month_match = re.search(r"\b(\d{4}-\d{2})\b", query)
    if month_match:
        return {"month": month_match.group(1)}

    if "today" in q_lower:
        return {"date": now.strftime("%Y-%m-%d")}
    elif "yesterday" in q_lower:
        yest = now - timedelta(days=1)
        return {"date": yest.strftime("%Y-%m-%d")}
    elif "last month" in q_lower:
        first_day_current_month = now.replace(day=1)
        last_day_prev_month = first_day_current_month - timedelta(days=1)
        return {"month": last_day_prev_month.strftime("%Y-%m")}
    elif "this month" in q_lower:
        return {"month": now.strftime("%Y-%m")}

    return None


class HybridRetriever:
    """High-performance multi-stage retriever coordinating dense semantic, sparse lexical, graph, and reranker stages."""

    def __init__(
        self,
        embedder: TextEmbedder | None = None,
        vector_store: QdrantVectorStore | None = None,
        bm25_index: BM25Index | None = None,
        knowledge_graph: VaultKnowledgeGraph | None = None,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        """Initialize HybridRetriever.

        Args:
            embedder: Dense vector embedder.
            vector_store: Qdrant vector database.
            bm25_index: BM25 sparse index.
            knowledge_graph: In-memory vault graph.
            reranker: Cross-encoder reranker.
        """
        self.embedder = embedder or TextEmbedder()
        self.vector_store = vector_store or get_vector_store()
        self.bm25_index = bm25_index or BM25Index()
        self.knowledge_graph = knowledge_graph or VaultKnowledgeGraph()
        self.reranker = reranker or CrossEncoderReranker()
        self._all_chunks_cache: dict[str, dict[str, Any]] = {}

    def update_chunk_cache(self, chunks: list[dict[str, Any]]) -> None:
        """Update local chunk cache and build BM25 index."""
        self._all_chunks_cache = {str(c.get("id", idx)): c for idx, c in enumerate(chunks)}
        self.bm25_index.build_index(chunks)

    def reciprocal_rank_fusion(
        self,
        dense_results: list[dict[str, Any]],
        sparse_results: list[dict[str, Any]],
        k_constant: float = 60.0,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> list[dict[str, Any]]:
        """Combine dense and sparse search rankings using Reciprocal Rank Fusion (RRF)."""
        rrf_scores: dict[str, float] = {}
        candidate_map: dict[str, dict[str, Any]] = {}
        dense_score_map: dict[str, float] = {}
        sparse_score_map: dict[str, float] = {}

        # 1. Score dense candidates
        for rank, item in enumerate(dense_results):
            cid = str(item.get("id", ""))
            if not cid:
                continue
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (dense_weight / (k_constant + rank + 1))
            candidate_map[cid] = item
            dense_score_map[cid] = float(item.get("dense_score", item.get("score", 0.0)))

        # 2. Score sparse candidates
        for rank, item in enumerate(sparse_results):
            cid = str(item.get("id", ""))
            if not cid:
                continue
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (sparse_weight / (k_constant + rank + 1))
            sparse_score_map[cid] = float(item.get("sparse_score", item.get("score", 0.0)))
            if cid not in candidate_map:
                candidate_map[cid] = item

        # Sort descending by fused score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        fused_candidates = []
        for cid in sorted_ids:
            cand = dict(candidate_map[cid])
            cand["rrf_score"] = rrf_scores[cid]
            cand["dense_score"] = dense_score_map.get(cid, 0.0)
            cand["sparse_score"] = sparse_score_map.get(cid, 0.0)
            fused_candidates.append(cand)

        return fused_candidates

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        expand_graph: bool = True,
    ) -> list[dict[str, Any]]:
        """Execute multi-stage hybrid search pipeline.

        Pipeline:
        1. Query preprocessing (temporal intent detection)
        2. Parallel Dense (Qdrant) + Sparse (BM25) search
        3. Reciprocal Rank Fusion (RRF) -> Candidate pool (30-50)
        4. Bounded Graph expansion (1-hop neighbors)
        5. Cross-Encoder reranking
        6. Top-k final context selection
        """
        start_time = time.time()
        clean_query = query.strip()
        if not clean_query:
            return []

        # 1. Preprocessing & temporal awareness
        temporal_info = extract_temporal_filters(clean_query) if getattr(settings, "TEMPORAL_ENABLED", True) else None

        dense_top_k = getattr(settings, "RETRIEVAL_DENSE_TOP_K", 30)
        sparse_top_k = getattr(settings, "RETRIEVAL_SPARSE_TOP_K", 30)
        reranker_top_k = getattr(settings, "RERANKER_TOP_K", 10)

        # 2. Parallel retrieval: Dense vector + Sparse BM25
        query_vec = await self.embedder.encode_async(clean_query)
        
        dense_hits, sparse_hits = await asyncio.gather(
            self.vector_store.search_dense_async(
                query_vector=query_vec,
                top_k=dense_top_k,
                filters=filters,
            ),
            asyncio.to_thread(self.bm25_index.search, clean_query, sparse_top_k),
        )

        # 3. Reciprocal Rank Fusion
        fused_candidates = self.reciprocal_rank_fusion(
            dense_results=dense_hits,
            sparse_results=sparse_hits,
        )

        rrf_candidate_count = len(fused_candidates)

        # 4. Temporal boosting if temporal query detected
        if temporal_info:
            target_date = temporal_info.get("date")
            target_month = temporal_info.get("month")
            for c in fused_candidates:
                payload = c.get("payload", {})
                created = payload.get("created", "") or payload.get("last_updated", "")
                f_path = payload.get("file_path", "")
                if target_date and (target_date in created or target_date in f_path):
                    c["rrf_score"] = c.get("rrf_score", 0.0) * 1.8
                elif target_month and (target_month in created or target_month in f_path):
                    c["rrf_score"] = c.get("rrf_score", 0.0) * 1.4

            fused_candidates.sort(key=lambda x: x.get("rrf_score", 0.0), reverse=True)

        # 5. Graph Expansion (bounded)
        graph_expanded_count = 0
        if expand_graph and getattr(settings, "GRAPH_ENABLED", True):
            top_notes = set()
            for c in fused_candidates[:5]:
                payload = c.get("payload", {})
                title = payload.get("title") or payload.get("file_name", "").replace(".md", "")
                if title:
                    top_notes.add(title)

            neighbor_titles = set()
            for t in top_notes:
                nb = self.knowledge_graph.get_neighbors(
                    t,
                    max_hops=getattr(settings, "GRAPH_MAX_HOPS", 1),
                    max_neighbors=getattr(settings, "GRAPH_MAX_NEIGHBORS", 3),
                )
                neighbor_titles.update(nb)

            # Find matching chunks for neighbor titles from chunk cache
            existing_ids = {str(c.get("id")) for c in fused_candidates}
            for nid, chunk in self._all_chunks_cache.items():
                if nid not in existing_ids:
                    payload = chunk.get("metadata", chunk.get("payload", {}))
                    chunk_title = payload.get("title") or payload.get("file_name", "").replace(".md", "")
                    if chunk_title in neighbor_titles:
                        expanded_item = {
                            "id": nid,
                            "content": chunk.get("content", ""),
                            "payload": payload,
                            "rrf_score": 0.015,  # Moderate entry score
                            "graph_expanded": True,
                        }
                        fused_candidates.append(expanded_item)
                        existing_ids.add(nid)
                        graph_expanded_count += 1

        # 6. Reranking Stage
        candidate_pool = fused_candidates[:reranker_top_k]
        reranked = await asyncio.to_thread(
            self.reranker.rerank,
            clean_query,
            candidate_pool,
            top_k,
        )

        latency_ms = int((time.time() - start_time) * 1000)

        if getattr(settings, "LOG_RETRIEVAL_DETAILS", True):
            logger.info(
                "Retrieval Stats [query='%s']: dense=%d, sparse=%d, rrf=%d, graph_exp=%d, reranked=%d, final=%d, latency=%dms",
                clean_query[:50],
                len(dense_hits),
                len(sparse_hits),
                rrf_candidate_count,
                graph_expanded_count,
                len(reranked),
                len(reranked[:top_k]),
                latency_ms,
            )

        return reranked[:top_k]


__all__ = ["HybridRetriever", "extract_temporal_filters"]
