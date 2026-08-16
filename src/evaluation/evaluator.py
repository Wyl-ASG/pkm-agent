"""Evaluation framework measuring Recall@K, MRR, citation accuracy, and latency."""

import asyncio
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any

from src.agents.qa import KnowledgeBaseQAAgent

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Aggregated retrieval and QA evaluation metrics."""

    total_queries: int
    recall_at_5: float
    recall_at_10: float
    mrr: float  # Mean Reciprocal Rank
    citation_accuracy: float
    avg_latency_ms: float
    reranker_enabled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "reranker_enabled": self.reranker_enabled,
            "recall_at_5": f"{self.recall_at_5:.4f}",
            "recall_at_10": f"{self.recall_at_10:.4f}",
            "mrr": f"{self.mrr:.4f}",
            "citation_accuracy": f"{self.citation_accuracy:.4f}",
            "avg_latency_ms": f"{self.avg_latency_ms:.1f}",
        }


class RetrievalEvaluator:
    """Evaluates retrieval accuracy and QA citation grounding."""

    def __init__(self, qa_agent: KnowledgeBaseQAAgent | None = None) -> None:
        """Initialize RetrievalEvaluator."""
        self.qa_agent = qa_agent or KnowledgeBaseQAAgent()

    def _match_source(self, retrieved_source: str, expected_source: str) -> bool:
        """Check if retrieved source matches expected note or path."""
        r_clean = retrieved_source.replace("[[", "").replace("]]", "").replace(".md", "").lower().strip()
        e_clean = expected_source.replace(".md", "").split("/")[-1].lower().strip()
        return r_clean == e_clean or e_clean in r_clean or r_clean in e_clean

    async def evaluate_dataset(
        self,
        dataset: list[dict[str, Any]],
        reranker_enabled: bool | None = None,
    ) -> EvaluationMetrics:
        """Run evaluation over a test dataset containing queries and expected sources.

        Args:
            dataset: List of dicts with 'question' and 'expected_sources'.
            reranker_enabled: Optional override for reranker active state.

        Returns:
            EvaluationMetrics instance with Recall@5, Recall@10, MRR, citation accuracy.
        """
        if not dataset:
            return EvaluationMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, False)

        # Apply reranker override if provided
        prev_reranker_state = None
        if reranker_enabled is not None and hasattr(self.qa_agent.retriever, "reranker"):
            prev_reranker_state = self.qa_agent.retriever.reranker.enabled
            self.qa_agent.retriever.reranker.enabled = reranker_enabled

        is_reranker_on = (
            self.qa_agent.retriever.reranker.enabled
            if hasattr(self.qa_agent.retriever, "reranker")
            else False
        )

        hits_at_5 = 0
        hits_at_10 = 0
        reciprocal_ranks = []
        citation_hits = 0
        total_latencies = 0.0

        try:
            for item in dataset:
                question = item["question"]
                expected_sources = item.get("expected_sources", [])

                t0 = time.time()
                res = await self.qa_agent.query(question, top_k=10)
                elapsed_ms = (time.time() - t0) * 1000
                total_latencies += elapsed_ms

                retrieved_sources = [c["file_path"] for c in res.context_chunks]

                # Evaluate Recall@5, Recall@10, MRR
                found_rank = None
                for rank, r_src in enumerate(retrieved_sources, 1):
                    if any(self._match_source(r_src, exp) for exp in expected_sources):
                        if rank <= 5:
                            hits_at_5 += 1
                        if rank <= 10:
                            hits_at_10 += 1
                        found_rank = rank
                        break

                if found_rank:
                    reciprocal_ranks.append(1.0 / found_rank)
                else:
                    reciprocal_ranks.append(0.0)

                # Evaluate QA Citation Accuracy
                cited = [c.note_title for c in res.citations]
                if any(any(self._match_source(c_name, exp) for exp in expected_sources) for c_name in cited):
                    citation_hits += 1

            n = len(dataset)
            return EvaluationMetrics(
                total_queries=n,
                recall_at_5=hits_at_5 / n if n else 0.0,
                recall_at_10=hits_at_10 / n if n else 0.0,
                mrr=sum(reciprocal_ranks) / n if n else 0.0,
                citation_accuracy=citation_hits / n if n else 0.0,
                avg_latency_ms=total_latencies / n if n else 0.0,
                reranker_enabled=is_reranker_on,
            )
        finally:
            if prev_reranker_state is not None and hasattr(self.qa_agent.retriever, "reranker"):
                self.qa_agent.retriever.reranker.enabled = prev_reranker_state

    async def evaluate_file(
        self,
        file_path: str | Path,
        reranker_enabled: bool | None = None,
    ) -> EvaluationMetrics:
        """Run evaluation from JSON dataset file with optional reranker toggle."""
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Evaluation dataset file not found: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        return await self.evaluate_dataset(data, reranker_enabled=reranker_enabled)


__all__ = ["RetrievalEvaluator", "EvaluationMetrics"]
