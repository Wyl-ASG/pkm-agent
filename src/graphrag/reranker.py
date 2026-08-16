"""Cross-Encoder reranking module for second-stage candidate precision."""

import asyncio
import gc
import logging
from typing import Any

from src.config import settings
from src.utils.resources import resource_manager

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Cross-encoder reranker for refining hybrid retrieval candidate pools."""

    def __init__(
        self,
        model_name: str | None = None,
        enabled: bool | None = None,
    ) -> None:
        """Initialize CrossEncoderReranker.

        Args:
            model_name: HuggingFace model identifier. Defaults to settings.RERANKER_MODEL_NAME.
            enabled: Flag to enable/disable reranking. Defaults to settings.RERANKER_ENABLED.
        """
        self.model_name = model_name or getattr(
            settings, "RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self.enabled = (
            enabled if enabled is not None else getattr(settings, "RERANKER_ENABLED", False)
        )
        self._model = None
        self._load_attempted = False

    @property
    def is_loaded(self) -> bool:
        """Check if CrossEncoder model is loaded in memory."""
        return self._model is not None

    def _get_model(self):
        """Lazy load CrossEncoder model instance safely with graceful fallback."""
        if not self.enabled:
            return None

        if self._model is not None:
            return self._model

        if self._load_attempted:
            return None

        self._load_attempted = True
        try:
            from sentence_transformers import CrossEncoder

            logger.info("Loading CrossEncoder reranker model '%s'...", self.model_name)
            self._model = CrossEncoder(self.model_name)
            logger.info("CrossEncoder reranker '%s' loaded successfully.", self.model_name)
            return self._model
        except Exception as err:
            logger.warning(
                "Could not load CrossEncoder model '%s' (%s); falling back to RRF rankings.",
                self.model_name,
                err,
            )
            self._model = None
            return None

    def unload_model(self) -> None:
        """Unload CrossEncoder model from RAM and run garbage collection."""
        if self._model is not None:
            logger.info("Unloading CrossEncoder model from RAM.")
            self._model = None
            self._load_attempted = False
            gc.collect()

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Rerank candidates using CrossEncoder score.

        Args:
            query: Query string.
            candidates: List of candidate result dictionaries.
            top_k: Number of top reranked items to return.

        Returns:
            Reranked list of candidate dictionaries with updated 'rerank_score'.
        """
        if not candidates:
            return []

        if not self.enabled:
            return candidates[:top_k]

        model = self._get_model()
        if model is None:
            # Fallback to existing RRF ranking
            return candidates[:top_k]

        try:
            pairs = []
            for c in candidates:
                content = c.get("content") or c.get("payload", {}).get("content", "")
                pairs.append((query, content))

            scores = model.predict(pairs)

            # Attach rerank score and sort
            scored_candidates = []
            for item, score in zip(candidates, scores):
                item_copy = dict(item)
                item_copy["rerank_score"] = float(score)
                scored_candidates.append(item_copy)

            scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
            return scored_candidates[:top_k]

        except Exception as err:
            logger.warning("Reranking failed (%s); falling back to RRF candidates.", err)
            return candidates[:top_k]

    async def rerank_async(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Asynchronously rerank candidates guarded by concurrency semaphore."""
        if not self.enabled or not candidates:
            return candidates[:top_k]

        async with resource_manager.reranker_semaphore:
            return await asyncio.to_thread(self.rerank, query, candidates, top_k)


__all__ = ["CrossEncoderReranker"]
