"""Configurable SentenceTransformers text embedding wrapper with multi-device support and concurrency limiting."""

import asyncio
import gc
import logging
from typing import Sequence
import torch

from src.config import settings
from src.utils.resources import resource_manager

logger = logging.getLogger(__name__)

KNOWN_DIMENSIONS = {
    "all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "BAAI/bge-m3": 1024,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
}


def resolve_torch_device(device_setting: str = "cpu") -> str:
    """Resolve compute device based on availability and configuration."""
    clean = device_setting.lower().strip()
    if clean in ("cpu", "cuda", "mps"):
        return clean

    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TextEmbedder:
    """Wrapper around SentenceTransformer for generating dense text vector embeddings."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        """Initialize TextEmbedder instance.

        Args:
            model_name: HuggingFace model identifier. Defaults to settings.EMBEDDING_MODEL_NAME.
            device: Device name ('auto', 'cpu', 'cuda', 'mps'). Defaults to settings.EMBEDDING_DEVICE.
        """
        self.model_name = model_name or getattr(settings, "EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
        self.device_setting = device or getattr(settings, "EMBEDDING_DEVICE", "cpu")
        self.resolved_device = resolve_torch_device(self.device_setting)
        self._model = None
        import threading
        self._model_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        """Check if embedding model is currently loaded in memory."""
        return self._model is not None

    @property
    def model(self):
        """Lazy load SentenceTransformer model instance on configured device."""
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

            logger.info("Loading SentenceTransformer model '%s' on device '%s'", self.model_name, self.resolved_device)
            try:
                self._model = SentenceTransformer(self.model_name, device=self.resolved_device)
            except Exception as err:
                logger.warning(
                    "Failed to load '%s' on %s (%s); attempting fallback on CPU...",
                    self.model_name,
                    self.resolved_device,
                    err,
                )
                try:
                    self._model = SentenceTransformer(self.model_name, device="cpu")
                    self.resolved_device = "cpu"
                except Exception as fallback_err:
                    logger.exception("Failed to load SentenceTransformer model '%s': %s", self.model_name, fallback_err)
                    raise
        return self._model

    @property
    def vector_size(self) -> int:
        """Return the vector embedding dimension size."""
        if not self.is_loaded and self.model_name in KNOWN_DIMENSIONS:
            return KNOWN_DIMENSIONS[self.model_name]
        size = self.model.get_sentence_embedding_dimension()
        return int(size) if size is not None else 384

    def unload_model(self) -> None:
        """Unload embedding model from RAM."""
        if self._model is not None:
            logger.info("Unloading SentenceTransformer model '%s' from RAM.", self.model_name)
            self._model = None
            gc.collect()

    def encode(self, text: str) -> list[float]:
        """Generate dense vector embedding for a single text string synchronously.

        Args:
            text: Input text string.

        Returns:
            List of floats representing the embedding vector.
        """
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()  # type: ignore[no-any-return]
        except Exception as err:
            logger.exception("Failed to generate embedding vector: %s", err)
            raise

    def encode_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate dense vector embeddings for a batch of text strings synchronously.

        Args:
            texts: List of input text strings.

        Returns:
            List of embedding vector float lists.
        """
        if not texts:
            return []
        try:
            embeddings = self.model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
            return embeddings.tolist()  # type: ignore[no-any-return]
        except Exception as err:
            logger.exception("Failed to generate batch embedding vectors: %s", err)
            raise

    def encode_sync(self, text: str) -> list[float]:
        """Synchronous alias for encode."""
        return self.encode(text)

    def encode_batch_sync(self, texts: Sequence[str]) -> list[list[float]]:
        """Synchronous alias for encode_batch."""
        return self.encode_batch(texts)

    async def encode_async(self, text: str) -> list[float]:
        """Asynchronously generate dense vector embedding, guarded by concurrency semaphore."""
        async with resource_manager.embedding_semaphore:
            return await asyncio.to_thread(self.encode, text)

    async def encode_batch_async(self, texts: Sequence[str]) -> list[list[float]]:
        """Asynchronously generate dense vector embeddings, guarded by concurrency semaphore."""
        async with resource_manager.embedding_semaphore:
            return await asyncio.to_thread(self.encode_batch, texts)


__all__ = ["TextEmbedder", "resolve_torch_device"]
