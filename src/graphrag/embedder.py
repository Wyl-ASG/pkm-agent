"""SentenceTransformers text embedding wrapper."""

import asyncio
import logging
from typing import Sequence
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class TextEmbedder:
    """Wrapper around SentenceTransformer for generating dense text vector embeddings."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        """Initialize TextEmbedder instance.

        Args:
            model_name: HuggingFace model identifier. Defaults to 'all-MiniLM-L6-v2'.
        """
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load SentenceTransformer model instance."""
        if self._model is None:
            logger.info("Loading SentenceTransformer model '%s'", self.model_name)
            try:
                self._model = SentenceTransformer(self.model_name)
            except Exception as err:
                logger.exception("Failed to load SentenceTransformer model '%s': %s", self.model_name, err)
                raise
        return self._model

    @property
    def vector_size(self) -> int:
        """Return the vector embedding dimension size."""
        size = self.model.get_sentence_embedding_dimension()
        return size if size is not None else 384

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
            embeddings = self.model.encode(list(texts), convert_to_numpy=True)
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
        """Asynchronously generate dense vector embedding for a text string."""
        return await asyncio.to_thread(self.encode, text)

    async def encode_batch_async(self, texts: Sequence[str]) -> list[list[float]]:
        """Asynchronously generate dense vector embeddings for a batch of text strings."""
        return await asyncio.to_thread(self.encode_batch, texts)

