import asyncio
import logging
from pathlib import Path
import threading
from typing import Any
import uuid
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from src.config import settings

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION_NAME = "pkm_notes"
DEFAULT_VECTOR_SIZE = 384


class QdrantVectorStore:
    """Thread-safe Singleton vector database manager for Qdrant.

    Supports embedded disk storage (zero Docker dependency), remote server,
    or in-memory mode, ensuring only a single QdrantClient instance is shared.
    """

    _instance: "QdrantVectorStore | None" = None
    _lock: threading.Lock = threading.Lock()

    def __new__(
        cls,
        host: str | None = None,
        port: int | None = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        in_memory: bool = False,
        storage_path: str | Path | None = None,
    ) -> "QdrantVectorStore":
        """Ensure only one shared QdrantVectorStore instance exists across the process."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        in_memory: bool = False,
        storage_path: str | Path | None = None,
    ) -> None:
        """Initialize shared Qdrant client once.

        Args:
            host: Qdrant host address. Defaults to settings.QDRANT_HOST.
            port: Qdrant port. Defaults to settings.QDRANT_PORT.
            collection_name: Target Qdrant collection name.
            in_memory: Force in-memory mode (:memory:).
            storage_path: Local directory path for embedded Qdrant storage mode.
        """
        if getattr(self, "_initialized", False):
            return

        with self._lock:
            if getattr(self, "_initialized", False):
                return

            self.collection_name = collection_name
            self._host = host or settings.QDRANT_HOST
            self._port = port or settings.QDRANT_PORT
            self._in_memory = in_memory
            self._storage_path = storage_path or settings.QDRANT_STORAGE_PATH
            self._client: QdrantClient | None = None
            self._init_client()
            self._initialized = True

    def _init_client(self) -> None:
        """Initialize the single underlying QdrantClient instance."""
        if self._in_memory or self._host == ":memory:":
            logger.info("Initializing in-memory Qdrant client")
            self._client = QdrantClient(":memory:")
            return

        # Embedded local disk storage (zero Docker dependency)
        if self._storage_path:
            storage_dir = Path(self._storage_path).resolve()
            storage_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Initializing embedded local Qdrant client at %s", storage_dir)
            try:
                self._client = QdrantClient(path=str(storage_dir))
                return
            except Exception as err:
                logger.error("Failed to initialize embedded Qdrant client at %s: %s", storage_dir, err)
                raise

        # Remote server connection
        try:
            logger.info("Connecting to Qdrant server at %s:%s", self._host, self._port)
            self._client = QdrantClient(host=self._host, port=self._port, timeout=10.0)
            self._client.get_collections()
        except Exception as err:
            logger.warning(
                "Could not connect to Qdrant server at %s:%s (%s); falling back to in-memory client.",
                self._host,
                self._port,
                err,
            )
            self._client = QdrantClient(":memory:")

    @property
    def client(self) -> QdrantClient:
        """Return the initialized shared QdrantClient instance."""
        if self._client is None:
            self._init_client()
        return self._client

    def close(self) -> None:
        """Safely close Qdrant storage connection on shutdown."""
        with self._lock:
            if self._client is not None:
                try:
                    self._client.close()
                    logger.info("Closed Qdrant client storage connection")
                except Exception as err:
                    logger.debug("Error closing Qdrant client: %s", err)
                self._client = None
            self._initialized = False
            QdrantVectorStore._instance = None

    def ensure_collection(self, vector_size: int = DEFAULT_VECTOR_SIZE) -> None:
        """Ensure Qdrant collection exists with dense vector config and full-text payload index.

        Args:
            vector_size: Size of dense embedding vectors. Defaults to 384.
        """
        try:
            if not self.client.collection_exists(self.collection_name):
                logger.info("Creating Qdrant collection '%s' (vector_size=%d)", self.collection_name, vector_size)
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                    ),
                )

            # Create full-text payload index on 'content' field if running against remote server
            if not self._storage_path and not self._in_memory and self._host != ":memory:":
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="content",
                        field_schema=models.TextIndexParams(
                            type=models.TextIndexType.TEXT,
                            tokenizer=models.TokenizerType.WORD,
                            lowercase=True,
                        ),
                    )
                    logger.info("Full-text payload index on 'content' field created for '%s'", self.collection_name)
                except Exception as index_err:
                    logger.debug("Payload index creation notice for '%s': %s", self.collection_name, index_err)

        except UnexpectedResponse as err:
            logger.error("Qdrant unexpected response while ensuring collection: %s", err)
            raise
        except Exception as err:
            logger.exception("Error ensuring Qdrant collection '%s': %s", self.collection_name, err)
            raise

    def upsert_documents(self, documents: list[dict[str, Any]]) -> bool:
        """Upsert documents into Qdrant vector store.

        Args:
            documents: List of dicts containing 'vector', 'content', and optional 'id', 'metadata'.

        Returns:
            True if upsert succeeded, False otherwise.
        """
        if not documents:
            return True

        self.ensure_collection()

        points: list[models.PointStruct] = []
        for idx, doc in enumerate(documents):
            vector = doc.get("vector")
            if not vector:
                logger.warning("Document at index %d missing 'vector'; skipping.", idx)
                continue

            content = doc.get("content", "")
            doc_id = doc.get("id")
            if doc_id is None:
                doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{content}_{idx}"))
            elif isinstance(doc_id, str):
                try:
                    uuid.UUID(doc_id)
                except ValueError:
                    doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id))

            payload = doc.get("metadata", {})
            if isinstance(payload, dict):
                payload = dict(payload)
            else:
                payload = {}
            payload["content"] = content

            points.append(
                models.PointStruct(
                    id=doc_id,
                    vector=vector,
                    payload=payload,
                )
            )

        if not points:
            return False

        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            logger.info("Successfully upserted %d points into collection '%s'", len(points), self.collection_name)
            return True
        except Exception as err:
            logger.exception("Failed to upsert points into Qdrant collection '%s': %s", self.collection_name, err)
            return False

    def upsert_documents_sync(self, documents: list[dict[str, Any]]) -> bool:
        """Synchronous alias for upsert_documents."""
        return self.upsert_documents(documents)

    def hybrid_search(
        self,
        query_text: str,
        query_vector: list[float],
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid search combining dense vector similarity with payload text matching.

        Args:
            query_text: Natural text query string for full-text payload match.
            query_vector: Dense embedding vector for semantic search.
            top_k: Maximum number of combined search results to return.
            score_threshold: Optional score threshold for vector similarity.

        Returns:
            List of result dictionaries containing id, score, dense_score, text_match flag, and payload.
        """
        self.ensure_collection(vector_size=len(query_vector))

        try:
            # 1. Dense vector search
            dense_response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k * 2,
                score_threshold=score_threshold,
            )
            dense_hits = dense_response.points if hasattr(dense_response, "points") else dense_response

            # 2. Text payload matching search
            text_hits: list[Any] = []
            clean_query_text = query_text.strip()
            if clean_query_text:
                text_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="content",
                            match=models.MatchText(text=clean_query_text),
                        )
                    ]
                )
                text_response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    query_filter=text_filter,
                    limit=top_k * 2,
                )
                text_hits = text_response.points if hasattr(text_response, "points") else text_response

            # 3. Reciprocal Rank Fusion (RRF) algorithm
            rrf_scores: dict[str | int, float] = {}
            point_map: dict[str | int, Any] = {}
            dense_score_map: dict[str | int, float] = {}
            text_matched_ids: set[str | int] = set()

            k_constant = 60.0

            for rank, point in enumerate(dense_hits):
                pid = point.id
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (1.0 / (k_constant + rank + 1))
                point_map[pid] = point
                dense_score_map[pid] = getattr(point, "score", 0.0)

            for rank, point in enumerate(text_hits):
                pid = point.id
                # Apply boost to points matching exact full-text payload criteria
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (1.5 / (k_constant + rank + 1))
                point_map[pid] = point
                text_matched_ids.add(pid)
                if pid not in dense_score_map:
                    dense_score_map[pid] = getattr(point, "score", 0.0)

            # Sort by fused score descending
            sorted_ids = sorted(rrf_scores.keys(), key=lambda pid: rrf_scores[pid], reverse=True)

            results: list[dict[str, Any]] = []
            for pid in sorted_ids[:top_k]:
                point = point_map[pid]
                results.append({
                    "id": pid,
                    "score": rrf_scores[pid],
                    "dense_score": dense_score_map.get(pid, 0.0),
                    "text_match": pid in text_matched_ids,
                    "payload": point.payload or {},
                })

            return results

        except Exception as err:
            logger.exception("Failed to execute hybrid search in Qdrant: %s", err)
            raise

    async def upsert_documents_async(self, documents: list[dict[str, Any]]) -> bool:
        """Asynchronously upsert documents into Qdrant vector store."""
        return await asyncio.to_thread(self.upsert_documents, documents)

    async def hybrid_search_async(
        self,
        query_text: str,
        query_vector: list[float],
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Asynchronously execute hybrid search in Qdrant."""
        return await asyncio.to_thread(
            self.hybrid_search, query_text, query_vector, top_k, score_threshold
        )


def get_vector_store(
    collection_name: str = DEFAULT_COLLECTION_NAME,
    storage_path: str | Path | None = None,
) -> QdrantVectorStore:
    """Return the singleton instance of QdrantVectorStore.

    Args:
        collection_name: Target Qdrant collection name. Defaults to 'pkm_notes'.
        storage_path: Optional path to embedded storage directory.

    Returns:
        Shared QdrantVectorStore singleton instance.
    """
    return QdrantVectorStore(
        collection_name=collection_name,
        storage_path=storage_path,
    )
