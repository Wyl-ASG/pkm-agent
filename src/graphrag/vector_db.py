"""Thread-safe Singleton vector database manager for Qdrant with dynamic collection sizing."""

import asyncio
import logging
from pathlib import Path
import threading
import time
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
    _op_lock: threading.RLock = threading.RLock()

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
        """Initialize shared Qdrant client once."""
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
            with self._op_lock:
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
        """Ensure Qdrant collection exists with the target vector dimension size."""
        with self._op_lock:
            try:
                if self.client.collection_exists(self.collection_name):
                    info = self.client.get_collection(self.collection_name)
                    existing_size = None
                    if info.config and info.config.params and info.config.params.vectors:
                        vec_params = info.config.params.vectors
                        if isinstance(vec_params, models.VectorParams):
                            existing_size = vec_params.size
                        elif isinstance(vec_params, dict) and "" in vec_params:
                            existing_size = vec_params[""].size

                    if existing_size is not None and existing_size != vector_size:
                        logger.error(
                            "Collection '%s' vector dimension mismatch (existing=%s, required=%d).",
                            self.collection_name,
                            existing_size,
                            vector_size,
                        )
                        raise ValueError(
                            f"Vector dimension mismatch: collection '{self.collection_name}' has size {existing_size} "
                            f"but configured size is {vector_size}. Manually delete the collection or migrate data "
                            f"before proceeding."
                        )

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
                    except Exception as index_err:
                        logger.debug("Payload index creation notice for '%s': %s", self.collection_name, index_err)

            except UnexpectedResponse as err:
                logger.error("Qdrant unexpected response while ensuring collection: %s", err)
                raise
            except Exception as err:
                logger.exception("Error ensuring Qdrant collection '%s': %s", self.collection_name, err)
                raise

    def upsert_documents(self, documents: list[dict[str, Any]]) -> bool:
        """Upsert documents into Qdrant vector store with concurrency locking and retry resilience."""
        if not documents:
            return True

        first_vec = documents[0].get("vector")
        vec_dim = len(first_vec) if first_vec else DEFAULT_VECTOR_SIZE

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

        with self._op_lock:
            self.ensure_collection(vector_size=vec_dim)
            max_retries = 3
            for attempt in range(1, max_retries + 1):
                try:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points,
                    )
                    logger.info("Successfully upserted %d points into collection '%s'", len(points), self.collection_name)
                    return True
                except Exception as err:
                    if attempt < max_retries:
                        logger.warning(
                            "Transient error upserting points to Qdrant (attempt %d/%d): %s; retrying...",
                            attempt,
                            max_retries,
                            err,
                        )
                        time.sleep(0.2 * attempt)
                    else:
                        logger.exception("Failed to upsert points into Qdrant collection '%s' after %d attempts: %s", self.collection_name, max_retries, err)
                        return False
            return False

    def search_dense(
        self,
        query_vector: list[float],
        top_k: int = 30,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Dense semantic vector similarity search."""
        with self._op_lock:
            self.ensure_collection(vector_size=len(query_vector))
            try:
                query_filter = None
                if filters:
                    conditions = []
                    for k, v in filters.items():
                        conditions.append(models.FieldCondition(key=k, match=models.MatchValue(value=v)))
                    query_filter = models.Filter(must=conditions)

                dense_response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    query_filter=query_filter,
                    limit=top_k,
                    score_threshold=score_threshold,
                )
                points = dense_response.points if hasattr(dense_response, "points") else dense_response

                results = []
                for p in points:
                    results.append({
                        "id": p.id,
                        "score": getattr(p, "score", 0.0),
                        "dense_score": getattr(p, "score", 0.0),
                        "payload": p.payload or {},
                        "content": (p.payload or {}).get("content", ""),
                    })
                return results
            except Exception as err:
                logger.exception("Error executing dense search in Qdrant: %s", err)
                return []

    def hybrid_search(
        self,
        query_text: str,
        query_vector: list[float],
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Legacy hybrid search combining dense similarity with text matching via RRF."""
        with self._op_lock:
            self.ensure_collection(vector_size=len(query_vector))
            try:
                dense_hits = self.search_dense(query_vector=query_vector, top_k=top_k * 2, score_threshold=score_threshold)

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

                # Reciprocal Rank Fusion
                rrf_scores: dict[str | int, float] = {}
                payload_map: dict[str | int, Any] = {}
                k_constant = 60.0

                for rank, item in enumerate(dense_hits):
                    pid = item["id"]
                    rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (1.0 / (k_constant + rank + 1))
                    payload_map[pid] = item["payload"]

                for rank, point in enumerate(text_hits):
                    pid = point.id
                    rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (1.5 / (k_constant + rank + 1))
                    if pid not in payload_map:
                        payload_map[pid] = point.payload or {}

                sorted_ids = sorted(rrf_scores.keys(), key=lambda pid: rrf_scores[pid], reverse=True)
                results = []
                for pid in sorted_ids[:top_k]:
                    payload = payload_map[pid]
                    results.append({
                        "id": pid,
                        "score": rrf_scores[pid],
                        "payload": payload,
                        "content": payload.get("content", ""),
                    })
                return results
            except Exception as err:
                logger.exception("Failed to execute hybrid search in Qdrant: %s", err)
                raise

    async def upsert_documents_async(self, documents: list[dict[str, Any]]) -> bool:
        """Asynchronously upsert documents."""
        return await asyncio.to_thread(self.upsert_documents, documents)

    async def search_dense_async(
        self,
        query_vector: list[float],
        top_k: int = 30,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Asynchronously execute dense search."""
        return await asyncio.to_thread(
            self.search_dense, query_vector, top_k, filters, score_threshold
        )

    async def hybrid_search_async(
        self,
        query_text: str,
        query_vector: list[float],
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Asynchronously execute hybrid search."""
        return await asyncio.to_thread(
            self.hybrid_search, query_text, query_vector, top_k, score_threshold
        )


def get_vector_store(
    collection_name: str = DEFAULT_COLLECTION_NAME,
    storage_path: str | Path | None = None,
) -> QdrantVectorStore:
    """Return singleton instance of QdrantVectorStore."""
    return QdrantVectorStore(
        collection_name=collection_name,
        storage_path=storage_path,
    )
