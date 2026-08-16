"""Automated real-time file watcher for Obsidian vault with debouncing and incremental updates."""

import asyncio
from datetime import datetime
import logging
from pathlib import Path
import threading
import time
from typing import Any
import uuid

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from src.config import settings
from src.graphrag.chunker import StructureAwareChunker
from src.graphrag.embedder import TextEmbedder
from src.graphrag.graph import VaultKnowledgeGraph
from src.graphrag.reindexer import IGNORED_DIRS
from src.graphrag.sparse import BM25Index
from src.graphrag.vector_db import QdrantVectorStore, get_vector_store
from src.graphrag.versioning import (
    HASH_REGISTRY_FILENAME,
    IndexVersionRegistry,
    compute_file_hash,
)
from src.utils.resources import resource_manager

logger = logging.getLogger(__name__)


class VaultChangeHandler(FileSystemEventHandler):
    """Event handler that detects Markdown file changes, debounces rapid writes, and performs incremental reindexing."""

    def __init__(
        self,
        vault_path: Path,
        embedder: TextEmbedder,
        vector_store: QdrantVectorStore,
        bm25_index: BM25Index | None = None,
        knowledge_graph: VaultKnowledgeGraph | None = None,
        debounce_seconds: float = 2.0,
    ) -> None:
        """Initialize VaultChangeHandler."""
        super().__init__()
        self.vault_path = vault_path.resolve()
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.knowledge_graph = knowledge_graph
        self.debounce_seconds = debounce_seconds
        self.chunker = StructureAwareChunker()
        self.registry = IndexVersionRegistry(self.vault_path / HASH_REGISTRY_FILENAME)
        self._pending_files: dict[str, float] = {}
        self._pending_deletes: set[str] = set()
        self._lock = threading.Lock()
        self._running = False
        self._worker_task: asyncio.Task | None = None
        self._worker_thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Start the debounce worker task or thread."""
        self._running = True
        if loop is not None:
            self._loop = loop
        else:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = None

        if self._loop and self._loop.is_running():
            self._worker_task = self._loop.create_task(self._async_debounce_worker())
        else:
            self._worker_thread = threading.Thread(target=self._sync_debounce_worker, daemon=True)
            self._worker_thread.start()

    def on_modified(self, event: FileSystemEvent) -> None:
        """Triggered on file modification."""
        if not event.is_directory:
            self._handle_event_path(event.src_path)

    def on_created(self, event: FileSystemEvent) -> None:
        """Triggered on file creation."""
        if not event.is_directory:
            self._handle_event_path(event.src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        """Triggered on file deletion."""
        if not event.is_directory:
            self._handle_delete_path(event.src_path)

    def _should_ignore(self, path: Path) -> bool:
        """Check if file should be ignored from indexing."""
        if path.suffix.lower() != ".md":
            return True
        if path.name.startswith("."):
            return True
        try:
            rel = path.relative_to(self.vault_path)
            if any(part in IGNORED_DIRS or part.startswith(".") for part in rel.parts):
                return True
        except ValueError:
            return True
        return False

    def _handle_event_path(self, raw_path: str) -> None:
        """Filter event path and add to debounce queue."""
        path = Path(raw_path).resolve()
        if self._should_ignore(path):
            return

        with self._lock:
            self._pending_files[str(path)] = time.time() + self.debounce_seconds
            self._pending_deletes.discard(str(path))

    def _handle_delete_path(self, raw_path: str) -> None:
        """Handle file deletion queueing."""
        path = Path(raw_path).resolve()
        if self._should_ignore(path):
            return

        with self._lock:
            self._pending_files.pop(str(path), None)
            self._pending_deletes.add(str(path))

    async def _async_handle_delete_file(self, path: Path) -> None:
        """Remove deleted file metadata from indices under semaphore."""
        try:
            rel_path = str(path.relative_to(self.vault_path))
            if self.bm25_index:
                await asyncio.to_thread(self.bm25_index.remove_file_chunks, rel_path)
            if self.knowledge_graph:
                await asyncio.to_thread(self.knowledge_graph.remove_file_note, path)
            await asyncio.to_thread(self.registry.load)
            if rel_path in self.registry.records:
                del self.registry.records[rel_path]
                await asyncio.to_thread(self.registry.save)
            logger.info("Removed deleted file '%s' from indices.", rel_path)
        except Exception as err:
            logger.debug("Error handling delete path %s: %s", path, err)

    async def _async_debounce_worker(self) -> None:
        """Async worker that polls debounced queue and processes ready files under background job semaphore."""
        while self._running:
            try:
                await asyncio.sleep(0.2)
            except asyncio.CancelledError:
                break

            ready_paths: list[str] = []
            ready_deletes: list[str] = []
            now = time.time()

            with self._lock:
                for path_str, fire_time in list(self._pending_files.items()):
                    if now >= fire_time:
                        ready_paths.append(path_str)
                        del self._pending_files[path_str]

                ready_deletes = list(self._pending_deletes)
                self._pending_deletes.clear()

            if not ready_paths and not ready_deletes:
                continue

            async with resource_manager.background_job_semaphore:
                for del_path_str in ready_deletes:
                    try:
                        await self._async_handle_delete_file(Path(del_path_str))
                    except Exception as err:
                        logger.exception("Error processing deleted file %s: %s", del_path_str, err)

                for path_str in ready_paths:
                    try:
                        await self._async_process_single_file(Path(path_str))
                    except Exception as err:
                        logger.exception("Error processing watched file %s: %s", path_str, err)

    def _sync_debounce_worker(self) -> None:
        """Fallback background thread worker when no external asyncio loop is active."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._async_debounce_worker())
        finally:
            loop.close()

    async def _async_process_single_file(self, file_path: Path) -> None:
        """Compute hash and reindex single file incrementally."""
        if not file_path.exists():
            return

        try:
            rel_path = str(file_path.relative_to(self.vault_path))
        except ValueError:
            return

        current_hash = await asyncio.to_thread(compute_file_hash, file_path)
        await asyncio.to_thread(self.registry.load)

        if not self.registry.is_file_stale(
            rel_path=rel_path,
            current_hash=current_hash,
            embedding_model=self.embedder.model_name,
        ):
            return  # No content changes

        logger.info("File watcher detected incremental modification in: %s", rel_path)
        chunks = await asyncio.to_thread(self.chunker.chunk_file, file_path, self.vault_path)
        if not chunks:
            return

        chunk_texts = [c.content for c in chunks]
        embeddings = await self.embedder.encode_batch_async(chunk_texts)

        documents_to_upsert: list[dict[str, Any]] = []
        for chunk, vector in zip(chunks, embeddings):
            doc_id = str(uuid.uuid5(
                uuid.NAMESPACE_DNS,
                f"{chunk.file_path}#{chunk.chunk_index}",
            ))
            documents_to_upsert.append({
                "id": doc_id,
                "vector": vector,
                "content": chunk.content,
                "metadata": chunk.to_dict(),
            })

        # 1. Update Qdrant
        await self.vector_store.upsert_documents_async(documents_to_upsert)

        # 2. Update BM25 incrementally
        if self.bm25_index:
            await asyncio.to_thread(self.bm25_index.upsert_file_chunks, rel_path, documents_to_upsert)

        # 3. Update Knowledge Graph incrementally
        if self.knowledge_graph:
            await asyncio.to_thread(self.knowledge_graph.update_file_note, file_path)

        # 4. Save Versioned Registry
        self.registry.record_indexed(
            rel_path=rel_path,
            content_hash=current_hash,
            embedding_model=self.embedder.model_name,
            chunks_count=len(chunks),
        )
        await asyncio.to_thread(self.registry.save)
        logger.info("Real-time watcher incrementally updated %d chunks for '%s'", len(documents_to_upsert), rel_path)

    def _process_single_file(self, file_path: Path) -> None:
        """Synchronous wrapper for legacy callers."""
        asyncio.run(self._async_process_single_file(file_path))

    def stop(self) -> None:
        """Stop background worker."""
        self._running = False
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)


class VaultWatcher:
    """Manager for real-time watchdog filesystem observer with debouncing and incremental updates."""

    def __init__(
        self,
        vault_path: str | Path | None = None,
        embedder: TextEmbedder | None = None,
        vector_store: QdrantVectorStore | None = None,
        bm25_index: BM25Index | None = None,
        knowledge_graph: VaultKnowledgeGraph | None = None,
        debounce_seconds: float = 2.0,
    ) -> None:
        """Initialize VaultWatcher."""
        self.vault_path = Path(vault_path or settings.VAULT_PATH).resolve()
        self.embedder = embedder or TextEmbedder(model_name=settings.EMBEDDING_MODEL_NAME)
        self.vector_store = vector_store or get_vector_store()
        self.bm25_index = bm25_index
        self.knowledge_graph = knowledge_graph
        self.debounce_seconds = debounce_seconds
        self._observer: Observer | None = None
        self._handler: VaultChangeHandler | None = None

    def start(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Start watchdog filesystem observer."""
        if not self.vault_path.exists():
            logger.warning("Vault path %s does not exist; file watcher not started.", self.vault_path)
            return

        try:
            self._handler = VaultChangeHandler(
                vault_path=self.vault_path,
                embedder=self.embedder,
                vector_store=self.vector_store,
                bm25_index=self.bm25_index,
                knowledge_graph=self.knowledge_graph,
                debounce_seconds=self.debounce_seconds,
            )
            self._handler.start(loop=loop)
            self._observer = Observer()
            self._observer.schedule(self._handler, str(self.vault_path), recursive=True)
            self._observer.start()
            logger.info("Vault real-time file watcher started on %s", self.vault_path)
        except Exception as err:
            logger.exception("Failed to start VaultWatcher observer: %s", err)

    def stop(self) -> None:
        """Stop watchdog filesystem observer."""
        if self._handler:
            self._handler.stop()
        if self._observer:
            try:
                self._observer.stop()
                self._observer.join(timeout=3.0)
                logger.info("Vault real-time file watcher stopped successfully.")
            except Exception as err:
                logger.warning("Error stopping file watcher: %s", err)


__all__ = ["VaultWatcher", "VaultChangeHandler"]
