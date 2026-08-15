"""Automated real-time file watcher for Obsidian vault using watchdog and Qdrant."""

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
from src.graphrag.embedder import TextEmbedder
from src.graphrag.reindexer import (
    IGNORED_DIRS,
    HASH_REGISTRY_FILENAME,
    chunk_markdown_file,
    compute_file_hash,
    load_hash_registry,
    save_hash_registry,
)
from src.graphrag.vector_db import QdrantVectorStore, get_vector_store

logger = logging.getLogger(__name__)


class VaultChangeHandler(FileSystemEventHandler):
    """Event handler that detects Markdown file changes and queues incremental reindexing."""

    def __init__(
        self,
        vault_path: Path,
        embedder: TextEmbedder,
        vector_store: QdrantVectorStore,
        debounce_seconds: float = 2.0,
    ) -> None:
        """Initialize VaultChangeHandler.

        Args:
            vault_path: Root path of the vault.
            embedder: TextEmbedder instance.
            vector_store: QdrantVectorStore instance.
            debounce_seconds: Seconds to debounce repeated file write events.
        """
        super().__init__()
        self.vault_path = vault_path.resolve()
        self.embedder = embedder
        self.vector_store = vector_store
        self.debounce_seconds = debounce_seconds
        self._registry_path = self.vault_path / HASH_REGISTRY_FILENAME
        self._pending_files: dict[str, float] = {}
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._worker_thread = threading.Thread(target=self._debounce_worker, daemon=True)
        self._running = True
        self._worker_thread.start()

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Set running asyncio event loop for scheduling async vector store tasks."""
        self._loop = loop

    def on_modified(self, event: FileSystemEvent) -> None:
        """Triggered on file modification."""
        if not event.is_directory:
            self._handle_event_path(event.src_path)

    def on_created(self, event: FileSystemEvent) -> None:
        """Triggered on file creation."""
        if not event.is_directory:
            self._handle_event_path(event.src_path)

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

    def _debounce_worker(self) -> None:
        """Background thread worker that processes debounced file changes."""
        while self._running:
            time.sleep(0.5)
            ready_paths: list[str] = []
            now = time.time()

            with self._lock:
                for path_str, fire_time in list(self._pending_files.items()):
                    if now >= fire_time:
                        ready_paths.append(path_str)
                        del self._pending_files[path_str]

            for path_str in ready_paths:
                try:
                    self._process_single_file(Path(path_str))
                except Exception as err:
                    logger.exception("Error processing watched file %s: %s", path_str, err)

    def _process_single_file(self, file_path: Path) -> None:
        """Compute hash and reindex single file if changed."""
        if not file_path.exists():
            return

        try:
            rel_path = str(file_path.relative_to(self.vault_path))
        except ValueError:
            return

        current_hash = compute_file_hash(file_path)
        registry = load_hash_registry(self._registry_path)

        if registry.get(rel_path) == current_hash:
            return  # No content changes

        logger.info("File watcher detected content modification in: %s", rel_path)
        chunks = chunk_markdown_file(file_path, self.vault_path)
        if not chunks:
            return

        # Prepare payloads
        chunk_texts = [c["content"] for c in chunks]
        embeddings = self.embedder.encode_batch(chunk_texts)

        documents_to_upsert: list[dict[str, Any]] = []
        for chunk, vector in zip(chunks, embeddings):
            doc_id = str(uuid.uuid5(
                uuid.NAMESPACE_DNS,
                f"{chunk['file_path']}#{chunk['chunk_index']}",
            ))
            documents_to_upsert.append({
                "id": doc_id,
                "vector": vector,
                "content": chunk["content"],
                "metadata": {
                    "file_path": rel_path,
                    "file_name": chunk["file_name"],
                    "header": chunk["header"],
                    "last_updated": chunk["last_updated"],
                    "chunk_index": chunk["chunk_index"],
                    "source": "file_watcher",
                },
            })

        self.vector_store.upsert_documents(documents_to_upsert)
        registry[rel_path] = current_hash
        save_hash_registry(self._registry_path, registry)
        logger.info("Real-time watcher re-indexed %d chunks for '%s'", len(documents_to_upsert), rel_path)

    def stop(self) -> None:
        """Stop background worker."""
        self._running = False


class VaultWatcher:
    """Manager for real-time watchdog filesystem observer."""

    def __init__(
        self,
        vault_path: str | Path | None = None,
        embedder: TextEmbedder | None = None,
        vector_store: QdrantVectorStore | None = None,
    ) -> None:
        """Initialize VaultWatcher."""
        self.vault_path = Path(vault_path or settings.VAULT_PATH).resolve()
        self.embedder = embedder or TextEmbedder(model_name=settings.EMBEDDING_MODEL_NAME)
        self.vector_store = vector_store or get_vector_store()
        self._observer: Observer | None = None
        self._handler: VaultChangeHandler | None = None

    def start(self) -> None:
        """Start watchdog filesystem observer."""
        if not self.vault_path.exists():
            logger.warning("Vault path %s does not exist; file watcher not started.", self.vault_path)
            return

        try:
            self._handler = VaultChangeHandler(
                vault_path=self.vault_path,
                embedder=self.embedder,
                vector_store=self.vector_store,
            )
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


__all__ = ["VaultWatcher"]
