"""Automated Vault Re-indexing Engine for PKM GraphRAG with versioned caching and graph building."""

import asyncio
from datetime import datetime
import logging
from pathlib import Path
from typing import Any
import uuid

from src.config import settings
from src.graphrag.chunker import StructureAwareChunker, chunk_markdown_file_structured
from src.graphrag.embedder import TextEmbedder
from src.graphrag.graph import VaultKnowledgeGraph
from src.graphrag.versioning import (
    CURRENT_CHUNKER_VERSION,
    CURRENT_EMBEDDING_VERSION,
    CURRENT_PARSER_VERSION,
    HASH_REGISTRY_FILENAME,
    IndexVersionRegistry,
    compute_file_hash,
)
from src.graphrag.vector_db import QdrantVectorStore, get_vector_store

logger = logging.getLogger(__name__)
IGNORED_DIRS = {".git", ".obsidian", ".trash", ".agent", ".venv", "__pycache__"}


def chunk_markdown_file(file_path: Path, vault_path: Path) -> list[dict[str, Any]]:
    """Backward-compatible helper parsing a Markdown file into structured chunk dictionaries."""
    return chunk_markdown_file_structured(file_path, vault_path)


class VaultReindexer:
    """Automated re-indexing engine for Obsidian vault Markdown files with versioning."""

    def __init__(
        self,
        vault_path: str | Path | None = None,
        embedder: TextEmbedder | None = None,
        vector_store: QdrantVectorStore | None = None,
        knowledge_graph: VaultKnowledgeGraph | None = None,
    ) -> None:
        """Initialize VaultReindexer."""
        self.vault_path = Path(vault_path or settings.VAULT_PATH).resolve()
        self.embedder = embedder or TextEmbedder()
        self.vector_store = vector_store or get_vector_store()
        self.knowledge_graph = knowledge_graph or VaultKnowledgeGraph(self.vault_path)
        self.chunker = StructureAwareChunker()
        self.registry = IndexVersionRegistry(self.vault_path / HASH_REGISTRY_FILENAME)
        self.cached_chunks: list[dict[str, Any]] = []

    def scan_vault_files(self) -> list[Path]:
        """Recursively scan vault path for valid Markdown (.md) files."""
        if not self.vault_path.exists():
            logger.warning("Vault path %s does not exist; skipping scan.", self.vault_path)
            return []

        md_files: list[Path] = []
        try:
            for path in self.vault_path.rglob("*.md"):
                parts = set(path.relative_to(self.vault_path).parts)
                if parts.intersection(IGNORED_DIRS):
                    continue
                if path.name.startswith("."):
                    continue
                if path.is_file():
                    md_files.append(path)
        except OSError as err:
            logger.exception("Error scanning vault directory %s: %s", self.vault_path, err)

        return sorted(md_files)

    def _evaluate_and_chunk_files(
        self,
        scanned_files: list[Path],
        force: bool,
        current_model_name: str,
    ) -> tuple[list[tuple[Path, str, list[Any]]], list[dict[str, Any]], int]:
        """Scan, hash, and chunk vault files synchronously on a worker thread."""
        files_to_process: list[tuple[Path, str, list[Any]]] = []
        all_vault_chunks: list[dict[str, Any]] = []
        skipped_count = 0

        for file_path in scanned_files:
            try:
                rel_path = str(file_path.relative_to(self.vault_path))
                current_hash = compute_file_hash(file_path)
                file_chunks = self.chunker.chunk_file(file_path, self.vault_path)
                for c in file_chunks:
                    doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{c.file_path}#{c.chunk_index}"))
                    all_vault_chunks.append({
                        "id": doc_id,
                        "content": c.content,
                        "metadata": c.to_dict(),
                    })

                is_stale = force or self.registry.is_file_stale(
                    rel_path=rel_path,
                    current_hash=current_hash,
                    embedding_model=current_model_name,
                )
                if is_stale:
                    files_to_process.append((file_path, current_hash, file_chunks))
                else:
                    skipped_count += 1
            except Exception as err:
                logger.exception("Failed to evaluate file %s during reindex: %s", file_path, err)

        return files_to_process, all_vault_chunks, skipped_count

    async def reindex_vault(self, force: bool = False) -> dict[str, Any]:
        """Scan vault, detect stale files via versioned registry, upsert chunks to Qdrant, and update graph."""
        logger.info("Starting vault re-indexing (path=%s, force=%s)", self.vault_path, force)

        if not self.vault_path.exists():
            self.vault_path.mkdir(parents=True, exist_ok=True)

        scanned_files = await asyncio.to_thread(self.scan_vault_files)
        scanned_count = len(scanned_files)

        # Reload registry
        await asyncio.to_thread(self.registry.load)
        current_model_name = self.embedder.model_name

        files_to_process, all_vault_chunks, skipped_count = await asyncio.to_thread(
            self._evaluate_and_chunk_files, scanned_files, force, current_model_name
        )
        self.cached_chunks = all_vault_chunks

        # Update Knowledge Graph asynchronously
        try:
            await asyncio.to_thread(self.knowledge_graph.build_graph)
        except Exception as graph_err:
            logger.warning("Knowledge Graph build notice: %s", graph_err)

        indexed_files_count = len(files_to_process)
        total_chunks_indexed = 0

        if not files_to_process:
            logger.info("Vault re-indexing complete: all %d files up-to-date.", scanned_count)
            return {
                "scanned": scanned_count,
                "indexed": 0,
                "skipped": skipped_count,
                "chunks": 0,
                "total_vault_chunks": len(all_vault_chunks),
            }

        logger.info("Found %d new or modified files to index in vault.", indexed_files_count)

        # Process chunks for modified files
        new_documents: list[dict[str, Any]] = []
        for file_path, current_hash, chunks in files_to_process:
            rel_path = str(file_path.relative_to(self.vault_path))
            for c in chunks:
                doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{c.file_path}#{c.chunk_index}"))
                new_documents.append({
                    "id": doc_id,
                    "content": c.content,
                    "metadata": c.to_dict(),
                })
            self.registry.record_indexed(
                rel_path=rel_path,
                content_hash=current_hash,
                embedding_model=current_model_name,
                chunks_count=len(chunks),
            )

        if new_documents:
            chunk_texts = [d["content"] for d in new_documents]
            logger.info("Generating embeddings for %d chunks using '%s'...", len(chunk_texts), current_model_name)
            embeddings = await self.embedder.encode_batch_async(chunk_texts)

            qdrant_payloads: list[dict[str, Any]] = []
            for doc, vec in zip(new_documents, embeddings):
                qdrant_payloads.append({
                    "id": doc["id"],
                    "vector": vec,
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                })

            await self.vector_store.upsert_documents_async(qdrant_payloads)
            total_chunks_indexed = len(qdrant_payloads)

        # Save updated versioned registry asynchronously
        await asyncio.to_thread(self.registry.save)

        stats = {
            "scanned": scanned_count,
            "indexed": indexed_files_count,
            "skipped": skipped_count,
            "chunks": total_chunks_indexed,
            "total_vault_chunks": len(all_vault_chunks),
        }
        logger.info("Vault re-indexing completed successfully: %s", stats)
        return stats

    def reindex_vault_sync(self, force: bool = False) -> dict[str, Any]:
        """Synchronous wrapper for reindex_vault."""
        return asyncio.run(self.reindex_vault(force=force))


async def reindex_vault(
    vault_path: str | Path | None = None,
    embedder: TextEmbedder | None = None,
    vector_store: QdrantVectorStore | None = None,
    knowledge_graph: VaultKnowledgeGraph | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Execute vault re-indexing asynchronously."""
    reindexer = VaultReindexer(
        vault_path=vault_path,
        embedder=embedder,
        vector_store=vector_store,
        knowledge_graph=knowledge_graph,
    )
    return await reindexer.reindex_vault(force=force)


__all__ = [
    "VaultReindexer",
    "reindex_vault",
    "chunk_markdown_file",
    "IGNORED_DIRS",
    "HASH_REGISTRY_FILENAME",
]
