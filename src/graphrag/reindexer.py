"""Automated Vault Re-indexing Engine for PKM GraphRAG."""

import asyncio
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import re
from typing import Any
import uuid

from src.config import settings
from src.graphrag.embedder import TextEmbedder
from src.graphrag.vector_db import QdrantVectorStore

logger = logging.getLogger(__name__)

HASH_REGISTRY_FILENAME = ".vault_hashes.json"
IGNORED_DIRS = {".git", ".obsidian", ".trash", ".agent", ".venv", "__pycache__"}


def compute_file_hash(file_path: Path) -> str:
    """Compute MD5 hash of file content.

    Args:
        file_path: Path to target file.

    Returns:
        Hexadecimal MD5 hash string.
    """
    content_bytes = file_path.read_bytes()
    return hashlib.md5(content_bytes).hexdigest()


def load_hash_registry(registry_path: Path) -> dict[str, str]:
    """Load existing hash registry from JSON file.

    Args:
        registry_path: Path to .vault_hashes.json file.

    Returns:
        Dictionary mapping relative file paths to MD5 hashes.
    """
    if not registry_path.exists():
        return {}

    try:
        content = registry_path.read_text(encoding="utf-8")
        data = json.loads(content)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
        return {}
    except (json.JSONDecodeError, OSError) as err:
        logger.warning("Could not read hash registry at %s (%s); starting fresh.", registry_path, err)
        return {}


def save_hash_registry(registry_path: Path, registry: dict[str, str]) -> None:
    """Save updated hash registry to JSON file.

    Args:
        registry_path: Path to .vault_hashes.json file.
        registry: Dictionary mapping relative file paths to MD5 hashes.
    """
    try:
        registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8")
        logger.info("Saved hash registry (%d entries) to %s", len(registry), registry_path)
    except OSError as err:
        logger.exception("Failed to save hash registry to %s: %s", registry_path, err)


def chunk_markdown_file(file_path: Path, vault_path: Path) -> list[dict[str, Any]]:
    """Parse a Markdown file into contextual text chunks with metadata.

    Args:
        file_path: Absolute or relative path to the Markdown file.
        vault_path: Path to the vault root directory.

    Returns:
        List of chunk dictionaries with 'content', 'header', 'file_path', 'file_name',
        'last_updated', and 'chunk_index'.
    """
    try:
        raw_content = file_path.read_text(encoding="utf-8")
    except Exception as err:
        logger.exception("Failed to read file %s for chunking: %s", file_path, err)
        return []

    rel_path = str(file_path.relative_to(vault_path))
    file_name = file_path.name
    try:
        last_updated = datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
    except OSError:
        last_updated = datetime.now().isoformat()

    chunks: list[dict[str, Any]] = []

    # Strip YAML frontmatter if present
    content = raw_content
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            content = parts[2].strip()

    # Split by markdown headers (# Header, ## Header, etc.)
    header_pattern = re.compile(r"^(#{1,6}\s+.+)$", re.MULTILINE)
    sections = header_pattern.split(content)

    chunk_index = 0
    default_header = file_path.stem

    if sections:
        first_text = sections[0].strip()
        if first_text:
            chunks.append({
                "content": first_text,
                "header": default_header,
                "file_path": rel_path,
                "file_name": file_name,
                "last_updated": last_updated,
                "chunk_index": chunk_index,
            })
            chunk_index += 1
        idx = 1
    else:
        idx = 0

    while idx < len(sections):
        header_line = sections[idx].strip()
        header_title = header_line.lstrip("#").strip()
        section_body = sections[idx + 1].strip() if idx + 1 < len(sections) else ""
        idx += 2

        if not section_body:
            continue

        full_chunk_text = f"{header_line}\n{section_body}"
        chunks.append({
            "content": full_chunk_text,
            "header": header_title,
            "file_path": rel_path,
            "file_name": file_name,
            "last_updated": last_updated,
            "chunk_index": chunk_index,
        })
        chunk_index += 1

    # Fallback for non-empty files with no parsed chunks
    if not chunks and raw_content.strip():
        chunks.append({
            "content": raw_content.strip(),
            "header": default_header,
            "file_path": rel_path,
            "file_name": file_name,
            "last_updated": last_updated,
            "chunk_index": 0,
        })

    return chunks


class VaultReindexer:
    """Automated re-indexing engine for Obsidian vault Markdown files."""

    def __init__(
        self,
        vault_path: str | Path | None = None,
        embedder: TextEmbedder | None = None,
        vector_store: QdrantVectorStore | None = None,
    ) -> None:
        """Initialize VaultReindexer.

        Args:
            vault_path: Path to local vault directory. Defaults to settings.VAULT_PATH.
            embedder: TextEmbedder instance. Defaults to new TextEmbedder.
            vector_store: QdrantVectorStore instance. Defaults to new QdrantVectorStore.
        """
        self.vault_path = Path(vault_path or settings.VAULT_PATH).resolve()
        self.embedder = embedder or TextEmbedder()
        self.vector_store = vector_store or QdrantVectorStore()
        self.registry_path = self.vault_path / HASH_REGISTRY_FILENAME

    def scan_vault_files(self) -> list[Path]:
        """Recursively scan vault path for valid Markdown (.md) files.

        Returns:
            List of Path objects for all scanned .md files.
        """
        if not self.vault_path.exists():
            logger.warning("Vault path %s does not exist; skipping scan.", self.vault_path)
            return []

        md_files: list[Path] = []
        try:
            for path in self.vault_path.rglob("*.md"):
                # Filter out ignored directories or hidden files
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

    async def reindex_vault(self, force: bool = False) -> dict[str, int]:
        """Scan vault, compute content hashes, and upsert new/modified note chunks to Qdrant.

        Args:
            force: If True, re-indexes all files regardless of cached hashes.

        Returns:
            Dictionary containing indexing statistics ('scanned', 'indexed', 'skipped', 'chunks').
        """
        logger.info("Starting vault re-indexing for path: %s (force=%s)", self.vault_path, force)

        if not self.vault_path.exists():
            logger.warning("Vault path %s does not exist; creating directory.", self.vault_path)
            self.vault_path.mkdir(parents=True, exist_ok=True)

        hash_registry = {} if force else load_hash_registry(self.registry_path)
        updated_registry = dict(hash_registry)

        scanned_files = self.scan_vault_files()
        files_to_process: list[tuple[Path, str]] = []

        scanned_count = len(scanned_files)
        skipped_count = 0

        for file_path in scanned_files:
            try:
                rel_path = str(file_path.relative_to(self.vault_path))
                current_hash = compute_file_hash(file_path)

                if force or hash_registry.get(rel_path) != current_hash:
                    files_to_process.append((file_path, current_hash))
                else:
                    skipped_count += 1
            except Exception as err:
                logger.exception("Failed to hash file %s: %s", file_path, err)

        indexed_files_count = len(files_to_process)
        total_chunks_indexed = 0

        if not files_to_process:
            logger.info("Vault re-indexing complete: all %d files are up-to-date.", scanned_count)
            return {
                "scanned": scanned_count,
                "indexed": 0,
                "skipped": skipped_count,
                "chunks": 0,
            }

        logger.info("Found %d new/modified files to index in vault.", indexed_files_count)

        # Extract chunks for all modified files
        all_chunks: list[dict[str, Any]] = []
        file_chunk_counts: dict[str, int] = {}

        for file_path, current_hash in files_to_process:
            rel_path = str(file_path.relative_to(self.vault_path))
            chunks = chunk_markdown_file(file_path, self.vault_path)
            file_chunk_counts[rel_path] = len(chunks)
            all_chunks.extend(chunks)
            updated_registry[rel_path] = current_hash

        if all_chunks:
            chunk_texts = [c["content"] for c in all_chunks]
            logger.info("Generating vector embeddings for %d text chunks...", len(chunk_texts))

            try:
                embeddings = await self.embedder.encode_batch_async(chunk_texts)
            except Exception as err:
                logger.exception("Failed to generate batch embeddings during re-indexing: %s", err)
                raise

            # Build Qdrant points payload
            documents: list[dict[str, Any]] = []
            for chunk, vec in zip(all_chunks, embeddings):
                doc_id = str(uuid.uuid5(
                    uuid.NAMESPACE_DNS,
                    f"{chunk['file_path']}#{chunk['chunk_index']}",
                ))
                documents.append({
                    "id": doc_id,
                    "vector": vec,
                    "content": chunk["content"],
                    "metadata": {
                        "file_path": chunk["file_path"],
                        "file_name": chunk["file_name"],
                        "header": chunk["header"],
                        "last_updated": chunk["last_updated"],
                        "chunk_index": chunk["chunk_index"],
                    },
                })

            try:
                logger.info("Upserting %d chunk documents into Qdrant...", len(documents))
                await self.vector_store.upsert_documents_async(documents)
                total_chunks_indexed = len(documents)
            except Exception as err:
                logger.exception("Failed to upsert chunk documents into Qdrant: %s", err)
                raise

        # Save updated hash registry
        save_hash_registry(self.registry_path, updated_registry)

        stats = {
            "scanned": scanned_count,
            "indexed": indexed_files_count,
            "skipped": skipped_count,
            "chunks": total_chunks_indexed,
        }
        logger.info("Vault re-indexing finished successfully: %s", stats)
        return stats

    def reindex_vault_sync(self, force: bool = False) -> dict[str, int]:
        """Synchronous wrapper for reindex_vault."""
        return asyncio.run(self.reindex_vault(force=force))


async def reindex_vault(
    vault_path: str | Path | None = None,
    embedder: TextEmbedder | None = None,
    vector_store: QdrantVectorStore | None = None,
    force: bool = False,
) -> dict[str, int]:
    """Execute vault re-indexing asynchronously.

    Args:
        vault_path: Optional path to vault. Defaults to settings.VAULT_PATH.
        embedder: Optional TextEmbedder instance.
        vector_store: Optional QdrantVectorStore instance.
        force: If True, forces re-indexing of all files.

    Returns:
        Statistics dictionary with 'scanned', 'indexed', 'skipped', and 'chunks'.
    """
    reindexer = VaultReindexer(
        vault_path=vault_path,
        embedder=embedder,
        vector_store=vector_store,
    )
    return await reindexer.reindex_vault(force=force)
