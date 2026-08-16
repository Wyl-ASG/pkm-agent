"""Index versioning and registry manager for PKM knowledge base."""

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from src.config import settings

logger = logging.getLogger(__name__)

CURRENT_EMBEDDING_VERSION = "2.0"
CURRENT_CHUNKER_VERSION = "2.0"
CURRENT_PARSER_VERSION = "2.0"
HASH_REGISTRY_FILENAME = ".vault_hashes.json"


@dataclass
class FileIndexRecord:
    """Metadata record representing the indexed state of a vault file."""

    path: str
    content_hash: str
    embedding_model: str
    embedding_version: str = CURRENT_EMBEDDING_VERSION
    chunker_version: str = CURRENT_CHUNKER_VERSION
    parser_version: str = CURRENT_PARSER_VERSION
    chunks_count: int = 0
    last_indexed: str = ""

    def is_stale(
        self,
        current_hash: str,
        current_embedding_model: str,
        current_embedding_version: str = CURRENT_EMBEDDING_VERSION,
        current_chunker_version: str = CURRENT_CHUNKER_VERSION,
        current_parser_version: str = CURRENT_PARSER_VERSION,
    ) -> bool:
        """Check if this file needs reindexing due to content hash or version changes."""
        if self.content_hash != current_hash:
            return True
        if self.embedding_model != current_embedding_model:
            return True
        if self.embedding_version != current_embedding_version:
            return True
        if self.chunker_version != current_chunker_version:
            return True
        if self.parser_version != current_parser_version:
            return True
        return False


def compute_file_hash(file_path: Path) -> str:
    """Compute MD5 hash of file contents."""
    content_bytes = file_path.read_bytes()
    return hashlib.md5(content_bytes).hexdigest()


class IndexVersionRegistry:
    """Manages versioned file hash and metadata registry in .vault_hashes.json."""

    def __init__(self, registry_path: Path) -> None:
        """Initialize IndexVersionRegistry."""
        self.registry_path = registry_path
        self.records: dict[str, FileIndexRecord] = {}
        self.global_metadata: dict[str, Any] = {
            "embedding_version": CURRENT_EMBEDDING_VERSION,
            "chunker_version": CURRENT_CHUNKER_VERSION,
            "parser_version": CURRENT_PARSER_VERSION,
            "last_updated": datetime.now().isoformat(),
        }
        self.load()

    def load(self) -> None:
        """Load registry from JSON file with backward-compatibility for flat dicts."""
        if not self.registry_path.exists():
            self.records = {}
            return

        try:
            content = self.registry_path.read_text(encoding="utf-8")
            data = json.loads(content)
            if not isinstance(data, dict):
                self.records = {}
                return

            # Check if this is new structured format or legacy flat format
            if "_metadata" in data or "files" in data:
                self.global_metadata = data.get("_metadata", self.global_metadata)
                raw_files = data.get("files", {})
                self.records = {}
                for path, item in raw_files.items():
                    if isinstance(item, dict):
                        self.records[path] = FileIndexRecord(
                            path=item.get("path", path),
                            content_hash=item.get("content_hash", ""),
                            embedding_model=item.get("embedding_model", ""),
                            embedding_version=item.get("embedding_version", "1.0"),
                            chunker_version=item.get("chunker_version", "1.0"),
                            parser_version=item.get("parser_version", "1.0"),
                            chunks_count=item.get("chunks_count", 0),
                            last_indexed=item.get("last_indexed", ""),
                        )
                    elif isinstance(item, str):
                        # Mixed entry
                        self.records[path] = FileIndexRecord(
                            path=path,
                            content_hash=item,
                            embedding_model="legacy",
                            embedding_version="1.0",
                            chunker_version="1.0",
                            parser_version="1.0",
                        )
            else:
                # Legacy flat format: {"Notes/Foo.md": "md5hash"}
                logger.info("Migrating legacy flat .vault_hashes.json to versioned format...")
                self.records = {}
                for path, hash_val in data.items():
                    self.records[str(path)] = FileIndexRecord(
                        path=str(path),
                        content_hash=str(hash_val),
                        embedding_model="legacy",
                        embedding_version="1.0",
                        chunker_version="1.0",
                        parser_version="1.0",
                    )

        except Exception as err:
            logger.warning("Could not parse index registry at %s: %s; starting fresh.", self.registry_path, err)
            self.records = {}

    def save(self) -> None:
        """Persist registry to JSON file safely."""
        try:
            self.global_metadata["last_updated"] = datetime.now().isoformat()
            payload = {
                "_metadata": self.global_metadata,
                "files": {path: asdict(rec) for path, rec in sorted(self.records.items())},
            }
            self.registry_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            logger.info("Saved index registry (%d files) to %s", len(self.records), self.registry_path)
        except OSError as err:
            logger.exception("Failed to save index registry to %s: %s", self.registry_path, err)

    def is_file_stale(self, rel_path: str, current_hash: str, embedding_model: str) -> bool:
        """Determine if a file needs reindexing."""
        if rel_path not in self.records:
            return True
        return self.records[rel_path].is_stale(
            current_hash=current_hash,
            current_embedding_model=embedding_model,
            current_embedding_version=CURRENT_EMBEDDING_VERSION,
            current_chunker_version=CURRENT_CHUNKER_VERSION,
            current_parser_version=CURRENT_PARSER_VERSION,
        )

    def record_indexed(
        self,
        rel_path: str,
        content_hash: str,
        embedding_model: str,
        chunks_count: int,
    ) -> None:
        """Record a successfully indexed file."""
        self.records[rel_path] = FileIndexRecord(
            path=rel_path,
            content_hash=content_hash,
            embedding_model=embedding_model,
            embedding_version=CURRENT_EMBEDDING_VERSION,
            chunker_version=CURRENT_CHUNKER_VERSION,
            parser_version=CURRENT_PARSER_VERSION,
            chunks_count=chunks_count,
            last_indexed=datetime.now().isoformat(),
        )

    def remove_file(self, rel_path: str) -> None:
        """Remove a file from the registry if deleted."""
        self.records.pop(rel_path, None)


__all__ = [
    "CURRENT_EMBEDDING_VERSION",
    "CURRENT_CHUNKER_VERSION",
    "CURRENT_PARSER_VERSION",
    "HASH_REGISTRY_FILENAME",
    "FileIndexRecord",
    "IndexVersionRegistry",
    "compute_file_hash",
]
