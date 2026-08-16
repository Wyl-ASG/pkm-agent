"""Structure-aware Markdown chunker for PKM knowledge base notes."""

from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path
import re
from typing import Any
import yaml

logger = logging.getLogger(__name__)


def extract_wikilinks(text: str) -> list[str]:
    """Extract Obsidian WikiLink target note names from text, stripping aliases."""
    matches = re.findall(r"\[\[([^\]\|]+)(?:\|[^\]]+)?\]\]", text)
    return [m.strip() for m in matches if m.strip()]


def extract_block_id(text: str) -> str | None:
    """Extract Obsidian block identifier anchor from text."""
    match = re.search(r"\^([a-zA-Z0-9_-]+)\s*$", text.strip())
    return match.group(1) if match else None


def extract_dataview_fields(text: str) -> dict[str, str]:
    """Extract Dataview inline fields formatted as [key:: value] from text."""
    fields: dict[str, str] = {}
    matches = re.findall(r"\[([a-zA-Z0-9_\-\s]+)::\s*([^\]]+)\]", text)
    for k, v in matches:
        fields[k.strip()] = v.strip()
    return fields


@dataclass
class MarkdownChunk:
    """Represents a structured, contextual text chunk extracted from a Markdown file."""

    content: str
    file_path: str
    file_name: str
    title: str
    heading: str
    heading_path: list[str]
    chunk_index: int
    tags: list[str] = field(default_factory=list)
    wikilinks: list[str] = field(default_factory=list)
    block_ids: list[str] = field(default_factory=list)
    created: str | None = None
    modified: str | None = None
    memory_type: str = "user_authored"  # fact, observation, decision, task, ai_summary, etc.
    source_type: str = "user_authored"  # user_authored, ai_generated, imported_external

    def to_dict(self) -> dict[str, Any]:
        """Convert chunk into payload dictionary."""
        return {
            "content": self.content,
            "file_path": self.file_path,
            "file_name": self.file_name,
            "title": self.title,
            "heading": self.heading,
            "heading_path": self.heading_path,
            "chunk_index": self.chunk_index,
            "tags": self.tags,
            "wikilinks": self.wikilinks,
            "block_ids": self.block_ids,
            "created": self.created,
            "modified": self.modified,
            "memory_type": self.memory_type,
            "source_type": self.source_type,
        }


def extract_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Extract YAML frontmatter and remaining body text from Markdown.

    Args:
        text: Raw Markdown document string.

    Returns:
        Tuple of (frontmatter_dict, remaining_body_text).
    """
    if not text.startswith("---"):
        return {}, text

    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text

    yaml_text = parts[1]
    body = parts[2].strip()

    try:
        data = yaml.safe_load(yaml_text)
        if isinstance(data, dict):
            normalized_fm = {}
            for k, v in data.items():
                if hasattr(v, "isoformat"):
                    normalized_fm[k] = v.isoformat()
                else:
                    normalized_fm[k] = v
            return normalized_fm, body
    except Exception as err:
        logger.debug("Failed to parse YAML frontmatter: %s", err)

    return {}, body


def extract_tags_from_text(text: str) -> list[str]:
    """Extract #hashtags from text, excluding URLs or headings."""
    tags = set()
    matches = re.findall(r"(?<!\w)#([a-zA-Z0-9_\-\/]+)\b", text)
    for m in matches:
        if not m.isdigit():
            tags.add(f"#{m.lower()}")
    return sorted(tags)


class StructureAwareChunker:
    """Structure-aware chunker that understands headings, code blocks, lists, callouts, and block IDs."""

    def __init__(self, max_chunk_chars: int = 1500, min_chunk_chars: int = 80) -> None:
        """Initialize chunker.

        Args:
            max_chunk_chars: Target maximum character length per chunk.
            min_chunk_chars: Minimum character length to form a standalone chunk.
        """
        self.max_chunk_chars = max_chunk_chars
        self.min_chunk_chars = min_chunk_chars

    def chunk_file(self, file_path: Path, vault_path: Path) -> list[MarkdownChunk]:
        """Parse and chunk a Markdown file into structured MarkdownChunk objects.

        Args:
            file_path: Path to Markdown file.
            vault_path: Vault root directory path.

        Returns:
            List of MarkdownChunk instances.
        """
        try:
            raw_text = file_path.read_text(encoding="utf-8")
        except Exception as err:
            logger.exception("Failed reading file %s: %s", file_path, err)
            return []

        rel_path = str(file_path.relative_to(vault_path))
        file_name = file_path.name
        title = file_path.stem

        try:
            modified_time = datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
        except OSError:
            modified_time = datetime.now().isoformat()

        frontmatter, body = extract_frontmatter(raw_text)

        created_time = str(frontmatter.get("created", "")) or modified_time
        fm_memory_type = str(frontmatter.get("memory_type", "")).lower()
        fm_source_type = str(frontmatter.get("source_type", "")).lower()

        # Frontmatter tags
        fm_tags = frontmatter.get("tags", [])
        base_tags = set()
        if isinstance(fm_tags, list):
            for t in fm_tags:
                base_tags.add(f"#{str(t).lstrip('#').lower()}")
        elif isinstance(fm_tags, str):
            for t in fm_tags.split(","):
                clean = t.strip()
                if clean:
                    base_tags.add(f"#{clean.lstrip('#').lower()}")

        if not body.strip():
            return []

        lines = body.splitlines()
        chunks: list[MarkdownChunk] = []
        chunk_idx = 0

        heading_stack: list[tuple[int, str]] = [(1, title)]
        current_block_lines: list[str] = []
        current_heading = title

        def get_heading_path() -> list[str]:
            return [h[1] for h in heading_stack]

        def flush_current_block() -> None:
            nonlocal chunk_idx, current_block_lines
            if not current_block_lines:
                return

            block_text = "\n".join(current_block_lines).strip()
            if not block_text:
                current_block_lines = []
                return

            block_wikilinks = extract_wikilinks(block_text)
            block_tags = sorted(base_tags.union(extract_tags_from_text(block_text)))

            found_block_ids = []
            for b_line in current_block_lines:
                b_id = extract_block_id(b_line)
                if b_id:
                    found_block_ids.append(f"^{b_id}")

            chunk_memory = fm_memory_type or ("task" if "- [ ]" in block_text or "- [x]" in block_text else "user_authored")
            chunk_source = fm_source_type or "user_authored"

            chunk = MarkdownChunk(
                content=block_text,
                file_path=rel_path,
                file_name=file_name,
                title=title,
                heading=current_heading,
                heading_path=get_heading_path(),
                chunk_index=chunk_idx,
                tags=block_tags,
                wikilinks=block_wikilinks,
                block_ids=found_block_ids,
                created=created_time,
                modified=modified_time,
                memory_type=chunk_memory,
                source_type=chunk_source,
            )
            chunks.append(chunk)
            chunk_idx += 1
            current_block_lines = []

        in_code_block = False

        for line in lines:
            stripped = line.strip()

            if stripped.startswith("```"):
                in_code_block = not in_code_block
                current_block_lines.append(line)
                continue

            if in_code_block:
                current_block_lines.append(line)
                continue

            heading_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
            if heading_match:
                flush_current_block()

                level = len(heading_match.group(1))
                h_text = heading_match.group(2).strip()
                current_heading = h_text

                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, h_text))

                current_block_lines.append(line)
                continue

            current_block_lines.append(line)
            current_chars = sum(len(l) + 1 for l in current_block_lines)

            if current_chars >= self.max_chunk_chars and stripped == "":
                flush_current_block()

        flush_current_block()
        return chunks


def chunk_markdown_file_structured(file_path: Path, vault_path: Path) -> list[dict[str, Any]]:
    """Convenience function returning chunk dictionaries matching new schema."""
    chunker = StructureAwareChunker()
    chunks = chunker.chunk_file(file_path, vault_path)
    return [c.to_dict() for c in chunks]


__all__ = [
    "MarkdownChunk",
    "StructureAwareChunker",
    "chunk_markdown_file_structured",
    "extract_frontmatter",
    "extract_tags_from_text",
    "extract_wikilinks",
    "extract_block_id",
    "extract_dataview_fields",
]
