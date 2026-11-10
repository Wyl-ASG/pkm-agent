"""Markdown file writer for Obsidian vault daily notes and atomic notes."""

import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import stat
import tempfile
import time
from typing import Any
import uuid

from filelock import FileLock

from src.agents.models import InterstitialEntry
from src.config import settings

logger = logging.getLogger(__name__)

STATE_DIR_NAME = ".pkm_state"
MAX_TASK_CORRECTIONS = 200


def atomic_write_text(path: Path, text: str) -> None:
    """Replace a file's contents atomically so a crash cannot leave a truncated note."""
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
        os.chmod(tmp_path, mode)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


@dataclass
class TaskItem:
    """Structured representation of a markdown task in Obsidian vault."""

    file_rel_path: str
    file_name: str
    line_number: int
    raw_line: str
    task_text: str
    completed: bool = False
    created_date: str | None = None
    due_date: str | None = None
    priority: str | None = None
    block_id: str | None = None
    daily_date: str | None = None
    task_id: str = ""

    def __post_init__(self) -> None:
        if not self.task_id:
            # Deterministic short 8-char hash for Telegram callback buttons (guaranteed <= 64 bytes)
            seed = f"{self.file_rel_path}:{self.line_number}:{self.task_text[:40]}"
            self.task_id = hashlib.md5(seed.encode("utf-8")).hexdigest()[:8]

    @property
    def source_note_display(self) -> str:
        """Display name of source note without .md extension."""
        return self.file_name[:-3] if self.file_name.endswith(".md") else self.file_name

    @property
    def clean_text_for_button(self) -> str:
        """Strip WikiLinks and extra symbols for clean plain text button display."""
        cleaned = re.sub(r"\[\[(?:[^\|\]]+\|)?([^\]]+)\]\]", r"\1", self.task_text)
        cleaned = re.sub(r"[#*`_~]", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if len(cleaned) > 36:
            cleaned = cleaned[:33] + "..."
        return cleaned


def parse_task_line(line: str, file_rel_path: str, file_name: str, line_number: int) -> TaskItem | None:
    """Parse a single markdown line into a TaskItem if it is a task checkbox."""
    stripped = line.strip()
    match = re.match(r"^[-*+]\s*\[([ xX])\]\s*(.*)$", stripped)
    if not match:
        return None

    completed_flag = match.group(1).lower() == "x"
    body = match.group(2).strip()

    # Extract created date ➕ YYYY-MM-DD
    created_match = re.search(r"➕\s*(\d{4}-\d{2}-\d{2})", body)
    created_date = created_match.group(1) if created_match else None

    # Extract due date (Obsidian Tasks 📅 YYYY-MM-DD or Dataview [due:: YYYY-MM-DD])
    due_match = re.search(r"📅\s*(\d{4}-\d{2}-\d{2})", body)
    if not due_match:
        due_match = re.search(r"\[due::\s*(\d{4}-\d{2}-\d{2})\]", body)
    due_date = due_match.group(1) if due_match else None

    # Extract block ID
    block_id = None
    block_match = re.search(r"\^([a-zA-Z0-9_-]+)\s*$", body)
    if block_match:
        block_id = block_match.group(1)

    # Extract priority
    priority = None
    prio_match = re.search(r"\[priority::\s*([^\]]+)\]", body, re.IGNORECASE)
    if prio_match:
        priority = prio_match.group(1).strip()
    elif re.search(r"#high\b", body, re.IGNORECASE) or "[#A]" in body:
        priority = "high"
    elif re.search(r"#medium\b", body, re.IGNORECASE) or "[#B]" in body:
        priority = "medium"
    elif re.search(r"#low\b", body, re.IGNORECASE) or "[#C]" in body:
        priority = "low"

    # Clean display text: remove dates, block IDs, dataview fields
    clean_text = body
    clean_text = re.sub(r"➕\s*\d{4}-\d{2}-\d{2}", "", clean_text)
    clean_text = re.sub(r"📅\s*\d{4}-\d{2}-\d{2}", "", clean_text)
    clean_text = re.sub(r"\[due::\s*[^\]]+\]", "", clean_text)
    clean_text = re.sub(r"\[category::\s*[^\]]+\]", "", clean_text)
    clean_text = re.sub(r"\[priority::\s*[^\]]+\]", "", clean_text)
    clean_text = re.sub(r"\[platform::\s*[^\]]+\]", "", clean_text)
    clean_text = re.sub(r"\[status::\s*[^\]]+\]", "", clean_text)
    clean_text = re.sub(r"\^[a-zA-Z0-9_-]+\s*$", "", clean_text)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()

    # Extract daily date if file is in Daily Notes or matches YYYY-MM-DD.md
    daily_date = None
    date_filename_match = re.match(r"^(\d{4}-\d{2}-\d{2})\.md$", file_name)
    if date_filename_match:
        daily_date = date_filename_match.group(1)

    return TaskItem(
        file_rel_path=file_rel_path,
        file_name=file_name,
        line_number=line_number,
        raw_line=line,
        task_text=clean_text,
        completed=completed_flag,
        created_date=created_date,
        due_date=due_date,
        priority=priority,
        block_id=block_id,
        daily_date=daily_date,
    )


from src.telegram.formatters import (
    format_daily_scheduled_message,
    format_pending_tasks_message,
)

# Standardized section headers specified in PKM coding conventions
HEADER_INBOX = "## 📥 Inbox (Quick Capture)"
HEADER_LOG = "## ⏱️ Log (Interstitial)"
HEADER_TASKS = "## 🎯 Priorities & Tasks"
HEADER_DISCOVERIES = "## 🧠 Discoveries & Learning"

CATEGORY_HEADER_MAP = {
    "inbox": HEADER_INBOX,
    "quick capture": HEADER_INBOX,
    "quick_capture": HEADER_INBOX,
    "capture": HEADER_INBOX,
    "log": HEADER_LOG,
    "interstitial": HEADER_LOG,
    "work": HEADER_LOG,
    "thought": HEADER_LOG,
    "journal": HEADER_LOG,
    "general": HEADER_LOG,
    "task": HEADER_TASKS,
    "tasks": HEADER_TASKS,
    "priority": HEADER_TASKS,
    "priorities": HEADER_TASKS,
    "todo": HEADER_TASKS,
    "discovery": HEADER_DISCOVERIES,
    "discoveries": HEADER_DISCOVERIES,
    "learning": HEADER_DISCOVERIES,
    "learn": HEADER_DISCOVERIES,
}


def sanitize_filename(filename: str) -> str:
    """Remove or replace characters invalid in file paths."""
    sanitized = re.sub(r'[\\/:*?"<>|]', "_", filename)
    return sanitized.strip()



def resolve_header(category: str) -> str:
    """Resolve entry category string to standardized daily note section header."""
    cleaned = category.lower().strip()
    return CATEGORY_HEADER_MAP.get(cleaned, HEADER_LOG)


def generate_daily_note_template(date_str: str) -> str:
    """Generate default Markdown template for a new Daily Note with YAML frontmatter."""
    return f"""---
created: {date_str}
type: daily-note
tags:
  - daily
---

# 📅 Daily Note: {date_str}

## 📥 Inbox (Quick Capture)

## ⏱️ Log (Interstitial)

## 🎯 Priorities & Tasks

## 🧠 Discoveries & Learning
"""


def format_interstitial_entry_line(
    entry: InterstitialEntry,
    date_str: str,
    atomic_note_title: str | None = None,
) -> str:
    """Format an InterstitialEntry into an Obsidian Markdown string conforming to the Obsidian Feature Rulebook.

    Supports:
    - Obsidian Tasks syntax: - [ ] {Task} ➕ YYYY-MM-DD 📅 YYYY-MM-DD
    - Dataview Inline Keys: [key:: value]
    - Block Identifiers: ^block-id at end of line
    - Callout syntax: > [!NOTE] or > [!WARNING]
    - WikiLink Aliases: [[Target|Alias]]
    """
    normalized_content = entry.content.strip()
    ts_display = entry.timestamp.split(" ")[-1] if " " in entry.timestamp else entry.timestamp

    is_task = entry.category.lower().strip() in ("task", "tasks", "priority", "priorities", "todo")
    content_is_task = normalized_content.lstrip().startswith("- [ ]") or normalized_content.lstrip().startswith("- [x]")
    content_is_callout = normalized_content.lstrip().startswith("> [!")

    # 1. Base content line
    if content_is_callout:
        base_line = normalized_content
    elif content_is_task:
        base_line = normalized_content
    elif is_task:
        clean_task_body = re.sub(r"^[-*+]\s*(\[\s*[xX]?\s*\])?\s*", "", normalized_content)
        clean_task_body = re.sub(r"➕\s*\d{4}-\d{2}-\d{2}", "", clean_task_body)
        clean_task_body = re.sub(r"📅\s*\d{4}-\d{2}-\d{2}", "", clean_task_body).strip()
        base_line = f"- [ ] {clean_task_body} ➕ {date_str}"
        if entry.due_date:
            base_line += f" 📅 {entry.due_date}"
    else:
        base_line = f"- **[{ts_display}]** {normalized_content}"

    # 2. Append Dataview inline fields: [key:: value]
    dv_items: list[str] = []
    if entry.dataview_fields:
        for k, v in entry.dataview_fields.items():
            dv_tag = f"[{k}:: {v}]"
            if dv_tag not in base_line:
                dv_items.append(dv_tag)
    if dv_items:
        base_line = f"{base_line} {' '.join(dv_items)}"

    # 3. Append Atomic Note inline link
    if atomic_note_title:
        atomic_link = f"[[{atomic_note_title}]]"
        if atomic_link not in base_line:
            base_line = f"{base_line} {atomic_link}"

    # 4. Append Block Identifier: ^block-id at the very end
    if entry.block_id:
        clean_id = entry.block_id.lstrip("^").strip()
        block_tag = f"^{clean_id}"
        if block_tag not in base_line:
            base_line = f"{base_line} {block_tag}"

    # 5. Wrap in Callout if callout_type is specified and not already formatted
    if entry.callout_type and not content_is_callout:
        callout_name = entry.callout_type.upper().strip()
        callout_lines = [
            f"> [!{callout_name}]",
            f"> {base_line}",
        ]
        return "\n".join(callout_lines)

    return base_line


def _read_json_state(path: Path) -> dict[str, dict[str, Any]]:
    """Read a JSON state file, treating a missing file as empty."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _task_line_to_text(line: str) -> str:
    """Remove task checkbox and date syntax from a line, keeping the user's words and links."""
    text = re.sub(r"^\s*[-*+]\s*\[[ xX]?\]\s*", "", line)
    text = re.sub(r"(?:➕|📅)\s*\d{4}-\d{2}-\d{2}", "", text)
    return re.sub(r"\s+", " ", text).strip()


class ObsidianVaultWriter:
    """Writer class responsible for modifying Obsidian daily notes and creating atomic notes."""

    def __init__(self, vault_path: str | Path | None = None) -> None:
        """Initialize ObsidianVaultWriter with local vault directory.

        Args:
            vault_path: Path to Obsidian vault directory. Defaults to settings.VAULT_PATH.
        """
        self.vault_path = Path(vault_path or settings.VAULT_PATH).resolve()
        self.daily_notes_dir = self.vault_path / "Daily Notes"
        self.notes_dir = self.vault_path / "Notes"
        self._cached_candidates: list[Path] = []
        self._cache_timestamp: float = 0.0
        self._cache_ttl: float = 5.0

    def _ensure_directories(self) -> None:
        """Ensure vault subdirectories exist."""
        try:
            self.daily_notes_dir.mkdir(parents=True, exist_ok=True)
            self.notes_dir.mkdir(parents=True, exist_ok=True)
        except Exception as err:
            logger.exception("Failed to create vault directories: %s", err)
            raise

    def get_candidate_files(self, force_refresh: bool = False) -> list[Path]:
        """Return all candidate markdown files across Notes, Daily Notes, and root with short TTL caching."""
        now = time.time()
        if not force_refresh and self._cached_candidates and (now - self._cache_timestamp) < self._cache_ttl:
            return self._cached_candidates

        candidates: list[Path] = []
        for search_dir in (self.notes_dir, self.daily_notes_dir, self.vault_path):
            if not search_dir.exists():
                continue
            for p in search_dir.glob("*.md"):
                if p.name.startswith(".") or not p.is_file():
                    continue
                if p not in candidates:
                    candidates.append(p)

        self._cached_candidates = candidates
        self._cache_timestamp = now
        return candidates

    def get_existing_note_titles(self) -> list[str]:
        """Scan vault directories and return clean titles of all existing atomic and concept notes."""
        candidates = self.get_candidate_files()
        return sorted({
            p.stem
            for p in candidates
            if p.parent != self.daily_notes_dir and p.name != "Dashboard.md"
        })

    async def get_existing_note_titles_async(self) -> list[str]:
        """Asynchronously scan vault directories and return clean titles of existing notes."""
        return await asyncio.to_thread(self.get_existing_note_titles)

    def find_note_by_title_or_prefix(self, query: str) -> Path | None:
        """Find a note file in Notes/, Daily Notes/, or root vault directory by exact name, prefix, or substring match.

        Args:
            query: Title, prefix, or filename hint (with or without .md or WikiLinks).

        Returns:
            Path to matching markdown file, or None if no match found.
        """
        if not query:
            return None

        clean_query = query.replace(".md", "").replace("[[", "").replace("]]", "").strip()
        if not clean_query:
            return None

        safe_query = sanitize_filename(clean_query)

        # 1. Exact match checks in standard directories
        for base_dir in (self.notes_dir, self.daily_notes_dir, self.vault_path):
            if not base_dir.exists():
                continue
            direct_path = base_dir / f"{clean_query}.md"
            if direct_path.exists() and direct_path.is_file() and direct_path.resolve().is_relative_to(self.vault_path):
                return direct_path
            sanitized_path = base_dir / f"{safe_query}.md"
            if sanitized_path.exists() and sanitized_path.is_file():
                return sanitized_path

        # 2. Gather candidate files from cache
        candidates = self.get_candidate_files()
        query_lower = clean_query.lower()

        # 3. Case-insensitive exact match
        for p in candidates:
            if p.stem.lower() == query_lower:
                return p

        # 4. Prefix match (query is prefix of stem, or stem is prefix of query)
        for p in candidates:
            stem_lower = p.stem.lower()
            if stem_lower.startswith(query_lower) or query_lower.startswith(stem_lower):
                return p

        # 5. Substring match
        for p in candidates:
            if query_lower in p.stem.lower():
                return p

        return None

    async def find_note_by_title_or_prefix_async(self, query: str) -> Path | None:
        """Asynchronously find note file by title, prefix, or substring."""
        return await asyncio.to_thread(self.find_note_by_title_or_prefix, query)

    def ensure_dashboard_exists(self) -> Path:
        """Ensure Dashboard.md exists at vault root."""
        self._ensure_directories()
        dash_path = self.vault_path / "Dashboard.md"
        with FileLock(f"{dash_path}.lock", timeout=10):
            if not dash_path.exists():
                content = (
                    "---\n"
                    "type: dashboard\n"
                    "created: " + datetime.now().strftime("%Y-%m-%d") + "\n"
                    "tags:\n"
                    "  - dashboard\n"
                    "  - pkm/overview\n"
                    "---\n\n"
                    "# 🚀 Second Brain Action Center\n\n"
                    "> [!TIP] Welcome back!\n"
                    "> This dashboard aggregates real-time tasks, logs, and concepts across your vault using the Dataview plugin.\n\n"
                    "---\n\n"
                    "## 🎯 Active Tasks (Incomplete)\n"
                    "```dataview\n"
                    "TASK\n"
                    'FROM "Daily Notes"\n'
                    "WHERE !completed\n"
                    "SORT file.name DESC\n"
                    "```\n\n"
                    "---\n\n"
                    "## ⏱️ Recent Daily Notes\n"
                    "```dataview\n"
                    'TABLE file.mtime as "Last Updated", tags as "Tags"\n'
                    'FROM "Daily Notes"\n'
                    "SORT file.name DESC\n"
                    "LIMIT 7\n"
                    "```\n\n"
                    "---\n\n"
                    "## 🧠 Knowledge Base (Concept & Atomic Notes)\n"
                    "```dataview\n"
                    'TABLE created as "Created", tags as "Tags", source_daily_note as "Origin"\n'
                    'FROM "Notes"\n'
                    "SORT file.ctime DESC\n"
                    "LIMIT 12\n"
                    "```\n"
                )
                atomic_write_text(dash_path, content)
                logger.info("Generated Dashboard.md at %s", dash_path)
        return dash_path

    async def ensure_dashboard_exists_async(self) -> Path:
        """Asynchronously ensure Dashboard.md exists."""
        return await asyncio.to_thread(self.ensure_dashboard_exists)

    def get_all_tasks(self, include_completed: bool = False) -> list[TaskItem]:
        """Scan all markdown files across the vault and return parsed TaskItems.

        Args:
            include_completed: If True, returns completed (- [x]) tasks as well.

        Returns:
            List of TaskItem instances found in the vault.
        """
        tasks: list[TaskItem] = []
        if not self.vault_path.exists():
            return tasks

        ignored_parts = {".git", ".obsidian", ".trash", ".agent", ".venv", "__pycache__", "node_modules"}

        try:
            for md_file in sorted(self.vault_path.rglob("*.md")):
                try:
                    rel = md_file.relative_to(self.vault_path)
                except ValueError:
                    continue
                if any(part in ignored_parts or part.startswith(".") for part in rel.parts):
                    continue

                try:
                    content = md_file.read_text(encoding="utf-8")
                    lines = content.splitlines()
                    for idx, line in enumerate(lines):
                        if "- [" in line or "* [" in line or "+ [" in line:
                            task = parse_task_line(
                                line=line,
                                file_rel_path=str(rel),
                                file_name=md_file.name,
                                line_number=idx,
                            )
                            if task:
                                if not task.completed or include_completed:
                                    tasks.append(task)
                except Exception as err:
                    logger.debug("Failed reading %s during task scan: %s", md_file, err)
        except Exception as err:
            logger.exception("Error scanning vault for tasks: %s", err)

        return tasks

    async def get_all_tasks_async(self, include_completed: bool = False) -> list[TaskItem]:
        """Asynchronously scan all markdown files across vault for tasks."""
        return await asyncio.to_thread(self.get_all_tasks, include_completed)

    def mark_task_by_id_or_pattern(
        self,
        task_id: str = "",
        daily_date: str = "",
        task_pattern: str = "",
    ) -> tuple[bool, str, str]:
        """Find an uncompleted task across vault files and mark it completed (- [x]).

        Args:
            task_id: Optional 8-character deterministic task hash ID.
            daily_date: Optional date string or note filename hint.
            task_pattern: Optional search substring to identify specific task line.

        Returns:
            Tuple of (success: bool, note_display_name: str, task_text: str).
        """
        if not task_id and not task_pattern:
            logger.warning("Refusing to complete a task without a task ID or search pattern.")
            return False, "", ""

        candidate_files: list[Path] = []

        # 1. Target specific note if daily_date / hint provided
        if daily_date:
            hint_clean = daily_date.replace(".md", "").strip()
            p1 = self.daily_notes_dir / f"{hint_clean}.md"
            if p1.exists():
                candidate_files.append(p1)
            p2 = self.notes_dir / f"{hint_clean}.md"
            if p2.exists() and p2 not in candidate_files:
                candidate_files.append(p2)
            p3 = self.vault_path / f"{hint_clean}.md"
            if p3.exists() and p3 not in candidate_files:
                candidate_files.append(p3)

        # 2. Add all vault markdown files as fallback candidates
        if self.daily_notes_dir.exists():
            for p in sorted(self.daily_notes_dir.glob("*.md"), reverse=True):
                if p not in candidate_files and not p.name.startswith("."):
                    candidate_files.append(p)

        if self.notes_dir.exists():
            for p in sorted(self.notes_dir.glob("*.md")):
                if p not in candidate_files and not p.name.startswith("."):
                    candidate_files.append(p)

        if self.vault_path.exists():
            for p in sorted(self.vault_path.glob("*.md")):
                if p not in candidate_files and not p.name.startswith("."):
                    candidate_files.append(p)

        # 3. Search and mark matching task line
        for target_file in candidate_files:
            try:
                with FileLock(f"{target_file}.lock", timeout=10):
                    rel_path = str(target_file.relative_to(self.vault_path))
                    content = target_file.read_text(encoding="utf-8")
                    lines = content.splitlines()
                    modified = False
                    matched_task_text = ""

                    for idx, line in enumerate(lines):
                        if "- [ ]" in line or "* [ ]" in line or "+ [ ]" in line:
                            parsed = parse_task_line(
                                line=line,
                                file_rel_path=rel_path,
                                file_name=target_file.name,
                                line_number=idx,
                            )
                            if not parsed or parsed.completed:
                                continue

                            # Check matches
                            is_match = False
                            if task_id and (parsed.task_id == task_id or parsed.block_id == task_id):
                                is_match = True
                            elif task_pattern and (task_pattern.lower() in line.lower() or task_pattern.lower() in parsed.task_text.lower()):
                                is_match = True

                            if is_match:
                                lines[idx] = re.sub(r"([-*+]\s*\[) (\])", r"\1x\2", line, count=1)
                                modified = True
                                matched_task_text = parsed.task_text
                                break

                    if modified:
                        atomic_write_text(target_file, "\n".join(lines) + "\n")
                        logger.info("Marked task completed in %s", target_file)
                        return True, target_file.stem, matched_task_text

            except Exception as err:
                logger.exception("Error updating task in %s: %s", target_file, err)

        return False, "", ""

    async def mark_task_by_id_or_pattern_async(
        self,
        task_id: str = "",
        daily_date: str = "",
        task_pattern: str = "",
    ) -> tuple[bool, str, str]:
        """Find an uncompleted task across vault files and mark it completed asynchronously on a worker thread."""
        return await asyncio.to_thread(
            self.mark_task_by_id_or_pattern,
            task_id=task_id,
            daily_date=daily_date,
            task_pattern=task_pattern,
        )

    def mark_task_completed(
        self,
        daily_date: str = "",
        task_pattern: str = "",
        task_id: str = "",
    ) -> bool:
        """Find an uncompleted task in the vault and mark it as completed (- [x]).

        Args:
            daily_date: Optional date string or note filename hint.
            task_pattern: Optional search substring to identify specific task line.
            task_id: Optional deterministic task hash ID.

        Returns:
            True if task was found and marked complete, False otherwise.
        """
        success, _, _ = self.mark_task_by_id_or_pattern(
            task_id=task_id, daily_date=daily_date, task_pattern=task_pattern
        )
        return success

    async def mark_task_completed_async(
        self,
        daily_date: str = "",
        task_pattern: str = "",
        task_id: str = "",
    ) -> bool:
        """Asynchronously mark task completed in daily note or vault."""
        return await asyncio.to_thread(self.mark_task_completed, daily_date, task_pattern, task_id)

    def _insert_entry_into_markdown(self, markdown_text: str, header: str, entry_line: str) -> str:
        """Insert formatted entry line under the specified markdown header."""
        lines = markdown_text.splitlines()
        header_idx = -1

        for idx, line in enumerate(lines):
            if line.strip() == header.strip():
                header_idx = idx
                break

        if header_idx != -1:
            # Locate position before next level-1 or level-2 header
            next_header_idx = len(lines)
            for idx in range(header_idx + 1, len(lines)):
                line_str = lines[idx].strip()
                if line_str.startswith("# ") or line_str.startswith("## "):
                    next_header_idx = idx
                    break

            lines.insert(next_header_idx, entry_line)
            result = "\n".join(lines)
            if markdown_text.endswith("\n") or not markdown_text:
                result += "\n"
            return result
        else:
            # Header does not exist; append header and entry gracefully at end of file
            stripped = markdown_text.rstrip()
            if stripped:
                return f"{stripped}\n\n{header}\n{entry_line}\n"
            else:
                return f"{header}\n{entry_line}\n"

    def _reserve_new_note_path(self, safe_title: str) -> Path:
        """Atomically claim an unused note filename, adding a numeric suffix when the title exists."""
        suffix = 1
        while True:
            name = safe_title if suffix == 1 else f"{safe_title} ({suffix})"
            candidate = self.notes_dir / f"{name}.md"
            try:
                os.close(os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644))
                if suffix > 1:
                    logger.warning("Note '%s' already exists; saving new note as '%s'.", safe_title, name)
                return candidate
            except FileExistsError:
                suffix += 1

    def create_atomic_note(
        self,
        title: str,
        content: str,
        tags: list[str] | None = None,
        timestamp: str | None = None,
        daily_note_name: str | None = None,
    ) -> Path:
        """Create a standalone atomic note inside the Notes/ directory with YAML frontmatter.

        An existing note is never overwritten; the new note gets a numbered filename instead.

        Args:
            title: Title of the atomic note.
            content: Markdown content of the atomic note.
            tags: Optional list of tags.
            timestamp: Optional creation timestamp.
            daily_note_name: Optional filename or wikilink to parent daily note.

        Returns:
            Path object pointing to the created atomic note file.
        """
        self._ensure_directories()
        safe_title = sanitize_filename(title)

        if timestamp:
            created_date = timestamp.split(" ")[0] if " " in timestamp else timestamp[:10]
        else:
            created_date = datetime.now().strftime("%Y-%m-%d")

        # Format tags as YAML list
        tag_lines: list[str] = []
        if tags:
            for t in tags:
                clean_tag = t.lstrip("#").strip()
                if clean_tag:
                    tag_lines.append(f"  - {clean_tag}")
        if not tag_lines:
            tag_lines.append("  - atomic-note")

        frontmatter_lines = [
            "---",
            f"created: {created_date}",
            "type: atomic-note",
            "tags:",
            *tag_lines,
        ]

        if daily_note_name:
            clean_parent = daily_note_name.replace(".md", "").replace("[[", "").replace("]]", "").strip()
            frontmatter_lines.append(f'source_daily_note: "[[{clean_parent}]]"')

        frontmatter_lines.extend([
            "---",
            "",
            f"# {title}",
            "",
            content.strip(),
            "",
        ])

        note_body = "\n".join(frontmatter_lines)

        note_file = self._reserve_new_note_path(safe_title)
        try:
            atomic_write_text(note_file, note_body)
            self._cache_timestamp = 0.0  # Invalidate candidate files cache
            logger.info("Created atomic note at %s with YAML frontmatter", note_file)
            return note_file
        except Exception as err:
            note_file.unlink(missing_ok=True)
            logger.exception("Failed to write atomic note to %s: %s", note_file, err)
            raise

    @property
    def _state_dir(self) -> Path:
        return self.vault_path / STATE_DIR_NAME

    def _append_capture_record(self, record: dict[str, Any]) -> None:
        """Durably append one JSON record to the current month's capture log."""
        now = datetime.now().astimezone()
        captures_dir = self._state_dir / "captures"
        captures_dir.mkdir(parents=True, exist_ok=True)
        log_path = captures_dir / f"{now:%Y-%m}.jsonl"
        line = json.dumps({"recorded_at": now.isoformat(), **record}, ensure_ascii=False)
        with FileLock(f"{log_path}.lock", timeout=10):
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    def append_capture_event(self, raw_text: str, modality: str = "text") -> str:
        """Record the exact raw input before any LLM processing and return its capture ID."""
        capture_id = uuid.uuid4().hex
        self._append_capture_record(
            {"event": "captured", "capture_id": capture_id, "modality": modality, "raw_text": raw_text}
        )
        return capture_id

    def record_capture_outcome(self, capture_id: str, status: str, note: str | None = None) -> None:
        """Link a raw capture to the note it produced, or record that processing failed."""
        self._append_capture_record(
            {"event": status, "capture_id": capture_id, "note": note}
        )

    async def append_capture_event_async(self, raw_text: str, modality: str = "text") -> str:
        """Asynchronously record raw input in the capture log."""
        return await asyncio.to_thread(self.append_capture_event, raw_text, modality)

    async def record_capture_outcome_async(self, capture_id: str, status: str, note: str | None = None) -> None:
        """Asynchronously record the outcome of a capture."""
        await asyncio.to_thread(self.record_capture_outcome, capture_id, status, note)

    def _proposal_store_path(self) -> Path:
        return self._state_dir / "pending_proposals.json"

    def _load_proposals(self) -> dict[str, dict[str, Any]]:
        return _read_json_state(self._proposal_store_path())

    def save_pending_proposal(self, proposal: dict[str, Any]) -> str:
        """Persist an atomic note proposal and return a short ID that fits Telegram callback data."""
        path = self._proposal_store_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        proposal_id = uuid.uuid4().hex[:12]
        with FileLock(f"{path}.lock", timeout=10):
            proposals = self._load_proposals()
            proposals[proposal_id] = proposal
            atomic_write_text(path, json.dumps(proposals, ensure_ascii=False, indent=2))
        return proposal_id

    def pop_pending_proposal(self, proposal_id: str) -> dict[str, Any] | None:
        """Remove and return a pending proposal, or None if it no longer exists."""
        path = self._proposal_store_path()
        if not path.exists():
            return None
        with FileLock(f"{path}.lock", timeout=10):
            proposals = self._load_proposals()
            proposal = proposals.pop(proposal_id, None)
            if proposal is not None:
                atomic_write_text(path, json.dumps(proposals, ensure_ascii=False, indent=2))
        return proposal

    async def save_pending_proposal_async(self, proposal: dict[str, Any]) -> str:
        """Asynchronously persist an atomic note proposal."""
        return await asyncio.to_thread(self.save_pending_proposal, proposal)

    async def pop_pending_proposal_async(self, proposal_id: str) -> dict[str, Any] | None:
        """Asynchronously remove and return a pending proposal."""
        return await asyncio.to_thread(self.pop_pending_proposal, proposal_id)

    def append_interstitial_entry(
        self,
        entry: InterstitialEntry,
        date_str: str | None = None,
    ) -> tuple[Path, Path | None]:
        """Append a structured InterstitialEntry to the corresponding daily note file.

        Args:
            entry: InterstitialEntry Pydantic model instance.
            date_str: Optional override for target daily note date ('YYYY-MM-DD').

        Returns:
            Tuple of (daily_note_path, atomic_note_path_or_none).
        """
        self._ensure_directories()

        if not date_str:
            if entry.timestamp and " " in entry.timestamp:
                date_str = entry.timestamp.split(" ")[0]
            elif entry.timestamp and len(entry.timestamp) >= 10:
                date_str = entry.timestamp[:10]
            else:
                date_str = datetime.now().strftime("%Y-%m-%d")

        daily_note_path = self.daily_notes_dir / f"{date_str}.md"
        atomic_note_path: Path | None = None

        # 1. Create atomic note if flagged (with YAML frontmatter and bidirectional link back to daily note)
        if entry.requires_atomic_note and entry.atomic_note_title and entry.atomic_note_content:
            atomic_note_path = self.create_atomic_note(
                title=entry.atomic_note_title,
                content=entry.atomic_note_content,
                tags=entry.extracted_tags,
                timestamp=entry.timestamp,
                daily_note_name=date_str,
            )

        # 2. Format line entry adhering to the Obsidian Feature Rulebook
        entry_line = format_interstitial_entry_line(
            entry=entry,
            date_str=date_str,
            atomic_note_title=atomic_note_path.stem if atomic_note_path else None,
        )

        header = resolve_header(entry.category)

        try:
            with FileLock(f"{daily_note_path}.lock", timeout=10):
                if daily_note_path.exists():
                    existing_content = daily_note_path.read_text(encoding="utf-8")
                else:
                    existing_content = generate_daily_note_template(date_str)

                updated_content = self._insert_entry_into_markdown(existing_content, header, entry_line)
                atomic_write_text(daily_note_path, updated_content)
                self._cache_timestamp = 0.0  # Invalidate candidate files cache
                logger.info("Appended entry to daily note at %s under header '%s'", daily_note_path, header)
                return daily_note_path, atomic_note_path

        except Exception as err:
            logger.exception("Failed to update daily note at %s: %s", daily_note_path, err)
            raise

    def _task_corrections_path(self) -> Path:
        return self._state_dir / "task_corrections.json"

    def save_task_correction(
        self,
        entry: InterstitialEntry,
        daily_note_path: Path,
        atomic_note_path: Path | None = None,
        capture_id: str | None = None,
    ) -> str:
        """Remember how to turn a stored task back into a note, returning an ID for the undo button."""
        date_str = daily_note_path.stem
        atomic_title = atomic_note_path.stem if atomic_note_path else None
        note_entry = entry.model_copy(update={
            "category": "log",
            "content": "\n".join(_task_line_to_text(line) for line in entry.content.splitlines()),
            "due_date": None,
            "memory_type": "observation",
        })
        record = {
            "daily_note": str(daily_note_path.relative_to(self.vault_path)),
            "task_block": format_interstitial_entry_line(entry, date_str, atomic_title),
            "note_block": format_interstitial_entry_line(note_entry, date_str, atomic_title),
            "capture_id": capture_id,
        }
        path = self._task_corrections_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        correction_id = uuid.uuid4().hex[:12]
        with FileLock(f"{path}.lock", timeout=10):
            corrections = _read_json_state(path)
            corrections[correction_id] = record
            while len(corrections) > MAX_TASK_CORRECTIONS:
                del corrections[next(iter(corrections))]
            atomic_write_text(path, json.dumps(corrections, ensure_ascii=False, indent=2))
        return correction_id

    def apply_task_correction(self, correction_id: str) -> bool:
        """Move a wrongly filed task to the Log section, unless it has changed since it was captured."""
        path = self._task_corrections_path()
        if not path.exists():
            return False
        with FileLock(f"{path}.lock", timeout=10):
            corrections = _read_json_state(path)
            record = corrections.get(correction_id)
            if record is None:
                return False

            daily_note_path = self.vault_path / record["daily_note"]
            if not daily_note_path.resolve().is_relative_to(self.vault_path):
                logger.warning("Refusing correction %s: its path is outside the vault.", correction_id)
                return False
            if not daily_note_path.exists():
                return False
            with FileLock(f"{daily_note_path}.lock", timeout=10):
                framed = f"\n{daily_note_path.read_text(encoding='utf-8').rstrip()}\n"
                framed_block = f"\n{record['task_block']}\n"
                if framed_block not in framed:
                    logger.info("Task for correction %s has changed; leaving it alone.", correction_id)
                    return False
                without_task = framed.replace(framed_block, "\n", 1).lstrip("\n")
                updated = self._insert_entry_into_markdown(without_task, HEADER_LOG, record["note_block"])
                atomic_write_text(daily_note_path, updated)

            del corrections[correction_id]
            atomic_write_text(path, json.dumps(corrections, ensure_ascii=False, indent=2))

        if record.get("capture_id"):
            try:
                self.record_capture_outcome(record["capture_id"], "corrected_not_task", daily_note_path.name)
            except Exception as err:
                logger.exception("Could not record task correction in capture log: %s", err)
        return True

    async def save_task_correction_async(
        self,
        entry: InterstitialEntry,
        daily_note_path: Path,
        atomic_note_path: Path | None = None,
        capture_id: str | None = None,
    ) -> str:
        """Asynchronously remember how to undo a task classification."""
        return await asyncio.to_thread(
            self.save_task_correction, entry, daily_note_path, atomic_note_path, capture_id
        )

    async def apply_task_correction_async(self, correction_id: str) -> bool:
        """Asynchronously move a wrongly filed task to the Log section."""
        return await asyncio.to_thread(self.apply_task_correction, correction_id)

    async def create_atomic_note_async(
        self,
        title: str,
        content: str,
        tags: list[str] | None = None,
        timestamp: str | None = None,
        daily_note_name: str | None = None,
    ) -> Path:
        """Asynchronously create an atomic note."""
        return await asyncio.to_thread(
            self.create_atomic_note, title, content, tags, timestamp, daily_note_name
        )

    async def append_interstitial_entry_async(
        self,
        entry: InterstitialEntry,
        date_str: str | None = None,
    ) -> tuple[Path, Path | None]:
        """Asynchronously append entry to daily note and optional atomic note."""
        return await asyncio.to_thread(self.append_interstitial_entry, entry, date_str)
