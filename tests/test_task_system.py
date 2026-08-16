"""Unit tests for deterministic task identification, block ID extraction, and task completion."""

import pytest
from pathlib import Path
from src.vault.md_writer import ObsidianVaultWriter, parse_task_line


def test_parse_task_line_with_block_id():
    line = "- [ ] Deploy FastAPI service to [[AWS ECS]] ➕ 2026-08-15 📅 2026-08-18 ^task-deploy-99"
    task = parse_task_line(line, "Daily Notes/2026-08-15.md", "2026-08-15.md", 10)

    assert task is not None
    assert not task.completed
    assert task.created_date == "2026-08-15"
    assert task.due_date == "2026-08-18"
    assert task.block_id == "task-deploy-99"
    assert len(task.task_id) == 8


def test_mark_task_completed(tmp_path):
    vault_dir = tmp_path / "vault"
    daily_dir = vault_dir / "Daily Notes"
    daily_dir.mkdir(parents=True)

    note_file = daily_dir / "2026-08-15.md"
    note_file.write_text("""# 📅 Daily Note: 2026-08-15

## 🎯 Priorities & Tasks
- [ ] Configure Qdrant vector database ➕ 2026-08-15 📅 2026-08-15 ^task-qdrant
- [ ] Review documentation
""", encoding="utf-8")

    writer = ObsidianVaultWriter(vault_path=vault_dir)
    tasks = writer.get_all_tasks(include_completed=False)
    assert len(tasks) == 2

    # Mark first task complete
    success, note_name, text = writer.mark_task_by_id_or_pattern(task_pattern="Configure Qdrant")
    assert success is True
    assert note_name == "2026-08-15"

    # Verify task is now completed in file
    updated_content = note_file.read_text(encoding="utf-8")
    assert "- [x] Configure Qdrant" in updated_content


def test_parse_task_line_edge_cases():
    """Test task parsing on various syntax combinations and malformed lines."""
    # 1. Non-task line
    assert parse_task_line("# Heading 1", "Notes/N.md", "N.md", 1) is None
    assert parse_task_line("Just some normal paragraph text", "Notes/N.md", "N.md", 2) is None

    # 2. Asterisk bullet with Dataview due date and priority field
    line_dv = "* [ ] Write report [due:: 2026-09-01] [priority:: high]"
    task_dv = parse_task_line(line_dv, "Daily Notes/2026-08-16.md", "2026-08-16.md", 5)
    assert task_dv is not None
    assert task_dv.due_date == "2026-09-01"
    assert task_dv.priority == "high"
    assert "Write report" in task_dv.task_text

    # 3. Plus bullet with Obsidian tag priority
    line_prio = "+ [ ] Fix critical issue #high"
    task_prio = parse_task_line(line_prio, "Daily Notes/2026-08-16.md", "2026-08-16.md", 6)
    assert task_prio is not None
    assert task_prio.priority == "high"

    # 4. Completed task line
    line_completed = "- [x] Finished task ➕ 2026-08-10 📅 2026-08-12"
    task_completed = parse_task_line(line_completed, "Daily Notes/2026-08-16.md", "2026-08-16.md", 7)
    assert task_completed is not None
    assert task_completed.completed is True
