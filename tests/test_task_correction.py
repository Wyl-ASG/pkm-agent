"""Tests for correcting a note that was wrongly filed as a task."""

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
import pytest

from src.agents.models import InterstitialEntry
from src.vault.md_writer import HEADER_LOG, HEADER_TASKS, ObsidianVaultWriter

WEBHOOK_HEADERS = {"X-Telegram-Bot-Api-Secret-Token": "test-secret"}


def _task(content: str = "- [ ] DNS TTL must be lowered ➕ 2026-08-30", **fields) -> InterstitialEntry:
    return InterstitialEntry(timestamp="2026-08-30 20:27", category="task", content=content, **fields)


def _section(note_text: str, header: str) -> str:
    after_header = note_text.split(header, 1)[1]
    return after_header.split("\n## ", 1)[0]


def _stored_task(writer: ObsidianVaultWriter, entry: InterstitialEntry, capture_id: str | None = None) -> tuple[Path, str]:
    daily_path, atomic_path = writer.append_interstitial_entry(entry, date_str="2026-08-30")
    correction_id = writer.save_task_correction(entry, daily_path, atomic_path, capture_id)
    return daily_path, correction_id


def test_correction_moves_entry_from_tasks_to_log(tmp_path):
    writer = ObsidianVaultWriter(vault_path=tmp_path)
    daily_path, correction_id = _stored_task(writer, _task())

    assert writer.apply_task_correction(correction_id) is True

    note = daily_path.read_text(encoding="utf-8")
    assert "DNS TTL must be lowered" not in _section(note, HEADER_TASKS)
    assert "- **[20:27]** DNS TTL must be lowered" in _section(note, HEADER_LOG)
    assert "- [ ]" not in note
    assert "➕" not in note


def test_correction_removes_due_date_and_keeps_links(tmp_path):
    writer = ObsidianVaultWriter(vault_path=tmp_path)
    entry = _task("- [ ] Use [[SQLite]] for the POC by 2026-09-01 ➕ 2026-08-30 📅 2026-09-01", due_date="2026-09-01")
    daily_path, correction_id = _stored_task(writer, entry)

    writer.apply_task_correction(correction_id)

    note = daily_path.read_text(encoding="utf-8")
    assert "Use [[SQLite]] for the POC by 2026-09-01" in _section(note, HEADER_LOG)
    assert "📅 2026-09-01" not in note


def test_correction_only_touches_its_own_task(tmp_path):
    writer = ObsidianVaultWriter(vault_path=tmp_path)
    daily_path, correction_id = _stored_task(writer, _task())
    writer.append_interstitial_entry(_task("- [ ] Buy milk ➕ 2026-08-30"), date_str="2026-08-30")

    writer.apply_task_correction(correction_id)

    assert "- [ ] Buy milk" in _section(daily_path.read_text(encoding="utf-8"), HEADER_TASKS)


def test_multi_line_task_capture_is_moved_completely(tmp_path):
    writer = ObsidianVaultWriter(vault_path=tmp_path)
    entry = _task("- [ ] First point ➕ 2026-08-30\n- [ ] Second point ➕ 2026-08-30")
    daily_path, correction_id = _stored_task(writer, entry)

    assert writer.apply_task_correction(correction_id) is True

    note = daily_path.read_text(encoding="utf-8")
    assert "- [ ]" not in note
    assert "First point" in _section(note, HEADER_LOG)
    assert "Second point" in _section(note, HEADER_LOG)


def test_second_tap_changes_nothing(tmp_path):
    writer = ObsidianVaultWriter(vault_path=tmp_path)
    daily_path, correction_id = _stored_task(writer, _task())
    writer.apply_task_correction(correction_id)
    after_first = daily_path.read_text(encoding="utf-8")

    assert writer.apply_task_correction(correction_id) is False
    assert daily_path.read_text(encoding="utf-8") == after_first


def test_task_edited_or_completed_since_capture_is_left_alone(tmp_path):
    writer = ObsidianVaultWriter(vault_path=tmp_path)
    daily_path, correction_id = _stored_task(writer, _task())
    assert writer.mark_task_by_id_or_pattern(task_pattern="DNS TTL")[0] is True
    completed = daily_path.read_text(encoding="utf-8")

    assert writer.apply_task_correction(correction_id) is False
    assert daily_path.read_text(encoding="utf-8") == completed


def test_unknown_correction_is_refused(tmp_path):
    assert ObsidianVaultWriter(vault_path=tmp_path).apply_task_correction("000000000000") is False


def test_tampered_correction_cannot_write_outside_the_vault(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("- [ ] DNS TTL must be lowered ➕ 2026-08-30\n", encoding="utf-8")
    writer = ObsidianVaultWriter(vault_path=vault)
    _, correction_id = _stored_task(writer, _task())

    state_path = vault / ".pkm_state" / "task_corrections.json"
    records = json.loads(state_path.read_text(encoding="utf-8"))
    records[correction_id]["daily_note"] = "../outside.md"
    state_path.write_text(json.dumps(records), encoding="utf-8")

    assert writer.apply_task_correction(correction_id) is False
    assert "- [ ] DNS TTL" in outside.read_text(encoding="utf-8")


def test_correction_is_recorded_in_the_capture_log(tmp_path):
    writer = ObsidianVaultWriter(vault_path=tmp_path)
    capture_id = writer.append_capture_event("DNS TTL must be lowered")
    _, correction_id = _stored_task(writer, _task(), capture_id)

    writer.apply_task_correction(correction_id)

    log_file = next((tmp_path / ".pkm_state" / "captures").glob("*.jsonl"))
    records = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()]
    assert records[-1]["event"] == "corrected_not_task"
    assert records[-1]["capture_id"] == capture_id


def test_old_correction_records_are_pruned(tmp_path):
    writer = ObsidianVaultWriter(vault_path=tmp_path)
    with patch("src.vault.md_writer.MAX_TASK_CORRECTIONS", 2):
        _, oldest = _stored_task(writer, _task("- [ ] One ➕ 2026-08-30"))
        _stored_task(writer, _task("- [ ] Two ➕ 2026-08-30"))
        _, newest = _stored_task(writer, _task("- [ ] Three ➕ 2026-08-30"))

    assert writer.apply_task_correction(oldest) is False
    assert writer.apply_task_correction(newest) is True


# Telegram -------------------------------------------------------------------


async def _capture_buttons(entry: InterstitialEntry) -> list[dict]:
    import src.main as app_main

    sent = AsyncMock()
    with (
        patch.object(app_main.parser_agent, "parse", new=AsyncMock(return_value=entry)),
        patch("src.main.send_telegram_message", sent),
        patch("src.main._async_background_post_ingestion", new_callable=AsyncMock),
    ):
        await app_main.process_telegram_ingestion(1, "raw")
    markup = sent.call_args.kwargs["reply_markup"]
    return [button for row in markup["inline_keyboard"] for button in row] if markup else []


@pytest.mark.parametrize(
    "content",
    [
        "- [ ] Lower the DNS TTL ➕ 2026-08-30",
        "- [ ] 提交季度财务报告并与团队一起审查所有数字和结论 ➕ 2026-08-30",
    ],
)
@pytest.mark.asyncio
async def test_task_confirmation_buttons_fit_telegram_limit(content):
    buttons = await _capture_buttons(_task(content))

    assert {b["callback_data"].split(":")[0] for b in buttons} == {"done", "not_task"}
    assert all(len(b["callback_data"].encode("utf-8")) <= 64 for b in buttons)


@pytest.mark.asyncio
async def test_non_task_confirmation_has_no_correction_button():
    buttons = await _capture_buttons(InterstitialEntry(timestamp="2026-08-30 20:27", category="log", content="A thought"))
    assert buttons == []


def _click(callback_data: str):
    import src.main as app_main

    payload = {
        "update_id": 1,
        "callback_query": {
            "id": "cb",
            "from": {"id": 1, "is_bot": False, "first_name": "T"},
            "data": callback_data,
            "message": {"message_id": 5, "chat": {"id": 1}, "text": "✅ Entry Logged (task)"},
        },
    }
    return TestClient(app_main.app).post("/webhook/telegram", json=payload, headers=WEBHOOK_HEADERS)


@pytest.mark.asyncio
async def test_not_a_task_button_moves_the_entry_and_updates_the_message():
    buttons = await _capture_buttons(_task("- [ ] Button test fact must hold ➕ 2026-08-30"))
    callback_data = next(b["callback_data"] for b in buttons if b["callback_data"].startswith("not_task:"))

    with (
        patch("src.main.answer_telegram_callback_query", new_callable=AsyncMock) as answered,
        patch("src.main.edit_telegram_message_text", new_callable=AsyncMock) as edited,
        patch("src.main._safe_git_commit_and_push", new_callable=AsyncMock) as committed,
    ):
        response = _click(callback_data)

    assert response.json()["status"] == "task_correction_applied"
    answered.assert_awaited_once()
    assert edited.await_args.kwargs.get("reply_markup") is None
    committed.assert_called_once()
    note = Path(os.environ["VAULT_PATH"]) / "Daily Notes" / "2026-08-30.md"
    assert "Button test fact must hold" in _section(note.read_text(encoding="utf-8"), HEADER_LOG)


def test_stale_not_a_task_button_reports_failure_without_committing():
    with (
        patch("src.main.answer_telegram_callback_query", new_callable=AsyncMock) as answered,
        patch("src.main.edit_telegram_message_text", new_callable=AsyncMock) as edited,
        patch("src.main._safe_git_commit_and_push", new_callable=AsyncMock) as committed,
    ):
        response = _click("not_task:000000000000")

    assert response.json()["status"] == "task_correction_unavailable"
    assert "⚠️" in answered.await_args.kwargs["text"]
    edited.assert_not_awaited()
    committed.assert_not_called()
