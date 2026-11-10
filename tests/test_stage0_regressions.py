"""Regression tests for Stage 0 data-safety, honesty, and privacy fixes."""

import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
import git
import pytest

from src.agents.models import ConsolidationProposal, InterstitialEntry
from src.agents.parser import (
    EntryParserAgent,
    extract_due_date_from_natural_language,
    is_natural_language_task,
)
from src.llm.mock_llm import MockLLM
from src.vault.git_engine import GitEngine
from src.vault.md_writer import ObsidianVaultWriter, format_interstitial_entry_line

API_HEADERS = {"x-api-key": "test-secret"}
WEBHOOK_HEADERS = {"X-Telegram-Bot-Api-Secret-Token": "test-secret"}


def _entry(**overrides) -> InterstitialEntry:
    values = {"timestamp": "2026-08-30 20:00", "category": "log", "content": "A note"}
    values.update(overrides)
    return InterstitialEntry(**values)


# Vault writer ---------------------------------------------------------------


def test_atomic_note_never_overwrites_existing_note(tmp_path):
    writer = ObsidianVaultWriter(vault_path=tmp_path)
    writer._ensure_directories()
    existing = writer.notes_dir / "Docker.md"
    existing.write_text("# Docker\n\nHand-written notes.\n", encoding="utf-8")

    created = writer.create_atomic_note("Docker", "Generated content.")

    assert "Hand-written notes." in existing.read_text(encoding="utf-8")
    assert created != existing
    assert created.stem == "Docker (2)"
    assert "Generated content." in created.read_text(encoding="utf-8")


def test_daily_note_links_to_renamed_atomic_note(tmp_path):
    writer = ObsidianVaultWriter(vault_path=tmp_path)
    writer._ensure_directories()
    (writer.notes_dir / "Docker.md").write_text("Existing", encoding="utf-8")
    entry = _entry(
        requires_atomic_note=True,
        atomic_note_title="Docker",
        atomic_note_content="Generated content.",
    )

    daily_path, atomic_path = writer.append_interstitial_entry(entry)

    assert atomic_path is not None and atomic_path.stem == "Docker (2)"
    assert "[[Docker (2)]]" in daily_path.read_text(encoding="utf-8")


def test_failed_write_leaves_original_note_intact(tmp_path):
    writer = ObsidianVaultWriter(vault_path=tmp_path)
    writer._ensure_directories()
    note = writer.daily_notes_dir / "2026-08-30.md"
    note.write_text("- [ ] Pay rent\n", encoding="utf-8")

    with patch("src.vault.md_writer.os.replace", side_effect=OSError("disk full")):
        marked, _, _ = writer.mark_task_by_id_or_pattern(task_pattern="Pay rent")

    assert marked is False
    assert note.read_text(encoding="utf-8") == "- [ ] Pay rent\n"
    assert [p.name for p in writer.daily_notes_dir.iterdir() if p.suffix == ".tmp"] == []


def test_empty_task_completion_marks_nothing(tmp_path):
    writer = ObsidianVaultWriter(vault_path=tmp_path)
    writer._ensure_directories()
    note = writer.daily_notes_dir / "2026-08-30.md"
    note.write_text("- [ ] Pay rent\n", encoding="utf-8")

    assert writer.mark_task_by_id_or_pattern() == (False, "", "")
    assert writer.mark_task_by_id_or_pattern(daily_date="2026-08-30") == (False, "", "")
    assert "- [ ] Pay rent" in note.read_text(encoding="utf-8")


def test_empty_task_completion_api_is_rejected():
    from src.main import app

    with patch("src.main.vault_writer.mark_task_by_id_or_pattern_async", new_callable=AsyncMock) as mark:
        response = TestClient(app).post("/tasks/complete", json={}, headers=API_HEADERS)

    assert response.status_code == 400
    mark.assert_not_called()


def test_capture_log_is_append_only_and_outside_note_indexes(tmp_path):
    writer = ObsidianVaultWriter(vault_path=tmp_path)

    first = writer.append_capture_event("First raw text", modality="text")
    second = writer.append_capture_event("Second raw text", modality="voice")

    log_files = list((tmp_path / ".pkm_state" / "captures").glob("*.jsonl"))
    assert len(log_files) == 1
    records = [json.loads(line) for line in log_files[0].read_text(encoding="utf-8").splitlines()]
    assert [r["raw_text"] for r in records] == ["First raw text", "Second raw text"]
    assert records[1]["modality"] == "voice"
    assert first != second
    assert writer.get_all_tasks(include_completed=True) == []


# Telegram ingestion ---------------------------------------------------------


@pytest.mark.asyncio
async def test_raw_capture_survives_processing_failure():
    import src.main as app_main

    with (
        patch.object(app_main.parser_agent, "parse", new=AsyncMock(side_effect=RuntimeError("boom"))),
        patch("src.main.send_telegram_message", new_callable=AsyncMock),
    ):
        await app_main.process_telegram_ingestion(1, "Do not lose this capture")

    log_dir = Path(os.environ["VAULT_PATH"]) / ".pkm_state" / "captures"
    contents = "\n".join(p.read_text(encoding="utf-8") for p in log_dir.glob("*.jsonl"))
    assert "Do not lose this capture" in contents


@pytest.mark.asyncio
async def test_approved_proposal_keeps_generated_content():
    import src.main as app_main

    entry = _entry(
        requires_atomic_note=True,
        atomic_note_confidence=0.6,
        atomic_note_title="Stage Zero Proposal",
        atomic_note_content="The real generated insight.",
    )
    sent = AsyncMock()
    with (
        patch.object(app_main.parser_agent, "parse", new=AsyncMock(return_value=entry)),
        patch("src.main.send_telegram_message", sent),
        patch("src.main._async_background_post_ingestion", new_callable=AsyncMock),
    ):
        await app_main.process_telegram_ingestion(1, "Raw idea")

    keyboard = sent.call_args.kwargs["reply_markup"]["inline_keyboard"]
    callback_data = next(
        button["callback_data"]
        for row in keyboard
        for button in row
        if button["callback_data"].startswith("create_atomic:")
    )
    assert len(callback_data.encode("utf-8")) <= 64

    payload = {
        "update_id": 1,
        "callback_query": {"id": "cb", "from": {"id": 1, "is_bot": False, "first_name": "T"}, "data": callback_data},
    }
    with (
        patch("src.main.answer_telegram_callback_query", new_callable=AsyncMock),
        patch("src.main.edit_telegram_message_text", new_callable=AsyncMock),
        patch("src.main._safe_git_commit_and_push", new_callable=AsyncMock),
    ):
        response = TestClient(app_main.app).post("/webhook/telegram", json=payload, headers=WEBHOOK_HEADERS)

    assert response.status_code == 200
    note = Path(os.environ["VAULT_PATH"]) / "Notes" / "Stage Zero Proposal.md"
    assert "The real generated insight." in note.read_text(encoding="utf-8")


# Git engine -----------------------------------------------------------------


def _engine_with_commit(path: Path) -> GitEngine:
    repo = git.Repo.init(path)
    repo.git.symbolic_ref("HEAD", "refs/heads/main")
    repo.git.config("user.email", "test@example.com")
    repo.git.config("user.name", "Test")
    return GitEngine(vault_path=path, branch="main")


def test_mass_deletion_is_kept_and_alerted_without_losing_edits(tmp_path):
    engine = _engine_with_commit(tmp_path)
    for i in range(60):
        (tmp_path / f"n{i}.md").write_text("x", encoding="utf-8")
    (tmp_path / "today.md").write_text("old", encoding="utf-8")
    assert engine.commit_and_push_sync("initial") is True

    for i in range(60):
        (tmp_path / f"n{i}.md").unlink()
    (tmp_path / "today.md").write_text("just captured", encoding="utf-8")

    assert engine.commit_and_push_sync("reorganise") is True

    assert not (tmp_path / "n0.md").exists()
    assert (tmp_path / "today.md").read_text(encoding="utf-8") == "just captured"
    alerts = engine.pop_alerts()
    assert len(alerts) == 1 and "60" in alerts[0]
    assert engine.pop_alerts() == []


def test_internal_files_are_not_committed_and_get_untracked(tmp_path):
    engine = _engine_with_commit(tmp_path)
    (tmp_path / ".vault_hashes.json").write_text("{}", encoding="utf-8")
    (tmp_path / "Note.md.lock").write_text("", encoding="utf-8")
    (tmp_path / "Note.md").write_text("Hello", encoding="utf-8")
    engine.repo.git.add(A=True)
    engine.repo.git.commit("-m", "history that tracked internal files")

    assert engine.commit_and_push_sync("clean up") is True

    tracked = engine.repo.git.ls_files().splitlines()
    assert "Note.md" in tracked and ".gitignore" in tracked
    assert ".vault_hashes.json" not in tracked and "Note.md.lock" not in tracked
    assert (tmp_path / ".vault_hashes.json").exists()

    (tmp_path / "Other.md.lock").write_text("", encoding="utf-8")
    assert engine.commit_and_push_sync("again") is True
    assert "Other.md.lock" not in engine.repo.git.ls_files().splitlines()


def test_repair_detects_head_pointing_at_missing_commit(tmp_path):
    engine = _engine_with_commit(tmp_path)
    (tmp_path / "One.md").write_text("one", encoding="utf-8")
    assert engine.commit_sync("first") is True
    (tmp_path / ".git" / "refs" / "heads" / "main").write_text("3f4ff950f915e15c5a6b3938bcf743da174ea32f\n")

    assert engine._repair_broken_head() is True
    (tmp_path / "Two.md").write_text("two", encoding="utf-8")
    assert engine.commit_sync("after repair") is True


# Parser ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw_text",
    [
        "I learned today that Qdrant supports payload indexes",
        "It has been replaced by cgroups, namespace and sec comp",
        "Have to use sudo to run the command.",
        "3/4 of users prefer dark mode",
        "Met Sarah on 30 Aug to discuss the roadmap",
    ],
)
@pytest.mark.asyncio
async def test_llm_non_task_classification_is_respected(raw_text):
    llm = MockLLM(default_json_response={"timestamp": "2026-08-30 20:00", "category": "discovery", "content": raw_text})

    entry = await EntryParserAgent(llm=llm).parse(raw_text)

    assert entry.category == "discovery"
    assert entry.content == raw_text
    assert "- [ ]" not in entry.content


@pytest.mark.asyncio
async def test_llm_failure_only_creates_task_for_explicit_commitment():
    class FailingLLM(MockLLM):
        async def generate_json(self, prompt, schema_model, system_prompt=None):
            raise ValueError("no output")

    parser = EntryParserAgent(llm=FailingLLM())

    note = await parser.parse("It has been replaced by cgroups, due to better isolation")
    task = await parser.parse("I need to do my bgs by next friday")

    assert note.category == "log" and "- [ ]" not in note.content
    assert task.category == "task" and task.content.startswith("- [ ] I need to do my bgs")


@pytest.mark.asyncio
async def test_task_without_stated_deadline_has_no_invented_due_date():
    llm = MockLLM(default_json_response={"timestamp": "2026-08-30 20:00", "category": "task", "content": "- [ ] Review the design"})

    entry = await EntryParserAgent(llm=llm).parse("Review the design")

    assert entry.due_date is None
    assert "📅" not in entry.content
    assert "📅" not in format_interstitial_entry_line(entry, date_str="2026-08-30")


def test_formatter_does_not_invent_due_date_for_plain_task_content():
    line = format_interstitial_entry_line(_entry(category="task", content="Review the design"), date_str="2026-08-30")
    assert line == "- [ ] Review the design ➕ 2026-08-30"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I need to do my bgs by next friday", "2026-09-04"),  # captured on Sunday 2026-08-30
        ("submit by next sunday", "2026-09-06"),
    ],
)
def test_next_weekday_is_parsed(text, expected):
    from datetime import datetime

    assert extract_due_date_from_natural_language(text, ref_date=datetime(2026, 8, 30)) == expected


def test_next_weekday_skips_the_current_week():
    from datetime import datetime

    wednesday = datetime(2026, 8, 26)
    assert extract_due_date_from_natural_language("by next friday", ref_date=wednesday) == "2026-09-04"
    assert extract_due_date_from_natural_language("by this friday", ref_date=wednesday) == "2026-08-28"


def test_explicit_task_heuristic_ignores_incidental_keywords():
    assert is_natural_language_task("I need to pay rent") is True
    assert is_natural_language_task("Finish offensive cybersecurity assignment by Sunday") is True
    assert is_natural_language_task("Upload my photo in the folder by this Friday") is True
    assert is_natural_language_task("I learned today that this works") is False
    assert is_natural_language_task("Have to use sudo to run the command.") is False
    assert is_natural_language_task("The assignment problem is solved by the Hungarian algorithm") is False


# LLM and Qdrant fallbacks ---------------------------------------------------


def test_missing_agy_fails_loudly_instead_of_using_mock():
    from src.llm.factory import LLMUnavailableError, get_llm_provider

    with pytest.raises(LLMUnavailableError):
        get_llm_provider("antigravity", binary_path="/nonexistent/agy")


def test_unknown_llm_provider_is_rejected():
    from src.llm.factory import LLMUnavailableError, get_llm_provider

    with pytest.raises(LLMUnavailableError):
        get_llm_provider("not-a-provider")


def test_mock_requires_explicit_selection():
    from src.llm.factory import get_llm_provider

    assert isinstance(get_llm_provider("mock"), MockLLM)


def test_unreachable_remote_qdrant_does_not_fall_back_to_memory():
    from src.graphrag.vector_db import QdrantVectorStore

    saved = QdrantVectorStore._instance
    QdrantVectorStore._instance = None
    try:
        with patch("src.graphrag.vector_db.settings.QDRANT_STORAGE_PATH", None):
            store = QdrantVectorStore(host="127.0.0.1", port=1)
            with pytest.raises(ConnectionError):
                _ = store.client
    finally:
        QdrantVectorStore._instance = saved


def test_importing_app_twice_does_not_lock_embedded_qdrant():
    code = (
        "import subprocess, sys; import src.main; "
        "child = subprocess.run([sys.executable, '-c', 'import src.main'], capture_output=True, text=True); "
        "sys.stderr.write(child.stderr[-500:]); sys.exit(child.returncode)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


# Startup and health ---------------------------------------------------------


def test_default_server_start_is_private_and_without_reload():
    import src.main as app_main

    with patch("uvicorn.run") as run, patch.object(sys, "argv", ["main.py"]):
        app_main.cli_entrypoint()

    assert run.call_args.kwargs["host"] == "127.0.0.1"
    assert run.call_args.kwargs["reload"] is False


def test_startup_refuses_to_run_without_llm():
    import src.main as app_main
    from src.llm.factory import LLMUnavailableError

    with patch.object(app_main.llm_driver, "resolve", side_effect=LLMUnavailableError("agy missing")):
        with pytest.raises(LLMUnavailableError):
            with TestClient(app_main.app):
                pass


def test_health_reports_actual_component_state():
    import src.main as app_main

    client = TestClient(app_main.app)
    healthy = client.get("/health").json()
    assert healthy["status"] == "healthy"
    assert healthy["components"]["llm"]["active"] == "MockLLM"
    assert healthy["components"]["qdrant"] == "ok"

    with patch.object(app_main.llm_driver, "resolve", side_effect=RuntimeError("agy missing")):
        response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "degraded"
    assert response.json()["components"]["llm"]["active"] is None

    with patch.object(app_main.vector_store, "probe", side_effect=RuntimeError("locked")):
        response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["components"]["qdrant"] == "error"


# Consolidation --------------------------------------------------------------


@pytest.mark.asyncio
async def test_consolidation_report_runs(tmp_path):
    from src.agents.consolidation import KnowledgeConsolidator
    from src.graphrag.graph import VaultKnowledgeGraph

    (tmp_path / "Note.md").write_text("# Note\n\nA useful thought.", encoding="utf-8")
    llm = MockLLM(default_json_response={"repeated_concepts": ["Thought"], "summary_markdown": "x"})
    consolidator = KnowledgeConsolidator(
        vault_path=tmp_path,
        knowledge_graph=VaultKnowledgeGraph(tmp_path),
        llm=llm,
    )

    report = await consolidator.generate_consolidation_report()

    assert isinstance(report, ConsolidationProposal)
    assert "[[Thought]]" in report.summary_markdown


# Logging --------------------------------------------------------------------


def test_bot_token_is_redacted_from_formatted_logs():
    from src.utils.logging_setup import RedactingFormatter

    formatter = RedactingFormatter("%(message)s")
    # Built at runtime so secret scanners don't mistake a test value for a real token.
    fake_secret = "x" * 35
    token_url = f"https://api.telegram.org/bot123456789:{fake_secret}/sendMessage"
    try:
        raise RuntimeError(f"failed: {token_url}")
    except RuntimeError:
        exc_info = sys.exc_info()
    record = logging.LogRecord("x", logging.ERROR, __file__, 1, token_url, None, exc_info)

    output = formatter.format(record)

    assert fake_secret not in output
    assert "bot<redacted>" in output


def test_httpx_request_logging_is_quiet():
    import src.main  # noqa: F401

    assert logging.getLogger("httpx").level == logging.WARNING


@pytest.mark.asyncio
async def test_note_text_is_not_logged_at_info(caplog):
    import src.main as app_main

    secret_query = "my-very-private-question"
    with (
        patch.object(app_main.qa_agent.retriever, "search", new=AsyncMock(return_value=[])),
        patch("src.main.send_telegram_message", new_callable=AsyncMock),
        caplog.at_level(logging.INFO),
    ):
        await app_main.process_telegram_query(1, secret_query)
        await app_main.process_telegram_tasks_query(1, secret_query)

    assert secret_query not in caplog.text
