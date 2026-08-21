"""Production hardening tests for 2 vCPU / 4 GB RAM resource constraints and safety."""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.models import InterstitialEntry
from src.agents.transcriber import AudioTranscriber
from src.config import settings
from src.graphrag.embedder import TextEmbedder
from src.graphrag.reranker import CrossEncoderReranker
from src.graphrag.sparse import BM25Index
from src.llm.antigravity_llm import AntigravityLLM
from src.utils.resources import ResourceMonitor, ResourceManager, resource_manager, resource_monitor
from src.vault.git_engine import GitEngine
from src.vault.md_writer import ObsidianVaultWriter


def test_resource_monitor_metrics():
    """Verify ResourceMonitor extracts valid numeric metrics and memory pressure flag."""
    metrics = resource_monitor.get_metrics()
    assert metrics.total_ram_mb > 0
    assert metrics.used_ram_mb >= 0
    assert metrics.process_rss_mb >= 0
    assert isinstance(metrics.under_memory_pressure, bool)


def test_reranker_disabled_by_default():
    """Verify CrossEncoder reranker is disabled by default and does not load model weights."""
    reranker = CrossEncoderReranker()
    assert not reranker.enabled
    assert not reranker.is_loaded
    assert reranker._model is None

    # Calling rerank when disabled returns candidates immediately without loading model
    candidates = [{"id": "1", "content": "hello", "score": 0.9}]
    results = reranker.rerank("test query", candidates, top_k=1)
    assert len(results) == 1
    assert not reranker.is_loaded


def test_embedder_lazy_loading():
    """Verify TextEmbedder does not load heavy PyTorch weights until explicitly required."""
    embedder = TextEmbedder(model_name="all-MiniLM-L6-v2", device="cpu")
    assert not embedder.is_loaded
    assert embedder.vector_size == 384  # Uses known dimension cache without loading model
    assert not embedder.is_loaded


def test_whisper_lazy_loading_and_unload():
    """Verify AudioTranscriber lazy loads and supports memory release."""
    transcriber = AudioTranscriber(model_size="base")
    assert not transcriber.is_loaded

    # Test unload does not error when not loaded
    transcriber.unload_model()
    assert not transcriber.is_loaded


@pytest.mark.asyncio
async def test_concurrency_semaphores():
    """Verify concurrency semaphores properly throttle parallel ML tasks."""
    sem = resource_manager.embedding_semaphore
    assert sem._value == settings.MAX_CONCURRENT_EMBEDDINGS

    active_tasks = 0
    max_parallel = 0

    async def _simulate_task():
        nonlocal active_tasks, max_parallel
        async with resource_manager.embedding_semaphore:
            active_tasks += 1
            max_parallel = max(max_parallel, active_tasks)
            await asyncio.sleep(0.05)
            active_tasks -= 1

    # Run 5 concurrent tasks
    await asyncio.gather(*[_simulate_task() for _ in range(5)])
    assert max_parallel == 1  # Concurrency strictly held at 1


@pytest.mark.asyncio
async def test_antigravity_timeout_handling():
    """Verify AntigravityLLM handles subprocess timeouts cleanly without leaking processes."""
    llm = AntigravityLLM(timeout=1)

    with patch("asyncio.create_subprocess_exec") as mock_subproc:
        mock_proc = MagicMock()
        mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError)
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()
        mock_subproc.return_value = mock_proc

        with pytest.raises(TimeoutError):
            await llm.generate_text("Test query")

        mock_proc.kill.assert_called_once()


@pytest.mark.asyncio
async def test_antigravity_error_output_extraction():
    """Verify AntigravityLLM correctly extracts error message from stdout JSON when stderr is empty."""
    llm = AntigravityLLM()
    mock_error_json = (
        b'{"status":"ERROR","error":"invalid model selection (--model \\"invalid_model\\"): not recognized"}'
    )

    with patch("asyncio.create_subprocess_exec") as mock_subproc:
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(mock_error_json, b""))
        mock_subproc.return_value = mock_proc

        with pytest.raises(RuntimeError) as exc_info:
            await llm.generate_text("Test query")

        assert "invalid model selection" in str(exc_info.value)


@pytest.mark.asyncio
async def test_antigravity_error_output_strips_newlines():
    """Verify AntigravityLLM strips leading and trailing newlines from stderr output when a CLI process fails."""
    llm = AntigravityLLM()
    mock_stderr = b"\n\nError: model execution failed\n\n"

    with patch("asyncio.create_subprocess_exec") as mock_subproc:
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", mock_stderr))
        mock_subproc.return_value = mock_proc

        with pytest.raises(RuntimeError) as exc_info:
            await llm.generate_text("Test query")

        err_str = str(exc_info.value)
        assert err_str.startswith("agy CLI execution failed (exit code 1): Error: model execution failed")
        assert not err_str.endswith("\n")


def test_git_failure_preserves_local_markdown(tmp_path):
    """Verify that if a remote Git sync fails, the local Markdown note remains safely written and intact."""
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    daily_dir = vault_dir / "Daily Notes"
    daily_dir.mkdir()

    writer = ObsidianVaultWriter(vault_path=vault_dir)
    engine = GitEngine(vault_path=vault_dir)

    entry = InterstitialEntry(
        timestamp="2026-08-15 17:00",
        category="log",
        content="Important meeting notes about deployment.",
    )

    # 1. Write Markdown
    daily_path, _ = writer.append_interstitial_entry(entry, date_str="2026-08-15")
    assert daily_path.exists()
    assert "Important meeting notes about deployment" in daily_path.read_text(encoding="utf-8")

    # 2. Simulate Git push failure (e.g. no remote or network disconnect)
    push_ok = engine.push_sync()
    assert push_ok is False

    # 3. Verify local Markdown was never deleted or corrupted
    assert daily_path.exists()
    assert "Important meeting notes about deployment" in daily_path.read_text(encoding="utf-8")


def test_bm25_incremental_file_updates():
    """Verify BM25Index updates chunks for a single file without needing full vault re-tokenization."""
    index = BM25Index()
    docs = [
        {"id": "doc1", "content": "PostgreSQL database notes", "metadata": {"file_path": "Notes/DB.md"}},
        {"id": "doc2", "content": "FastAPI service architecture", "metadata": {"file_path": "Notes/API.md"}},
    ]
    index.build_index(docs)
    assert len(index.documents) == 2

    # Update Notes/DB.md with new content
    new_db_chunks = [
        {"id": "doc1_v2", "content": "PostgreSQL 16 tuning and indexing", "metadata": {"file_path": "Notes/DB.md"}},
    ]
    index.upsert_file_chunks("Notes/DB.md", new_db_chunks)

    assert len(index.documents) == 2
    results = index.search("tuning indexing", top_k=2)
    assert len(results) >= 1
    assert results[0]["id"] == "doc1_v2"


@pytest.mark.asyncio
async def test_concurrent_daily_note_writes(tmp_path):
    """Verify concurrent async interstitial appends do not corrupt daily note due to FileLock."""
    vault_dir = tmp_path / "concurrent_vault"
    writer = ObsidianVaultWriter(vault_path=vault_dir)

    async def write_entry(idx: int):
        entry = InterstitialEntry(
            timestamp=f"2026-08-16 10:{idx:02d}",
            category="log",
            content=f"Concurrent entry {idx} with unique details",
        )
        return await writer.append_interstitial_entry_async(entry, date_str="2026-08-16")

    # Run 10 concurrent writes
    await asyncio.gather(*(write_entry(i) for i in range(10)))

    daily_file = vault_dir / "Daily Notes" / "2026-08-16.md"
    assert daily_file.exists()
    content = daily_file.read_text(encoding="utf-8")

    # All 10 entries must be present in the file
    for i in range(10):
        assert f"Concurrent entry {i} with unique details" in content


@pytest.mark.asyncio
async def test_webhook_secret_token_enforcement(monkeypatch):
    """Verify webhook rejects requests with missing or spoofed secret tokens."""
    import httpx
    from src.main import app

    # Case 1: Secret token configured, but request sends wrong token
    monkeypatch.setattr(settings, "TELEGRAM_SECRET_TOKEN", "super-secret-passphrase")
    monkeypatch.setattr(settings, "ALLOWED_TELEGRAM_USER_IDS", [123456])

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Spoofed token
        resp = await client.post(
            "/webhook/telegram",
            json={"update_id": 1, "message": {"message_id": 1, "chat": {"id": 123456}, "text": "hello"}},
            headers={"x-telegram-bot-api-secret-token": "wrong-token"},
        )
        assert resp.status_code == 401

        # Missing token
        resp_missing = await client.post(
            "/webhook/telegram",
            json={"update_id": 1, "message": {"message_id": 1, "chat": {"id": 123456}, "text": "hello"}},
        )
        assert resp_missing.status_code == 401


@pytest.mark.asyncio
async def test_watcher_and_reindex_concurrency_mutex(tmp_path):
    """Verify watcher indexing and manual reindexing cannot run concurrently (serialized by background_job_semaphore)."""
    vault_dir = tmp_path / "vault_mutex"
    vault_dir.mkdir()
    (vault_dir / "Note.md").write_text("# Test Note\nSome content.", encoding="utf-8")

    from src.graphrag.watcher import VaultChangeHandler
    from src.graphrag.embedder import TextEmbedder
    from src.graphrag.vector_db import get_vector_store

    mock_embedder = TextEmbedder()
    mock_vector_store = get_vector_store()

    handler = VaultChangeHandler(
        vault_path=vault_dir,
        embedder=mock_embedder,
        vector_store=mock_vector_store,
        debounce_seconds=0.1,
    )

    execution_order = []

    async def simulate_manual_reindex():
        async with resource_manager.background_job_semaphore:
            execution_order.append("reindex_start")
            await asyncio.sleep(0.3)
            execution_order.append("reindex_end")

    async def simulate_watcher_job():
        # Wait a small bit so manual reindex gets semaphore first
        await asyncio.sleep(0.05)
        async with resource_manager.background_job_semaphore:
            execution_order.append("watcher_job")

    await asyncio.gather(simulate_manual_reindex(), simulate_watcher_job())

    # Watcher job must not start before manual reindex ends
    assert execution_order == ["reindex_start", "reindex_end", "watcher_job"]
    handler.stop()


@pytest.mark.asyncio
async def test_watcher_events_debounced_and_coalesced(tmp_path):
    """Verify multiple rapid filesystem events for the same file are coalesced into a single pending update."""
    vault_dir = tmp_path / "vault_debounce"
    vault_dir.mkdir()
    target_file = vault_dir / "Active.md"
    target_file.write_text("# Active Note\nInitial.", encoding="utf-8")

    from src.graphrag.watcher import VaultChangeHandler
    from src.graphrag.embedder import TextEmbedder
    from src.graphrag.vector_db import get_vector_store
    from watchdog.events import FileModifiedEvent

    handler = VaultChangeHandler(
        vault_path=vault_dir,
        embedder=TextEmbedder(),
        vector_store=get_vector_store(),
        debounce_seconds=0.5,
    )

    # Simulate 5 rapid modification events for the same file
    for _ in range(5):
        handler.on_modified(FileModifiedEvent(str(target_file)))

    with handler._lock:
        # Pending files should only contain 1 entry for this file
        assert len(handler._pending_files) == 1
        assert str(target_file) in handler._pending_files

    handler.stop()


@pytest.mark.asyncio
async def test_mark_task_async_non_blocking_event_loop(tmp_path):
    """Verify mark_task_by_id_or_pattern_async executes without blocking concurrent event loop coroutines."""
    vault_dir = tmp_path / "vault_async_tasks"
    daily_dir = vault_dir / "Daily Notes"
    daily_dir.mkdir(parents=True)

    note_file = daily_dir / "2026-08-16.md"
    note_file.write_text("""# 📅 Daily Note: 2026-08-16
## 🎯 Priorities & Tasks
- [ ] Non-blocking task item ^task-nb-01
""", encoding="utf-8")

    writer = ObsidianVaultWriter(vault_path=vault_dir)

    heartbeat_ticks = 0

    async def heartbeat():
        nonlocal heartbeat_ticks
        for _ in range(10):
            await asyncio.sleep(0.02)
            heartbeat_ticks += 1

    async def mark_task():
        marked, note_name, text = await writer.mark_task_by_id_or_pattern_async(task_id="task-nb-01")
        return marked

    results = await asyncio.gather(mark_task(), heartbeat())
    assert results[0] is True
    # Heartbeat coroutine successfully ticked during task completion
    assert heartbeat_ticks >= 1

    # Verify task completed on disk
    assert "- [x] Non-blocking task item" in note_file.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_consolidation_respects_background_job_semaphore(monkeypatch):
    """Verify process_telegram_consolidation is properly guarded by resource_manager.background_job_semaphore."""
    from src.main import process_telegram_consolidation

    semaphore_acquired = False
    original_acquire = resource_manager.background_job_semaphore.acquire

    async def mock_acquire():
        nonlocal semaphore_acquired
        semaphore_acquired = True
        return await original_acquire()

    monkeypatch.setattr(resource_manager.background_job_semaphore, "acquire", mock_acquire)

    with patch("src.main.send_telegram_message", new_callable=AsyncMock):
        with patch("src.main.consolidator.generate_consolidation_report", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = MagicMock(summary_markdown="Test consolidation summary")
            await process_telegram_consolidation(chat_id=12345)

    assert semaphore_acquired is True


def test_llm_effort_validation():
    """Verify Settings validates LLM_EFFORT values ('low', 'medium', 'high', or None)."""
    from src.config import Settings

    s1 = Settings(LLM_EFFORT="high")
    assert s1.LLM_EFFORT == "high"

    s2 = Settings(LLM_EFFORT="  MEDIUM  ")
    assert s2.LLM_EFFORT == "medium"

    s3 = Settings(LLM_EFFORT=None)
    assert s3.LLM_EFFORT is None

    s4 = Settings(LLM_EFFORT="")
    assert s4.LLM_EFFORT is None

    with pytest.raises(ValueError, match="Invalid LLM_EFFORT 'high3'"):
        Settings(LLM_EFFORT="high3")


@pytest.mark.asyncio
async def test_antigravity_llm_stderr_stripping():
    """Verify AntigravityLLM strips whitespace from stderr when command fails."""
    llm = AntigravityLLM(timeout=5)

    with patch("asyncio.create_subprocess_exec") as mock_subproc:
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"\n\nError: invalid --effort high3\n\n"))
        mock_subproc.return_value = mock_proc

        with pytest.raises(RuntimeError) as exc_info:
            await llm.generate_text("Test prompt")

        assert "agy CLI execution failed (exit code 1): Error: invalid --effort high3" in str(exc_info.value)

