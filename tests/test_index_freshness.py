"""Tests that the search index follows the vault, using the real filesystem watcher."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.qa import KnowledgeBaseQAAgent
from src.graphrag.reindexer import VaultReindexer
from src.graphrag.sparse import BM25Index
from src.graphrag.vector_db import QdrantVectorStore
from src.graphrag.watcher import VaultChangeHandler, VaultWatcher
from src.llm.mock_llm import MockLLM
from src.vault.md_writer import ObsidianVaultWriter, atomic_write_text

VECTOR = [0.1, 0.2, 0.3, 0.4]


def _fake_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.model_name = "test-model"
    embedder.encode_batch_async = AsyncMock(side_effect=lambda texts: [VECTOR for _ in texts])
    return embedder


def _fake_store() -> MagicMock:
    store = MagicMock()
    store.upsert_documents_async = AsyncMock(return_value=True)
    store.delete_file_chunks_async = AsyncMock()
    store.indexed_file_paths_async = AsyncMock(return_value=set())
    return store


def _handler(vault: Path, store: MagicMock, bm25: BM25Index | None = None) -> VaultChangeHandler:
    return VaultChangeHandler(
        vault_path=vault,
        embedder=_fake_embedder(),
        vector_store=store,
        bm25_index=bm25,
        debounce_seconds=0.05,
    )


async def _wait_until(condition, timeout_seconds: float = 15.0) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while asyncio.get_running_loop().time() < deadline:
        if condition():
            return True
        await asyncio.sleep(0.1)
    return False


# Real watcher ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_capture_written_atomically_reaches_the_index(tmp_path):
    """Regression: atomic writes looked like deletions, so captures were never indexed."""
    from src.agents.models import InterstitialEntry

    vault = tmp_path.resolve()
    store = _fake_store()
    bm25 = BM25Index()
    watcher = VaultWatcher(
        vault_path=vault,
        embedder=_fake_embedder(),
        vector_store=store,
        bm25_index=bm25,
        debounce_seconds=0.1,
    )
    writer = ObsidianVaultWriter(vault_path=vault)
    writer._ensure_directories()
    watcher.start()
    try:
        entry = InterstitialEntry(timestamp="2026-11-09 18:55", category="discovery", content="Cgroups replaced it")
        await asyncio.to_thread(writer.append_interstitial_entry, entry, "2026-11-09")
        indexed = await _wait_until(lambda: any("Cgroups" in d["content"] for d in bm25.documents))
    finally:
        watcher.stop()

    assert indexed
    upserted = [doc for call in store.upsert_documents_async.await_args_list for doc in call.args[0]]
    assert any("Cgroups" in doc["content"] for doc in upserted)


@pytest.mark.asyncio
async def test_rewriting_a_note_atomically_keeps_it_in_the_index(tmp_path):
    vault = tmp_path.resolve()
    note = vault / "Note.md"
    note.write_text("# Note\n\nFirst version of the text.\n", encoding="utf-8")
    store = _fake_store()
    bm25 = BM25Index()
    watcher = VaultWatcher(
        vault_path=vault,
        embedder=_fake_embedder(),
        vector_store=store,
        bm25_index=bm25,
        debounce_seconds=0.1,
    )
    watcher.start()
    try:
        await asyncio.sleep(0.5)
        atomic_write_text(note, "# Note\n\nSecond version of the text.\n")
        indexed = await _wait_until(lambda: any("Second version" in d["content"] for d in bm25.documents))
        await asyncio.sleep(0.5)
    finally:
        watcher.stop()

    assert indexed
    assert any("Second version" in d["content"] for d in bm25.documents)


# Watcher events -------------------------------------------------------------


def _pending(handler: VaultChangeHandler) -> tuple[set[str], set[str]]:
    return set(handler._pending_files), set(handler._pending_deletes)


def test_rename_queues_the_new_path_and_removes_the_old_one(tmp_path):
    vault = tmp_path.resolve()
    handler = _handler(vault, _fake_store())
    event = MagicMock(is_directory=False, src_path=str(vault / "Old.md"), dest_path=str(vault / "New.md"))

    handler.on_moved(event)

    assert _pending(handler) == ({str(vault / "New.md")}, {str(vault / "Old.md")})


def test_temp_file_renamed_over_a_note_only_queues_the_note(tmp_path):
    vault = tmp_path.resolve()
    handler = _handler(vault, _fake_store())
    event = MagicMock(is_directory=False, src_path=str(vault / ".Note.md.abc.tmp"), dest_path=str(vault / "Note.md"))

    handler.on_moved(event)

    assert _pending(handler) == ({str(vault / "Note.md")}, set())


@pytest.mark.asyncio
async def test_delete_event_for_a_file_that_still_exists_reindexes_it(tmp_path):
    vault = tmp_path.resolve()
    note = vault / "Note.md"
    note.write_text("# Note\n\nStill here.\n", encoding="utf-8")
    store = _fake_store()
    bm25 = BM25Index()
    handler = _handler(vault, store, bm25)
    handler.on_deleted(MagicMock(is_directory=False, src_path=str(note)))

    worker = asyncio.create_task(handler._async_debounce_worker())
    handler._running = True
    try:
        assert await _wait_until(lambda: store.upsert_documents_async.await_count > 0, 5.0)
    finally:
        handler._running = False
        worker.cancel()

    store.delete_file_chunks_async.assert_awaited_with("Note.md")
    assert any("Still here" in d["content"] for d in bm25.documents)


@pytest.mark.asyncio
async def test_real_deletion_removes_vectors(tmp_path):
    vault = tmp_path.resolve()
    store = _fake_store()
    handler = _handler(vault, store)

    await handler._async_handle_delete_file(vault / "Gone.md")

    store.delete_file_chunks_async.assert_awaited_once_with("Gone.md")


@pytest.mark.asyncio
async def test_old_vectors_are_removed_before_new_ones_are_added(tmp_path):
    vault = tmp_path.resolve()
    note = vault / "Note.md"
    note.write_text("# Note\n\nText.\n", encoding="utf-8")
    calls: list[str] = []
    store = _fake_store()
    store.delete_file_chunks_async.side_effect = lambda path: calls.append("delete")
    store.upsert_documents_async.side_effect = lambda docs: calls.append("upsert") or True

    await _handler(vault, store)._async_process_single_file(note)

    assert calls == ["delete", "upsert"]


@pytest.mark.asyncio
async def test_emptied_note_loses_its_old_vectors(tmp_path):
    vault = tmp_path.resolve()
    note = vault / "Note.md"
    note.write_text("", encoding="utf-8")
    store = _fake_store()

    await _handler(vault, store)._async_process_single_file(note)

    store.delete_file_chunks_async.assert_awaited_once_with("Note.md")
    store.upsert_documents_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_upsert_is_not_recorded_as_indexed(tmp_path):
    vault = tmp_path.resolve()
    note = vault / "Note.md"
    note.write_text("# Note\n\nText.\n", encoding="utf-8")
    store = _fake_store()
    store.upsert_documents_async.return_value = False
    handler = _handler(vault, store)

    await handler._async_process_single_file(note)

    assert "Note.md" not in handler.registry.records


def test_watcher_ignores_the_dashboard(tmp_path):
    vault = tmp_path.resolve()
    assert _handler(vault, _fake_store())._should_ignore(vault / "Dashboard.md") is True


# Reindexer ------------------------------------------------------------------


def _reindexer(vault: Path, store: MagicMock) -> VaultReindexer:
    return VaultReindexer(vault_path=vault, embedder=_fake_embedder(), vector_store=store, knowledge_graph=MagicMock())


@pytest.mark.asyncio
async def test_reindex_skips_dashboard_and_prunes_files_no_longer_in_vault(tmp_path):
    vault = tmp_path.resolve()
    (vault / "Dashboard.md").write_text("# Dashboard\n\nGenerated.\n", encoding="utf-8")
    (vault / "Note.md").write_text("# Note\n\nReal text.\n", encoding="utf-8")
    store = _fake_store()
    store.indexed_file_paths_async.return_value = {"Dashboard.md", "Deleted.md", "Note.md"}

    stats = await _reindexer(vault, store).reindex_vault()

    assert stats["scanned"] == 1
    assert stats["pruned"] == 2
    pruned = {call.args[0] for call in store.delete_file_chunks_async.await_args_list}
    assert {"Dashboard.md", "Deleted.md"} <= pruned


@pytest.mark.asyncio
async def test_reindex_does_not_record_files_when_upsert_fails(tmp_path):
    vault = tmp_path.resolve()
    (vault / "Note.md").write_text("# Note\n\nReal text.\n", encoding="utf-8")
    store = _fake_store()
    store.upsert_documents_async.return_value = False
    reindexer = _reindexer(vault, store)

    await reindexer.reindex_vault()

    assert "Note.md" not in reindexer.registry.records


# Vector store ---------------------------------------------------------------


@pytest.fixture
def real_store(tmp_path):
    saved = QdrantVectorStore._instance
    QdrantVectorStore._instance = None
    store = QdrantVectorStore(collection_name="freshness", storage_path=tmp_path / "qdrant")
    yield store
    store.close()
    QdrantVectorStore._instance = saved


def _document(doc_id: str, file_path: str) -> dict:
    return {"id": doc_id, "vector": VECTOR, "content": "x", "metadata": {"file_path": file_path}}


def test_vector_store_deletes_only_the_given_file(real_store):
    real_store.upsert_documents([_document("a", "One.md"), _document("b", "One.md"), _document("c", "Two.md")])
    assert real_store.indexed_file_paths() == {"One.md", "Two.md"}

    real_store.delete_file_chunks("One.md")

    assert real_store.indexed_file_paths() == {"Two.md"}


def test_vector_store_handles_a_missing_collection(real_store):
    real_store.delete_file_chunks("One.md")
    assert real_store.indexed_file_paths() == set()


# Answers --------------------------------------------------------------------


def _agent(answer: str) -> KnowledgeBaseQAAgent:
    retriever = AsyncMock()
    retriever.search.return_value = [{"content": "x", "payload": {"title": "Dashboard", "file_path": "Dashboard.md"}}]
    return KnowledgeBaseQAAgent(retriever=retriever, llm=MockLLM(default_text_response=answer))


@pytest.mark.parametrize(
    "answer",
    ["I couldn't find evidence in your vault that confirms this.", "I couldn’t find evidence in your vault."],
)
@pytest.mark.asyncio
async def test_no_evidence_answer_lists_no_sources(answer):
    response = await _agent(answer).query("what replaced chroot?")
    assert "Sources" not in response.answer
    assert response.citations == []
    assert response.sources == []


@pytest.mark.asyncio
async def test_real_answer_still_lists_sources():
    response = await _agent("Cgroups replaced it.\nSOURCES: 1").query("what replaced chroot?")
    assert "**Sources:**" in response.answer


# Telegram status text -------------------------------------------------------


def test_status_messages_do_not_render_as_fake_links():
    source = (Path(__file__).resolve().parent.parent / "src" / "main.py").read_text(encoding="utf-8")
    assert "✨ *[" not in source
