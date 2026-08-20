"""FastAPI webhook server, resource diagnostics, and knowledge base API endpoints for PKM AI Agent."""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import logging
from pathlib import Path
import re
import secrets
import sys
from typing import Any
import uuid

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from src.agents.consolidation import KnowledgeConsolidator
from src.agents.models import ConsolidationProposal, InterstitialEntry, QueryRequest, QueryResponse
from src.agents.multimodal import IngestionItem, ModalityType, MultimodalIngestionPipeline
from src.agents.parser import EntryParserAgent
from src.agents.qa import KnowledgeBaseQAAgent, query_knowledge_base
from src.agents.transcriber import AudioTranscriber
from src.config import settings
from src.evaluation.evaluator import RetrievalEvaluator
from src.graphrag.embedder import TextEmbedder
from src.graphrag.graph import VaultKnowledgeGraph
from src.graphrag.reindexer import reindex_vault
from src.graphrag.reranker import CrossEncoderReranker
from src.graphrag.resolver import WikiLinkResolver
from src.graphrag.retriever import HybridRetriever
from src.graphrag.sparse import BM25Index
from src.graphrag.vector_db import get_vector_store
from src.graphrag.watcher import VaultWatcher
from src.llm.factory import get_llm_provider
from src.telegram import (
    TelegramUpdate,
    answer_telegram_callback_query,
    download_telegram_file,
    edit_telegram_message_text,
    format_daily_scheduled_message,
    format_pending_tasks_message,
    is_task_query_intent,
    send_telegram_message,
)
from src.utils.resources import resource_manager, resource_monitor
from src.vault.git_engine import VaultGitEngine
from src.vault.md_writer import ObsidianVaultWriter

# Standardized logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Global shared singleton component instances (lazy-loaded as needed)
embedder = TextEmbedder(
    model_name=settings.EMBEDDING_MODEL_NAME,
    device=settings.EMBEDDING_DEVICE,
)
vector_store = get_vector_store(collection_name=settings.QDRANT_COLLECTION_NAME)
bm25_index = BM25Index()
knowledge_graph = VaultKnowledgeGraph(vault_path=settings.VAULT_PATH)
reranker = CrossEncoderReranker(
    model_name=settings.RERANKER_MODEL_NAME,
    enabled=settings.RERANKER_ENABLED,
)
retriever = HybridRetriever(
    embedder=embedder,
    vector_store=vector_store,
    bm25_index=bm25_index,
    knowledge_graph=knowledge_graph,
    reranker=reranker,
)

vault_writer = ObsidianVaultWriter()
git_engine = VaultGitEngine()
audio_transcriber = AudioTranscriber(model_size=settings.WHISPER_MODEL_SIZE)
multimodal_pipeline = MultimodalIngestionPipeline(audio_transcriber=audio_transcriber)

vault_watcher = (
    VaultWatcher(
        vault_path=settings.VAULT_PATH,
        embedder=embedder,
        vector_store=vector_store,
        bm25_index=bm25_index,
        knowledge_graph=knowledge_graph,
    )
    if getattr(settings, "ENABLE_FILE_WATCHER", True)
    else None
)

llm_driver = get_llm_provider()
wiki_resolver = WikiLinkResolver(confidence_threshold=settings.WIKILINKS_CONFIDENCE_THRESHOLD)
parser_agent = EntryParserAgent(llm=llm_driver, resolver=wiki_resolver)
qa_agent = KnowledgeBaseQAAgent(
    retriever=retriever,
    embedder=embedder,
    vector_store=vector_store,
    llm=llm_driver,
)
consolidator = KnowledgeConsolidator(
    vault_path=settings.VAULT_PATH,
    knowledge_graph=knowledge_graph,
    llm=llm_driver,
)


async def run_startup_reindex() -> None:
    """Run vault re-indexing safely in background on startup and populate retriever cache."""
    async with resource_manager.background_job_semaphore:
        try:
            logger.info("Initiating automated vault re-indexing background task on startup...")
            stats = await reindex_vault(
                vault_path=settings.VAULT_PATH,
                embedder=embedder,
                vector_store=vector_store,
                knowledge_graph=knowledge_graph,
            )
            logger.info("Automated startup vault re-indexing completed: %s", stats)

            # Build in-memory BM25 and graph cache from scanned chunks on worker thread
            def _build_and_update_retriever_cache() -> int:
                from src.graphrag.reindexer import VaultReindexer
                reindexer = VaultReindexer(
                    vault_path=settings.VAULT_PATH,
                    embedder=embedder,
                    vector_store=vector_store,
                    knowledge_graph=knowledge_graph,
                )
                scanned_files = reindexer.scan_vault_files()
                all_chunks = []
                for f in scanned_files:
                    for c in reindexer.chunker.chunk_file(f, reindexer.vault_path):
                        cid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{c.file_path}#{c.chunk_index}"))
                        all_chunks.append({"id": cid, "content": c.content, "metadata": c.to_dict()})
                retriever.update_chunk_cache(all_chunks)
                return len(all_chunks)

            chunks_count = await asyncio.to_thread(_build_and_update_retriever_cache)
            logger.info("Initialized in-memory BM25 index with %d chunks.", chunks_count)

        except Exception as err:
            logger.exception("Background startup vault re-indexing encountered error: %s", err)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager: sets up file watcher, ensures dashboard exists, and triggers startup indexing and daily scheduler."""
    logger.info(
        "FastAPI application starting up (Profile: %s, 2 vCPU / 4 GB Target)...",
        settings.APP_PROFILE,
    )

    # 1. Ensure master Dashboard note exists in Obsidian vault
    try:
        vault_writer.ensure_dashboard_exists()
    except Exception as err:
        logger.warning("Could not initialize Dashboard.md: %s", err)

    # 2. Start real-time file watcher
    if vault_watcher:
        vault_watcher.start()

    # 3. Trigger initial startup reindexing in background
    startup_task = asyncio.create_task(run_startup_reindex())
    _background_tasks.add(startup_task)
    startup_task.add_done_callback(_background_tasks.discard)

    # 4. Start daily 8:30 AM morning briefing background scheduler
    scheduler_task = asyncio.create_task(daily_task_digest_scheduler())

    # 5. Start Memory Watchdog to proactively free RAM
    async def memory_watchdog() -> None:
        while True:
            try:
                await asyncio.sleep(60)
                if resource_monitor.is_under_pressure():
                    logger.warning("System under memory pressure! Evicting idle models...")
                    audio_transcriber.unload_model()
                    embedder.unload_model()
                    reranker.unload_model()
            except asyncio.CancelledError:
                break

    watchdog_task = asyncio.create_task(memory_watchdog())

    yield

    logger.info("FastAPI application shutting down...")
    scheduler_task.cancel()
    watchdog_task.cancel()
    try:
        await scheduler_task
    except asyncio.CancelledError:
        pass

    if vault_watcher:
        vault_watcher.stop()
    vector_store.close()


app = FastAPI(
    title="PKM AI Agent",
    description="Modern Local-First AI Personal Knowledge Management System (2 vCPU / 4 GB Target)",
    version="2.1.0",
    lifespan=lifespan,
)


# ==============================================================================
# Telegram Background Pipelines (Asynchronous & Non-Blocking)
# ==============================================================================


async def process_telegram_tasks_query(chat_id: int, query_text: str = "") -> None:
    """Handle request for pending tasks list, render formatted template, and reply with interactive buttons."""
    logger.info("Processing pending tasks query for chat %d: '%s'", chat_id, query_text)
    try:
        tasks = await vault_writer.get_all_tasks_async(include_completed=False)
        message_text, inline_buttons = format_pending_tasks_message(
            tasks=tasks, today=datetime.now().date()
        )
        reply_markup = {"inline_keyboard": inline_buttons} if inline_buttons else None
        await send_telegram_message(chat_id, message_text, reply_markup=reply_markup)
    except Exception as err:
        logger.exception("Failed to process pending tasks query for chat %d: %s", chat_id, err)
        await send_telegram_message(chat_id, "⚠️ Failed to retrieve pending tasks from vault.")


async def send_daily_scheduled_briefing(chat_id: int | None = None) -> None:
    """Generate and send 8:30 AM daily task briefing to authorized Telegram users."""
    try:
        tasks = await vault_writer.get_all_tasks_async(include_completed=False)
        message_text, inline_buttons = format_daily_scheduled_message(
            tasks=tasks, today=datetime.now()
        )
        reply_markup = {"inline_keyboard": inline_buttons} if inline_buttons else None

        recipient_chats = [chat_id] if chat_id else settings.ALLOWED_TELEGRAM_USER_IDS
        for cid in recipient_chats:
            try:
                await send_telegram_message(cid, message_text, reply_markup=reply_markup)
                logger.info("Sent daily scheduled task briefing to chat ID %d", cid)
            except Exception as send_err:
                logger.exception("Failed to send daily scheduled task briefing to chat ID %d: %s", cid, send_err)
    except Exception as err:
        logger.exception("Failed to generate daily scheduled task briefing: %s", err)


async def daily_task_digest_scheduler() -> None:
    """Background scheduler task that sends the morning briefing every day at 08:30 AM."""
    logger.info(
        "Daily task digest scheduler initialized (Target: %s daily)",
        settings.SCHEDULED_BRIEFING_TIME,
    )

    try:
        time_parts = settings.SCHEDULED_BRIEFING_TIME.split(":")
        target_hour = int(time_parts[0])
        target_minute = int(time_parts[1]) if len(time_parts) > 1 else 0
    except Exception:
        target_hour, target_minute = 8, 30

    while True:
        try:
            now = datetime.now()
            target_dt = now.replace(
                hour=target_hour, minute=target_minute, second=0, microsecond=0
            )
            if now >= target_dt:
                target_dt += timedelta(days=1)

            wait_seconds = (target_dt - now).total_seconds()
            logger.info(
                "Daily 8:30 AM briefing scheduled for %s (waiting %.1f seconds)",
                target_dt.strftime("%Y-%m-%d %H:%M:%S"),
                wait_seconds,
            )
            while datetime.now() < target_dt:
                await asyncio.sleep(60)

            if settings.SCHEDULED_BRIEFING_ENABLED:
                logger.info("Triggering automated daily 8:30 AM task briefing...")
                await send_daily_scheduled_briefing()

            await asyncio.sleep(65)
        except asyncio.CancelledError:
            logger.info("Daily task digest scheduler received cancellation signal.")
            break
        except Exception as err:
            logger.exception("Unexpected error in daily task digest scheduler loop: %s", err)
            await asyncio.sleep(60)


async def process_telegram_query(chat_id: int, query_text: str) -> None:
    """Process search query from Telegram via /ask, perform hybrid multi-stage search, and synthesize QA answer."""
    logger.info("Processing Telegram query for chat %d: '%s'", chat_id, query_text)
    try:
        response = await qa_agent.query(
            query_text=query_text,
            top_k=settings.RETRIEVAL_FINAL_TOP_K,
        )
        await send_telegram_message(chat_id, response.answer)
    except Exception as err:
        logger.exception("Failed to process Telegram query for chat %d: %s", chat_id, err)
        await send_telegram_message(
            chat_id, "⚠️ Sorry, an error occurred while processing your query."
        )


async def process_telegram_consolidation(chat_id: int) -> None:
    """Generate and send knowledge consolidation report via Telegram under the background job semaphore."""
    logger.info("Generating knowledge consolidation report for chat %d", chat_id)
    await send_telegram_message(chat_id, "🔍 *Analyzing vault knowledge evolution and maintenance opportunities...*")
    async with resource_manager.background_job_semaphore:
        try:
            proposal = await consolidator.generate_consolidation_report()
            await send_telegram_message(chat_id, proposal.summary_markdown)
        except Exception as err:
            logger.exception("Consolidation report failed: %s", err)
            await send_telegram_message(chat_id, "⚠️ Failed to generate knowledge consolidation report.")


# Global set to hold references to fire-and-forget background tasks (prevents GC mid-execution)
_background_tasks: set[asyncio.Task] = set()


async def _safe_git_commit_and_push(commit_msg: str) -> None:
    """Execute Git commit+push under the background job semaphore to prevent .git/index.lock collisions."""
    async with resource_manager.background_job_semaphore:
        try:
            await git_engine.commit_and_push(commit_msg)
        except Exception as err:
            logger.warning("Background git sync failed: %s", err)


async def _async_background_post_ingestion(
    entry: InterstitialEntry,
    daily_note_name: str,
    atomic_note_name: str | None = None,
) -> None:
    """Execute non-blocking background indexing and Git synchronization after entry is persisted to Markdown."""
    async with resource_manager.background_job_semaphore:
        try:
            # Graph Update (offloaded from event loop)
            await asyncio.to_thread(
                knowledge_graph.update_file_note,
                vault_writer.daily_notes_dir / daily_note_name,
            )

            # 3. Git Commit and Push
            commit_msg = f"Vault update: {entry.category} entry at {entry.timestamp}"
            await git_engine.commit_and_push(commit_msg)

        except Exception as err:
            logger.warning("Background post-ingestion indexing or git sync encountered issue: %s", err)


async def process_telegram_ingestion(
    chat_id: int,
    raw_text: str,
    modality: ModalityType = ModalityType.TEXT,
    item_metadata: dict[str, Any] | None = None,
) -> None:
    """Process Telegram ingestion: Parse -> Write Markdown -> Send user confirmation -> Background index & Git."""
    logger.info("Processing Telegram capture for chat %d (modality=%s)", chat_id, modality)

    try:
        # 1. Fetch existing vault note titles for entity grounding
        existing_notes = await vault_writer.get_existing_note_titles_async()

        # 2. Parse raw text into structured InterstitialEntry with grounded WikiLinks
        entry: InterstitialEntry = await parser_agent.parse(
            raw_text=raw_text,
            existing_notes=existing_notes,
        )

        # 3. Atomic Note Confidence Decision
        auto_threshold = getattr(settings, "ATOMIC_NOTES_AUTO_CREATE_THRESHOLD", 0.85)
        proposal_threshold = getattr(settings, "ATOMIC_NOTES_PROPOSAL_THRESHOLD", 0.50)
        auto_create_enabled = getattr(settings, "ATOMIC_NOTES_AUTO_CREATE", True)

        should_auto_create = (
            entry.requires_atomic_note
            and auto_create_enabled
            and entry.atomic_note_confidence >= auto_threshold
        )
        should_propose = (
            entry.requires_atomic_note
            and not should_auto_create
            and entry.atomic_note_confidence >= proposal_threshold
        )

        # If only proposing, temporarily disable auto-creation during append
        original_requires = entry.requires_atomic_note
        if not should_auto_create:
            entry.requires_atomic_note = False

        # 4. Write Markdown to Obsidian Daily Note & optional atomic note immediately on disk
        daily_note_path, atomic_note_path = await vault_writer.append_interstitial_entry_async(entry)
        entry.requires_atomic_note = original_requires

        # 5. Build user confirmation reply with interactive inline keyboard
        confirmation = f"✅ **Entry Logged** ({entry.category})\n"
        confirmation += f"📄 **Content**: {entry.content}\n"
        if entry.extracted_wikilinks:
            confirmation += f"🔗 **WikiLinks**: {', '.join(['[[' + w + ']]' for w in entry.extracted_wikilinks])}\n"
        if atomic_note_path and entry.atomic_note_title:
            confirmation += f"📌 **Atomic Note Created**: [[{entry.atomic_note_title}]]\n"

        # Build inline action buttons
        inline_buttons: list[list[dict[str, str]]] = []
        date_str = (
            entry.timestamp.split(" ")[0]
            if " " in entry.timestamp
            else entry.timestamp[:10]
        )

        is_task = entry.category.lower().strip() in (
            "task", "tasks", "priority", "priorities", "todo"
        )
        if is_task:
            task_snippet = entry.content[:30].replace(":", "")
            inline_buttons.append([
                {"text": "✅ Mark Done", "callback_data": f"done:{date_str}:{task_snippet}"}
            ])

        if atomic_note_path and entry.atomic_note_title:
            btn_title = entry.atomic_note_title
            btn_label = btn_title if len(btn_title) <= 24 else btn_title[:21] + "..."
            cb_title = entry.atomic_note_title.encode("utf-8")[:58].decode("utf-8", "ignore")
            inline_buttons.append([
                {
                    "text": f"📌 View [[{btn_label}]]",
                    "callback_data": f"view:{cb_title}",
                }
            ])

        # Handle Medium-Confidence Proposal
        if should_propose and entry.atomic_note_title:
            proposal_note_title = entry.atomic_note_title
            confirmation += f"\n💡 *Possible new concept detected*: **[[{proposal_note_title}]]**\n"
            if entry.atomic_note_reason:
                confirmation += f"_{entry.atomic_note_reason}_\n"
            btn_label = proposal_note_title if len(proposal_note_title) <= 20 else proposal_note_title[:17] + "..."
            cb_prop = proposal_note_title.encode("utf-8")[:48].decode("utf-8", "ignore")
            inline_buttons.append([
                {
                    "text": f"✨ Create [[{btn_label}]]",
                    "callback_data": f"create_atomic:{cb_prop}",
                },
                {
                    "text": "❌ Ignore",
                    "callback_data": f"ignore_atomic:{cb_prop}",
                }
            ])

        # 6. Send Telegram response immediately
        reply_markup = {"inline_keyboard": inline_buttons} if inline_buttons else None
        await send_telegram_message(chat_id, confirmation, reply_markup=reply_markup)

        # 7. Queue background indexing and Git sync (non-blocking)
        task = asyncio.create_task(
            _async_background_post_ingestion(
                entry=entry,
                daily_note_name=daily_note_path.name,
                atomic_note_name=atomic_note_path.name if atomic_note_path else None,
            )
        )
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

    except Exception as err:
        logger.exception("Failed to process Telegram capture for chat %d: %s", chat_id, err)
        await send_telegram_message(chat_id, "⚠️ Failed to log entry to vault.")


async def process_telegram_audio_ingestion(
    chat_id: int, file_id: str, caption: str | None = None
) -> None:
    """Download Telegram voice/audio, transcribe using Whisper worker, and process ingestion."""
    logger.info("Processing Telegram audio ingestion for chat %d (file_id=%s)", chat_id, file_id)

    temp_audio_path: Path | None = None
    try:
        temp_audio_path = await download_telegram_file(file_id)
        if not temp_audio_path or not temp_audio_path.exists():
            await send_telegram_message(chat_id, "⚠️ Failed to download voice memo.")
            return

        item = IngestionItem(
            modality=ModalityType.VOICE,
            content="",
            file_path=temp_audio_path,
            caption=caption,
            sender_id=chat_id,
        )
        full_content = await multimodal_pipeline.process(item)

        if not full_content.strip():
            await send_telegram_message(
                chat_id, "⚠️ Voice memo was empty or could not be transcribed."
            )
            return

        logger.info("Voice transcription completed for chat %d: '%s'", chat_id, full_content)
        await send_telegram_message(chat_id, f"🎙️ **Transcribed**: *\"{full_content}\"*")
        await process_telegram_ingestion(chat_id, full_content, modality=ModalityType.VOICE)

    except Exception as err:
        logger.exception("Audio ingestion failed for chat %d: %s", chat_id, err)
        await send_telegram_message(chat_id, "⚠️ Audio transcription error occurred.")
    finally:
        if temp_audio_path and temp_audio_path.exists():
            try:
                temp_audio_path.unlink()
            except OSError:
                pass


import pyotp
import dotenv

# In-memory set to track if a chat is waiting for a TOTP code
pending_resets: set[int] = set()

async def initiate_telegram_mfa_setup(chat_id: int) -> None:
    """Generate and send MFA secret for TOTP setup."""
    from src.config import settings
    import pyotp
    import dotenv
    import os
    
    secret = pyotp.random_base32()
    
    def _save_secret() -> None:
        dotenv.set_key(".env", "MFA_SECRET", secret)
        os.chmod(".env", 0o600)

    await asyncio.to_thread(_save_secret)
    settings.MFA_SECRET = secret
    
    setup_msg = (
        "🔐 *MFA Setup Required*\n\n"
        "I have generated a new secret for your Authenticator app (e.g., Microsoft Authenticator).\n"
        f"**Secret Key:** `{secret}`\n\n"
        "Please add this manually to your app. It will be required for sensitive actions like `/reset`."
    )
    await send_telegram_message(chat_id, setup_msg)


async def initiate_telegram_reset_db(chat_id: int) -> None:
    """Initiate a database reset using TOTP Authenticator."""
    from src.config import settings
    
    if not settings.MFA_SECRET:
        await initiate_telegram_mfa_setup(chat_id)
        return

    pending_resets.add(chat_id)
    
    warning_msg = (
        "⚠️ *WARNING: DATABASE RESET* ⚠️\n\n"
        "You are about to completely wipe all notes and data from the vault. "
        "This action is **irreversible**.\n\n"
        "To confirm, please open your **Microsoft/Google Authenticator** app and reply with the current 6-digit TOTP code."
    )
    await send_telegram_message(chat_id, warning_msg)

async def process_telegram_reset_db(chat_id: int) -> None:
    """Clear all markdown notes from the vault except essential templates/configs, triggering a database wipe."""
    logger.info("Executing database reset via Telegram command for chat %d", chat_id)
    await send_telegram_message(chat_id, "🧹 *Initiating complete database reset. Deleting all notes...*")
    
    async with resource_manager.background_job_semaphore:
        try:
            import shutil
            import os
            from src.config import settings
            vault_path = Path(settings.VAULT_PATH).resolve()
            
            allowed_items = {".obsidian", "templates", "dashboard.md", ".gitignore", ".git"}
            
            def _wipe_vault() -> int:
                count = 0
                if not vault_path.exists():
                    return 0
                for item in vault_path.iterdir():
                    if item.name.lower() not in allowed_items:
                        if item.is_dir():
                            count += sum(len(files) for _, _, files in os.walk(item))
                            shutil.rmtree(item)
                        else:
                            item.unlink()
                            count += 1
                return count
            
            deleted_count = await asyncio.to_thread(_wipe_vault)
            
            # Commit and push, bypassing the self-recovery mechanism so the deletion is permanent
            await git_engine.commit_and_push("Vault reset: wiped all notes", bypass_recovery=True)
            await send_telegram_message(chat_id, f"✅ *Database successfully reset.* (Deleted {deleted_count} notes/files)")
        except Exception as err:
            logger.exception("Database reset failed: %s", err)
            await send_telegram_message(chat_id, "⚠️ Failed to reset database. Check server logs.")


# ==============================================================================
# FastAPI REST Endpoints
# ==============================================================================


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Comprehensive service health and resource diagnostics endpoint."""
    metrics = resource_monitor.get_metrics()
    return {
        "status": "healthy",
        "service": "pkm-agent",
        "version": "2.1.0",
        "profile": settings.APP_PROFILE,
        "resources": {
            "ram_total_mb": metrics.total_ram_mb,
            "ram_used_mb": metrics.used_ram_mb,
            "ram_free_mb": metrics.free_ram_mb,
            "ram_percent": metrics.percent_ram,
            "process_rss_mb": metrics.process_rss_mb,
            "cpu_percent": metrics.cpu_percent,
            "memory_pressure": metrics.under_memory_pressure,
        },
        "components": {
            "qdrant": "ok",
            "watcher": "active" if vault_watcher else "disabled",
            "embedding": {
                "model": embedder.model_name,
                "loaded": embedder.is_loaded,
                "device": embedder.resolved_device,
            },
            "reranker": {
                "model": reranker.model_name,
                "enabled": reranker.enabled,
                "loaded": reranker.is_loaded,
            },
            "whisper": {
                "model": audio_transcriber.model_size,
                "loaded": audio_transcriber.is_loaded,
            },
            "llm": {
                "provider": settings.LLM_PROVIDER,
            },
            "knowledge_graph": {
                "notes_count": len(knowledge_graph.nodes),
                "edges_count": knowledge_graph.graph.number_of_edges(),
            },
            "bm25": {
                "indexed_chunks": len(bm25_index.documents),
            },
        },
    }


from fastapi import Depends, Header, HTTPException

def verify_api_key(api_key: str | None = Header(default=None, alias="x-api-key")) -> None:
    expected = settings.TELEGRAM_SECRET_TOKEN
    if not expected:
        raise HTTPException(status_code=500, detail="API Token not configured on server")
    if not api_key or not secrets.compare_digest(api_key, expected):
        logger.warning("Rejected unauthorized API request.")
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.post("/ask", response_model=QueryResponse, dependencies=[Depends(verify_api_key)])
@app.post("/query", response_model=QueryResponse, dependencies=[Depends(verify_api_key)])
async def ask_knowledge_base(request: QueryRequest) -> QueryResponse:
    """Query knowledge base and synthesize an answer using modern hybrid search, reranking, and provenance."""
    return await qa_agent.query(
        query_text=request.query,
        top_k=request.top_k,
        filters=request.filters,
        expand_graph=request.expand_graph,
    )


@app.post("/vault/reindex", dependencies=[Depends(verify_api_key)])
async def trigger_vault_reindex(
    background_tasks: BackgroundTasks,
    force: bool = False,
) -> dict[str, str]:
    """Trigger vault re-indexing as a background task."""

    async def _run_reindex() -> None:
        async with resource_manager.background_job_semaphore:
            try:
                stats = await reindex_vault(
                    vault_path=settings.VAULT_PATH,
                    embedder=embedder,
                    vector_store=vector_store,
                    knowledge_graph=knowledge_graph,
                    force=force,
                )
                logger.info("Manual vault re-indexing completed: %s", stats)
            except Exception as err:
                logger.exception("Manual vault re-indexing failed: %s", err)

    background_tasks.add_task(_run_reindex)
    return {"status": "reindexing_started", "force": str(force)}


@app.post("/vault/consolidate", response_model=ConsolidationProposal, dependencies=[Depends(verify_api_key)])
async def trigger_vault_consolidation() -> ConsolidationProposal:
    """Run knowledge consolidation and evolution report on demand under background job semaphore."""
    async with resource_manager.background_job_semaphore:
        return await consolidator.generate_consolidation_report()


@app.post("/webhook/telegram")
async def telegram_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_telegram_bot_api_secret_token: str | None = Header(default=None),
) -> dict[str, str]:
    """Telegram bot webhook endpoint for receiving text, voice notes, and interactive button callbacks."""
    # 1. Header token verification
    expected_token = settings.TELEGRAM_SECRET_TOKEN
    if not expected_token:
        logger.error("TELEGRAM_SECRET_TOKEN is not configured — rejecting all webhook requests.")
        raise HTTPException(status_code=500, detail="TELEGRAM_SECRET_TOKEN not configured")

    received_token = x_telegram_bot_api_secret_token or ""
    if not secrets.compare_digest(received_token, expected_token):
        logger.warning("Unauthorized webhook request: secret token mismatch.")
        raise HTTPException(status_code=401, detail="Invalid secret token")

    # 2. Parse request JSON payload
    try:
        body = await request.json()
        update = TelegramUpdate.model_validate(body)
    except Exception as err:
        logger.error("Invalid JSON body received on Telegram webhook: %s", err)
        raise HTTPException(status_code=400, detail="Invalid update payload") from err

    # 3. Handle Interactive Button Callback Queries
    if update.callback_query:
        sender_id = update.callback_query.from_user.id
        if sender_id not in settings.ALLOWED_TELEGRAM_USER_IDS:
            logger.warning("Rejected callback query from unauthorized user %s", sender_id)
            raise HTTPException(status_code=403, detail="Forbidden")

        cb_data = update.callback_query.data or ""
        cb_id = update.callback_query.id
        chat_id = (
            update.callback_query.message.chat.id
            if update.callback_query.message
            else sender_id
        )

        if cb_data.startswith("done:"):
            parts = cb_data.split(":", 2)
            target_hint = parts[1] if len(parts) > 1 else ""
            param = parts[2] if len(parts) > 2 else ""

            if len(parts) == 3:
                if len(param) == 8 and re.match(r"^[0-9a-fA-F]{8}$", param):
                    marked, note_name, task_text = await vault_writer.mark_task_by_id_or_pattern_async(
                        task_id=param, daily_date=target_hint
                    )
                else:
                    marked, note_name, task_text = await vault_writer.mark_task_by_id_or_pattern_async(
                        daily_date=target_hint, task_pattern=param
                    )
            elif len(parts) == 2:
                if len(target_hint) == 8 and re.match(r"^[0-9a-fA-F]{8}$", target_hint):
                    marked, note_name, task_text = await vault_writer.mark_task_by_id_or_pattern_async(
                        task_id=target_hint
                    )
                else:
                    marked, note_name, task_text = await vault_writer.mark_task_by_id_or_pattern_async(
                        task_pattern=target_hint
                    )
            else:
                marked, note_name, task_text = False, "", ""

            if marked:
                display_name = task_text or "task"
                # Acknowledge callback query immediately to dismiss UI loading spinner
                await answer_telegram_callback_query(
                    cb_id, text=f"✅ Completed: {display_name[:45]}"
                )
                task = asyncio.create_task(
                    _safe_git_commit_and_push(
                        f"Completed task in vault note '{note_name}': {display_name[:40]}"
                    )
                )
                _background_tasks.add(task)
                task.add_done_callback(_background_tasks.discard)
                if update.callback_query.message:
                    orig_text = update.callback_query.message.text or ""
                    if "Entry Logged" in orig_text:
                        await edit_telegram_message_text(
                            chat_id=chat_id,
                            message_id=update.callback_query.message.message_id,
                            text=f"{orig_text}\n\n✨ *[Status: Marked Complete in Vault]*",
                        )
                    else:
                        try:
                            remaining_tasks = await vault_writer.get_all_tasks_async(
                                include_completed=False
                            )
                            if (
                                "Morning Briefing" in orig_text
                                or "Daily Morning" in orig_text
                            ):
                                updated_text, updated_buttons = format_daily_scheduled_message(
                                    tasks=remaining_tasks, today=datetime.now()
                                )
                            else:
                                updated_text, updated_buttons = format_pending_tasks_message(
                                    tasks=remaining_tasks,
                                    today=datetime.now().date(),
                                )
                            reply_markup = (
                                {"inline_keyboard": updated_buttons}
                                if updated_buttons
                                else None
                            )
                            await edit_telegram_message_text(
                                chat_id=chat_id,
                                message_id=update.callback_query.message.message_id,
                                text=updated_text,
                                reply_markup=reply_markup,
                            )
                        except Exception as edit_err:
                            logger.debug("Could not live-refresh task list: %s", edit_err)
                            await edit_telegram_message_text(
                                chat_id=chat_id,
                                message_id=update.callback_query.message.message_id,
                                text=f"{orig_text}\n\n✨ *[Completed: {display_name[:50]}]*",
                            )
            else:
                await answer_telegram_callback_query(
                    cb_id, text="⚠️ Task already completed or not found."
                )
            return {"status": "callback_handled"}

        elif cb_data.startswith("view:"):
            note_title = cb_data.replace("view:", "").strip()
            # 1. Immediate acknowledgement dismisses Telegram button loading spinner
            await answer_telegram_callback_query(cb_id)

            # 2. Locate note across vault via exact match, prefix, or substring
            note_file = await vault_writer.find_note_by_title_or_prefix_async(note_title)
            if note_file and note_file.exists():
                try:
                    full_content = await asyncio.to_thread(note_file.read_text, encoding="utf-8")
                    # Strip frontmatter
                    body = re.sub(r"^---\s*\n.*?\n---\s*\n", "", full_content, flags=re.DOTALL).strip()
                    # Strip redundant top-level H1 header if matching filename stem
                    body = re.sub(rf"^#\s+{re.escape(note_file.stem)}\s*\n+", "", body, flags=re.IGNORECASE).strip()
                    
                    view_header = f"📌 **[[{note_file.stem}]]**:\n\n"
                    full_note_text = f"{view_header}{body}" if body else f"📌 **[[{note_file.stem}]]** *(Empty note)*"
                    await send_telegram_message(chat_id, full_note_text)
                except Exception as read_err:
                    logger.exception("Failed reading note %s: %s", note_file, read_err)
                    await send_telegram_message(
                        chat_id, f"⚠️ Error reading note **[[{note_file.stem}]]**."
                    )
            else:
                await send_telegram_message(
                    chat_id, f"⚠️ Note **[[{note_title}]]** could not be found in the vault."
                )
            return {"status": "callback_handled"}

        elif cb_data.startswith("create_atomic:"):
            proposed_title = cb_data.replace("create_atomic:", "").strip()
            await answer_telegram_callback_query(cb_id, text=f"✅ Created [[{proposed_title}]]")
            created_path = await vault_writer.create_atomic_note_async(
                title=proposed_title,
                content=f"# {proposed_title}\n\nConcept captured and approved via Telegram.",
                tags=["atomic-note", "approved"],
            )
            task = asyncio.create_task(
                _safe_git_commit_and_push(f"Created atomic note [[{proposed_title}]] via Telegram approval")
            )
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)
            if update.callback_query.message:
                orig_text = update.callback_query.message.text or ""
                await edit_telegram_message_text(
                    chat_id=chat_id,
                    message_id=update.callback_query.message.message_id,
                    text=f"{orig_text}\n\n✨ *[Approved & Created Note: [[{proposed_title}]]]*",
                )
            return {"status": "atomic_note_created"}

        elif cb_data.startswith("ignore_atomic:"):
            await answer_telegram_callback_query(cb_id, text="Proposal dismissed.")
            if update.callback_query.message:
                orig_text = update.callback_query.message.text or ""
                await edit_telegram_message_text(
                    chat_id=chat_id,
                    message_id=update.callback_query.message.message_id,
                    text=f"{orig_text}\n\n✨ *[Note Proposal Dismissed]*",
                )
            return {"status": "atomic_note_ignored"}

        await answer_telegram_callback_query(cb_id)
        return {"status": "callback_acknowledged"}

    if not update.message:
        return {"status": "ignored"}

    # 4. Security Whitelist Check
    sender_id = (
        update.message.from_user.id
        if update.message.from_user
        else update.message.chat.id
    )
    if not settings.ALLOWED_TELEGRAM_USER_IDS:
        logger.warning("Access denied: ALLOWED_TELEGRAM_USER_IDS not configured.")
        raise HTTPException(
            status_code=403,
            detail="Forbidden: ALLOWED_TELEGRAM_USER_IDS not configured.",
        )

    if sender_id not in settings.ALLOWED_TELEGRAM_USER_IDS:
        logger.warning(
            "Access denied: User ID %s not in ALLOWED_TELEGRAM_USER_IDS.",
            sender_id,
        )
        raise HTTPException(status_code=403, detail="Forbidden: User ID not authorized.")

    chat_id = update.message.chat.id

    # 5. Handle Voice Memos and Audio Files (Immediate Acknowledgment -> Async Whisper Worker)
    if update.message.voice or update.message.audio:
        file_id = (
            update.message.voice.file_id
            if update.message.voice
            else update.message.audio.file_id
        )
        caption = update.message.caption

        # Immediate acknowledgement to Telegram
        await send_telegram_message(
            chat_id,
            "🎙️ *Voice memo queued for transcription...*",
        )
        background_tasks.add_task(
            process_telegram_audio_ingestion,
            chat_id,
            file_id,
            caption,
        )
        return {"status": "voice_ingestion_queued"}

    if not update.message.text:
        return {"status": "ignored"}

    raw_text = update.message.text.strip()

    # MFA Check for pending reset
    if chat_id in pending_resets:
        pending_resets.remove(chat_id)
        if settings.MFA_SECRET:
            totp = pyotp.TOTP(settings.MFA_SECRET)
            if totp.verify(raw_text):
                background_tasks.add_task(process_telegram_reset_db, chat_id)
                return {"status": "reset_db_processing"}
            else:
                background_tasks.add_task(send_telegram_message, chat_id, "❌ Incorrect Authenticator code. Database reset cancelled.")
                return {"status": "reset_db_cancelled"}

    # 6. Route commands vs queries vs text ingestion
    if raw_text.startswith("/start"):
        if not settings.MFA_SECRET:
            background_tasks.add_task(initiate_telegram_mfa_setup, chat_id)
        else:
            background_tasks.add_task(send_telegram_message, chat_id, "🤖 Welcome back to PKM Agent! I am ready to process your notes and queries.")
        return {"status": "start_processed"}
    elif raw_text.startswith(("/reset_db", "/reset")):
        background_tasks.add_task(initiate_telegram_reset_db, chat_id)
        return {"status": "reset_db_mfa_sent"}
    elif raw_text.startswith(("/consolidate", "/maintenance", "/review")):
        background_tasks.add_task(process_telegram_consolidation, chat_id)
        return {"status": "consolidation_processing"}
    elif is_task_query_intent(raw_text):
        background_tasks.add_task(process_telegram_tasks_query, chat_id, raw_text)
        return {"status": "task_query_processing"}
    elif raw_text.startswith(("/ask", "/query")):
        clean_query = re.sub(r"^/(?:ask|query)\s*", "", raw_text).strip()
        if is_task_query_intent(clean_query):
            background_tasks.add_task(process_telegram_tasks_query, chat_id, clean_query)
            return {"status": "task_query_processing"}
        background_tasks.add_task(process_telegram_query, chat_id, raw_text)
        return {"status": "query_processing"}
    else:
        # Process text capture
        background_tasks.add_task(process_telegram_ingestion, chat_id, raw_text)
        return {"status": "ingestion_queued"}


class MarkTaskRequest(BaseModel):
    """Schema for marking a task complete via API."""

    task_id: str | None = Field(default=None, description="Deterministic 8-char task hash ID")
    daily_date: str | None = Field(default=None, description="Daily note date or file hint")
    task_pattern: str | None = Field(default=None, description="Search substring matching task line")


@app.get("/tasks/pending", dependencies=[Depends(verify_api_key)])
async def get_pending_tasks_endpoint() -> dict[str, Any]:
    """Return all active pending tasks from the vault in structured JSON and Markdown template format."""
    tasks = await vault_writer.get_all_tasks_async(include_completed=False)
    template_text, _ = format_pending_tasks_message(tasks, today=datetime.now().date())
    return {
        "count": len(tasks),
        "tasks": [
            {
                "task_id": t.task_id,
                "task_text": t.task_text,
                "file_rel_path": t.file_rel_path,
                "file_name": t.file_name,
                "source_note": t.source_note_display,
                "due_date": t.due_date,
                "created_date": t.created_date,
                "priority": t.priority,
            }
            for t in tasks
        ],
        "template_preview": template_text,
    }


@app.post("/tasks/complete", dependencies=[Depends(verify_api_key)])
async def mark_task_complete_endpoint(req: MarkTaskRequest) -> dict[str, Any]:
    """Mark a task complete in the vault via API."""
    marked, note_name, task_text = await vault_writer.mark_task_by_id_or_pattern_async(
        task_id=req.task_id or "",
        daily_date=req.daily_date or "",
        task_pattern=req.task_pattern or "",
    )
    if marked:
        await _safe_git_commit_and_push(
            f"Completed task in vault note '{note_name}': {task_text[:40]}"
        )
        return {"status": "success", "note_name": note_name, "task_text": task_text}
    else:
        raise HTTPException(status_code=404, detail="Task not found or already completed.")


@app.post("/tasks/send-briefing", dependencies=[Depends(verify_api_key)])
async def trigger_send_briefing_endpoint(
    background_tasks: BackgroundTasks,
    chat_id: int | None = None,
) -> dict[str, str]:
    """Trigger the 8:30 AM daily task briefing to be sent via Telegram."""
    background_tasks.add_task(send_daily_scheduled_briefing, chat_id)
    return {
        "status": "briefing_queued",
        "target_chat": str(chat_id or settings.ALLOWED_TELEGRAM_USER_IDS),
    }


def print_health_report() -> None:
    """Print comprehensive resource and component health diagnostics to console."""
    metrics = resource_monitor.get_metrics()
    print("=" * 60)
    print("🧠 PKM AI Second Brain — System & Component Health")
    print("=" * 60)
    print(f"Deployment Profile:    {settings.APP_PROFILE} (2 vCPU / 4 GB Target)")
    print(f"System RAM:            {metrics.used_ram_mb:.1f} MB / {metrics.total_ram_mb:.1f} MB ({metrics.percent_ram}%)")
    print(f"Process RSS Memory:    {metrics.process_rss_mb:.1f} MB")
    print(f"CPU Utilization:       {metrics.cpu_percent}%")
    print(f"Memory Pressure State: {'⚠️ HIGH PRESSURE' if metrics.under_memory_pressure else '✅ NORMAL'}")
    print("-" * 60)
    print(f"Qdrant Vector Store:   OK (Collection: '{settings.QDRANT_COLLECTION_NAME}')")
    print(f"Real-Time Watcher:     {'ACTIVE' if vault_watcher else 'DISABLED'}")
    print(f"Embedding Model:       {embedder.model_name} (loaded: {embedder.is_loaded}, device: {embedder.resolved_device})")
    print(f"CrossEncoder Reranker: {reranker.model_name} (enabled: {reranker.enabled}, loaded: {reranker.is_loaded})")
    print(f"Whisper Transcriber:   faster-whisper (model: {audio_transcriber.model_size}, loaded: {audio_transcriber.is_loaded})")
    print(f"LLM Provider:          {settings.LLM_PROVIDER} ({llm_driver.__class__.__name__})")
    print(f"Knowledge Graph:       {len(knowledge_graph.nodes)} notes, {knowledge_graph.graph.number_of_edges()} WikiLink edges")
    print(f"BM25 Lexical Index:    {len(bm25_index.documents)} document chunks indexed")

    git_stat = git_engine.check_working_tree()
    print(f"Git Working Tree:      {'DIRTY' if git_stat.get('is_dirty') else 'CLEAN'} ({len(git_stat.get('modified', []))} modified, {len(git_stat.get('conflicts', []))} conflicts)")
    print("=" * 60)


def cli_entrypoint() -> None:
    """CLI handler supporting 'reindex', 'evaluate', 'health', and web server start."""
    if len(sys.argv) > 1 and sys.argv[1] == "health":
        print_health_report()
    elif len(sys.argv) > 1 and sys.argv[1] == "reindex":
        force = "--force" in sys.argv or "-f" in sys.argv
        print(f"🔄 Starting Vault reindexing (force={force})...")
        stats = asyncio.run(
            reindex_vault(
                vault_path=settings.VAULT_PATH,
                embedder=embedder,
                vector_store=vector_store,
                knowledge_graph=knowledge_graph,
                force=force,
            )
        )
        print(f"✅ Reindexing finished: {stats}")
    elif len(sys.argv) > 1 and sys.argv[1] == "evaluate":
        evaluator = RetrievalEvaluator(qa_agent=qa_agent)
        dataset_path = Path("src/evaluation/sample_dataset.json")

        reranker_flag = None
        if "--reranker" in sys.argv:
            idx = sys.argv.index("--reranker")
            if idx + 1 < len(sys.argv):
                val = sys.argv[idx + 1].lower()
                if val in ("on", "true", "1", "enabled"):
                    reranker_flag = True
                elif val in ("off", "false", "0", "disabled"):
                    reranker_flag = False

        state_str = (
            "ON" if reranker_flag is True else ("OFF" if reranker_flag is False else f"Default ({reranker.enabled})")
        )
        print(f"📊 Running retrieval & QA evaluation against {dataset_path} (Reranker: {state_str})...")
        metrics = asyncio.run(evaluator.evaluate_file(dataset_path, reranker_enabled=reranker_flag))
        print(f"✅ Evaluation Results:\n{metrics.to_dict()}")
    else:
        import uvicorn
        uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    cli_entrypoint()
