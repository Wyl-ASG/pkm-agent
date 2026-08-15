"""FastAPI webhook server and knowledge base API endpoints for PKM AI Agent."""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import logging
from pathlib import Path
import re
from typing import Any
import uuid

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from src.agents.models import InterstitialEntry, QueryRequest, QueryResponse
from src.agents.parser import EntryParserAgent
from src.agents.qa import query_knowledge_base
from src.agents.transcriber import AudioTranscriber
from src.config import settings
from src.graphrag.embedder import TextEmbedder
from src.graphrag.reindexer import reindex_vault
from src.graphrag.vector_db import get_vector_store
from src.graphrag.watcher import VaultWatcher
from src.llm.antigravity_llm import AntigravityLLM
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
from src.vault.git_engine import VaultGitEngine
from src.vault.md_writer import ObsidianVaultWriter

# Standardized logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Global shared component instances
embedder = TextEmbedder(model_name=settings.EMBEDDING_MODEL_NAME)
vector_store = get_vector_store(collection_name=settings.QDRANT_COLLECTION_NAME)
vault_writer = ObsidianVaultWriter()
git_engine = VaultGitEngine()
audio_transcriber = AudioTranscriber(model_size=settings.WHISPER_MODEL_SIZE)
vault_watcher = (
    VaultWatcher(
        vault_path=settings.VAULT_PATH,
        embedder=embedder,
        vector_store=vector_store,
    )
    if getattr(settings, "ENABLE_FILE_WATCHER", True)
    else None
)

llm_driver = AntigravityLLM(
    binary_path=settings.AGY_PATH,
    model=settings.LLM_MODEL,
    effort=settings.LLM_EFFORT,
)
parser_agent = EntryParserAgent(llm=llm_driver)


async def run_startup_reindex() -> None:
    """Run vault re-indexing safely in background on startup."""
    try:
        logger.info("Initiating automated vault re-indexing background task on startup...")
        stats = await reindex_vault(
            vault_path=settings.VAULT_PATH,
            embedder=embedder,
            vector_store=vector_store,
        )
        logger.info("Automated startup vault re-indexing completed: %s", stats)
    except Exception as err:
        logger.exception("Background startup vault re-indexing encountered error: %s", err)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager: sets up file watcher, ensures dashboard exists, and triggers startup indexing and daily scheduler."""
    logger.info("FastAPI application starting up...")

    # 1. Ensure master Dashboard note exists in Obsidian vault
    try:
        vault_writer.ensure_dashboard_exists()
    except Exception as err:
        logger.warning("Could not initialize Dashboard.md: %s", err)

    # 2. Start real-time file watcher
    if vault_watcher:
        vault_watcher.start()

    # 3. Trigger initial startup reindexing
    asyncio.create_task(run_startup_reindex())

    # 4. Start daily 8:30 AM morning briefing background scheduler
    scheduler_task = asyncio.create_task(daily_task_digest_scheduler())

    yield

    logger.info("FastAPI application shutting down...")
    scheduler_task.cancel()
    try:
        await scheduler_task
    except asyncio.CancelledError:
        pass

    if vault_watcher:
        vault_watcher.stop()
    vector_store.close()


app = FastAPI(
    title="PKM AI Agent",
    description="Local-first AI Personal Knowledge Management System",
    version="1.2.0",
    lifespan=lifespan,
)


# ==============================================================================
# Telegram Background Pipelines
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
            await send_telegram_message(cid, message_text, reply_markup=reply_markup)
            logger.info("Sent daily scheduled task briefing to chat ID %d", cid)
    except Exception as err:
        logger.exception("Failed to send daily scheduled task briefing: %s", err)


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
                "Daily 8:30 AM briefing scheduled for %s (waiting %.1f seconds / %.1f hours)",
                target_dt.strftime("%Y-%m-%d %H:%M:%S"),
                wait_seconds,
                wait_seconds / 3600.0,
            )

            await asyncio.sleep(wait_seconds)

            if settings.SCHEDULED_BRIEFING_ENABLED:
                logger.info("Triggering automated daily 8:30 AM task briefing...")
                await send_daily_scheduled_briefing()

            # Wait 65s to advance past the current target minute
            await asyncio.sleep(65)
        except asyncio.CancelledError:
            logger.info("Daily task digest scheduler received cancellation signal.")
            break
        except Exception as err:
            logger.exception("Unexpected error in daily task digest scheduler loop: %s", err)
            await asyncio.sleep(60)


async def process_telegram_query(chat_id: int, query_text: str) -> None:
    """Process search query from Telegram via /ask, perform hybrid search, synthesize LLM answer, and reply."""
    logger.info("Processing Telegram query for chat %d: '%s'", chat_id, query_text)
    try:
        response = await query_knowledge_base(
            query_text=query_text,
            embedder=embedder,
            vector_store=vector_store,
            llm=llm_driver,
        )
        await send_telegram_message(chat_id, response.answer)
    except Exception as err:
        logger.exception("Failed to process Telegram query for chat %d: %s", chat_id, err)
        await send_telegram_message(
            chat_id, "⚠️ Sorry, an error occurred while processing your query."
        )


async def process_telegram_ingestion(chat_id: int, raw_text: str) -> None:
    """Process raw text ingestion in background task: pull git, parse with existing notes, append, embed, sync."""
    logger.info("Processing Telegram ingestion background task for chat %d", chat_id)

    try:
        # 1. Pull latest changes from remote Git before modifying local notes
        await git_engine.pull(rebase=True, autostash=True)

        # 2. Fetch existing vault note titles for entity grounding
        existing_notes = await vault_writer.get_existing_note_titles_async()

        # 3. Parse raw text into structured InterstitialEntry with grounded WikiLinks
        entry: InterstitialEntry = await parser_agent.parse(
            raw_text=raw_text,
            existing_notes=existing_notes,
        )

        # 4. Append entry to Obsidian Daily Note & optional atomic note
        daily_note_path, atomic_note_path = await vault_writer.append_interstitial_entry_async(entry)

        # 5. Generate embedding vector and index document in Qdrant
        doc_vec = await embedder.encode_async(entry.content)
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{entry.timestamp}_{entry.content}"))
        doc_payload = {
            "id": doc_id,
            "vector": doc_vec,
            "content": entry.content,
            "metadata": {
                "timestamp": entry.timestamp,
                "category": entry.category,
                "extracted_tags": entry.extracted_tags,
                "extracted_wikilinks": entry.extracted_wikilinks,
                "daily_note": daily_note_path.name,
                "atomic_note": atomic_note_path.name if atomic_note_path else None,
            },
        }
        await vector_store.upsert_documents_async([doc_payload])

        # 6. Commit and push changes via Git
        commit_msg = f"Vault update via Telegram: {entry.category} entry at {entry.timestamp}"
        await git_engine.commit_and_push(commit_msg)

        # 7. Build user confirmation reply with interactive inline keyboard
        confirmation = f"✅ **Entry Logged** ({entry.category})\n"
        confirmation += f"📄 **Content**: {entry.content}\n"
        if entry.extracted_wikilinks:
            confirmation += f"🔗 **WikiLinks**: {', '.join(['[[' + w + ']]' for w in entry.extracted_wikilinks])}\n"
        if atomic_note_path and entry.atomic_note_title:
            confirmation += f"📌 **Atomic Note**: [[{entry.atomic_note_title}]]"

        # Build inline action buttons
        inline_buttons: list[list[dict[str, str]]] = []
        date_str = (
            entry.timestamp.split(" ")[0]
            if " " in entry.timestamp
            else entry.timestamp[:10]
        )

        is_task = entry.category.lower().strip() in (
            "task",
            "tasks",
            "priority",
            "priorities",
            "todo",
        )
        if is_task:
            task_snippet = entry.content[:30].replace(":", "")
            inline_buttons.append([
                {"text": "✅ Mark Done", "callback_data": f"done:{date_str}:{task_snippet}"}
            ])

        if atomic_note_path and entry.atomic_note_title:
            inline_buttons.append([
                {
                    "text": f"📌 View [[{entry.atomic_note_title[:20]}]]",
                    "callback_data": f"view:{entry.atomic_note_title[:30]}",
                }
            ])

        reply_markup = {"inline_keyboard": inline_buttons} if inline_buttons else None
        await send_telegram_message(chat_id, confirmation, reply_markup=reply_markup)

    except Exception as err:
        logger.exception("Failed to process Telegram ingestion for chat %d: %s", chat_id, err)
        await send_telegram_message(chat_id, "⚠️ Failed to log entry to vault.")


async def process_telegram_audio_ingestion(
    chat_id: int, file_id: str, caption: str | None = None
) -> None:
    """Download Telegram voice/audio, transcribe using faster-whisper, and process ingestion."""
    logger.info("Processing Telegram audio ingestion for chat %d (file_id=%s)", chat_id, file_id)
    await send_telegram_message(chat_id, "🎙️ *Transcribing voice memo...*")

    temp_audio_path: Path | None = None
    try:
        temp_audio_path = await download_telegram_file(file_id)
        if not temp_audio_path or not temp_audio_path.exists():
            await send_telegram_message(chat_id, "⚠️ Failed to download voice memo.")
            return

        transcribed_text = await audio_transcriber.transcribe_async(temp_audio_path)
        if not transcribed_text.strip():
            await send_telegram_message(
                chat_id, "⚠️ Voice memo was empty or could not be transcribed."
            )
            return

        full_content = transcribed_text.strip()
        if caption:
            full_content = f"{caption.strip()} - {full_content}"

        logger.info("Voice transcription completed for chat %d: '%s'", chat_id, full_content)
        await send_telegram_message(chat_id, f"🎙️ **Transcribed**: *\"{full_content}\"*")
        await process_telegram_ingestion(chat_id, full_content)

    except Exception as err:
        logger.exception("Audio ingestion failed for chat %d: %s", chat_id, err)
        await send_telegram_message(chat_id, "⚠️ Audio transcription error occurred.")
    finally:
        if temp_audio_path and temp_audio_path.exists():
            try:
                temp_audio_path.unlink()
            except OSError:
                pass


# ==============================================================================
# FastAPI REST Endpoints
# ==============================================================================


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Service health check endpoint."""
    return {"status": "healthy", "service": "pkm-agent"}


@app.post("/ask", response_model=QueryResponse)
@app.post("/query", response_model=QueryResponse)
async def ask_knowledge_base(request: QueryRequest) -> QueryResponse:
    """Query knowledge base and synthesize an answer using hybrid search and LLM."""
    return await query_knowledge_base(
        query_text=request.query,
        top_k=request.top_k,
        filters=request.filters,
        embedder=embedder,
        vector_store=vector_store,
        llm=llm_driver,
    )


@app.post("/vault/reindex")
async def trigger_vault_reindex(
    background_tasks: BackgroundTasks,
    force: bool = False,
) -> dict[str, str]:
    """Trigger vault re-indexing as a background task."""

    async def _run_reindex() -> None:
        try:
            stats = await reindex_vault(
                vault_path=settings.VAULT_PATH,
                embedder=embedder,
                vector_store=vector_store,
                force=force,
            )
            logger.info("Manual vault re-indexing completed: %s", stats)
        except Exception as err:
            logger.exception("Manual vault re-indexing failed: %s", err)

    background_tasks.add_task(_run_reindex)
    return {"status": "reindexing_started", "force": str(force)}


@app.post("/webhook/telegram")
async def telegram_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_telegram_bot_api_secret_token: str | None = Header(default=None),
) -> dict[str, str]:
    """Telegram bot webhook endpoint for receiving text, voice notes, and interactive button callbacks."""
    # 1. Header token verification
    expected_token = settings.TELEGRAM_SECRET_TOKEN
    if expected_token:
        if x_telegram_bot_api_secret_token != expected_token:
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
                    marked, note_name, task_text = vault_writer.mark_task_by_id_or_pattern(
                        task_id=param, daily_date=target_hint
                    )
                else:
                    marked, note_name, task_text = vault_writer.mark_task_by_id_or_pattern(
                        daily_date=target_hint, task_pattern=param
                    )
            elif len(parts) == 2:
                if len(target_hint) == 8 and re.match(r"^[0-9a-fA-F]{8}$", target_hint):
                    marked, note_name, task_text = vault_writer.mark_task_by_id_or_pattern(
                        task_id=target_hint
                    )
                else:
                    marked, note_name, task_text = vault_writer.mark_task_by_id_or_pattern(
                        task_pattern=target_hint
                    )
            else:
                marked, note_name, task_text = False, "", ""

            if marked:
                display_name = task_text or "task"
                await git_engine.commit_and_push(
                    f"Completed task in vault note '{note_name}': {display_name[:40]}"
                )
                await answer_telegram_callback_query(
                    cb_id, text=f"✅ Completed: {display_name[:45]}"
                )
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
                            logger.debug(
                                "Could not live-refresh task list message: %s",
                                edit_err,
                            )
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
            note_file = vault_writer.notes_dir / f"{note_title}.md"
            if note_file.exists():
                snippet_text = note_file.read_text(encoding="utf-8")[:350]
                await answer_telegram_callback_query(cb_id)
                await send_telegram_message(
                    chat_id, f"📌 **[[{note_title}]]**:\n\n{snippet_text}..."
                )
            else:
                await answer_telegram_callback_query(
                    cb_id, text=f"Note [[{note_title}]] not found."
                )
            return {"status": "callback_handled"}

        await answer_telegram_callback_query(cb_id)
        return {"status": "callback_acknowledged"}

    if not update.message:
        logger.info("Ignored non-message Telegram update %d", update.update_id)
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

    # 5. Handle Voice Memos and Audio Files
    if update.message.voice:
        voice_file_id = update.message.voice.file_id
        background_tasks.add_task(
            process_telegram_audio_ingestion,
            chat_id,
            voice_file_id,
            update.message.caption,
        )
        return {"status": "voice_ingestion_queued"}

    if update.message.audio:
        audio_file_id = update.message.audio.file_id
        background_tasks.add_task(
            process_telegram_audio_ingestion,
            chat_id,
            audio_file_id,
            update.message.caption,
        )
        return {"status": "audio_ingestion_queued"}

    if not update.message.text:
        logger.info(
            "Ignored non-text message %d from user %d",
            update.message.message_id,
            sender_id,
        )
        return {"status": "ignored"}

    raw_text = update.message.text.strip()

    # 6. Route '/tasks', '/todo', '/briefing', or task queries vs '/ask' vs text ingestion
    if is_task_query_intent(raw_text):
        background_tasks.add_task(process_telegram_tasks_query, chat_id, raw_text)
        return {"status": "task_query_processing"}
    elif raw_text.startswith("/ask") or raw_text.startswith("/query"):
        clean_query = re.sub(r"^/(?:ask|query)\s*", "", raw_text).strip()
        if is_task_query_intent(clean_query):
            background_tasks.add_task(process_telegram_tasks_query, chat_id, clean_query)
            return {"status": "task_query_processing"}
        background_tasks.add_task(process_telegram_query, chat_id, raw_text)
        return {"status": "query_processing"}
    else:
        background_tasks.add_task(process_telegram_ingestion, chat_id, raw_text)
        return {"status": "ingestion_queued"}


class MarkTaskRequest(BaseModel):
    """Schema for marking a task complete via API."""

    task_id: str | None = Field(default=None, description="Deterministic 8-char task hash ID")
    daily_date: str | None = Field(default=None, description="Daily note date or file hint")
    task_pattern: str | None = Field(default=None, description="Search substring matching task line")


@app.get("/tasks/pending")
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


@app.post("/tasks/complete")
async def mark_task_complete_endpoint(req: MarkTaskRequest) -> dict[str, Any]:
    """Mark a task complete in the vault via API."""
    marked, note_name, task_text = vault_writer.mark_task_by_id_or_pattern(
        task_id=req.task_id or "",
        daily_date=req.daily_date or "",
        task_pattern=req.task_pattern or "",
    )
    if marked:
        await git_engine.commit_and_push(
            f"Completed task in vault note '{note_name}': {task_text[:40]}"
        )
        return {"status": "success", "note_name": note_name, "task_text": task_text}
    else:
        raise HTTPException(status_code=404, detail="Task not found or already completed.")


@app.post("/tasks/send-briefing")
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
