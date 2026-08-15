# Python & Project Coding Standards for PKM AI Agent

You are an expert Python software engineer building an enterprise-grade, local-first AI Personal Knowledge Management (PKM) system. Follow these instructions strictly when generating or modifying code in this repository.

---

## 1. Core Architecture & Layer Boundaries

1. **Clean Architecture & Separation of Concerns**:
   - `src/config.py`: Application configuration and environment variable validation via `pydantic-settings`.
   - `src/agents/`: Pydantic data schemas (`models.py`), LLM prompt orchestration (`parser.py`), and local audio transcription (`transcriber.py`).
   - `src/llm/`: CLI drivers and wrapper interfaces for local LLM execution (`antigravity_llm.py`).
   - `src/vault/`: Obsidian filesystem operations, template rendering, Markdown AST manipulation (`md_writer.py`), and Git version control engine (`git_engine.py`).
   - `src/graphrag/`: Dense vector embeddings (`embedder.py`), Qdrant vector database (`vector_db.py`), hash-based incremental indexing (`reindexer.py`), and real-time filesystem observation (`watcher.py`).
   - `src/main.py`: FastAPI server setup, lifecycle management (`lifespan`), HTTP endpoints, and high-level request routing.

2. **Single Responsibility Principle (SRP) & Layer Decoupling**:
   - **Vault Layer Purity**: The vault layer (`src/vault/`) must strictly manage file I/O, Git operations, and Markdown parsing. It MUST NOT contain presentation-layer logic or external platform constructs (e.g. Telegram inline keyboard matrices).
   - **Decoupled Business Logic**: Business logic and external API integrations must never depend directly on FastAPI `Request` objects. Functions must accept clean Python types or Pydantic models.
   - **Module Sizing & Cohesion**: Keep modules focused and cohesive. Avoid monolithic files exceeding 400–500 lines by delegating distinct concerns (e.g. Telegram bot communication, background schedulers, RAG synthesis) to dedicated helper modules or services.

---

## 2. Python Code Style & Modern Syntax

1. **PEP 8 Compliance & Modern Typing**:
   - Adhere strictly to PEP 8 standards.
   - Use Python 3.10+ modern typing syntax (`list[str]`, `dict[str, Any]`, `str | None`, `tuple[Path, Path | None]`) instead of legacy `typing.List`, `typing.Optional`, or `typing.Union`.
   - Every function and method signature MUST include explicit input parameter type hints and return type hints.

2. **Pydantic v2 Standards**:
   - Always use Pydantic v2 (`pydantic.BaseModel`, `Field`, `ConfigDict`, `pydantic_settings.BaseSettings`).
   - Use `model_config = ConfigDict(str_strip_whitespace=True, ...)` instead of deprecated inner `class Config`.
   - Always provide clear `description` metadata fields inside `Field()` definitions to assist LLM structured output schema generation.

3. **Asynchronous & Non-Blocking Patterns**:
   - Use `async def` for FastAPI endpoints and I/O-bound webhook operations.
   - Offload heavy blocking tasks (Git operations, disk writes, vector embedding generation, audio transcription) using `asyncio.to_thread` or FastAPI `BackgroundTasks` to keep the event loop responsive.
   - Never call `asyncio.run()` inside an active asyncio event loop.

4. **Thread-Safe Singleton Resource Management**:
   - Heavy shared components (e.g. `QdrantVectorStore`, `AudioTranscriber`, `TextEmbedder`) must implement thread-safe singleton or lazy-loading patterns to avoid redundant resource allocation or GPU/CPU memory duplication.
   - Heavy models (`faster-whisper`, `SentenceTransformer`) must be lazy-loaded on first use.

---

## 3. Error Handling, Resource Management & Logging

1. **Structured Logging**:
   - Use Python's standard `logging` module (`logger = logging.getLogger(__name__)`).
   - Do NOT use bare `print()` statements anywhere in `src/`.
   - Use lazy string formatting in logger calls (`logger.info("Processed %d items", count)`) rather than f-strings (`logger.info(f"Processed {count} items")`).
   - Log at appropriate severity levels (`logger.debug()`, `logger.info()`, `logger.warning()`, `logger.error()`, `logger.exception()`).

2. **Defensive Error Handling**:
   - Wrap all external I/O interactions (Git operations, Qdrant database calls, network HTTP requests, file reads/writes, subprocesses) in explicit `try...except` blocks.
   - Catch specific exceptions (`git.GitCommandError`, `json.JSONDecodeError`, `subprocess.CalledProcessError`, `httpx.HTTPError`) before generic `Exception`.
   - Always use `logger.exception()` to capture full tracebacks when handling non-fatal errors.

3. **Deterministic Resource Cleanup**:
   - Temporary files (e.g., downloaded voice memos) must be cleaned up deterministically in `finally:` blocks.
   - External connections (e.g., Qdrant storage client, filesystem watchers, background schedulers) must be cleanly closed and cancelled in the FastAPI `lifespan` context manager during application shutdown.

---

## 4. Telegram Bot & Webhook Conventions

1. **Callback Data Size Limit (<= 64 Bytes)**:
   - Telegram API imposes a strict 64-byte limit on `callback_data` payloads.
   - Always use compact formats with deterministic short hashes (e.g. 8-character MD5 hash for tasks: `done:<hint>:<hash8>`).

2. **Security & Authorization**:
   - Validate incoming webhook requests using `x-telegram-bot-api-secret-token` against `settings.TELEGRAM_SECRET_TOKEN`.
   - Enforce user ID authorization against `settings.ALLOWED_TELEGRAM_USER_IDS` on both messages and callback queries.

3. **Message Formatting & Resilient Fallback**:
   - Send formatted responses in Markdown (`parse_mode="Markdown"`).
   - If Telegram API rejects a message due to Markdown formatting errors, automatically catch the error and retry sending as plain text.

---

## 5. Obsidian Markdown & Knowledge Graph Integrity

1. **Wikilink & Alias Formatting**:
   - Wrap concepts, tools, and notes in double brackets: `[[Concept Name]]` or `[[Target Note|Display Alias]]`.
   - When extracting note titles from aliased links, always extract the raw target note title without brackets or aliases.
   - Ensure no single-bracket links (`[AWS]`) or malformed nested brackets (`[[[Concept]]]`) are generated.

2. **Strict Daily Note Section Insertion**:
   - Interstitial journal entries must target predefined standard headers:
     - `## 📥 Inbox (Quick Capture)`
     - `## ⏱️ Log (Interstitial)`
     - `## 🎯 Priorities & Tasks`
     - `## 🧠 Discoveries & Learning`
   - If a target header is missing in an existing note, create the header gracefully before appending the entry.

3. **Obsidian Tasks & Dataview Syntax**:
   - Format action items using strict Obsidian Tasks syntax: `- [ ] {Task Description} ➕ YYYY-MM-DD 📅 YYYY-MM-DD`.
   - Format inline Dataview metadata as `[key:: value]`.
   - Format Obsidian block anchors as `^block-id` at the end of the entry line.
   - Format callouts strictly as `> [!NOTE]`, `> [!WARNING]`, `> [!TIP]`, etc.

4. **Atomic Note Creation & Bidirectional Linking**:
   - Standalone concept notes must be stored in `/Notes/{Title}.md` with YAML frontmatter (`type: atomic-note`, `created: YYYY-MM-DD`, `tags: [...]`).
   - Include a bidirectional link `source_daily_note: "[[YYYY-MM-DD]]"` back to the originating daily note.

---

## 6. Environment & Deployment Safety

1. **Zero-Hardcoding Configuration**:
   - Never hardcode secrets, bot tokens, absolute paths, or credentials.
   - Access all configuration exclusively through `src.config.settings`.

2. **Local-First & Embedded Defaults**:
   - Default to local embedded storage (`qdrant_storage`) and local path resolution (`./vault`) with zero external container dependencies required for basic development.
