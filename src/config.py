"""Configuration settings for PKM AI Agent using pydantic-settings."""

from functools import lru_cache
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    # Deployment Profile
    APP_PROFILE: str = Field(
        default="production-small",
        description="Deployment profile: 'production-small' (2 vCPU / 4 GB RAM) or 'performance'.",
    )

    TELEGRAM_BOT_TOKEN: str = Field(
        default="",
        description="API token for Telegram Bot authentication.",
    )
    TELEGRAM_SECRET_TOKEN: str = Field(
        default="",
        description="Secret token for Telegram webhook validation.",
    )
    GIT_REPO_URL: str = Field(
        default="",
        description="Git repository URL for Obsidian vault synchronization.",
    )
    GIT_BRANCH: str = Field(
        default="main",
        description="Git branch name for Obsidian vault repository.",
    )
    SSH_KEY_PATH: str | None = Field(
        default=None,
        description="Path to SSH private key used for Git repository operations (e.g. ~/.ssh/id_ed25519).",
    )
    VAULT_PATH: str = Field(
        default="./vault",
        description="Path to local Obsidian vault directory.",
    )
    QDRANT_STORAGE_PATH: str | None = Field(
        default="./qdrant_storage",
        description="Local directory path for embedded Qdrant storage mode (zero-Docker). Set to None or empty when using a remote Qdrant server.",
    )
    QDRANT_HOST: str = Field(
        default="localhost",
        description="Host address for Qdrant vector database server (used if QDRANT_STORAGE_PATH is not set).",
    )
    QDRANT_PORT: int = Field(
        default=6333,
        description="Port number for Qdrant vector database server.",
    )

    ALLOWED_TELEGRAM_USER_IDS: list[int] = Field(
        default_factory=list,
        description="Whitelisted Telegram user IDs permitted to interact with the PKM agent.",
    )
    MFA_SECRET: str | None = Field(
        default=None,
        description="Base32 secret key for TOTP (e.g. Microsoft Authenticator) database reset verification.",
    )

    # Vector Storage & Embeddings
    QDRANT_COLLECTION_NAME: str = Field(
        default="pkm_notes",
        description="Qdrant collection name for PKM notes.",
    )
    EMBEDDING_PROVIDER: str = Field(
        default="sentence_transformers",
        description="Embedding provider engine (e.g., 'sentence_transformers').",
    )
    EMBEDDING_MODEL_NAME: str = Field(
        default="all-MiniLM-L6-v2",
        description="HuggingFace model identifier for generating text embeddings (e.g., 'all-MiniLM-L6-v2', 'BAAI/bge-m3').",
    )
    EMBEDDING_DEVICE: str = Field(
        default="cpu",
        description="Device for embedding model computation ('cpu', 'auto', 'cuda', 'mps').",
    )

    # Retrieval & Ranking Pipeline
    RETRIEVAL_DENSE_TOP_K: int = Field(
        default=30,
        description="Number of candidate chunks retrieved from dense vector search.",
    )
    RETRIEVAL_SPARSE_TOP_K: int = Field(
        default=30,
        description="Number of candidate chunks retrieved from sparse / lexical BM25 search.",
    )
    RETRIEVAL_FINAL_TOP_K: int = Field(
        default=5,
        description="Final number of top context chunks provided to QA synthesis.",
    )
    RETRIEVAL_FUSION: str = Field(
        default="rrf",
        description="Rank fusion algorithm for dense + sparse candidate merging ('rrf', 'weighted').",
    )

    # Reranker Configuration (Disabled by default for 2 vCPU / 4 GB production target)
    RERANKER_ENABLED: bool = Field(
        default=False,
        description="Whether cross-encoder reranking is enabled after hybrid fusion. Disabled by default for 2 vCPU / 4 GB servers.",
    )
    RERANKER_MODEL_NAME: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="HuggingFace model identifier for local cross-encoder reranking.",
    )
    RERANKER_TOP_K: int = Field(
        default=10,
        description="Maximum candidates passed to reranker before final context trimming.",
    )

    # Knowledge Graph Configuration
    GRAPH_ENABLED: bool = Field(
        default=True,
        description="Whether WikiLink graph-aware retrieval and neighborhood expansion is enabled.",
    )
    GRAPH_MAX_HOPS: int = Field(
        default=1,
        description="Maximum graph traversal distance (hops) for context expansion.",
    )
    GRAPH_MAX_NEIGHBORS: int = Field(
        default=5,
        description="Maximum number of graph neighbor notes to pull into candidate fusion pool.",
    )

    # WikiLinks & Entity Resolution
    WIKILINKS_CONFIDENCE_THRESHOLD: float = Field(
        default=0.65,
        description="Confidence threshold for resolving proposed entities to existing note titles/aliases.",
    )

    # Atomic Note Creation Thresholds
    ATOMIC_NOTES_AUTO_CREATE: bool = Field(
        default=True,
        description="Whether high-confidence atomic notes can be created automatically.",
    )
    ATOMIC_NOTES_AUTO_CREATE_THRESHOLD: float = Field(
        default=0.85,
        description="Confidence threshold for automatic atomic note generation.",
    )
    ATOMIC_NOTES_PROPOSAL_THRESHOLD: float = Field(
        default=0.50,
        description="Confidence threshold for proposing atomic notes to the user via Telegram.",
    )

    # Temporal & Provenance Intelligence
    TEMPORAL_ENABLED: bool = Field(
        default=True,
        description="Whether temporal query understanding and date-based metadata filtering are enabled.",
    )
    PROVENANCE_ENABLED: bool = Field(
        default=True,
        description="Whether block-level and heading-level provenance tracking is enabled in QA.",
    )
    CONSOLIDATION_ENABLED: bool = Field(
        default=True,
        description="Whether knowledge evolution and periodic vault consolidation are enabled.",
    )

    # LLM Provider Abstraction
    LLM_PROVIDER: str = Field(
        default="antigravity",
        description="Active LLM provider backend ('antigravity', 'ollama', 'mock').",
    )
    AGY_PATH: str | None = Field(
        default=None,
        description="Explicit path to the agy CLI binary (auto-discovered if not set).",
    )
    LLM_MODEL: str | None = Field(
        default=None,
        description="Model identifier passed to agy CLI or LLM provider (e.g., 'flash', 'pro').",
    )
    LLM_EFFORT: str | None = Field(
        default=None,
        description="Reasoning effort level passed to agy CLI ('low', 'medium', 'high').",
    )
    ANTIGRAVITY_TIMEOUT_SECONDS: int = Field(
        default=120,
        description="Maximum timeout in seconds for Antigravity CLI subprocess execution.",
    )
    ANTIGRAVITY_MAX_CONCURRENT: int = Field(
        default=1,
        description="Maximum concurrent Antigravity CLI subprocess calls.",
    )
    OLLAMA_BASE_URL: str = Field(
        default="http://localhost:11434",
        description="Base URL for Ollama local LLM server.",
    )
    OLLAMA_MODEL: str = Field(
        default="llama3.2",
        description="Model tag for Ollama LLM provider.",
    )

    # Audio Transcription & Watcher
    WHISPER_MODEL_SIZE: str = Field(
        default="base",
        description="Model size for faster-whisper audio transcription ('base', 'tiny', 'small', etc.).",
    )
    WHISPER_FALLBACK_MODEL_SIZE: str = Field(
        default="tiny",
        description="Fallback model size if primary Whisper model encounters memory or load errors.",
    )
    WHISPER_DEVICE: str = Field(
        default="cpu",
        description="Compute device for Whisper ('cpu', 'cuda', 'auto').",
    )
    WHISPER_COMPUTE_TYPE: str = Field(
        default="int8",
        description="Quantization compute type for CPU Whisper ('int8', 'float32').",
    )
    WHISPER_IDLE_TIMEOUT_SECONDS: int = Field(
        default=180,
        description="Idle timeout in seconds before releasing Whisper model from RAM.",
    )
    ENABLE_FILE_WATCHER: bool = Field(
        default=True,
        description="Whether to enable real-time vault file watcher for automated reindexing.",
    )

    # Concurrency & Resource Limits (2 vCPU / 4 GB Target)
    MAX_CONCURRENT_EMBEDDINGS: int = Field(
        default=1,
        description="Maximum concurrent embedding generation tasks.",
    )
    MAX_CONCURRENT_TRANSCRIPTIONS: int = Field(
        default=1,
        description="Maximum concurrent Whisper audio transcriptions.",
    )
    MAX_CONCURRENT_RERANKING: int = Field(
        default=1,
        description="Maximum concurrent CrossEncoder reranking executions.",
    )
    MAX_BACKGROUND_JOBS: int = Field(
        default=1,
        description="Maximum concurrent background indexing and maintenance tasks.",
    )
    MAX_WORKER_THREADS: int = Field(
        default=2,
        description="Maximum worker threads for threadpool operations.",
    )
    MAX_MEMORY_PRESSURE_PERCENT: float = Field(
        default=85.0,
        description="System RAM threshold percentage above which non-critical ML tasks are throttled.",
    )

    # Scheduled Briefings
    SCHEDULED_BRIEFING_TIME: str = Field(
        default="08:30",
        description="Time of day for automated daily scheduled task briefing (HH:MM in 24h format).",
    )
    SCHEDULED_BRIEFING_ENABLED: bool = Field(
        default=True,
        description="Whether to automatically send the daily 8:30 AM task briefing to Telegram.",
    )
    TIMEZONE: str = Field(
        default="Asia/Singapore",
        description="Timezone for scheduled jobs and timestamps.",
    )

    # Observability
    LOG_RETRIEVAL_DETAILS: bool = Field(
        default=True,
        description="Whether to log detailed structured statistics for retrieval and QA queries.",
    )

    @field_validator("ALLOWED_TELEGRAM_USER_IDS", mode="before")
    @classmethod
    def parse_allowed_user_ids(cls, v: object) -> list[int]:
        """Parse comma-separated string, single integer, or list into list of integers."""
        if isinstance(v, str):
            v_clean = v.strip()
            if not v_clean:
                return []
            if v_clean.startswith("[") and v_clean.endswith("]"):
                import json
                try:
                    parsed = json.loads(v_clean)
                    return [int(x) for x in parsed]
                except Exception:
                    pass
            parts = [p.strip() for p in v_clean.split(",") if p.strip()]
            return [int(p) for p in parts if p.isdigit() or (p.startswith("-") and p[1:].isdigit())]
        elif isinstance(v, int):
            return [v]
        elif isinstance(v, (list, tuple, set)):
            return [int(x) for x in v]
        return []

    @field_validator("LLM_EFFORT", mode="before")
    @classmethod
    def validate_llm_effort(cls, v: object) -> str | None:
        """Validate LLM reasoning effort level ('low', 'medium', 'high')."""
        if v is None:
            return None
        if isinstance(v, str):
            v_clean = v.strip().lower()
            if not v_clean:
                return None
            if v_clean in ("low", "medium", "high"):
                return v_clean
            raise ValueError(
                f"Invalid LLM_EFFORT '{v}'. Must be one of: 'low', 'medium', 'high', or unset."
            )
        return None

    model_config = SettingsConfigDict(
        env_file=(".env.local", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings instance."""
    return Settings()


settings = get_settings()
