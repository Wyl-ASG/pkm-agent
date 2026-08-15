"""Configuration settings for PKM AI Agent using pydantic-settings."""

from functools import lru_cache
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

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
        default="./secrets/pkm_deploy_key",
        description="Path to SSH private key used for Git repository operations.",
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

    # Optional GraphRAG & LLM Configurations
    QDRANT_COLLECTION_NAME: str = Field(
        default="pkm_notes",
        description="Qdrant collection name for PKM notes.",
    )
    EMBEDDING_MODEL_NAME: str = Field(
        default="all-MiniLM-L6-v2",
        description="HuggingFace model identifier for generating text embeddings.",
    )
    AGY_PATH: str | None = Field(
        default=None,
        description="Explicit path to the agy CLI binary (auto-discovered if not set).",
    )
    LLM_MODEL: str | None = Field(
        default=None,
        description="Model identifier passed to agy CLI (e.g., 'flash', 'pro').",
    )
    LLM_EFFORT: str | None = Field(
        default=None,
        description="Reasoning effort level passed to agy CLI ('low', 'medium', 'high').",
    )
    WHISPER_MODEL_SIZE: str = Field(
        default="base",
        description="Model size for faster-whisper audio transcription ('tiny', 'base', 'small', etc.).",
    )
    ENABLE_FILE_WATCHER: bool = Field(
        default=True,
        description="Whether to enable real-time vault file watcher for automated reindexing.",
    )
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
