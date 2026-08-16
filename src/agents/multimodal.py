"""Extensible Multimodal Ingestion Pipeline for text, voice, images, documents, URLs, and forwards."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
from pathlib import Path
from typing import Any, Callable, Coroutine
import httpx

from src.agents.transcriber import AudioTranscriber

logger = logging.getLogger(__name__)


class ModalityType(str, Enum):
    """Supported content modalities for knowledge ingestion."""

    TEXT = "text"
    VOICE = "voice"
    IMAGE = "image"
    DOCUMENT = "document"
    URL = "url"
    FORWARDED = "forwarded"


@dataclass
class IngestionItem:
    """Standardized representation of incoming capture content."""

    modality: ModalityType
    content: str
    file_path: Path | None = None
    caption: str | None = None
    sender_id: int | None = None
    sender_name: str | None = None
    source_url: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M"))
    metadata: dict[str, Any] = field(default_factory=dict)


class MultimodalIngestionPipeline:
    """Extensible ingestion processor that routes different media types into standard text for the parser agent."""

    def __init__(
        self,
        audio_transcriber: AudioTranscriber | None = None,
    ) -> None:
        """Initialize MultimodalIngestionPipeline."""
        self.audio_transcriber = audio_transcriber or AudioTranscriber()
        self._handlers: dict[ModalityType, Callable[[IngestionItem], Coroutine[Any, Any, str]]] = {
            ModalityType.TEXT: self._process_text,
            ModalityType.VOICE: self._process_voice,
            ModalityType.IMAGE: self._process_image,
            ModalityType.DOCUMENT: self._process_document,
            ModalityType.URL: self._process_url,
            ModalityType.FORWARDED: self._process_forwarded,
        }

    def register_handler(
        self,
        modality: ModalityType,
        handler: Callable[[IngestionItem], Coroutine[Any, Any, str]],
    ) -> None:
        """Register a custom handler for a modality type."""
        self._handlers[modality] = handler

    async def process(self, item: IngestionItem) -> str:
        """Process incoming IngestionItem and return canonical Markdown text ready for vault parsing."""
        handler = self._handlers.get(item.modality, self._process_text)
        try:
            return await handler(item)
        except Exception as err:
            logger.exception("Error processing modality %s: %s", item.modality, err)
            raise

    async def _process_text(self, item: IngestionItem) -> str:
        """Process plain text input."""
        return item.content.strip()

    async def _process_voice(self, item: IngestionItem) -> str:
        """Transcribe voice memo and combine with caption."""
        if not item.file_path or not item.file_path.exists():
            raise FileNotFoundError("Audio file path missing or not found.")

        transcription = await self.audio_transcriber.transcribe_async(item.file_path)
        full_text = transcription.strip()
        if item.caption:
            full_text = f"{item.caption.strip()} - {full_text}"
        return full_text

    async def _process_image(self, item: IngestionItem) -> str:
        """Process image memo with caption or metadata."""
        caption = item.caption.strip() if item.caption else "Visual capture"
        filename = item.file_path.name if item.file_path else "attachment.png"
        return f"🖼️ {caption} [Attachment: ![[{filename}]]]"

    async def _process_document(self, item: IngestionItem) -> str:
        """Process text/markdown/pdf document file."""
        if item.file_path and item.file_path.exists():
            if item.file_path.suffix.lower() in (".md", ".txt"):
                return await asyncio.to_thread(item.file_path.read_text, encoding="utf-8")
        return item.content or (item.caption or "Document capture")

    async def _process_url(self, item: IngestionItem) -> str:
        """Process URL bookmark with title."""
        url = item.source_url or item.content.strip()
        caption = item.caption or "Web resource"
        return f"🔗 [{caption}]({url})"

    async def _process_forwarded(self, item: IngestionItem) -> str:
        """Process forwarded message preserving sender attribution."""
        origin = item.sender_name or f"User {item.sender_id}" if item.sender_id else "Forwarded"
        return f"Forwarded from {origin}:\n{item.content.strip()}"


__all__ = [
    "ModalityType",
    "IngestionItem",
    "MultimodalIngestionPipeline",
]
