"""Audio transcription engine for Telegram voice memos and audio files using faster-whisper."""

import asyncio
import gc
import logging
from pathlib import Path
import threading
import time
from typing import Any

from src.config import settings
from src.utils.resources import resource_manager

logger = logging.getLogger(__name__)


class AudioTranscriber:
    """Thread-safe, lazy-loading audio transcriber using faster-whisper with automatic CPU fallback and memory unloading."""

    _instance: "AudioTranscriber | None" = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls, model_size: str | None = None) -> "AudioTranscriber":
        """Singleton pattern to ensure only one model instance resides in memory."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, model_size: str | None = None) -> None:
        """Initialize AudioTranscriber with target model size.

        Args:
            model_size: Size of faster-whisper model (e.g., 'base', 'tiny').
        """
        if getattr(self, "_initialized", False):
            return

        with self._lock:
            if getattr(self, "_initialized", False):
                return

            self.model_size = model_size or getattr(settings, "WHISPER_MODEL_SIZE", "base")
            self.fallback_model_size = getattr(settings, "WHISPER_FALLBACK_MODEL_SIZE", "tiny")
            self._model = None
            self._loaded_model_name: str | None = None
            self._model_lock = threading.Lock()
            self._last_used: float = 0.0
            self._initialized = True

    @property
    def is_loaded(self) -> bool:
        """Check if Whisper model is currently resident in RAM."""
        return self._model is not None

    def _update_last_used(self) -> None:
        """Safely update model last-used timestamp under lock."""
        with self._model_lock:
            self._last_used = time.time()

    def _get_model(self):
        """Lazy-load the WhisperModel instance safely with automatic CPU int8 fallback to tiny."""
        if self._model is not None:
            self._update_last_used()
            return self._model

        with self._model_lock:
            if self._model is not None:
                self._last_used = time.time()
                return self._model

            from faster_whisper import WhisperModel

            compute_type = getattr(settings, "WHISPER_COMPUTE_TYPE", "int8")
            device = getattr(settings, "WHISPER_DEVICE", "cpu")

            try:
                logger.info(
                    "Loading faster-whisper model '%s' (device=%s, compute_type=%s)...",
                    self.model_size,
                    device,
                    compute_type,
                )
                self._model = WhisperModel(self.model_size, device=device, compute_type=compute_type)
                self._loaded_model_name = self.model_size
                self._last_used = time.time()
                logger.info("Successfully loaded faster-whisper model '%s'", self.model_size)
                return self._model
            except Exception as primary_err:
                logger.warning(
                    "Failed loading primary Whisper model '%s': %s. Falling back to '%s'...",
                    self.model_size,
                    primary_err,
                    self.fallback_model_size,
                )
                try:
                    self._model = WhisperModel(
                        self.fallback_model_size, device="cpu", compute_type="int8"
                    )
                    self._loaded_model_name = self.fallback_model_size
                    self._last_used = time.time()
                    logger.info("Successfully loaded fallback Whisper model '%s'", self.fallback_model_size)
                    return self._model
                except Exception as fallback_err:
                    logger.exception("Failed loading fallback Whisper model: %s", fallback_err)
                    raise

    def unload_model(self) -> None:
        """Unload Whisper model from RAM and trigger garbage collection."""
        with self._model_lock:
            if self._model is not None:
                logger.info("Unloading Whisper model '%s' from RAM to conserve resources.", self._loaded_model_name)
                self._model = None
                self._loaded_model_name = None
                gc.collect()

    def transcribe(self, audio_path: str | Path) -> str:
        """Transcribe an audio file into text synchronously.

        Args:
            audio_path: Path to the audio file (.ogg, .mp3, .wav, .m4a).

        Returns:
            Transcribed text string.
        """
        path_obj = Path(audio_path).resolve()
        if not path_obj.exists():
            raise FileNotFoundError(f"Audio file not found at: {path_obj}")

        logger.info("Transcribing audio file at %s", path_obj)
        try:
            model = self._get_model()
            segments, info = model.transcribe(str(path_obj), beam_size=5)
            transcription_parts = [segment.text.strip() for segment in segments if segment.text.strip()]
            full_text = " ".join(transcription_parts).strip()
            self._update_last_used()
            logger.info(
                "Transcribed audio (%s, language: %s, duration: %.1fs): '%s'",
                path_obj.name,
                info.language,
                info.duration,
                full_text,
            )
            return full_text
        except Exception as err:
            logger.exception("Error during audio transcription of %s: %s", path_obj, err)
            raise

    async def transcribe_async(self, audio_path: str | Path) -> str:
        """Transcribe an audio file into text asynchronously, guarded by concurrency semaphore."""
        async with resource_manager.whisper_semaphore:
            result = await asyncio.to_thread(self.transcribe, audio_path)
            return result


__all__ = ["AudioTranscriber"]
