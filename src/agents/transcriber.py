"""Audio transcription engine for Telegram voice memos and audio files using faster-whisper."""

import asyncio
import logging
from pathlib import Path
import threading
from src.config import settings

logger = logging.getLogger(__name__)


class AudioTranscriber:
    """Thread-safe, lazy-loading audio transcriber using faster-whisper."""

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
            model_size: Size of faster-whisper model (e.g., 'tiny', 'base', 'small').
        """
        if getattr(self, "_initialized", False):
            return

        with self._lock:
            if getattr(self, "_initialized", False):
                return

            self.model_size = model_size or getattr(settings, "WHISPER_MODEL_SIZE", "base")
            self._model = None
            self._model_lock = threading.Lock()
            self._initialized = True

    def _get_model(self):
        """Lazy-load the WhisperModel instance safely."""
        if self._model is not None:
            return self._model

        with self._model_lock:
            if self._model is not None:
                return self._model

            try:
                from faster_whisper import WhisperModel

                logger.info("Loading faster-whisper model '%s' (CPU / int8)...", self.model_size)
                # Use CPU and int8 quantization for lightweight local performance
                self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
                logger.info("Successfully loaded faster-whisper model '%s'", self.model_size)
                return self._model
            except Exception as err:
                logger.exception("Failed to load faster-whisper model '%s': %s", self.model_size, err)
                raise

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
            logger.info("Transcribed audio (%s, language: %s, duration: %.1fs): '%s'",
                        path_obj.name, info.language, info.duration, full_text)
            return full_text
        except Exception as err:
            logger.exception("Error during audio transcription of %s: %s", path_obj, err)
            raise

    async def transcribe_async(self, audio_path: str | Path) -> str:
        """Transcribe an audio file into text asynchronously."""
        return await asyncio.to_thread(self.transcribe, audio_path)


__all__ = ["AudioTranscriber"]
