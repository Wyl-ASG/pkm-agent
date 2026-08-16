"""Lightweight resource monitoring and concurrency limiter for 2 vCPU / 4 GB RAM servers."""

import asyncio
from dataclasses import dataclass
import logging
import os
import time
from typing import Any

from src.config import settings

logger = logging.getLogger(__name__)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


@dataclass
class SystemResourceMetrics:
    """Snapshot of system memory and CPU utilization."""

    total_ram_mb: float
    used_ram_mb: float
    free_ram_mb: float
    percent_ram: float
    process_rss_mb: float
    cpu_percent: float
    under_memory_pressure: bool


class ResourceMonitor:
    """Lightweight resource inspector monitoring memory and CPU."""

    @staticmethod
    def get_metrics() -> SystemResourceMetrics:
        """Collect current system and process memory / CPU metrics."""
        process_rss_mb = 0.0
        total_ram_mb = 4096.0
        used_ram_mb = 1024.0
        free_ram_mb = 3072.0
        percent_ram = 25.0
        cpu_percent = 0.0

        if HAS_PSUTIL:
            try:
                proc = psutil.Process(os.getpid())
                process_rss_mb = proc.memory_info().rss / (1024 * 1024)
                cpu_percent = psutil.cpu_percent(interval=None)

                vm = psutil.virtual_memory()
                total_ram_mb = vm.total / (1024 * 1024)
                used_ram_mb = vm.used / (1024 * 1024)
                free_ram_mb = vm.available / (1024 * 1024)
                percent_ram = vm.percent
            except Exception as err:
                logger.debug("psutil metrics collection error: %s", err)
        else:
            try:
                import resource
                rusage = resource.getrusage(resource.RUSAGE_SELF)
                # ru_maxrss is in bytes on macOS, kilobytes on Linux
                import sys
                if sys.platform == "darwin":
                    process_rss_mb = rusage.ru_maxrss / (1024 * 1024)
                else:
                    process_rss_mb = rusage.ru_maxrss / 1024
            except Exception:
                pass

        under_pressure = percent_ram >= getattr(settings, "MAX_MEMORY_PRESSURE_PERCENT", 85.0)

        return SystemResourceMetrics(
            total_ram_mb=round(total_ram_mb, 1),
            used_ram_mb=round(used_ram_mb, 1),
            free_ram_mb=round(free_ram_mb, 1),
            percent_ram=round(percent_ram, 1),
            process_rss_mb=round(process_rss_mb, 1),
            cpu_percent=round(cpu_percent, 1),
            under_memory_pressure=under_pressure,
        )

    @classmethod
    def is_under_pressure(cls) -> bool:
        """Check if system is under high memory pressure."""
        return cls.get_metrics().under_memory_pressure


class ResourceManager:
    """Global manager providing async concurrency semaphores and resource tracking."""

    _instance: "ResourceManager | None" = None

    def __init__(self) -> None:
        """Initialize concurrency semaphores."""
        self._embedding_sem: asyncio.Semaphore | None = None
        self._whisper_sem: asyncio.Semaphore | None = None
        self._reranker_sem: asyncio.Semaphore | None = None
        self._background_job_sem: asyncio.Semaphore | None = None
        self._antigravity_sem: asyncio.Semaphore | None = None

        self.active_background_jobs = 0
        self.total_background_jobs_completed = 0
        self._lock = asyncio.Lock()

    @classmethod
    def get_instance(cls) -> "ResourceManager":
        """Get or create singleton ResourceManager instance."""
        if cls._instance is None:
            cls._instance = ResourceManager()
        return cls._instance

    @property
    def embedding_semaphore(self) -> asyncio.Semaphore:
        """Semaphore guarding embedding generation concurrency."""
        if self._embedding_sem is None:
            self._embedding_sem = asyncio.Semaphore(settings.MAX_CONCURRENT_EMBEDDINGS)
        return self._embedding_sem

    @property
    def whisper_semaphore(self) -> asyncio.Semaphore:
        """Semaphore guarding Whisper transcription concurrency."""
        if self._whisper_sem is None:
            self._whisper_sem = asyncio.Semaphore(settings.MAX_CONCURRENT_TRANSCRIPTIONS)
        return self._whisper_sem

    @property
    def reranker_semaphore(self) -> asyncio.Semaphore:
        """Semaphore guarding CrossEncoder reranker concurrency."""
        if self._reranker_sem is None:
            self._reranker_sem = asyncio.Semaphore(settings.MAX_CONCURRENT_RERANKING)
        return self._reranker_sem

    @property
    def background_job_semaphore(self) -> asyncio.Semaphore:
        """Semaphore guarding background indexing and maintenance concurrency."""
        if self._background_job_sem is None:
            self._background_job_sem = asyncio.Semaphore(settings.MAX_BACKGROUND_JOBS)
        return self._background_job_sem

    @property
    def antigravity_semaphore(self) -> asyncio.Semaphore:
        """Semaphore guarding Antigravity CLI subprocess execution concurrency."""
        if self._antigravity_sem is None:
            self._antigravity_sem = asyncio.Semaphore(settings.ANTIGRAVITY_MAX_CONCURRENT)
        return self._antigravity_sem


resource_manager = ResourceManager.get_instance()
resource_monitor = ResourceMonitor()


__all__ = [
    "SystemResourceMetrics",
    "ResourceMonitor",
    "ResourceManager",
    "resource_manager",
    "resource_monitor",
]
