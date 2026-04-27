"""
Lightweight asynchronous hardware resource monitoring.

The monitor is designed for real-time vision apps:
- no per-frame polling
- no blocking psutil CPU intervals
- optional NVIDIA/NVML support with graceful fallback
- latest-value queue semantics so UI consumers never process stale samples
"""

from __future__ import annotations

import importlib
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

BYTES_PER_MB = 1024 * 1024


@dataclass(frozen=True)
class SystemMetrics:
    cpu_percent: float
    ram_total_mb: float
    ram_used_mb: float
    ram_percent: float
    error: str = ""


@dataclass(frozen=True)
class GpuMetrics:
    available: bool
    name: str = ""
    utilization_percent: float = 0.0
    vram_total_mb: float = 0.0
    vram_used_mb: float = 0.0
    vram_percent: float = 0.0
    temperature_c: float = 0.0
    error: str = ""


@dataclass(frozen=True)
class ResourceMetrics:
    timestamp: float
    system: SystemMetrics
    gpu: GpuMetrics


def put_latest(output_queue: "queue.Queue[Any]", item: Any) -> None:
    """Put item into a queue, replacing one stale item if the queue is full."""
    try:
        output_queue.put_nowait(item)
        return
    except queue.Full:
        pass

    try:
        output_queue.get_nowait()
    except queue.Empty:
        pass

    try:
        output_queue.put_nowait(item)
    except queue.Full:
        pass


class ResourceMetricsCollector:
    """Collect CPU/RAM and optional NVIDIA GPU metrics without blocking."""

    def __init__(
        self,
        psutil_module: Optional[Any] = None,
        pynvml_module: Optional[Any] = None,
        gpu_index: int = 0,
        time_fn=time.time,
    ):
        self._psutil = psutil_module if psutil_module is not None else _optional_import("psutil")
        self._pynvml = pynvml_module if pynvml_module is not None else _optional_import("pynvml")
        self._gpu_index = max(0, int(gpu_index))
        self._time_fn = time_fn
        self._gpu_checked = False
        self._gpu_handle = None
        self._gpu_name = ""
        self._gpu_error = ""

        if self._psutil is not None:
            try:
                self._psutil.cpu_percent(interval=None)
            except Exception:
                pass

    def collect(self) -> ResourceMetrics:
        return ResourceMetrics(
            timestamp=float(self._time_fn()),
            system=self._collect_system(),
            gpu=self._collect_gpu(),
        )

    def _collect_system(self) -> SystemMetrics:
        if self._psutil is None:
            return SystemMetrics(0.0, 0.0, 0.0, 0.0, "psutil is not installed")

        try:
            cpu_percent = float(self._psutil.cpu_percent(interval=None))
            memory = self._psutil.virtual_memory()
            return SystemMetrics(
                cpu_percent=cpu_percent,
                ram_total_mb=float(memory.total) / BYTES_PER_MB,
                ram_used_mb=float(memory.used) / BYTES_PER_MB,
                ram_percent=float(memory.percent),
            )
        except Exception as exc:
            return SystemMetrics(0.0, 0.0, 0.0, 0.0, str(exc))

    def _collect_gpu(self) -> GpuMetrics:
        self._ensure_gpu_handle()
        if self._gpu_handle is None:
            return GpuMetrics(False, error=self._gpu_error)

        try:
            util = self._pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
            memory = self._pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
            temperature = self._pynvml.nvmlDeviceGetTemperature(
                self._gpu_handle,
                self._pynvml.NVML_TEMPERATURE_GPU,
            )
            total_mb = float(memory.total) / BYTES_PER_MB
            used_mb = float(memory.used) / BYTES_PER_MB
            vram_percent = (used_mb / total_mb * 100.0) if total_mb > 0 else 0.0
            return GpuMetrics(
                available=True,
                name=self._gpu_name,
                utilization_percent=float(util.gpu),
                vram_total_mb=total_mb,
                vram_used_mb=used_mb,
                vram_percent=vram_percent,
                temperature_c=float(temperature),
            )
        except Exception as exc:
            return GpuMetrics(False, name=self._gpu_name, error=str(exc))

    def _ensure_gpu_handle(self) -> None:
        if self._gpu_checked:
            return
        self._gpu_checked = True

        if self._pynvml is None:
            self._gpu_error = "pynvml is not installed"
            return

        try:
            self._pynvml.nvmlInit()
            self._gpu_handle = self._pynvml.nvmlDeviceGetHandleByIndex(self._gpu_index)
            raw_name = self._pynvml.nvmlDeviceGetName(self._gpu_handle)
            self._gpu_name = raw_name.decode("utf-8", errors="replace") if isinstance(raw_name, bytes) else str(raw_name)
        except Exception as exc:
            self._gpu_handle = None
            self._gpu_error = str(exc)


class SystemMonitorAgent:
    """Background worker that publishes ResourceMetrics every poll_interval seconds."""

    def __init__(
        self,
        output_queue: "queue.Queue[ResourceMetrics]",
        collector: Optional[ResourceMetricsCollector] = None,
        poll_interval: float = 2.0,
        gpu_index: int = 0,
    ):
        self.output_queue = output_queue
        self.collector = collector or ResourceMetricsCollector(gpu_index=gpu_index)
        self.poll_interval = max(0.5, float(poll_interval))
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.thread_ident: Optional[int] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="SystemMonitorAgent",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        self.thread_ident = threading.get_ident()
        while not self._stop_event.is_set():
            try:
                put_latest(self.output_queue, self.collector.collect())
            except Exception as exc:
                logger.warning("Resource monitor sample failed: %s", exc)
            self._stop_event.wait(self.poll_interval)


def _optional_import(module_name: str) -> Optional[Any]:
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None
