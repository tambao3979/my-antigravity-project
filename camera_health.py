from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from alerts import AlertManager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraHealthSnapshot:
    source_name: str
    display_name: str
    connected: bool
    disconnected_for_seconds: float
    reconnect_count: int
    last_error: str = ""


class CameraHealthState:
    def __init__(self, source_name: str, display_name: str):
        self.source_name = str(source_name)
        self.display_name = str(display_name)
        self.connected = False
        self.disconnected_since: Optional[float] = None
        self.last_connected_at: Optional[float] = None
        self.last_error = ""
        self.reconnect_count = 0
        self._lock = threading.Lock()

    def mark_connected(self, now: Optional[float] = None) -> None:
        now = time.time() if now is None else float(now)
        with self._lock:
            self.connected = True
            self.disconnected_since = None
            self.last_connected_at = now
            self.last_error = ""
            self.reconnect_count = 0

    def mark_disconnected(self, reason: str, now: Optional[float] = None) -> None:
        now = time.time() if now is None else float(now)
        with self._lock:
            if self.connected or self.disconnected_since is None:
                self.disconnected_since = now
            self.connected = False
            self.last_error = str(reason)

    def record_reconnect_attempt(self) -> int:
        with self._lock:
            self.reconnect_count += 1
            return self.reconnect_count

    def disconnected_for(self, now: Optional[float] = None) -> float:
        now = time.time() if now is None else float(now)
        with self._lock:
            if self.connected or self.disconnected_since is None:
                return 0.0
            return max(0.0, now - self.disconnected_since)

    def snapshot(self, now: Optional[float] = None) -> CameraHealthSnapshot:
        now = time.time() if now is None else float(now)
        with self._lock:
            disconnected_for_seconds = (
                0.0
                if self.connected or self.disconnected_since is None
                else max(0.0, now - self.disconnected_since)
            )
            return CameraHealthSnapshot(
                source_name=self.source_name,
                display_name=self.display_name,
                connected=self.connected,
                disconnected_for_seconds=disconnected_for_seconds,
                reconnect_count=self.reconnect_count,
                last_error=self.last_error,
            )


class ExponentialBackoff:
    def __init__(self, base_delay_seconds: float = 1.0, max_delay_seconds: float = 30.0):
        self.base_delay_seconds = max(0.01, float(base_delay_seconds))
        self.max_delay_seconds = max(self.base_delay_seconds, float(max_delay_seconds))
        self._current_delay = self.base_delay_seconds

    def next_delay(self) -> float:
        delay = self._current_delay
        self._current_delay = min(self._current_delay * 2.0, self.max_delay_seconds)
        return delay

    def reset(self) -> None:
        self._current_delay = self.base_delay_seconds


class CameraHealthWatchdog:
    def __init__(
        self,
        streams_provider: Callable[[], Iterable[object]],
        alert_manager: AlertManager,
        check_interval_seconds: float = 1.0,
        time_fn: Callable[[], float] = time.time,
    ):
        self.streams_provider = streams_provider
        self.alert_manager = alert_manager
        self.check_interval_seconds = max(0.1, float(check_interval_seconds))
        self.time_fn = time_fn
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="CameraHealthWatchdog",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def sample_once(self) -> None:
        now = float(self.time_fn())
        try:
            streams = list(self.streams_provider())
        except Exception as exc:
            logger.warning("Camera health watchdog could not list streams: %s", exc)
            return

        for stream in streams:
            try:
                snapshot = stream.health_snapshot(now)
            except Exception as exc:
                logger.warning("Camera health watchdog sample failed: %s", exc)
                continue

            if snapshot.connected:
                continue

            self.alert_manager.record_camera_disconnected(
                source_name=snapshot.display_name or snapshot.source_name,
                disconnected_for_seconds=snapshot.disconnected_for_seconds,
                reconnect_count=snapshot.reconnect_count,
                now=now,
            )

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self.sample_once()
            self._stop_event.wait(self.check_interval_seconds)
