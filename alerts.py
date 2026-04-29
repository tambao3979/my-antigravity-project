from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class AlertType(str, Enum):
    CAMERA_DISCONNECTED_TOO_LONG = "camera_disconnected_too_long"
    GPU_USAGE_HIGH = "gpu_usage_high"
    INFERENCE_SLOW = "inference_slow"


@dataclass(frozen=True)
class AlertEvent:
    type: AlertType
    message: str
    timestamp: float
    source_name: str = ""
    value: float = 0.0
    threshold: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class AlertManager:
    def __init__(
        self,
        camera_disconnect_timeout_seconds: float = 30.0,
        gpu_usage_threshold_percent: float = 90.0,
        gpu_consecutive_samples: int = 3,
        inference_latency_threshold_ms: float = 250.0,
        inference_consecutive_samples: int = 3,
        alert_repeat_cooldown_seconds: float = 60.0,
        event_sink: Optional[Callable[[AlertEvent], None]] = None,
        logger_obj: Optional[logging.Logger] = None,
    ):
        self.camera_disconnect_timeout_seconds = max(0.0, float(camera_disconnect_timeout_seconds))
        self.gpu_usage_threshold_percent = float(gpu_usage_threshold_percent)
        self.gpu_consecutive_samples = max(1, int(gpu_consecutive_samples))
        self.inference_latency_threshold_ms = float(inference_latency_threshold_ms)
        self.inference_consecutive_samples = max(1, int(inference_consecutive_samples))
        self.alert_repeat_cooldown_seconds = max(0.0, float(alert_repeat_cooldown_seconds))
        self.event_sink = event_sink
        self.logger = logger_obj or logger

        self._lock = threading.Lock()
        self._last_alert_at: dict[tuple[AlertType, str], float] = {}
        self._gpu_high_count = 0
        self._inference_slow_counts: dict[str, int] = {}

    @classmethod
    def from_config(cls, config, event_sink: Optional[Callable[[AlertEvent], None]] = None) -> "AlertManager":
        return cls(
            camera_disconnect_timeout_seconds=getattr(config, "CAMERA_DISCONNECT_TIMEOUT_SECONDS", 30.0),
            gpu_usage_threshold_percent=getattr(config, "GPU_ALERT_USAGE_THRESHOLD_PERCENT", 90.0),
            gpu_consecutive_samples=getattr(config, "GPU_ALERT_CONSECUTIVE_SAMPLES", 3),
            inference_latency_threshold_ms=getattr(config, "INFERENCE_LATENCY_ALERT_THRESHOLD_MS", 250.0),
            inference_consecutive_samples=getattr(config, "INFERENCE_LATENCY_ALERT_CONSECUTIVE_SAMPLES", 3),
            alert_repeat_cooldown_seconds=getattr(config, "SYSTEM_ALERT_REPEAT_COOLDOWN_SECONDS", 60.0),
            event_sink=event_sink,
        )

    def record_camera_disconnected(
        self,
        source_name: str,
        disconnected_for_seconds: float,
        reconnect_count: int,
        now: Optional[float] = None,
    ) -> Optional[AlertEvent]:
        now = time.time() if now is None else float(now)
        disconnected_for_seconds = float(disconnected_for_seconds)
        if disconnected_for_seconds < self.camera_disconnect_timeout_seconds:
            return None

        message = (
            f"Camera '{source_name}' disconnected for {disconnected_for_seconds:.1f}s "
            f"(reconnect attempts: {int(reconnect_count)})."
        )
        return self._emit(
            AlertType.CAMERA_DISCONNECTED_TOO_LONG,
            key=str(source_name),
            message=message,
            timestamp=now,
            source_name=str(source_name),
            value=disconnected_for_seconds,
            threshold=self.camera_disconnect_timeout_seconds,
            metadata={"reconnect_count": int(reconnect_count)},
        )

    def record_gpu_sample(self, utilization_percent: float, now: Optional[float] = None) -> Optional[AlertEvent]:
        now = time.time() if now is None else float(now)
        utilization_percent = float(utilization_percent)
        with self._lock:
            if utilization_percent < self.gpu_usage_threshold_percent:
                self._gpu_high_count = 0
                return None
            self._gpu_high_count += 1
            if self._gpu_high_count < self.gpu_consecutive_samples:
                return None

        message = (
            f"GPU usage is high: {utilization_percent:.1f}% "
            f"(threshold {self.gpu_usage_threshold_percent:.1f}%)."
        )
        return self._emit(
            AlertType.GPU_USAGE_HIGH,
            key="gpu",
            message=message,
            timestamp=now,
            value=utilization_percent,
            threshold=self.gpu_usage_threshold_percent,
            metadata={"consecutive_samples": self._gpu_high_count},
        )

    def record_inference_latency(
        self,
        source_name: str,
        latency_ms: float,
        now: Optional[float] = None,
    ) -> Optional[AlertEvent]:
        now = time.time() if now is None else float(now)
        source_name = str(source_name)
        latency_ms = float(latency_ms)
        with self._lock:
            if latency_ms < self.inference_latency_threshold_ms:
                self._inference_slow_counts[source_name] = 0
                return None
            count = self._inference_slow_counts.get(source_name, 0) + 1
            self._inference_slow_counts[source_name] = count
            if count < self.inference_consecutive_samples:
                return None

        message = (
            f"Inference for '{source_name}' is slow: {latency_ms:.1f}ms "
            f"(threshold {self.inference_latency_threshold_ms:.1f}ms)."
        )
        return self._emit(
            AlertType.INFERENCE_SLOW,
            key=source_name,
            message=message,
            timestamp=now,
            source_name=source_name,
            value=latency_ms,
            threshold=self.inference_latency_threshold_ms,
            metadata={"consecutive_samples": self._inference_slow_counts[source_name]},
        )

    def _emit(
        self,
        alert_type: AlertType,
        key: str,
        message: str,
        timestamp: float,
        source_name: str = "",
        value: float = 0.0,
        threshold: float = 0.0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[AlertEvent]:
        alert_key = (alert_type, key)
        with self._lock:
            last_at = self._last_alert_at.get(alert_key)
            if last_at is not None and timestamp - last_at < self.alert_repeat_cooldown_seconds:
                return None
            self._last_alert_at[alert_key] = timestamp

        event = AlertEvent(
            type=alert_type,
            message=message,
            timestamp=timestamp,
            source_name=source_name,
            value=value,
            threshold=threshold,
            metadata=metadata or {},
        )
        self.logger.warning("ALERT [%s] %s", alert_type.value, message)
        if self.event_sink is not None:
            try:
                self.event_sink(event)
            except Exception:
                self.logger.exception("Alert event sink failed")
        return event
