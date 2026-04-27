"""
Asynchronous MQTT alert publisher.

paho-mqtt is optional. If it is not installed, alerts are dropped with a log
message instead of blocking or crashing the GUI.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from typing import Any

logger = logging.getLogger(__name__)


class AsyncMQTTPublisher:
    def __init__(
        self,
        host: str = "",
        port: int = 1883,
        topic: str = "camera_ai/alerts",
        client_id: str = "camera-ai-edge",
        enabled: bool = False,
    ):
        self.host = host
        self.port = port
        self.topic = topic
        self.client_id = client_id
        self.enabled = enabled and bool(host)
        self._queue: "queue.SimpleQueue[dict[str, Any] | None]" = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        self._client = None

    def start(self) -> None:
        if not self.enabled or self._thread:
            return
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._queue.put(None)

    def publish_alert(self, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self.start()
        self._queue.put(payload)

    def _worker(self) -> None:
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            logger.warning("paho-mqtt is not installed; MQTT publishing disabled.")
            self.enabled = False
            return

        try:
            self._client = mqtt.Client(client_id=self.client_id)
            self._client.connect(self.host, self.port, keepalive=30)
            self._client.loop_start()
        except Exception as exc:
            logger.error("MQTT connect failed: %s", exc)
            self.enabled = False
            return

        while True:
            item = self._queue.get()
            if item is None:
                break
            try:
                self._client.publish(
                    self.topic,
                    json.dumps(item, ensure_ascii=False, separators=(",", ":")),
                    qos=0,
                    retain=False,
                )
            except Exception as exc:
                logger.error("MQTT publish failed: %s", exc)

        try:
            self._client.loop_stop()
            self._client.disconnect()
        except Exception:
            pass
