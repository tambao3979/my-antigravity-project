"""
SQLite-backed event logging and anomaly snapshotting.

Disk work is isolated here so the GUI can call it from a background thread.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class EventRecord:
    event_id: int
    timestamp: str
    event_type: str
    source_name: str
    detection_count: int
    snapshot_path: str


class EventStore:
    """Persist critical events and cropped anomaly images locally."""

    def __init__(self, db_path: str = "events/events.db", snapshot_dir: str = "events/snapshots"):
        self.db_path = db_path
        self.snapshot_dir = snapshot_dir
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        os.makedirs(snapshot_dir, exist_ok=True)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    detection_count INTEGER NOT NULL,
                    snapshot_path TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)"
            )
            conn.commit()

    def log_event(
        self,
        event_type: str,
        source_name: str,
        frame: np.ndarray,
        detections: Sequence[Any],
        metrics: dict[str, Any] | None = None,
    ) -> EventRecord:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        snapshot_path = self._write_snapshot(timestamp, event_type, source_name, frame, detections)
        metadata = {
            "metrics": metrics or {},
            "detections": [self._detection_to_dict(det) for det in detections],
        }

        with self._lock, closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.execute(
                """
                INSERT INTO events (
                    timestamp, event_type, source_name, detection_count,
                    snapshot_path, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    event_type,
                    source_name,
                    len(detections),
                    snapshot_path,
                    json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
                ),
            )
            event_id = int(cursor.lastrowid)
            cursor.close()
            conn.commit()

        return EventRecord(
            event_id=event_id,
            timestamp=timestamp,
            event_type=event_type,
            source_name=source_name,
            detection_count=len(detections),
            snapshot_path=snapshot_path,
        )

    def recent_events(self, limit: int = 200) -> list[dict[str, Any]]:
        try:
            safe_limit = int(limit)
        except (TypeError, ValueError):
            safe_limit = 200
        if safe_limit <= 0:
            return []
        safe_limit = min(safe_limit, 1000)

        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT id, timestamp, event_type, source_name, detection_count,
                       snapshot_path, metadata_json
                FROM events
                ORDER BY id DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()

        return [dict(row) for row in rows]

    def _write_snapshot(
        self,
        timestamp: str,
        event_type: str,
        source_name: str,
        frame: np.ndarray,
        detections: Sequence[Any],
    ) -> str:
        crop = crop_anomaly(frame, detections)
        stamp = timestamp.replace(":", "").replace("-", "").replace("+", "_")
        safe_source = "".join(ch if ch.isalnum() else "_" for ch in source_name).strip("_")
        filename = f"{event_type}_{safe_source or 'source'}_{stamp}.jpg"
        path = os.path.join(self.snapshot_dir, filename)
        if crop is None or getattr(crop, "size", 0) == 0:
            raise RuntimeError("Cannot write event snapshot from an empty frame")
        if not cv2.imwrite(path, crop, [cv2.IMWRITE_JPEG_QUALITY, 85]):
            raise RuntimeError(f"Failed to write event snapshot: {path}")
        return path

    def _detection_to_dict(self, detection: Any) -> dict[str, Any]:
        return {
            "class_name": getattr(detection, "class_name", "unknown"),
            "confidence": float(getattr(detection, "confidence", 0.0)),
            "box": [
                int(getattr(detection, "x1", 0)),
                int(getattr(detection, "y1", 0)),
                int(getattr(detection, "x2", 0)),
                int(getattr(detection, "y2", 0)),
            ],
        }


def crop_anomaly(frame: np.ndarray, detections: Sequence[Any], padding: int = 24) -> np.ndarray:
    """Crop around all detections, falling back to the full frame."""
    if frame is None or frame.size == 0 or not detections:
        return frame

    h, w = frame.shape[:2]
    x1 = max(0, min(int(getattr(det, "x1", 0)) for det in detections) - padding)
    y1 = max(0, min(int(getattr(det, "y1", 0)) for det in detections) - padding)
    x2 = min(w, max(int(getattr(det, "x2", w)) for det in detections) + padding)
    y2 = min(h, max(int(getattr(det, "y2", h)) for det in detections) + padding)

    if x2 <= x1 or y2 <= y1:
        return frame
    return frame[y1:y2, x1:x2]
