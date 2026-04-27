import os
import sqlite3
import tempfile
import unittest
from contextlib import closing

import numpy as np

from detector import Detection
from event_store import EventStore


class EventStoreTests(unittest.TestCase):
    def test_log_event_writes_sqlite_row_and_cropped_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = EventStore(
                db_path=os.path.join(tmp, "events.db"),
                snapshot_dir=os.path.join(tmp, "snapshots"),
            )
            frame = np.full((100, 140, 3), 255, dtype=np.uint8)
            detections = [Detection("fire", 0.93, 20, 10, 70, 60)]

            record = store.log_event(
                event_type="fire_confirmed",
                source_name="cam-1",
                frame=frame,
                detections=detections,
                metrics={"fps": 24.5, "latency_ms": 12.2},
            )

            self.assertTrue(os.path.exists(record.snapshot_path))
            self.assertGreater(os.path.getsize(record.snapshot_path), 0)

            with closing(sqlite3.connect(os.path.join(tmp, "events.db"))) as conn:
                rows = conn.execute(
                    "SELECT event_type, source_name, detection_count, snapshot_path FROM events"
                ).fetchall()

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][0], "fire_confirmed")
            self.assertEqual(rows[0][1], "cam-1")
            self.assertEqual(rows[0][2], 1)
            self.assertEqual(rows[0][3], record.snapshot_path)


if __name__ == "__main__":
    unittest.main()
