import unittest

from alerts import AlertManager, AlertType
from camera_health import (
    CameraHealthSnapshot,
    CameraHealthState,
    CameraHealthWatchdog,
    ExponentialBackoff,
)


class CameraHealthTests(unittest.TestCase):
    def test_health_state_tracks_disconnect_duration_and_recovery(self):
        state = CameraHealthState(source_name="0", display_name="Webcam 0")

        state.mark_connected(now=10.0)
        state.mark_disconnected("read failed", now=15.0)

        self.assertFalse(state.connected)
        self.assertEqual(state.disconnected_for(now=18.5), 3.5)
        self.assertEqual(state.last_error, "read failed")

        state.mark_connected(now=20.0)

        self.assertTrue(state.connected)
        self.assertEqual(state.disconnected_for(now=25.0), 0.0)
        self.assertEqual(state.last_error, "")

    def test_exponential_backoff_caps_and_resets(self):
        backoff = ExponentialBackoff(base_delay_seconds=1.0, max_delay_seconds=8.0)

        self.assertEqual(
            [backoff.next_delay() for _ in range(6)],
            [1.0, 2.0, 4.0, 8.0, 8.0, 8.0],
        )

        backoff.reset()

        self.assertEqual(backoff.next_delay(), 1.0)

    def test_watchdog_emits_long_disconnect_alert_from_stream_snapshot(self):
        events = []
        alert_manager = AlertManager(
            camera_disconnect_timeout_seconds=5.0,
            alert_repeat_cooldown_seconds=60.0,
            event_sink=events.append,
        )
        snapshot = CameraHealthSnapshot(
            source_name="rtsp://gate",
            display_name="Gate",
            connected=False,
            disconnected_for_seconds=6.0,
            reconnect_count=4,
            last_error="read failed",
        )

        class FakeStream:
            def health_snapshot(self, now):
                return snapshot

        watchdog = CameraHealthWatchdog(
            streams_provider=lambda: [FakeStream()],
            alert_manager=alert_manager,
            check_interval_seconds=0.01,
            time_fn=lambda: 100.0,
        )

        watchdog.sample_once()

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].type, AlertType.CAMERA_DISCONNECTED_TOO_LONG)
        self.assertEqual(events[0].source_name, "Gate")


if __name__ == "__main__":
    unittest.main()
