import unittest

from alerts import AlertManager, AlertType


class AlertManagerTests(unittest.TestCase):
    def make_manager(self, events):
        return AlertManager(
            camera_disconnect_timeout_seconds=5.0,
            gpu_usage_threshold_percent=90.0,
            gpu_consecutive_samples=3,
            inference_latency_threshold_ms=100.0,
            inference_consecutive_samples=2,
            alert_repeat_cooldown_seconds=60.0,
            event_sink=events.append,
        )

    def test_camera_disconnect_alert_fires_after_timeout_and_respects_cooldown(self):
        events = []
        manager = self.make_manager(events)

        self.assertIsNone(
            manager.record_camera_disconnected(
                source_name="Gate 1",
                disconnected_for_seconds=4.9,
                reconnect_count=2,
                now=10.0,
            )
        )

        event = manager.record_camera_disconnected(
            source_name="Gate 1",
            disconnected_for_seconds=5.1,
            reconnect_count=3,
            now=11.0,
        )

        self.assertIsNotNone(event)
        self.assertEqual(event.type, AlertType.CAMERA_DISCONNECTED_TOO_LONG)
        self.assertEqual(event.source_name, "Gate 1")
        self.assertEqual(len(events), 1)

        duplicate = manager.record_camera_disconnected(
            source_name="Gate 1",
            disconnected_for_seconds=20.0,
            reconnect_count=8,
            now=50.0,
        )
        self.assertIsNone(duplicate)
        self.assertEqual(len(events), 1)

        repeated = manager.record_camera_disconnected(
            source_name="Gate 1",
            disconnected_for_seconds=70.0,
            reconnect_count=12,
            now=72.0,
        )
        self.assertIsNotNone(repeated)
        self.assertEqual(len(events), 2)

    def test_gpu_alert_requires_consecutive_high_samples_and_resets_on_normal_sample(self):
        events = []
        manager = self.make_manager(events)

        self.assertIsNone(manager.record_gpu_sample(91.0, now=1.0))
        self.assertIsNone(manager.record_gpu_sample(89.0, now=2.0))
        self.assertIsNone(manager.record_gpu_sample(92.0, now=3.0))
        self.assertIsNone(manager.record_gpu_sample(93.0, now=4.0))

        event = manager.record_gpu_sample(94.0, now=5.0)

        self.assertIsNotNone(event)
        self.assertEqual(event.type, AlertType.GPU_USAGE_HIGH)
        self.assertIn("94.0%", event.message)
        self.assertEqual(len(events), 1)

    def test_inference_alert_is_tracked_per_source(self):
        events = []
        manager = self.make_manager(events)

        self.assertIsNone(manager.record_inference_latency("Gate 1", 120.0, now=1.0))
        self.assertIsNone(manager.record_inference_latency("Gate 2", 130.0, now=2.0))

        event = manager.record_inference_latency("Gate 1", 140.0, now=3.0)

        self.assertIsNotNone(event)
        self.assertEqual(event.type, AlertType.INFERENCE_SLOW)
        self.assertEqual(event.source_name, "Gate 1")
        self.assertEqual(len(events), 1)

        self.assertIsNone(manager.record_inference_latency("Gate 2", 80.0, now=4.0))
        self.assertIsNone(manager.record_inference_latency("Gate 2", 150.0, now=5.0))
        second = manager.record_inference_latency("Gate 2", 160.0, now=6.0)

        self.assertIsNotNone(second)
        self.assertEqual(second.source_name, "Gate 2")
        self.assertEqual(len(events), 2)


if __name__ == "__main__":
    unittest.main()
