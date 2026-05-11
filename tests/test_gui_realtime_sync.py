import unittest

import numpy as np

from detector import Detection
from config import Config
from gui import CameraStream, RECORD_FPS, SecurityApp, dedupe_overlay_detections


class _FakeDetector:
    def __init__(self):
        self.batch_sizes = []

    def detect_batch(self, frames, allowed_class_ids=None):
        self.batch_sizes.append(len(frames))
        return [[] for _ in frames]


class _FakeAlertManager:
    def record_inference_latency(self, display_name, latency_ms, now=None):
        pass


class _FailingOnceInferenceApp:
    def __init__(self):
        self.is_inferencing = True
        self.calls = 0

    def _run_inference_once(self, last_processed_frames):
        self.calls += 1
        self.is_inferencing = False
        raise ValueError("model backend exploded")


class _RaisingCapture:
    def read(self):
        raise RuntimeError("driver read crashed")


class _EmptyFrameCapture:
    def read(self):
        return True, None


class CameraStreamRealtimeSyncTests(unittest.TestCase):
    def test_render_result_uses_newest_frame_with_latest_inference_detections(self):
        stream = CameraStream("0")
        first_frame = np.full((4, 4, 3), 10, dtype=np.uint8)
        newer_frame = np.full((4, 4, 3), 20, dtype=np.uint8)

        first_frame_id = stream.publish_frame(first_frame)
        snapshot = stream.snapshot_for_inference(last_processed_frame_id=None)

        newer_frame_id = stream.publish_frame(newer_frame)
        first_frame[:, :, :] = 99
        detections = [Detection("person", 0.91, 0, 0, 2, 2)]
        stream.publish_detections(snapshot, detections, latency_ms=12.5, completed_at=123.0)

        render_result = stream.get_render_result()

        self.assertEqual(render_result.frame_id, newer_frame_id)
        self.assertEqual(render_result.detection_frame_id, first_frame_id)
        self.assertEqual(render_result.detections, detections)
        self.assertEqual(int(render_result.frame[0, 0, 0]), 20)

    def test_snapshot_for_inference_skips_frames_that_were_already_processed(self):
        stream = CameraStream("0")
        stream.publish_frame(np.zeros((2, 2, 3), dtype=np.uint8))

        snapshot = stream.snapshot_for_inference(last_processed_frame_id=None)

        self.assertIsNone(stream.snapshot_for_inference(last_processed_frame_id=snapshot.frame_id))

    def test_render_result_uses_latest_frame_without_stale_detections_before_first_inference(self):
        stream = CameraStream("0")
        frame_id = stream.publish_frame(np.full((3, 3, 3), 7, dtype=np.uint8))

        render_result = stream.get_render_result()

        self.assertEqual(render_result.frame_id, frame_id)
        self.assertEqual(render_result.detections, [])
        self.assertEqual(int(render_result.frame[0, 0, 0]), 7)

    def test_recording_prebuffer_is_sampled_at_recording_fps(self):
        stream = CameraStream("0")
        interval = 1.0 / RECORD_FPS

        self.assertTrue(stream.should_buffer_record_frame(10.0))
        self.assertFalse(stream.should_buffer_record_frame(10.0 + interval * 0.5))
        self.assertTrue(stream.should_buffer_record_frame(10.0 + interval))

    def test_output_recording_is_sampled_at_recording_fps(self):
        stream = CameraStream("0")
        interval = 1.0 / RECORD_FPS

        self.assertTrue(stream.should_record_output_frame(20.0))
        self.assertFalse(stream.should_record_output_frame(20.0 + interval * 0.5))
        self.assertTrue(stream.should_record_output_frame(20.0 + interval))

    def test_unknown_class_visibility_defaults_true_without_allocating_tk_variable(self):
        dummy_app = type("DummyApp", (), {"class_visibility": {}})()

        self.assertTrue(SecurityApp._is_class_visible(dummy_app, "new-object"))

    def test_overlay_dedup_uses_camera_iou_to_make_nms_slider_visible(self):
        detections = [
            Detection("fire", 0.91, 100, 100, 220, 220),
            Detection("fire", 0.82, 125, 125, 245, 245),
            Detection("fire", 0.77, 500, 100, 620, 220),
        ]

        low_iou = dedupe_overlay_detections(detections, {"iou": 0.30})
        high_iou = dedupe_overlay_detections(detections, {"iou": 0.90})

        self.assertEqual(len(low_iou), 2)
        self.assertEqual(len(high_iou), 3)

    def test_overlay_dedup_keeps_nested_boxes_at_highest_iou(self):
        detections = [
            Detection("fire", 0.91, 100, 100, 300, 300),
            Detection("fire", 0.82, 130, 130, 230, 230),
        ]

        low_iou = dedupe_overlay_detections(detections, {"iou": 0.10})
        highest_iou = dedupe_overlay_detections(detections, {"iou": 0.95})

        self.assertEqual(len(low_iou), 1)
        self.assertEqual(len(highest_iou), 2)

    def test_inference_once_uses_realtime_micro_batches(self):
        original_batch_size = getattr(Config, "INFERENCE_BATCH_SIZE", None)
        Config.INFERENCE_BATCH_SIZE = 1
        try:
            streams = {}
            for index in range(3):
                stream = CameraStream(str(index))
                stream.is_running = True
                stream.publish_frame(np.full((2, 2, 3), index, dtype=np.uint8))
                streams[str(index)] = stream

            app = object.__new__(SecurityApp)
            app.streams = streams
            app.detector = _FakeDetector()
            app.alert_manager = _FakeAlertManager()

            processed = SecurityApp._run_inference_once(app, {})

            self.assertTrue(processed)
            self.assertEqual(app.detector.batch_sizes, [1, 1, 1])
        finally:
            if original_batch_size is None:
                delattr(Config, "INFERENCE_BATCH_SIZE")
            else:
                Config.INFERENCE_BATCH_SIZE = original_batch_size

    def test_inference_loop_survives_non_runtime_detector_exception(self):
        app = _FailingOnceInferenceApp()

        SecurityApp._multi_inference_loop(app)

        self.assertEqual(app.calls, 1)

    def test_capture_loop_reconnects_when_camera_read_raises(self):
        stream = CameraStream("rtsp://gate")
        stream.is_running = True
        stream.is_video_file = False
        stream.cap = _RaisingCapture()
        reconnect_reasons = []

        def reconnect(reason):
            reconnect_reasons.append(reason)
            stream.is_running = False

        stream._attempt_reconnect = reconnect

        stream._capture_loop()

        self.assertEqual(reconnect_reasons, ["Camera read raised: driver read crashed"])

    def test_capture_loop_reconnects_when_camera_returns_empty_frame(self):
        stream = CameraStream("rtsp://gate")
        stream.is_running = True
        stream.is_video_file = False
        stream.cap = _EmptyFrameCapture()
        reconnect_reasons = []

        def reconnect(reason):
            reconnect_reasons.append(reason)
            stream.is_running = False

        stream._attempt_reconnect = reconnect

        stream._capture_loop()

        self.assertEqual(reconnect_reasons, ["Camera returned an empty frame"])


if __name__ == "__main__":
    unittest.main()
