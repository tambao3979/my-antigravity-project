import unittest

import numpy as np

from config import Config
from detector import (
    Detection,
    ObjectDetector,
    TileRegion,
    build_tile_regions,
    merge_detections,
    remap_tile_detection,
)


class DetectorTilingTests(unittest.TestCase):
    def test_build_tile_regions_covers_full_frame_with_overlap(self):
        regions = build_tile_regions(width=1280, height=720, tile_size=512, overlap=128)

        self.assertEqual(regions[0], TileRegion(0, 0, 512, 512))
        self.assertTrue(any(region.x2 == 1280 for region in regions))
        self.assertTrue(any(region.y2 == 720 for region in regions))

        for y in range(0, 720, 80):
            for x in range(0, 1280, 80):
                self.assertTrue(
                    any(region.x1 <= x < region.x2 and region.y1 <= y < region.y2 for region in regions),
                    f"Point {(x, y)} is not covered by any tile",
                )

    def test_remap_tile_detection_offsets_and_clips_box_to_original_frame(self):
        detection = Detection("fire", 0.91, 10, 20, 140, 160)
        tile = TileRegion(300, 200, 812, 712)

        remapped = remap_tile_detection(
            detection,
            tile,
            frame_width=420,
            frame_height=350,
        )

        self.assertEqual(remapped.class_name, "fire")
        self.assertEqual(remapped.confidence, 0.91)
        self.assertEqual((remapped.x1, remapped.y1, remapped.x2, remapped.y2), (310, 220, 420, 350))

    def test_merge_detections_removes_duplicate_tiles_but_keeps_overlapping_objects(self):
        detections = [
            Detection("fire", 0.70, 100, 100, 200, 200),
            Detection("fire", 0.92, 102, 102, 202, 202),
            Detection("fire", 0.80, 140, 140, 240, 240),
            Detection("smoke", 0.85, 102, 102, 202, 202),
        ]

        merged = merge_detections(detections, iou_threshold=0.85)

        self.assertEqual(len(merged), 3)
        self.assertIn(("fire", 0.92, 102, 102, 202, 202), self._as_tuples(merged))
        self.assertIn(("fire", 0.80, 140, 140, 240, 240), self._as_tuples(merged))
        self.assertIn(("smoke", 0.85, 102, 102, 202, 202), self._as_tuples(merged))

    def test_detect_batch_runs_tiled_hazard_pass_and_remaps_results(self):
        old_values = {
            "TILED_DETECTION_ENABLED": Config.TILED_DETECTION_ENABLED,
            "TILED_DETECTION_TILE_SIZE": Config.TILED_DETECTION_TILE_SIZE,
            "TILED_DETECTION_OVERLAP": Config.TILED_DETECTION_OVERLAP,
            "TILED_DETECTION_BATCH_SIZE": Config.TILED_DETECTION_BATCH_SIZE,
            "TILED_DETECTION_MERGE_IOU": Config.TILED_DETECTION_MERGE_IOU,
            "YOLO_IMGSZ": Config.YOLO_IMGSZ,
        }
        try:
            Config.TILED_DETECTION_ENABLED = True
            Config.TILED_DETECTION_TILE_SIZE = 512
            Config.TILED_DETECTION_OVERLAP = 128
            Config.TILED_DETECTION_BATCH_SIZE = 10
            Config.TILED_DETECTION_MERGE_IOU = 0.85
            Config.YOLO_IMGSZ = 960

            fake_model = FakeModel()
            detector = ObjectDetector.__new__(ObjectDetector)
            detector.model = fake_model
            detector.class_names = list(fake_model.names.values())
            detector._min_conf = Config.YOLO_MIN_CONF
            detector._hazard_ids = {3, 14}
            detector._tile_class_ids = {3, 14}

            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            detections = detector.detect_batch([frame])[0]

            self.assertEqual(len(fake_model.calls), 2)
            self.assertEqual(set(fake_model.calls[1]["classes"]), {3, 14})
            self.assertEqual(fake_model.calls[0]["imgsz"], 960)
            self.assertIn(("fire", 0.95, 5, 6, 30, 36), self._as_tuples(detections))
        finally:
            for key, value in old_values.items():
                setattr(Config, key, value)

    def _as_tuples(self, detections):
        return [
            (det.class_name, det.confidence, det.x1, det.y1, det.x2, det.y2)
            for det in detections
        ]


class FakeBox:
    def __init__(self, class_id, confidence, xyxy):
        self.cls = [class_id]
        self.conf = [confidence]
        self.xyxy = [np.array(xyxy)]


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeModel:
    names = {3: "fire", 10: "person", 14: "smoke"}

    def __init__(self):
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        source = kwargs["source"]
        classes = set(kwargs["classes"] or [])

        if classes == {3, 14}:
            return [FakeResult([FakeBox(3, 0.95, (5, 6, 30, 36))])] + [
                FakeResult([]) for _ in source[1:]
            ]

        return [FakeResult([]) for _ in source]


if __name__ == "__main__":
    unittest.main()
