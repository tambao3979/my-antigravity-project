import unittest

from detector import Detection
from roi_tools import filter_detections_by_roi, point_in_polygon


class RoiToolsTests(unittest.TestCase):
    def test_point_in_polygon_accepts_points_inside_roi(self):
        roi = [(10, 10), (110, 10), (110, 90), (10, 90)]

        self.assertTrue(point_in_polygon((60, 50), roi))
        self.assertFalse(point_in_polygon((150, 50), roi))

    def test_filter_detections_by_roi_keeps_center_inside_polygon(self):
        roi = [(0, 0), (100, 0), (100, 100), (0, 100)]
        detections = [
            Detection("fire", 0.91, 10, 10, 40, 40),
            Detection("person", 0.72, 120, 10, 170, 60),
        ]

        filtered = filter_detections_by_roi(detections, roi)

        self.assertEqual([det.class_name for det in filtered], ["fire"])

    def test_filter_detections_returns_all_when_roi_is_incomplete(self):
        detections = [Detection("fire", 0.91, 10, 10, 40, 40)]

        self.assertIs(filter_detections_by_roi(detections, []), detections)
        self.assertIs(filter_detections_by_roi(detections, [(0, 0), (10, 0)]), detections)


if __name__ == "__main__":
    unittest.main()
