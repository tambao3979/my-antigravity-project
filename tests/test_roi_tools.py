import unittest

from detector import Detection
from roi_tools import filter_detections_by_roi, frame_point_from_widget_event, point_in_polygon


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

    def test_frame_point_from_widget_event_uses_root_coordinates_for_nested_ctk_widgets(self):
        display = {
            "frame_w": 1280,
            "frame_h": 720,
            "image_w": 640,
            "image_h": 360,
            "offset_x": 80,
            "offset_y": 20,
        }
        outer_label = FakeWidget(root_x=1000, root_y=500)
        event = FakeEvent(
            x=10,
            y=10,
            x_root=1000 + 80 + 320,
            y_root=500 + 20 + 180,
        )

        point = frame_point_from_widget_event(event, display, outer_label)

        self.assertEqual(point, (640, 360))


class FakeWidget:
    def __init__(self, root_x, root_y):
        self._root_x = root_x
        self._root_y = root_y

    def winfo_rootx(self):
        return self._root_x

    def winfo_rooty(self):
        return self._root_y


class FakeEvent:
    def __init__(self, x, y, x_root, y_root):
        self.x = x
        self.y = y
        self.x_root = x_root
        self.y_root = y_root


if __name__ == "__main__":
    unittest.main()
