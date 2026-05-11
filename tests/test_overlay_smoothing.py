import unittest

from detector import Detection
from overlay_smoothing import BoxMotionSmoother


class BoxMotionSmootherTests(unittest.TestCase):
    def test_predicts_box_forward_between_inference_frames(self):
        smoother = BoxMotionSmoother(max_predict_frames=6, smoothing_alpha=1.0)
        first = [Detection("person", 0.90, 100, 100, 160, 220)]
        second = [Detection("person", 0.92, 112, 100, 172, 220)]

        smoother.render(first, detection_frame_id=10, render_frame_id=10)
        predicted = smoother.render(second, detection_frame_id=12, render_frame_id=14)

        self.assertEqual((predicted[0].x1, predicted[0].y1, predicted[0].x2, predicted[0].y2), (124, 100, 184, 220))

    def test_default_smoothing_does_not_lag_constant_motion(self):
        smoother = BoxMotionSmoother(max_predict_frames=6, smoothing_alpha=0.65)
        first = [Detection("person", 0.90, 100, 100, 160, 220)]
        second = [Detection("person", 0.92, 112, 100, 172, 220)]

        smoother.render(first, detection_frame_id=10, render_frame_id=10)
        predicted = smoother.render(second, detection_frame_id=12, render_frame_id=14)

        self.assertEqual((predicted[0].x1, predicted[0].y1, predicted[0].x2, predicted[0].y2), (124, 100, 184, 220))

    def test_default_prediction_horizon_covers_slow_inference_gap(self):
        smoother = BoxMotionSmoother(smoothing_alpha=1.0)
        first = [Detection("person", 0.90, 0, 0, 20, 40)]
        second = [Detection("person", 0.92, 10, 0, 30, 40)]

        smoother.render(first, detection_frame_id=0, render_frame_id=0)
        predicted = smoother.render(second, detection_frame_id=10, render_frame_id=40)

        self.assertEqual((predicted[0].x1, predicted[0].y1, predicted[0].x2, predicted[0].y2), (40, 0, 60, 40))

    def test_caps_prediction_so_stale_boxes_do_not_run_away(self):
        smoother = BoxMotionSmoother(max_predict_frames=2, smoothing_alpha=1.0)
        first = [Detection("fire", 0.90, 50, 50, 150, 150)]
        second = [Detection("fire", 0.92, 70, 50, 170, 150)]

        smoother.render(first, detection_frame_id=1, render_frame_id=1)
        predicted = smoother.render(second, detection_frame_id=2, render_frame_id=10)

        self.assertEqual((predicted[0].x1, predicted[0].y1, predicted[0].x2, predicted[0].y2), (110, 50, 210, 150))

    def test_smoothing_reduces_single_frame_box_jitter(self):
        smoother = BoxMotionSmoother(max_predict_frames=0, smoothing_alpha=0.5)
        first = [Detection("smoke", 0.90, 100, 100, 200, 200)]
        jittered = [Detection("smoke", 0.91, 108, 100, 208, 200)]

        smoother.render(first, detection_frame_id=1, render_frame_id=1)
        smoothed = smoother.render(jittered, detection_frame_id=2, render_frame_id=2)

        self.assertEqual((smoothed[0].x1, smoothed[0].x2), (104, 204))

    def test_same_inference_frame_rebuilds_tracks_when_visible_detections_change(self):
        smoother = BoxMotionSmoother(max_predict_frames=6, smoothing_alpha=1.0)
        fire = Detection("fire", 0.90, 10, 10, 50, 50)
        person = Detection("person", 0.95, 200, 100, 260, 220)

        smoother.render([fire, person], detection_frame_id=7, render_frame_id=7)
        smoothed = smoother.render([person], detection_frame_id=7, render_frame_id=8)

        self.assertEqual(len(smoothed), 1)
        self.assertEqual(smoothed[0].class_name, "person")
        self.assertEqual((smoothed[0].x1, smoothed[0].y1, smoothed[0].x2, smoothed[0].y2), (200, 100, 260, 220))


if __name__ == "__main__":
    unittest.main()
