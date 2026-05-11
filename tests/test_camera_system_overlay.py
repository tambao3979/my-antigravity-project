import unittest

import numpy as np

from camera import CameraSystem


class _FakeNotifier:
    def cooldown_remaining(self):
        return 12.3


class CameraSystemOverlayTests(unittest.TestCase):
    def test_info_panel_calls_notifier_cooldown_method_before_formatting(self):
        camera_system = object.__new__(CameraSystem)
        camera_system._fps = 24.0
        camera_system.notifier = _FakeNotifier()
        frame = np.zeros((160, 360, 3), dtype=np.uint8)

        camera_system._draw_info_panel(frame, total_det=1, fire_det=1, w=360, h=160)

        self.assertGreater(int(frame.sum()), 0)


if __name__ == "__main__":
    unittest.main()
