import unittest

from gui import compute_display_geometry


class DisplayGeometryTests(unittest.TestCase):
    def test_compute_display_geometry_preserves_aspect_ratio_and_centers_image(self):
        geometry = compute_display_geometry(
            frame_width=1280,
            frame_height=720,
            container_width=640,
            container_height=500,
        )

        self.assertEqual(geometry, (640, 360, 0, 70))

    def test_compute_display_geometry_rejects_too_small_target(self):
        self.assertIsNone(
            compute_display_geometry(
                frame_width=1280,
                frame_height=720,
                container_width=8,
                container_height=500,
            )
        )


if __name__ == "__main__":
    unittest.main()
