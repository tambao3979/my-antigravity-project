import unittest
from unittest.mock import patch

from config import Config
from detector import Detection
from fire_tracker import FireTracker


def fire_detection():
    return Detection("fire", 0.80, 10, 10, 40, 40)


class FireTrackerGraceTests(unittest.TestCase):
    def setUp(self):
        self._confirm_seconds = Config.FIRE_CONFIRM_SECONDS
        self._missing_grace = getattr(Config, "FIRE_MISSING_GRACE_SECONDS", None)
        Config.FIRE_CONFIRM_SECONDS = 5.0
        Config.FIRE_MISSING_GRACE_SECONDS = 1.0

    def tearDown(self):
        Config.FIRE_CONFIRM_SECONDS = self._confirm_seconds
        if self._missing_grace is None:
            delattr(Config, "FIRE_MISSING_GRACE_SECONDS")
        else:
            Config.FIRE_MISSING_GRACE_SECONDS = self._missing_grace

    @patch("fire_tracker.time.time")
    def test_short_missing_gap_keeps_confirmation_window_alive(self, mock_time):
        tracker = FireTracker()

        mock_time.return_value = 0.0
        self.assertFalse(tracker.update([fire_detection()]))

        mock_time.return_value = 0.4
        self.assertFalse(tracker.update([]))
        self.assertTrue(tracker.is_tracking)

        mock_time.return_value = 5.1
        self.assertTrue(tracker.update([fire_detection()]))

    @patch("fire_tracker.time.time")
    def test_long_missing_gap_resets_confirmation_window(self, mock_time):
        tracker = FireTracker()

        mock_time.return_value = 0.0
        tracker.update([fire_detection()])

        mock_time.return_value = 1.2
        self.assertFalse(tracker.update([]))
        self.assertFalse(tracker.is_tracking)


if __name__ == "__main__":
    unittest.main()
