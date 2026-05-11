import os
import sys
import tempfile
import unittest
from unittest.mock import patch

from config import _default_runtime_path
from logger_config import _default_log_dir


class RuntimePathTests(unittest.TestCase):
    def test_runtime_data_paths_stay_relative_in_source_mode(self):
        with patch.object(sys, "frozen", False, create=True):
            self.assertEqual(_default_runtime_path("events"), "events")
            self.assertEqual(_default_runtime_path("exports"), "exports")

    def test_runtime_data_paths_use_appdata_when_packaged(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.dict(os.environ, {"APPDATA": temp_dir}),
                patch.object(sys, "frozen", True, create=True),
            ):
                self.assertEqual(
                    _default_runtime_path("events"),
                    os.path.join(temp_dir, "CameraAI", "events"),
                )

    def test_log_dir_uses_appdata_when_packaged(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.dict(os.environ, {"APPDATA": temp_dir}),
                patch.object(sys, "frozen", True, create=True),
            ):
                self.assertEqual(
                    _default_log_dir(),
                    os.path.join(temp_dir, "CameraAI", "logs"),
                )


if __name__ == "__main__":
    unittest.main()
