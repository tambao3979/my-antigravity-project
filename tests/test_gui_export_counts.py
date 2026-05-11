import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from gui import SecurityApp


class ImmediateThread:
    def __init__(self, target, daemon=False):
        self.target = target
        self.daemon = daemon

    def start(self):
        self.target()


class ExportCountsUiTests(unittest.TestCase):
    def test_export_counts_without_rows_updates_status(self):
        app = SimpleNamespace(
            _count_history=[],
            statuses=[],
            _set_status=lambda msg, color, icon="": app.statuses.append((msg, icon)),
        )

        SecurityApp._export_counts(app, "csv")

        self.assertEqual(app.statuses[-1], ("Chưa có dữ liệu để xuất", "!"))

    def test_export_counts_csv_reports_saved_file_and_opens_export_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = os.path.join(tmp, "counts.csv")
            app = SimpleNamespace(
                _count_history=[{"timestamp": "now", "source": "cam-1", "total": 1}],
                statuses=[],
                opened_folders=[],
                after=lambda _delay, callback: callback(),
                _run_on_ui_thread=lambda callback: callback(),
                _set_status=lambda msg, color, icon="": app.statuses.append((msg, icon)),
                _open_export_folder=lambda path: app.opened_folders.append(path),
            )

            with (
                patch("gui.threading.Thread", ImmediateThread),
                patch("gui.export_count_rows_csv", return_value=output_path),
                patch("gui.Config.EXPORT_DIR", tmp),
            ):
                SecurityApp._export_counts(app, "csv")

        self.assertEqual(app.statuses[-1], ("Đã xuất CSV: counts.csv", "OK"))
        self.assertEqual(app.opened_folders, [output_path])

    def test_export_counts_failure_reports_error_after_ui_callback_runs(self):
        app = SimpleNamespace(
            _count_history=[{"timestamp": "now", "source": "cam-1", "total": 1}],
            statuses=[],
            ui_callbacks=[],
            _run_on_ui_thread=lambda callback: app.ui_callbacks.append(callback),
            _set_status=lambda msg, color, icon="": app.statuses.append((msg, icon)),
            _open_export_folder=lambda _path: None,
        )

        with (
            patch("gui.threading.Thread", ImmediateThread),
            patch("gui.export_count_rows_csv", side_effect=OSError("disk full")),
        ):
            SecurityApp._export_counts(app, "csv")

        self.assertEqual(len(app.ui_callbacks), 1)
        app.ui_callbacks[0]()
        self.assertEqual(app.statuses[-1][1], "!")
        self.assertIn("CSV", app.statuses[-1][0])
        self.assertIn("disk full", app.statuses[-1][0])


if __name__ == "__main__":
    unittest.main()
