import csv
import os
import tempfile
import unittest

from export_utils import export_count_rows_csv


class ExportUtilsTests(unittest.TestCase):
    def test_export_count_rows_csv_writes_header_and_rows(self):
        rows = [
            {
                "timestamp": "2026-04-27T10:00:00",
                "source": "cam-1",
                "total": 3,
                "fire": 1,
                "smoke": 0,
                "person": 2,
            }
        ]

        with tempfile.TemporaryDirectory() as tmp:
            path = export_count_rows_csv(rows, os.path.join(tmp, "counts.csv"))

            with open(path, newline="", encoding="utf-8") as f:
                data = list(csv.DictReader(f))

        self.assertEqual(data[0]["source"], "cam-1")
        self.assertEqual(data[0]["total"], "3")
        self.assertEqual(data[0]["person"], "2")


if __name__ == "__main__":
    unittest.main()
