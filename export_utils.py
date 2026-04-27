"""Export helpers for count/event data."""

from __future__ import annotations

import csv
import os
from typing import Mapping, Sequence


DEFAULT_COUNT_FIELDS = ["timestamp", "source", "total", "fire", "smoke"]


def export_count_rows_csv(rows: Sequence[Mapping[str, object]], path: str) -> str:
    """Write count rows to CSV and return the output path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = _fieldnames(rows)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def export_count_rows_excel(rows: Sequence[Mapping[str, object]], path: str) -> str:
    """
    Write count rows to xlsx when openpyxl is available.

    The dependency is optional to keep edge deployments light.
    """
    try:
        from openpyxl import Workbook
    except ImportError as exc:
        raise RuntimeError("Excel export requires optional dependency: openpyxl") from exc

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = _fieldnames(rows)
    wb = Workbook()
    ws = wb.active
    ws.title = "Counts"
    ws.append(fieldnames)
    for row in rows:
        ws.append([row.get(field, "") for field in fieldnames])
    wb.save(path)
    return path


def _fieldnames(rows: Sequence[Mapping[str, object]]) -> list[str]:
    fields = list(DEFAULT_COUNT_FIELDS)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    return fields
