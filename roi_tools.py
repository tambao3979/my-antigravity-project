"""
Small ROI helpers used by both the GUI and tests.

The functions are deliberately dependency-light so ROI filtering can run in the
render path without allocating masks every frame.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple, TypeVar

Point = Tuple[int, int]
TDetection = TypeVar("TDetection")


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Return True when point is inside polygon using ray casting."""
    if len(polygon) < 3:
        return True

    x, y = point
    inside = False
    j = len(polygon) - 1

    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        crosses = (yi > y) != (yj > y)
        if crosses:
            x_at_y = (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
            if x < x_at_y:
                inside = not inside
        j = i

    return inside


def filter_detections_by_roi(
    detections: list[TDetection],
    polygon: Sequence[Point],
) -> list[TDetection]:
    """Keep detections whose center point falls inside polygon."""
    if len(polygon) < 3:
        return detections

    return [
        detection
        for detection in detections
        if point_in_polygon(getattr(detection, "center"), polygon)
    ]


def nearest_vertex(
    point: Point,
    polygon: Sequence[Point],
    max_distance: float = 18.0,
) -> int | None:
    """Return nearest ROI vertex index within max_distance pixels."""
    if not polygon:
        return None

    px, py = point
    max_dist_sq = max_distance * max_distance
    best_index = None
    best_dist_sq = max_dist_sq

    for index, (x, y) in enumerate(polygon):
        dist_sq = (px - x) ** 2 + (py - y) ** 2
        if dist_sq <= best_dist_sq:
            best_dist_sq = dist_sq
            best_index = index

    return best_index


def class_counts(detections: Iterable[TDetection]) -> dict[str, int]:
    """Build a compact class-name count map."""
    counts: dict[str, int] = {}
    for detection in detections:
        class_name = str(getattr(detection, "class_name", "unknown"))
        counts[class_name] = counts.get(class_name, 0) + 1
    return counts
