"""
Small ROI helpers used by both the GUI and tests.

The functions are deliberately dependency-light so ROI filtering can run in the
render path without allocating masks every frame.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence, Tuple, TypeVar

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


def frame_point_from_widget_event(
    event: Any,
    display: Mapping[str, int],
    widget: Any,
) -> Point | None:
    """
    Convert a Tk event to original-frame coordinates.

    CustomTkinter labels bind events on internal child widgets. Root coordinates
    keep the calculation stable regardless of which child received the click.
    """
    if not display or widget is None:
        return None

    try:
        local_x = int(getattr(event, "x_root")) - int(widget.winfo_rootx())
        local_y = int(getattr(event, "y_root")) - int(widget.winfo_rooty())
    except Exception:
        local_x = int(getattr(event, "x", -1))
        local_y = int(getattr(event, "y", -1))

    image_x = local_x - int(display.get("offset_x", 0))
    image_y = local_y - int(display.get("offset_y", 0))
    image_w = max(1, int(display.get("image_w", 1)))
    image_h = max(1, int(display.get("image_h", 1)))
    frame_w = max(1, int(display.get("frame_w", 1)))
    frame_h = max(1, int(display.get("frame_h", 1)))

    if image_x < 0 or image_y < 0 or image_x > image_w or image_y > image_h:
        return None

    frame_x = int(image_x / image_w * frame_w)
    frame_y = int(image_y / image_h * frame_h)
    return (
        max(0, min(frame_x, frame_w - 1)),
        max(0, min(frame_y, frame_h - 1)),
    )


def class_counts(detections: Iterable[TDetection]) -> dict[str, int]:
    """Build a compact class-name count map."""
    counts: dict[str, int] = {}
    for detection in detections:
        class_name = str(getattr(detection, "class_name", "unknown"))
        counts[class_name] = counts.get(class_name, 0) + 1
    return counts
