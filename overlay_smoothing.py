from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Iterable

from detector import Detection, box_iou


Box = tuple[float, float, float, float]


@dataclass
class _Track:
    track_id: int
    class_name: str
    confidence: float
    smoothed_box: Box
    last_detection_box: Box
    velocity: Box
    last_detection_frame_id: int


class BoxMotionSmoother:
    """Smooth and lightly predict overlay boxes between model inference frames."""

    def __init__(
        self,
        *,
        max_predict_frames: int = 60,
        smoothing_alpha: float = 0.65,
        match_iou_threshold: float = 0.20,
        center_distance_ratio: float = 0.45,
        motion_distance_ratio: float = 0.12,
    ):
        self.max_predict_frames = max(0, int(max_predict_frames))
        self.smoothing_alpha = max(0.0, min(float(smoothing_alpha), 1.0))
        self.match_iou_threshold = max(0.0, float(match_iou_threshold))
        self.center_distance_ratio = max(0.0, float(center_distance_ratio))
        self.motion_distance_ratio = max(0.0, float(motion_distance_ratio))
        self._tracks: dict[int, _Track] = {}
        self._active_track_ids: list[int] = []
        self._last_detection_frame_id: int | None = None
        self._last_detection_signature: tuple | None = None
        self._next_track_id = 1

    def render(
        self,
        detections: Iterable[Detection],
        *,
        detection_frame_id: int | None,
        render_frame_id: int,
    ) -> list[Detection]:
        detections = list(detections)
        if detection_frame_id is None:
            detection_frame_id = render_frame_id

        detection_signature = _detections_signature(detections)
        if detection_frame_id != self._last_detection_frame_id or detection_signature != self._last_detection_signature:
            self._active_track_ids = self._update_tracks(detections, detection_frame_id)
            self._last_detection_frame_id = detection_frame_id
            self._last_detection_signature = detection_signature

        smoothed: list[Detection] = []
        for track_id, fallback in zip(self._active_track_ids, detections):
            track = self._tracks.get(track_id)
            if track is None:
                smoothed.append(fallback)
                continue
            box = self._project_box(track, render_frame_id)
            smoothed.append(
                Detection(
                    fallback.class_name,
                    fallback.confidence,
                    round(box[0]),
                    round(box[1]),
                    round(box[2]),
                    round(box[3]),
                )
            )
        return smoothed

    def _update_tracks(self, detections: list[Detection], frame_id: int) -> list[int]:
        active_track_ids: list[int] = []
        matched_track_ids: set[int] = set()

        for detection in detections:
            box = _box_from_detection(detection)
            track = self._match_track(detection, matched_track_ids)
            if track is None:
                track = self._new_track(detection, box, frame_id)
            else:
                matched_track_ids.add(track.track_id)
                frame_delta = max(1, frame_id - track.last_detection_frame_id)
                velocity = _scale_box(_subtract_box(box, track.last_detection_box), 1.0 / frame_delta)
                if self._should_follow_motion(box, track.last_detection_box, frame_delta):
                    smoothed_box = box
                else:
                    projected_box = _add_box(track.smoothed_box, _scale_box(track.velocity, frame_delta))
                    smoothed_box = _blend_box(projected_box, box, self.smoothing_alpha)
                track.class_name = detection.class_name
                track.confidence = detection.confidence
                track.smoothed_box = smoothed_box
                track.last_detection_box = box
                track.velocity = velocity
                track.last_detection_frame_id = frame_id

            active_track_ids.append(track.track_id)

        self._tracks = {track_id: self._tracks[track_id] for track_id in active_track_ids if track_id in self._tracks}
        return active_track_ids

    def _new_track(self, detection: Detection, box: Box, frame_id: int) -> _Track:
        track = _Track(
            track_id=self._next_track_id,
            class_name=detection.class_name,
            confidence=detection.confidence,
            smoothed_box=box,
            last_detection_box=box,
            velocity=(0.0, 0.0, 0.0, 0.0),
            last_detection_frame_id=frame_id,
        )
        self._tracks[track.track_id] = track
        self._next_track_id += 1
        return track

    def _match_track(self, detection: Detection, matched_track_ids: set[int]) -> _Track | None:
        best_track: _Track | None = None
        best_score = -1.0

        for track in self._tracks.values():
            if track.track_id in matched_track_ids:
                continue
            if track.class_name.lower() != detection.class_name.lower():
                continue

            iou = box_iou(detection, _detection_from_box(track.class_name, track.confidence, track.last_detection_box))
            center_score = self._center_score(detection, track.last_detection_box)
            if iou < self.match_iou_threshold and center_score <= 0.0:
                continue

            score = max(iou, center_score)
            if score > best_score:
                best_score = score
                best_track = track

        return best_track

    def _center_score(self, detection: Detection, track_box: Box) -> float:
        det_cx, det_cy = detection.center
        tr_cx = (track_box[0] + track_box[2]) / 2.0
        tr_cy = (track_box[1] + track_box[3]) / 2.0
        det_w = max(1.0, detection.x2 - detection.x1)
        det_h = max(1.0, detection.y2 - detection.y1)
        tr_w = max(1.0, track_box[2] - track_box[0])
        tr_h = max(1.0, track_box[3] - track_box[1])
        max_distance = max(det_w, det_h, tr_w, tr_h) * self.center_distance_ratio
        distance = hypot(det_cx - tr_cx, det_cy - tr_cy)
        if distance > max_distance:
            return 0.0
        return 1.0 - (distance / max_distance if max_distance > 0 else 1.0)

    def _should_follow_motion(self, current_box: Box, previous_box: Box, frame_delta: int) -> bool:
        center_distance = _box_center_distance(current_box, previous_box)
        if center_distance <= 0.0:
            return False
        if frame_delta > 1:
            return True
        return center_distance >= _box_max_dimension(current_box, previous_box) * self.motion_distance_ratio

    def _project_box(self, track: _Track, render_frame_id: int) -> Box:
        frame_gap = max(0, render_frame_id - track.last_detection_frame_id)
        frame_gap = min(frame_gap, self.max_predict_frames)
        return _add_box(track.smoothed_box, _scale_box(track.velocity, frame_gap))


def _box_from_detection(detection: Detection) -> Box:
    return (float(detection.x1), float(detection.y1), float(detection.x2), float(detection.y2))


def _detections_signature(detections: Iterable[Detection]) -> tuple:
    return tuple(
        (
            detection.class_name.lower(),
            round(float(detection.confidence), 6),
            int(detection.x1),
            int(detection.y1),
            int(detection.x2),
            int(detection.y2),
        )
        for detection in detections
    )


def _detection_from_box(class_name: str, confidence: float, box: Box) -> Detection:
    return Detection(class_name, confidence, round(box[0]), round(box[1]), round(box[2]), round(box[3]))


def _add_box(first: Box, second: Box) -> Box:
    return tuple(a + b for a, b in zip(first, second))  # type: ignore[return-value]


def _subtract_box(first: Box, second: Box) -> Box:
    return tuple(a - b for a, b in zip(first, second))  # type: ignore[return-value]


def _scale_box(box: Box, scale: float) -> Box:
    return tuple(value * scale for value in box)  # type: ignore[return-value]


def _blend_box(first: Box, second: Box, alpha: float) -> Box:
    return tuple(a * (1.0 - alpha) + b * alpha for a, b in zip(first, second))  # type: ignore[return-value]


def _box_center_distance(first: Box, second: Box) -> float:
    first_cx = (first[0] + first[2]) / 2.0
    first_cy = (first[1] + first[3]) / 2.0
    second_cx = (second[0] + second[2]) / 2.0
    second_cy = (second[1] + second[3]) / 2.0
    return hypot(first_cx - second_cx, first_cy - second_cy)


def _box_max_dimension(first: Box, second: Box) -> float:
    return max(
        1.0,
        first[2] - first[0],
        first[3] - first[1],
        second[2] - second[0],
        second[3] - second[1],
    )
