"""
detector.py - YOLO model wrapper cho phát hiện đối tượng.
"""

import logging
import functools
import types
from dataclasses import dataclass
from typing import List, Optional, Set

import numpy as np


def _patch_singledispatch_for_nuitka() -> None:
    """
    Python 3.14 + Nuitka may compile functions without ``__annotate__``.
    ``functools.singledispatch.register`` then rejects plain ``@register`` use.
    This patch retries registration from ``__annotations__`` when needed.
    """
    if getattr(functools, "_nuitka_singledispatch_patched", False):
        return

    original_singledispatch = functools.singledispatch

    def compat_singledispatch(func):
        dispatcher = original_singledispatch(func)
        original_register = dispatcher.register

        def compat_register(cls, func=None):
            try:
                return original_register(cls, func=func)
            except TypeError as exc:
                if func is not None:
                    raise
                message = str(exc)
                if "Use either `@register(some_class)`" not in message:
                    raise
                if not callable(cls):
                    raise

                annotations = getattr(cls, "__annotations__", {}) or {}
                if not annotations:
                    raise

                first_annot = next(iter(annotations.values()))
                if isinstance(first_annot, str):
                    try:
                        first_annot = eval(first_annot, getattr(cls, "__globals__", {}), {})
                    except Exception:
                        raise exc

                if isinstance(first_annot, (type, types.UnionType)):
                    return original_register(first_annot, func=cls)
                raise

        dispatcher.register = compat_register
        return dispatcher

    functools.singledispatch = compat_singledispatch
    functools._nuitka_singledispatch_patched = True


_patch_singledispatch_for_nuitka()

from ultralytics import YOLO

from config import Config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TileRegion:
    """A rectangular crop in original-frame coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class Detection:
    """Kết quả phát hiện một đối tượng."""

    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def is_fire(self) -> bool:
        """True nếu class thuộc nhóm nguy hiểm (fire/smoke)."""
        return self.class_name.lower() in Config.HAZARD_CLASS_NAMES

    @property
    def center(self) -> tuple:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


def build_tile_regions(width: int, height: int, tile_size: int, overlap: int) -> List[TileRegion]:
    """Build overlapping tile regions that cover the full frame."""
    if width <= 0 or height <= 0 or tile_size <= 0:
        return []

    tile_w = min(tile_size, width)
    tile_h = min(tile_size, height)
    safe_overlap = max(0, min(overlap, tile_size - 1))

    x_starts = _axis_starts(width, tile_w, safe_overlap)
    y_starts = _axis_starts(height, tile_h, safe_overlap)

    return [
        TileRegion(x, y, min(x + tile_w, width), min(y + tile_h, height))
        for y in y_starts
        for x in x_starts
    ]


def _axis_starts(length: int, tile_size: int, overlap: int) -> List[int]:
    if length <= tile_size:
        return [0]

    step = max(1, tile_size - overlap)
    starts = [0]
    current = 0

    while current + tile_size < length:
        current = min(current + step, length - tile_size)
        if current == starts[-1]:
            break
        starts.append(current)

    return starts


def remap_tile_detection(
    detection: Detection,
    tile: TileRegion,
    frame_width: int,
    frame_height: int,
) -> Detection:
    """Move a tile-local detection back into original-frame coordinates."""
    return Detection(
        class_name=detection.class_name,
        confidence=detection.confidence,
        x1=_clip_int(tile.x1 + detection.x1, 0, frame_width),
        y1=_clip_int(tile.y1 + detection.y1, 0, frame_height),
        x2=_clip_int(tile.x1 + detection.x2, 0, frame_width),
        y2=_clip_int(tile.y1 + detection.y2, 0, frame_height),
    )


def merge_detections(
    detections: List[Detection],
    iou_threshold: float,
    class_agnostic: bool = False,
) -> List[Detection]:
    """Remove near-identical boxes while preserving overlapping objects."""
    kept: List[Detection] = []

    for detection in sorted(detections, key=lambda det: det.confidence, reverse=True):
        if detection.x2 <= detection.x1 or detection.y2 <= detection.y1:
            continue

        is_duplicate = False
        for existing in kept:
            same_class = detection.class_name.lower() == existing.class_name.lower()
            if not class_agnostic and not same_class:
                continue
            if box_iou(detection, existing) >= iou_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(detection)

    return kept


def box_iou(first: Detection, second: Detection) -> float:
    x1 = max(first.x1, second.x1)
    y1 = max(first.y1, second.y1)
    x2 = min(first.x2, second.x2)
    y2 = min(first.y2, second.y2)

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    intersection = inter_w * inter_h
    if intersection == 0:
        return 0.0

    first_area = max(0, first.x2 - first.x1) * max(0, first.y2 - first.y1)
    second_area = max(0, second.x2 - second.x1) * max(0, second.y2 - second.y1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _clip_int(value: int, low: int, high: int) -> int:
    return max(low, min(int(value), high))


class ObjectDetector:
    """
    Wrapper cho YOLOv8 model.
    Tải model một lần, gọi detect() cho mỗi frame.
    """

    def __init__(self):
        logger.info("Đang tải model từ: %s", Config.MODEL_PATH)
        self.model = YOLO(Config.MODEL_PATH)
        self.class_names: List[str] = list(self.model.names.values())
        logger.info("Tải model thành công. Các nhãn: %s", self.class_names)

        # Min confidence cho YOLO predict (thấp nhất để post-filter ở GUI)
        self._min_conf: float = Config.YOLO_MIN_CONF

        # Cache: hazard class IDs (luôn bảo vệ khi lọc)
        self._hazard_ids: Set[int] = set()
        self._tile_class_ids: Set[int] = set()
        for cid, cname in self.model.names.items():
            if cname.lower() in Config.HAZARD_CLASS_NAMES:
                self._hazard_ids.add(cid)
            if cname.lower() in Config.TILED_DETECTION_CLASSES:
                self._tile_class_ids.add(cid)

        # Cảnh báo nếu model không có nhãn fire
        if Config.FIRE_CLASS_NAME.lower() not in {c.lower() for c in self.class_names}:
            logger.warning(
                "Không tìm thấy nhãn '%s' trong model. Kiểm tra FIRE_CLASS_NAME.",
                Config.FIRE_CLASS_NAME,
            )

    def get_class_id(self, class_name: str) -> int:
        """Trả về class index theo tên, -1 nếu không tìm thấy."""
        for cid, cname in self.model.names.items():
            if cname.lower() == class_name.lower():
                return cid
        return -1

    def detect_batch(
        self,
        frames: List[np.ndarray],
        allowed_class_ids: Optional[Set[int]] = None,
    ) -> List[List[Detection]]:
        """
        Phát hiện đối tượng trên một mảng (batch) các frame cùng lúc.
        """
        if not frames:
            return []

        # Merge allowed + hazard IDs
        predict_classes = None
        if allowed_class_ids is not None:
            merged = allowed_class_ids | self._hazard_ids
            predict_classes = list(merged) if merged else None

        results = self._predict_frames(frames, predict_classes)

        batch_detections = []
        for result in results:
            detections: List[Detection] = []
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    confidence = float(box.conf[0])

                    # Post-filtering sẽ được thực hiện ở lớp GUI (per-camera)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append(
                        Detection(
                            class_name=class_name,
                            confidence=confidence,
                            x1=x1, y1=y1, x2=x2, y2=y2,
                        )
                    )
            batch_detections.append(detections)

        self._add_tiled_hazard_detections(frames, batch_detections)

        return batch_detections

    def detect(
        self,
        frame: np.ndarray,
        allowed_class_ids: Optional[Set[int]] = None,
    ) -> List[Detection]:
        """
        Phát hiện đối tượng trong một frame (sử dụng detect_batch).
        """
        batch_results = self.detect_batch([frame], allowed_class_ids)
        return batch_results[0] if batch_results else []

    def _predict_frames(
        self,
        frames: List[np.ndarray],
        class_ids: Optional[Set[int]] = None,
    ):
        predict_kwargs = {
            "source": frames,
            "conf": self._min_conf,
            "classes": sorted(class_ids) if class_ids is not None else None,
            "verbose": False,
            "imgsz": Config.YOLO_IMGSZ,
            "iou": Config.YOLO_IOU,
            "max_det": Config.YOLO_MAX_DET,
            "agnostic_nms": Config.YOLO_AGNOSTIC_NMS,
        }
        if Config.YOLO_AUGMENT:
            predict_kwargs["augment"] = True

        return self.model.predict(**predict_kwargs)

    def _detections_from_result(self, result) -> List[Detection]:
        detections: List[Detection] = []
        if result.boxes is None:
            return detections

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=confidence,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )

        return detections

    def _add_tiled_hazard_detections(
        self,
        frames: List[np.ndarray],
        batch_detections: List[List[Detection]],
    ) -> None:
        if not Config.TILED_DETECTION_ENABLED or not self._tile_class_ids:
            return

        tile_frames: List[np.ndarray] = []
        tile_meta = []
        tile_size = max(1, int(Config.TILED_DETECTION_TILE_SIZE))
        overlap = max(0, int(Config.TILED_DETECTION_OVERLAP))

        for frame_index, frame in enumerate(frames):
            height, width = frame.shape[:2]
            regions = build_tile_regions(width, height, tile_size, overlap)
            if len(regions) <= 1:
                continue

            for region in regions:
                crop = frame[region.y1:region.y2, region.x1:region.x2]
                if crop.size == 0:
                    continue
                tile_frames.append(crop)
                tile_meta.append((frame_index, region, width, height))

        if not tile_frames:
            return

        batch_size = max(1, int(Config.TILED_DETECTION_BATCH_SIZE))
        for start in range(0, len(tile_frames), batch_size):
            chunk = tile_frames[start:start + batch_size]
            chunk_meta = tile_meta[start:start + batch_size]
            results = self._predict_frames(chunk, self._tile_class_ids)

            for result, (frame_index, region, width, height) in zip(results, chunk_meta):
                for detection in self._detections_from_result(result):
                    remapped = remap_tile_detection(detection, region, width, height)
                    if remapped.x2 > remapped.x1 and remapped.y2 > remapped.y1:
                        batch_detections[frame_index].append(remapped)

        for index, detections in enumerate(batch_detections):
            batch_detections[index] = merge_detections(
                detections,
                iou_threshold=Config.TILED_DETECTION_MERGE_IOU,
            )
