"""
detector.py - YOLO model wrapper cho phát hiện đối tượng.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Set

import numpy as np
from ultralytics import YOLO

from config import Config

logger = logging.getLogger(__name__)


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

        # Cache: tên class (lowercase) → ngưỡng confidence
        self._conf_map: dict = {}
        for cname in self.class_names:
            key = cname.lower()
            if key == Config.FIRE_CLASS_NAME.lower():
                self._conf_map[key] = Config.FIRE_CONFIDENCE_THRESHOLD
            elif key == Config.SMOKE_CLASS_NAME.lower():
                self._conf_map[key] = Config.SMOKE_CONFIDENCE_THRESHOLD
            else:
                self._conf_map[key] = Config.CONFIDENCE_THRESHOLD

        # Cache: hazard class IDs (luôn bảo vệ khi lọc)
        self._hazard_ids: Set[int] = set()
        for cid, cname in self.model.names.items():
            if cname.lower() in Config.HAZARD_CLASS_NAMES:
                self._hazard_ids.add(cid)

        # Min confidence cho YOLO predict
        self._min_conf: float = (
            min(self._conf_map.values()) if self._conf_map else Config.CONFIDENCE_THRESHOLD
        )

        # Cảnh báo nếu model không có nhãn fire
        if Config.FIRE_CLASS_NAME.lower() not in {c.lower() for c in self.class_names}:
            logger.warning(
                "Không tìm thấy nhãn '%s' trong model. Kiểm tra FIRE_CLASS_NAME.",
                Config.FIRE_CLASS_NAME,
            )

    def update_thresholds(self, fire_conf: float = None, smoke_conf: float = None, default_conf: float = None):
        """Cập nhật độ nhạy (confidence) ngay trong lúc đang chạy (theo UI)."""
        if fire_conf is not None:
            self._conf_map[Config.FIRE_CLASS_NAME.lower()] = fire_conf
        if smoke_conf is not None:
            self._conf_map[Config.SMOKE_CLASS_NAME.lower()] = smoke_conf
        if default_conf is not None:
            # Cập nhật cho tất cả các class còn lại
            for cname in self.class_names:
                key = cname.lower()
                if key != Config.FIRE_CLASS_NAME.lower() and key != Config.SMOKE_CLASS_NAME.lower():
                    self._conf_map[key] = default_conf
            
        self._min_conf = min(self._conf_map.values()) if self._conf_map else Config.CONFIDENCE_THRESHOLD
        logger.info(f"[Detector] Đã cập nhật Thresholds -> Lửa: {fire_conf}, Khói: {smoke_conf}, Chung: {default_conf}")

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

        results = self.model.predict(
            source=frames,
            conf=self._min_conf,
            classes=predict_classes,
            verbose=False,
        )

        batch_detections = []
        for result in results:
            detections: List[Detection] = []
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    confidence = float(box.conf[0])

                    req_conf = self._conf_map.get(class_name.lower(), Config.CONFIDENCE_THRESHOLD)
                    if confidence < req_conf:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append(
                        Detection(
                            class_name=class_name,
                            confidence=confidence,
                            x1=x1, y1=y1, x2=x2, y2=y2,
                        )
                    )
            batch_detections.append(detections)

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
