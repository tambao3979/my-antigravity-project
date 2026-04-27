"""
fire_tracker.py - Theo dõi trạng thái đám cháy và xác nhận sau 5 giây liên tục
"""

import logging
import time
from typing import List, Optional
from detector import Detection
from config import Config


class FireTracker:
    """
    Theo dõi việc phát hiện lửa liên tục.

    Logic:
        - Khi nhận được frame có fire → ghi nhận thời điểm bắt đầu (nếu chưa có)
        - Nếu lửa biến mất trong bất kỳ frame nào → reset thời gian đếm
        - Nếu lửa hiện diện >= FIRE_CONFIRM_SECONDS giây → xác nhận cháy
    """

    def __init__(self):
        self._start_time: Optional[float] = None
        self._last_seen_time: Optional[float] = None
        self._confirmed: bool = False
        self._pause_time: Optional[float] = None

    @property
    def is_tracking(self) -> bool:
        """True nếu đang trong quá trình đếm ngược"""
        return self._start_time is not None

    @property
    def elapsed(self) -> float:
        """Số giây lửa đã được phát hiện liên tục (0 nếu không đang theo dõi)"""
        if self._start_time is None:
            return 0.0
        if self._pause_time is not None:
            return self._pause_time - self._start_time
        return time.time() - self._start_time

    def pause(self):
        """Tạm dừng đếm ngược"""
        if self._start_time is not None and self._pause_time is None:
            self._pause_time = time.time()

    def resume(self):
        """Tiếp tục đếm ngược"""
        if self._pause_time is not None:
            now = time.time()
            paused_duration = now - self._pause_time
            if self._start_time is not None:
                self._start_time += paused_duration
            if self._last_seen_time is not None:
                self._last_seen_time += paused_duration
            self._pause_time = None

    @property
    def progress(self) -> float:
        """Tiến độ xác nhận từ 0.0 → 1.0"""
        if not self.is_tracking:
            return 0.0
        return min(self.elapsed / Config.FIRE_CONFIRM_SECONDS, 1.0)

    @property
    def remaining(self) -> float:
        """Số giây còn lại để xác nhận"""
        return max(0.0, Config.FIRE_CONFIRM_SECONDS - self.elapsed)

    def update(self, detections: List[Detection]) -> bool:
        """
        Cập nhật tracker với danh sách phát hiện mới.

        Args:
            detections: Kết quả detect từ frame hiện tại

        Returns:
            True nếu đây là thời điểm xác nhận cháy (lần đầu đủ 5 giây)
        """
        now = time.time()
        fire_detected = any(d.is_fire for d in detections)

        if fire_detected:
            self._last_seen_time = now
            if self._start_time is None:
                # Bắt đầu đếm
                self._start_time = now
                self._confirmed = False

            if self.elapsed >= Config.FIRE_CONFIRM_SECONDS and not self._confirmed:
                self._confirmed = True
                return True  # 🔥 Xác nhận cháy!
        elif self._start_time is not None:
            last_seen = self._last_seen_time or self._start_time
            missing_for = now - last_seen
            if missing_for <= Config.FIRE_MISSING_GRACE_SECONDS:
                return False

            # Lửa biến mất → reset
            self.reset()

        return False

    def reset(self):
        """Reset toàn bộ trạng thái tracker"""
        self._start_time = None
        self._last_seen_time = None
        self._confirmed = False

    def get_fire_detections(self, detections: List[Detection]) -> List[Detection]:
        """Lọc chỉ lấy các detection là lửa"""
        return [d for d in detections if d.is_fire]
