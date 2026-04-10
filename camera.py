"""
camera.py - Vòng lặp chính: đọc camera, phát hiện, hiển thị, và gửi cảnh báo
"""

import logging
import time

import cv2
import numpy as np
from typing import List, Optional

logger = logging.getLogger(__name__)

from config import Config
from detector import ObjectDetector, Detection
from fire_tracker import FireTracker
from telegram_notifier import TelegramNotifier


class CameraSystem:
    """
    Hệ thống camera AI phát hiện cháy.

    Quy trình:
        1. Đọc frame từ camera
        2. Chạy YOLO detect
        3. Cập nhật FireTracker (đếm ngược 5 giây)
        4. Nếu xác nhận cháy → gửi cảnh báo Telegram (cooldown 60s)
        5. Vẽ overlay lên frame và hiển thị
    """

    def __init__(self):
        self.detector = ObjectDetector()
        self.tracker = FireTracker()
        self.notifier = TelegramNotifier()
        self.cap: Optional[cv2.VideoCapture] = None
        self._fps: float = 0.0
        self._frame_count: int = 0
        self._fps_start: float = time.time()
        self.class_colors = {}
        self.is_video_file: bool = False

    # Bảng màu cố định cho các lớp vật thể thông thường (BGR)
    _OBJECT_COLOR_PALETTE = [
        (255, 180, 0),   # Xanh trời sáng
        (0, 220, 100),   # Xanh lá mint
        (200, 0, 255),   # Hồng tím
        (0, 180, 255),   # Cam sáng
        (120, 0, 200),   # Tím đậm
        (0, 255, 180),   # Xanh lơ
        (255, 80, 180),  # Hồng sen
        (30, 200, 255),  # Vàng chanh
        (200, 120, 0),   # Xanh đậm
        (0, 80, 255),    # Cam đỏ
    ]

    def _get_class_color(self, class_name: str) -> tuple:
        """Trả về màu cố định duy nhất cho từng tên class (không random)."""
        if class_name not in self.class_colors:
            idx = len(self.class_colors) % len(self._OBJECT_COLOR_PALETTE)
            self.class_colors[class_name] = self._OBJECT_COLOR_PALETTE[idx]
        return self.class_colors[class_name]

    def open_camera(self, video_source) -> bool:
        """Mở camera/video dựa trên video_source (int, path, hoặc URL)"""
        logger.info(f"[Camera] Đang mở nguồn video: {video_source}...")
        self.cap = cv2.VideoCapture(video_source)

        if not self.cap.isOpened():
            logger.info(f"[Camera] ❌ Không thể mở nguồn video: {video_source}")
            return False

        # Thiết lập cờ để nhận biết có phải là file video không
        self.is_video_file = isinstance(video_source, str) and not video_source.startswith(('http://', 'https://', 'rtsp://'))

        # Thiết lập độ phân giải
        if Config.FRAME_WIDTH > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"[Camera] ✅ Nguồn video mở thành công. Độ phân giải: {actual_w}x{actual_h}")
        return True

    def run(self, video_source):
        """Vòng lặp chính. Nhấn 'q' để thoát."""
        if not self.open_camera(video_source):
            return

        logger.info("[Camera] 🚀 Bắt đầu phát hiện. Nhấn 'q' để thoát, 'p' hoặc SPACE để tạm dừng/tiếp tục.")
        cv2.namedWindow(Config.WINDOW_NAME, cv2.WINDOW_NORMAL)
        
        self.is_paused = False
        self.last_display_frame = None

        try:
            while True:
                # ── Xử lý phím ──
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # q hoặc ESC
                    logger.info("[Camera] Đang thoát...")
                    break
                elif key == ord('p') or key == 32:  # 'p' hoặc Space
                    self.is_paused = not getattr(self, 'is_paused', False)
                    logger.info(f"\n[Camera] ⏸️ Đã {'TẠM DỪNG' if self.is_paused else 'TIẾP TỤC'} video.")

                if getattr(self, 'is_paused', False) and getattr(self, 'last_display_frame', None) is not None:
                    # Giữ nguyên frame hiện tại trên màn hình
                    cv2.imshow(Config.WINDOW_NAME, self.last_display_frame)
                    continue

                ret, frame = self.cap.read()
                if not ret:
                    if getattr(self, 'is_video_file', False):
                        logger.info("[Camera] 🎬 Đã phát hết video. Đang thoát...")
                        break
                    logger.info("[Camera] ⚠️  Không đọc được frame. Thử lại...")
                    time.sleep(0.1)
                    continue

                # ── Phát hiện đối tượng ──
                detections = self.detector.detect(frame)

                # ── Cập nhật tracker ──
                fire_confirmed = self.tracker.update(detections)

                # ── Vẽ lên frame ──
                self._update_fps()
                display_frame = self._draw_overlay(frame, detections)
                self.last_display_frame = display_frame

                # ── Gửi cảnh báo nếu xác nhận cháy ──
                if fire_confirmed:
                    logger.info("[System] 🔥 ĐÁM CHÁY ĐÃ ĐƯỢC XÁC NHẬN! Đang gửi cảnh báo...")
                    fire_dets = self.tracker.get_fire_detections(detections)
                    self.notifier.send_fire_alert(display_frame.copy(), len(fire_dets))

                # ── Hiển thị ──
                cv2.imshow(Config.WINDOW_NAME, display_frame)

        except KeyboardInterrupt:
            logger.info("\n[Camera] Dừng bởi người dùng (Ctrl+C)")
        finally:
            self._cleanup()

    def _update_fps(self):
        """Tính FPS thực tế"""
        self._frame_count += 1
        elapsed = time.time() - self._fps_start
        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_start = time.time()

    def _draw_overlay(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """
        Vẽ toàn bộ overlay lên frame:
        - Bounding box từng đối tượng
        - Thanh countdown lửa
        - Panel thông tin (góc trên trái)
        - Cảnh báo cháy lớn (khi đang xác nhận hoặc đã xác nhận)
        """
        frame = frame.copy()
        h, w = frame.shape[:2]
        fire_dets = []

        # ── Vẽ bounding box ──
        is_confirmed = self.tracker.is_tracking and self.tracker.elapsed >= Config.FIRE_CONFIRM_SECONDS

        for det in detections:
            if det.is_fire:
                fire_dets.append(det)
                is_smoke = det.class_name.lower() == Config.SMOKE_CLASS_NAME.lower()

                if is_confirmed:
                    # Sau 5 giây: CẢ lửa và khói đều chuyển đỏ
                    color = Config.COLOR_FIRE
                else:
                    # Chưa xác nhận: lửa=vàng, khói=xám
                    color = Config.COLOR_SMOKE if is_smoke else Config.COLOR_FIRE_UNCONFIRMED

                label_text = "SMOKE" if is_smoke else "FIRE"
                label = f"{label_text} {det.confidence:.0%}"
            else:
                color = self._get_class_color(det.class_name)
                label = f"{det.class_name} {det.confidence:.0%}"

            # Box
            cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), color, 4)

            # Label background
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                frame,
                (det.x1, det.y1 - lh - 8),
                (det.x1 + lw + 4, det.y1),
                color, -1
            )
            cv2.putText(
                frame, label,
                (det.x1 + 2, det.y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )

        # ── Thanh countdown lửa ──
        if self.tracker.is_tracking:
            self._draw_countdown_bar(frame, w, h)

        # ── Cảnh báo lớn khi đủ 5 giây ──
        if self.tracker.is_tracking and self.tracker.elapsed >= Config.FIRE_CONFIRM_SECONDS:
            self._draw_fire_alert_banner(frame, w, h)

        # ── Panel thông tin ──
        self._draw_info_panel(frame, len(detections), len(fire_dets), w, h)

        return frame

    def _draw_countdown_bar(self, frame: np.ndarray, w: int, h: int):
        """Vẽ thanh tiến trình countdown phía dưới màn hình"""
        bar_h = 30
        bar_y = h - bar_h - 10
        bar_x1, bar_x2 = 10, w - 10
        bar_w = bar_x2 - bar_x1

        # Background
        cv2.rectangle(frame, (bar_x1, bar_y), (bar_x2, bar_y + bar_h), (50, 50, 50), -1)

        # Progress fill
        filled_w = int(bar_w * self.tracker.progress)
        if filled_w > 0:
            cv2.rectangle(
                frame,
                (bar_x1, bar_y),
                (bar_x1 + filled_w, bar_y + bar_h),
                Config.COLOR_DANGER, -1
            )

        # Border
        cv2.rectangle(frame, (bar_x1, bar_y), (bar_x2, bar_y + bar_h), (200, 200, 200), 2)

        # Text
        remaining = self.tracker.remaining
        text = f"Xac nhan chay: {remaining:.1f}s con lai ({Config.FIRE_CONFIRM_SECONDS:.0f}s)"
        cv2.putText(
            frame, text,
            (bar_x1 + 5, bar_y + bar_h - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1
        )

    def _draw_fire_alert_banner(self, frame: np.ndarray, w: int, h: int):
        """Vẽ banner cảnh báo đỏ khi đám cháy được xác nhận"""
        overlay = frame.copy()
        banner_h = 60
        cv2.rectangle(overlay, (0, h // 2 - banner_h // 2), (w, h // 2 + banner_h // 2), (0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        text = "!!! DAM CHAY DA DUOC XAC NHAN !!!"
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
        tx = (w - tw) // 2
        cv2.putText(
            frame, text,
            (tx, h // 2 + 10),
            cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2
        )

    def _draw_info_panel(self, frame: np.ndarray, total_det: int, fire_det: int, w: int, h: int):
        """Vẽ panel thông tin góc trên trái"""
        panel_w, panel_h = 320, 130
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        lines = [
            f"FPS: {self._fps:.1f}",
            f"Doi tuong: {total_det} | Lua: {fire_det}",
            f"Trang thai: {'PHAT HIEN LUA!' if fire_det > 0 else 'Binh thuong'}",
            f"Telegram cooldown: {self.notifier.cooldown_remaining:.0f}s",
        ]

        for i, line in enumerate(lines):
            color = (0, 100, 255) if ("LUA" in line or "CHAY" in line) else (220, 220, 220)
            cv2.putText(
                frame, line,
                (10, 22 + i * 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1
            )

        # Dòng hướng dẫn nhỏ ở góc dưới phải
        hint = "Nhan 'Q' hoac ESC de thoat"
        cv2.putText(
            frame, hint,
            (w - 270, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1
        )

    def _cleanup(self):
        """Dọn dẹp tài nguyên"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        # Thêm vài lần waitKey(1) để xử lý triệt để event đóng cửa sổ (sửa lỗi treo trên PyCharm/Windows)
        for _ in range(5):
            cv2.waitKey(1)
        logger.info("[Camera] Đã giải phóng tài nguyên.")
