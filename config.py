"""
config.py - Cấu hình trung tâm cho hệ thống FireWatch AI
Load từ file .env hoặc dùng giá trị mặc định.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ──────────────────────────────────────────────
    # Model YOLO
    # ──────────────────────────────────────────────
    MODEL_PATH: str = os.getenv(
        "MODEL_PATH",
        r"C:\Users\MAITAM-DNI\PycharmProjects\alarm\runs\detect\train3\weights\best.pt",
    )

    # ──────────────────────────────────────────────
    # Tên nhãn lớp đặc biệt (dùng cho logic fire/smoke)
    # ──────────────────────────────────────────────
    FIRE_CLASS_NAME: str = os.getenv("FIRE_CLASS_NAME", "fire")
    SMOKE_CLASS_NAME: str = os.getenv("SMOKE_CLASS_NAME", "smoke")

    # ──────────────────────────────────────────────
    # Class Registry — metadata hiển thị cho tất cả lớp
    #   icon  : emoji hiển thị trên GUI
    #   color : hex color cho icon / checkbox
    #   hazard: True → class an toàn, luôn giám sát
    # Thêm class mới? Chỉ cần thêm 1 dòng vào đây.
    # ──────────────────────────────────────────────
    CLASS_REGISTRY: dict = {
        "fire":         {"icon": "🔥", "color": "#ff6b35", "hazard": True},
        "smoke":        {"icon": "💨", "color": "#adb5bd", "hazard": True},
        "forklift":     {"icon": "🚜", "color": "#f59f00", "hazard": False},
        "person":       {"icon": "👤", "color": "#4dabf7", "hazard": False},
        "truck":        {"icon": "🚚", "color": "#51cf66", "hazard": False},
        "work_clothes": {"icon": "🦺", "color": "#ff922b", "hazard": False},
        "helmet":       {"icon": "⛑️", "color": "#ffd43b", "hazard": False},
        "barcode":      {"icon": "📷", "color": "#a9e34b", "hazard": False},
    }

    # Tập tên class nguy hiểm (tự tính từ registry)
    HAZARD_CLASS_NAMES: set = {
        name for name, meta in CLASS_REGISTRY.items() if meta.get("hazard")
    }

    # ──────────────────────────────────────────────
    # Ngưỡng độ tin cậy
    # ──────────────────────────────────────────────
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
    FIRE_CONFIDENCE_THRESHOLD: float = float(os.getenv("FIRE_CONFIDENCE_THRESHOLD", "0.05"))
    SMOKE_CONFIDENCE_THRESHOLD: float = float(os.getenv("SMOKE_CONFIDENCE_THRESHOLD", "0.05"))

    # ──────────────────────────────────────────────
    # Quy tắc xác nhận cháy
    # ──────────────────────────────────────────────
    FIRE_CONFIRM_SECONDS: float = float(os.getenv("FIRE_CONFIRM_SECONDS", "5"))

    # ──────────────────────────────────────────────
    # Telegram
    # ──────────────────────────────────────────────
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
    ALERT_COOLDOWN_SECONDS: float = float(os.getenv("ALERT_COOLDOWN_SECONDS", "60"))

    # ──────────────────────────────────────────────
    # Camera
    # ──────────────────────────────────────────────
    CAMERA_INDEX: int = int(os.getenv("CAMERA_INDEX", "0"))
    FRAME_WIDTH: int = int(os.getenv("FRAME_WIDTH", "1280"))
    FRAME_HEIGHT: int = int(os.getenv("FRAME_HEIGHT", "720"))

    # ──────────────────────────────────────────────
    # Màu bounding box (B, G, R)
    # ──────────────────────────────────────────────
    COLOR_FIRE = (0, 0, 255)
    COLOR_FIRE_UNCONFIRMED = (0, 255, 255)
    COLOR_SMOKE = (150, 150, 150)
    COLOR_DANGER = (0, 165, 255)

    WINDOW_NAME: str = "🔥 AI Fire Detection Camera"
