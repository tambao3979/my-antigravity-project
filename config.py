"""
config.py - Cấu hình trung tâm cho hệ thống FireWatch AI
Load từ file .env hoặc dùng giá trị mặc định.
"""

import os

from persistent_env import load_application_env

load_application_env()


def _resolve_model_path() -> str:
    """Resolve model path with safe fallback for packaged app."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    bundled_default = os.path.join(base_dir, "weights", "best.pt")

    configured_path = os.getenv("MODEL_PATH", bundled_default).strip()
    if not configured_path:
        return bundled_default

    if not os.path.isabs(configured_path):
        configured_path = os.path.join(base_dir, configured_path)

    if os.path.exists(configured_path):
        return configured_path
    return bundled_default


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_class_names(name: str, default: str) -> set:
    return {
        item.strip().lower()
        for item in os.getenv(name, default).split(",")
        if item.strip()
    }


class Config:
    # ──────────────────────────────────────────────
    # Model YOLO
    # ──────────────────────────────────────────────
    MODEL_PATH: str = _resolve_model_path()

    # Stronger inference defaults for small flames and overlapping objects.
    YOLO_MIN_CONF: float = float(os.getenv("YOLO_MIN_CONF", "0.01"))
    YOLO_IMGSZ: int = int(os.getenv("YOLO_IMGSZ", "960"))
    YOLO_IOU: float = float(os.getenv("YOLO_IOU", "0.85"))
    YOLO_MAX_DET: int = int(os.getenv("YOLO_MAX_DET", "300"))
    YOLO_AGNOSTIC_NMS: bool = _env_bool("YOLO_AGNOSTIC_NMS", "false")
    YOLO_AUGMENT: bool = _env_bool("YOLO_AUGMENT", "false")

    # ──────────────────────────────────────────────
    # Tên nhãn lớp đặc biệt (dùng cho logic fire/smoke)
    # ──────────────────────────────────────────────
    FIRE_CLASS_NAME: str = os.getenv("FIRE_CLASS_NAME", "fire")
    SMOKE_CLASS_NAME: str = os.getenv("SMOKE_CLASS_NAME", "smoke")

    # Extra tiled pass for tiny hazards. Only fire/smoke are tiled by default.
    TILED_DETECTION_ENABLED: bool = _env_bool("TILED_DETECTION_ENABLED", "true")
    TILED_DETECTION_CLASSES: set = _env_class_names(
        "TILED_DETECTION_CLASSES",
        f"{FIRE_CLASS_NAME},{SMOKE_CLASS_NAME}",
    )
    TILED_DETECTION_TILE_SIZE: int = int(os.getenv("TILED_DETECTION_TILE_SIZE", "512"))
    TILED_DETECTION_OVERLAP: int = int(os.getenv("TILED_DETECTION_OVERLAP", "128"))
    TILED_DETECTION_BATCH_SIZE: int = int(os.getenv("TILED_DETECTION_BATCH_SIZE", "4"))
    TILED_DETECTION_MERGE_IOU: float = float(os.getenv("TILED_DETECTION_MERGE_IOU", "0.85"))
    FIRE_MISSING_GRACE_SECONDS: float = float(os.getenv("FIRE_MISSING_GRACE_SECONDS", "1.0"))

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
    FIRE_CONFIDENCE_THRESHOLD: float = float(os.getenv("FIRE_CONFIDENCE_THRESHOLD", "0.01"))
    SMOKE_CONFIDENCE_THRESHOLD: float = float(os.getenv("SMOKE_CONFIDENCE_THRESHOLD", "0.01"))

    # ──────────────────────────────────────────────
    # Quy tắc xác nhận cháy
    # ──────────────────────────────────────────────
    FIRE_CONFIRM_SECONDS: float = float(os.getenv("FIRE_CONFIRM_SECONDS", "5"))

    # ──────────────────────────────────────────────
    # Telegram & Webhook API
    # ──────────────────────────────────────────────
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
    WEBHOOK_URL: str = os.getenv("WEBHOOK_URL", "")
    ALERT_COOLDOWN_SECONDS: float = float(os.getenv("ALERT_COOLDOWN_SECONDS", "60"))

    # ──────────────────────────────────────────────
    # Zalo OA (Official Account) API
    #   ZALO_OA_TOKEN : OA Access Token
    #   ZALO_USER_ID  : ID người dùng Zalo nhận thông báo
    # ──────────────────────────────────────────────
    ZALO_OA_TOKEN: str = os.getenv("ZALO_OA_TOKEN", "")
    ZALO_USER_ID: str = os.getenv("ZALO_USER_ID", "")

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
    
    # ──────────────────────────────────────────────
    # Ghi nhận Sự cố (Event Recording)
    # ──────────────────────────────────────────────
    EVENT_DIR: str = os.getenv("EVENT_DIR", "events")
    EVENT_DB_PATH: str = os.getenv("EVENT_DB_PATH", os.path.join(EVENT_DIR, "events.db"))
    EVENT_SNAPSHOT_DIR: str = os.getenv("EVENT_SNAPSHOT_DIR", os.path.join(EVENT_DIR, "snapshots"))
    EXPORT_DIR: str = os.getenv("EXPORT_DIR", "exports")

    # MQTT alerts are optional to keep edge deployments lightweight.
    MQTT_ENABLED: bool = _env_bool("MQTT_ENABLED", "false")
    MQTT_HOST: str = os.getenv("MQTT_HOST", "")
    MQTT_PORT: int = int(os.getenv("MQTT_PORT", "1883"))
    MQTT_TOPIC: str = os.getenv("MQTT_TOPIC", "camera_ai/alerts")
    MQTT_CLIENT_ID: str = os.getenv("MQTT_CLIENT_ID", "camera-ai-edge")

    # Lightweight hardware resource monitor.
    RESOURCE_MONITOR_ENABLED: bool = _env_bool("RESOURCE_MONITOR_ENABLED", "true")
    RESOURCE_MONITOR_INTERVAL_SECONDS: float = float(os.getenv("RESOURCE_MONITOR_INTERVAL_SECONDS", "2.0"))
    RESOURCE_MONITOR_GPU_INDEX: int = int(os.getenv("RESOURCE_MONITOR_GPU_INDEX", "0"))

    # Health checks, alert thresholds, and local RBAC credentials.
    CAMERA_HEALTH_CHECK_INTERVAL_SECONDS: float = float(os.getenv("CAMERA_HEALTH_CHECK_INTERVAL_SECONDS", "1.0"))
    CAMERA_DISCONNECT_TIMEOUT_SECONDS: float = float(os.getenv("CAMERA_DISCONNECT_TIMEOUT_SECONDS", "30.0"))
    CAMERA_RECONNECT_BASE_DELAY_SECONDS: float = float(os.getenv("CAMERA_RECONNECT_BASE_DELAY_SECONDS", "1.0"))
    CAMERA_RECONNECT_MAX_DELAY_SECONDS: float = float(os.getenv("CAMERA_RECONNECT_MAX_DELAY_SECONDS", "30.0"))
    GPU_ALERT_USAGE_THRESHOLD_PERCENT: float = float(os.getenv("GPU_ALERT_USAGE_THRESHOLD_PERCENT", "90.0"))
    GPU_ALERT_CONSECUTIVE_SAMPLES: int = int(os.getenv("GPU_ALERT_CONSECUTIVE_SAMPLES", "3"))
    INFERENCE_LATENCY_ALERT_THRESHOLD_MS: float = float(os.getenv("INFERENCE_LATENCY_ALERT_THRESHOLD_MS", "250.0"))
    INFERENCE_LATENCY_ALERT_CONSECUTIVE_SAMPLES: int = int(os.getenv("INFERENCE_LATENCY_ALERT_CONSECUTIVE_SAMPLES", "3"))
    SYSTEM_ALERT_REPEAT_COOLDOWN_SECONDS: float = float(os.getenv("SYSTEM_ALERT_REPEAT_COOLDOWN_SECONDS", "60.0"))
    AUTH_ADMIN_USERNAME: str = os.getenv("AUTH_ADMIN_USERNAME", "admin")
    AUTH_ADMIN_PASSWORD: str = os.getenv("AUTH_ADMIN_PASSWORD", "admin")
    AUTH_USER_USERNAME: str = os.getenv("AUTH_USER_USERNAME", "user")
    AUTH_USER_PASSWORD: str = os.getenv("AUTH_USER_PASSWORD", "user")
