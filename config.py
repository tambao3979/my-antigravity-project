"""
config.py - Cấu hình trung tâm cho hệ thống FireWatch AI
Load từ file .env hoặc dùng giá trị mặc định.
"""

import logging
import math
import os
import sys

from persistent_env import get_persistent_app_dir, load_application_env

load_application_env()

logger = logging.getLogger(__name__)


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


def _clamp(value: int | float, min_value: int | float | None, max_value: int | float | None):
    if min_value is not None and value < min_value:
        return min_value
    if max_value is not None and value > max_value:
        return max_value
    return value


def _env_float(
    name: str,
    default: str,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    raw_value = os.getenv(name, default).strip()
    try:
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError("not finite")
    except (TypeError, ValueError):
        logger.warning("Invalid float for %s=%r; using default %s", name, raw_value, default)
        value = float(default)
    return float(_clamp(value, min_value, max_value))


def _env_int(
    name: str,
    default: str,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    raw_value = os.getenv(name, default).strip()
    try:
        parsed = float(raw_value)
        if not math.isfinite(parsed):
            raise ValueError("not finite")
        value = int(parsed)
    except (TypeError, ValueError):
        logger.warning("Invalid integer for %s=%r; using default %s", name, raw_value, default)
        value = int(float(default))
    return int(_clamp(value, min_value, max_value))


def _env_class_names(name: str, default: str) -> set:
    return {
        item.strip().lower()
        for item in os.getenv(name, default).split(",")
        if item.strip()
    }


def _default_runtime_path(dirname: str) -> str:
    if getattr(sys, "frozen", False):
        return os.path.join(get_persistent_app_dir(), dirname)
    return dirname


class Config:
    # ──────────────────────────────────────────────
    # Model YOLO
    # ──────────────────────────────────────────────
    MODEL_PATH: str = _resolve_model_path()

    # Stronger inference defaults for small flames and overlapping objects.
    YOLO_MIN_CONF: float = _env_float("YOLO_MIN_CONF", "0.01", min_value=0.0, max_value=1.0)
    YOLO_IMGSZ: int = _env_int("YOLO_IMGSZ", "960", min_value=32)
    YOLO_IOU: float = _env_float("YOLO_IOU", "0.85", min_value=0.0, max_value=1.0)
    YOLO_MAX_DET: int = _env_int("YOLO_MAX_DET", "300", min_value=1)
    YOLO_AGNOSTIC_NMS: bool = _env_bool("YOLO_AGNOSTIC_NMS", "false")
    YOLO_AUGMENT: bool = _env_bool("YOLO_AUGMENT", "false")
    YOLO_DEVICE: str = os.getenv("YOLO_DEVICE", "").strip()
    YOLO_HALF: bool = _env_bool("YOLO_HALF", "false")

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
    TILED_DETECTION_TILE_SIZE: int = _env_int("TILED_DETECTION_TILE_SIZE", "512", min_value=32)
    TILED_DETECTION_OVERLAP: int = _env_int("TILED_DETECTION_OVERLAP", "128", min_value=0)
    TILED_DETECTION_BATCH_SIZE: int = _env_int("TILED_DETECTION_BATCH_SIZE", "4", min_value=1)
    TILED_DETECTION_MERGE_IOU: float = _env_float("TILED_DETECTION_MERGE_IOU", "0.85", min_value=0.0, max_value=1.0)
    OVERLAY_DEDUP_ENABLED: bool = _env_bool("OVERLAY_DEDUP_ENABLED", "true")
    OVERLAY_DEDUP_IOU: float = _env_float("OVERLAY_DEDUP_IOU", "0.45", min_value=0.0, max_value=1.0)
    OVERLAY_DEDUP_CONTAINMENT: float = _env_float("OVERLAY_DEDUP_CONTAINMENT", "0.75", min_value=0.0, max_value=1.0)
    DISPLAY_TARGET_FPS: int = _env_int("DISPLAY_TARGET_FPS", "60", min_value=1, max_value=240)
    LIVE_CAPTURE_TARGET_FPS: int = _env_int("LIVE_CAPTURE_TARGET_FPS", "60", min_value=1, max_value=240)
    INFERENCE_BATCH_SIZE: int = _env_int("INFERENCE_BATCH_SIZE", "1", min_value=1)
    BOX_SMOOTHING_ENABLED: bool = _env_bool("BOX_SMOOTHING_ENABLED", "true")
    BOX_SMOOTHING_ALPHA: float = _env_float("BOX_SMOOTHING_ALPHA", "0.65", min_value=0.0, max_value=1.0)
    BOX_PREDICT_MAX_FRAMES: int = _env_int("BOX_PREDICT_MAX_FRAMES", "60", min_value=0)
    BOX_MATCH_IOU: float = _env_float("BOX_MATCH_IOU", "0.20", min_value=0.0, max_value=1.0)
    BOX_MATCH_CENTER_RATIO: float = _env_float("BOX_MATCH_CENTER_RATIO", "0.45", min_value=0.0)
    FIRE_MISSING_GRACE_SECONDS: float = _env_float("FIRE_MISSING_GRACE_SECONDS", "1.0", min_value=0.0)

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
    } | {
        name.strip().lower()
        for name in (FIRE_CLASS_NAME, SMOKE_CLASS_NAME)
        if name and name.strip()
    }

    # ──────────────────────────────────────────────
    # Ngưỡng độ tin cậy
    # ──────────────────────────────────────────────
    CONFIDENCE_THRESHOLD: float = _env_float("CONFIDENCE_THRESHOLD", "0.25", min_value=0.0, max_value=1.0)
    FIRE_CONFIDENCE_THRESHOLD: float = _env_float("FIRE_CONFIDENCE_THRESHOLD", "0.01", min_value=0.0, max_value=1.0)
    SMOKE_CONFIDENCE_THRESHOLD: float = _env_float("SMOKE_CONFIDENCE_THRESHOLD", "0.01", min_value=0.0, max_value=1.0)

    # ──────────────────────────────────────────────
    # Quy tắc xác nhận cháy
    # ──────────────────────────────────────────────
    FIRE_CONFIRM_SECONDS: float = _env_float("FIRE_CONFIRM_SECONDS", "5", min_value=0.1)

    # ──────────────────────────────────────────────
    # Telegram & Webhook API
    # ──────────────────────────────────────────────
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
    WEBHOOK_URL: str = os.getenv("WEBHOOK_URL", "")
    ALERT_COOLDOWN_SECONDS: float = _env_float("ALERT_COOLDOWN_SECONDS", "60", min_value=0.0)
    ALERT_SEND_RETRIES: int = _env_int("ALERT_SEND_RETRIES", "2", min_value=1, max_value=5)
    ALERT_RETRY_BACKOFF_SECONDS: float = _env_float("ALERT_RETRY_BACKOFF_SECONDS", "0.5", min_value=0.0)

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
    CAMERA_INDEX: int = _env_int("CAMERA_INDEX", "0", min_value=0)
    FRAME_WIDTH: int = _env_int("FRAME_WIDTH", "1280", min_value=0)
    FRAME_HEIGHT: int = _env_int("FRAME_HEIGHT", "720", min_value=0)

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
    EVENT_DIR: str = os.getenv("EVENT_DIR", _default_runtime_path("events"))
    EVENT_DB_PATH: str = os.getenv("EVENT_DB_PATH", os.path.join(EVENT_DIR, "events.db"))
    EVENT_SNAPSHOT_DIR: str = os.getenv("EVENT_SNAPSHOT_DIR", os.path.join(EVENT_DIR, "snapshots"))
    EXPORT_DIR: str = os.getenv("EXPORT_DIR", _default_runtime_path("exports"))

    # MQTT alerts are optional to keep edge deployments lightweight.
    MQTT_ENABLED: bool = _env_bool("MQTT_ENABLED", "false")
    MQTT_HOST: str = os.getenv("MQTT_HOST", "")
    MQTT_PORT: int = _env_int("MQTT_PORT", "1883", min_value=1, max_value=65535)
    MQTT_TOPIC: str = os.getenv("MQTT_TOPIC", "camera_ai/alerts")
    MQTT_CLIENT_ID: str = os.getenv("MQTT_CLIENT_ID", "camera-ai-edge")

    # Lightweight hardware resource monitor.
    RESOURCE_MONITOR_ENABLED: bool = _env_bool("RESOURCE_MONITOR_ENABLED", "true")
    RESOURCE_MONITOR_INTERVAL_SECONDS: float = _env_float("RESOURCE_MONITOR_INTERVAL_SECONDS", "2.0", min_value=0.5)
    RESOURCE_MONITOR_GPU_INDEX: int = _env_int("RESOURCE_MONITOR_GPU_INDEX", "0", min_value=0)

    # Health checks, alert thresholds, and local RBAC credentials.
    CAMERA_HEALTH_CHECK_INTERVAL_SECONDS: float = _env_float("CAMERA_HEALTH_CHECK_INTERVAL_SECONDS", "1.0", min_value=0.1)
    CAMERA_DISCONNECT_TIMEOUT_SECONDS: float = _env_float("CAMERA_DISCONNECT_TIMEOUT_SECONDS", "30.0", min_value=0.0)
    CAMERA_RECONNECT_BASE_DELAY_SECONDS: float = _env_float("CAMERA_RECONNECT_BASE_DELAY_SECONDS", "1.0", min_value=0.01)
    CAMERA_RECONNECT_MAX_DELAY_SECONDS: float = _env_float("CAMERA_RECONNECT_MAX_DELAY_SECONDS", "30.0", min_value=0.01)
    GPU_ALERT_USAGE_THRESHOLD_PERCENT: float = _env_float("GPU_ALERT_USAGE_THRESHOLD_PERCENT", "90.0", min_value=0.0, max_value=100.0)
    GPU_ALERT_CONSECUTIVE_SAMPLES: int = _env_int("GPU_ALERT_CONSECUTIVE_SAMPLES", "3", min_value=1)
    INFERENCE_LATENCY_ALERT_THRESHOLD_MS: float = _env_float("INFERENCE_LATENCY_ALERT_THRESHOLD_MS", "250.0", min_value=0.0)
    INFERENCE_LATENCY_ALERT_CONSECUTIVE_SAMPLES: int = _env_int("INFERENCE_LATENCY_ALERT_CONSECUTIVE_SAMPLES", "3", min_value=1)
    SYSTEM_ALERT_REPEAT_COOLDOWN_SECONDS: float = _env_float("SYSTEM_ALERT_REPEAT_COOLDOWN_SECONDS", "60.0", min_value=0.0)
    AUTH_ADMIN_USERNAME: str = os.getenv("AUTH_ADMIN_USERNAME", "admin")
    AUTH_ADMIN_PASSWORD: str = os.getenv("AUTH_ADMIN_PASSWORD", "")
    AUTH_USER_USERNAME: str = os.getenv("AUTH_USER_USERNAME", "user")
    AUTH_USER_PASSWORD: str = os.getenv("AUTH_USER_PASSWORD", "")
