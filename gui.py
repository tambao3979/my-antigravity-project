"""
gui.py - Camera AI – Hệ Thống Giám Sát Đa Màn Hình
"""

import logging
import os
import sys
import threading
import time
import json
import queue
import unicodedata
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

from config import Config
from logger_config import setup_logging
from detector import ObjectDetector
from event_store import EventStore
from export_utils import export_count_rows_csv, export_count_rows_excel
from fire_tracker import FireTracker
from mqtt_publisher import AsyncMQTTPublisher
from persistent_env import get_persistent_env_path, save_env_value
from resource_monitor import ResourceMetrics, SystemMonitorAgent
from roi_tools import class_counts, filter_detections_by_roi, frame_point_from_widget_event, nearest_vertex
from telegram_notifier import TelegramNotifier

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
#  THEME CONSTANTS
# ─────────────────────────────────────────────────────────────
CLR_BG          = "#0f1117"
CLR_PANEL       = "#1a1d27"
CLR_PANEL2      = "#222535"
CLR_BORDER      = "#2e3250"
CLR_ACCENT      = "#3b5bdb"
CLR_ACCENT_DARK = "#2d47b5"
CLR_GREEN       = "#12b886"
CLR_YELLOW      = "#fab005"
CLR_RED         = "#fa5252"
CLR_TEXT        = "#e9ecef"
CLR_TEXT_DIM    = "#868e96"
CLR_CAM_SEL     = "#3b5bdb"
CLR_CAM_IDLE    = "#2a2d3e"

# ─────────────────────────────────────────────────────────────
#  APP CONSTANTS (extracted from magic numbers)
# ─────────────────────────────────────────────────────────────
MAX_RECONNECT_RETRIES = 5          # Số lần retry tối đa khi camera mất kết nối
RECONNECT_DELAY_SEC = 3            # Delay giữa các lần reconnect

RING_BUFFER_SIZE = 150             # ~5 giây buffer ở 30fps
RECORD_FUTURE_FRAMES = 150        # ~5 giây ghi thêm sau sự cố
RECORD_FPS = 30.0                  # FPS khi ghi video sự cố
VIDEO_ASPECT_RATIO = 9 / 16       # Tỉ lệ chiều cao/rộng (16:9)
CAM_TILE_HEADER_HEIGHT = 30       # Chiều cao header mỗi camera tile (px)
RENDER_INTERVAL_MS = 30            # Khoảng cách giữa các lần render (ms)
IDLE_RENDER_INTERVAL_MS = 50       # Render interval khi idle
MIN_CELL_WIDTH = 200               # Chiều rộng tối thiểu mỗi camera cell
SIDE_PANEL_TOTAL_WIDTH = 550       # Ước tính tổng width 2 side panel

# Login credentials
LOGIN_USERNAME = "admin"
LOGIN_PASSWORD = "admin"
MAX_LOGIN_ATTEMPTS = 3
LOGIN_LOCKOUT_SECONDS = 5

_OBJECT_PALETTE = [
    (255, 180,   0), (  0, 220, 100), (200,   0, 255),
    (  0, 180, 255), (120,   0, 200), (  0, 255, 180),
    (255,  80, 180), ( 30, 200, 255), (200, 120,   0),
    (  0,  80, 255),
]


class _TextboxHandler(logging.Handler):
    def __init__(self, message_queue: "queue.SimpleQueue[str]"):
        super().__init__()
        self.message_queue = message_queue

    def emit(self, record: logging.LogRecord):
        try:
            self.message_queue.put_nowait(self.format(record) + "\n")
        except Exception:
            pass

class _StdoutRedirect:
    def __init__(self, message_queue: "queue.SimpleQueue[str]"):
        self.message_queue = message_queue

    def write(self, s: str):
        if not s.strip():
            return
        try:
            self.message_queue.put_nowait(s)
        except Exception:
            pass

    def flush(self): pass


# ─────────────────────────────────────────────────────────────
#  LOGIN APP  (root Tk riêng — tránh hoàn toàn zero-window bug)
# ─────────────────────────────────────────────────────────────

class LoginApp(ctk.CTk):
    """
    Cửa sổ đăng nhập chạy như một Tk root độc lập.
    Sau khi mainloop() kết thúc, kiểm tra self.authenticated để
    quyết định có khởi động SecurityApp hay không.
    """

    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("🔐 Đăng Nhập – Camera AI")
        self.geometry("420x500")
        self.resizable(False, False)
        self.configure(fg_color=CLR_BG)

        # Đặt icon taskbar / tiêu đề cửa sổ
        _ico_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.ico")
        if os.path.exists(_ico_path):
            try:
                self.iconbitmap(_ico_path)
            except Exception:
                pass

        self.authenticated = False
        self._attempts = 0
        self._locked_until = 0.0

        # Center on screen
        self.update_idletasks()
        x = (self.winfo_screenwidth() - 420) // 2
        y = (self.winfo_screenheight() - 500) // 2
        self.geometry(f"420x500+{x}+{y}")

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build_ui()

    def _build_ui(self):
        # Logo area
        logo_frame = ctk.CTkFrame(self, fg_color="transparent")
        logo_frame.pack(pady=(40, 10))

        # Hiển thị logo công ty từ file ICO
        _ico_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.ico")
        _logo_shown = False
        if os.path.exists(_ico_path):
            try:
                _pil_img = Image.open(_ico_path).convert("RGBA").resize((72, 72), Image.LANCZOS)
                self._login_logo_img = ctk.CTkImage(light_image=_pil_img, dark_image=_pil_img, size=(72, 72))
                ctk.CTkLabel(logo_frame, image=self._login_logo_img, text="").pack()
                _logo_shown = True
            except Exception:
                pass
        if not _logo_shown:
            ctk.CTkLabel(logo_frame, text="⬤", font=ctk.CTkFont(size=36), text_color=CLR_RED).pack()

        ctk.CTkLabel(logo_frame, text="Camera AI", font=ctk.CTkFont(family="Segoe UI", size=28, weight="bold"), text_color=CLR_TEXT).pack(pady=(8, 0))
        ctk.CTkLabel(logo_frame, text="Hệ Thống Giám Sát Đa Màn Hình", font=ctk.CTkFont(family="Segoe UI", size=13), text_color=CLR_TEXT_DIM).pack(pady=(2, 0))

        # Divider
        ctk.CTkFrame(self, fg_color=CLR_BORDER, height=1).pack(fill="x", padx=40, pady=(20, 20))

        # Form
        form = ctk.CTkFrame(self, fg_color="transparent")
        form.pack(padx=40, fill="x")

        ctk.CTkLabel(form, text="👤  Tên đăng nhập", font=ctk.CTkFont(size=12, weight="bold"), text_color=CLR_TEXT_DIM, anchor="w").pack(fill="x", pady=(0, 4))
        self.entry_user = ctk.CTkEntry(form, height=40, fg_color=CLR_PANEL2, border_color=CLR_BORDER, text_color=CLR_TEXT, placeholder_text="Nhập tên đăng nhập...", font=ctk.CTkFont(size=13))
        self.entry_user.pack(fill="x", pady=(0, 12))

        ctk.CTkLabel(form, text="🔒  Mật khẩu", font=ctk.CTkFont(size=12, weight="bold"), text_color=CLR_TEXT_DIM, anchor="w").pack(fill="x", pady=(0, 4))
        self.entry_pass = ctk.CTkEntry(form, height=40, fg_color=CLR_PANEL2, border_color=CLR_BORDER, text_color=CLR_TEXT, placeholder_text="Nhập mật khẩu...", show="●", font=ctk.CTkFont(size=13))
        self.entry_pass.pack(fill="x", pady=(0, 20))

        # Bind Enter key
        self.entry_pass.bind("<Return>", lambda e: self._do_login())
        self.entry_user.bind("<Return>", lambda e: self.entry_pass.focus())

        self.btn_login = ctk.CTkButton(
            form, text="Đăng Nhập", height=42,
            fg_color=CLR_ACCENT, hover_color=CLR_ACCENT_DARK,
            font=ctk.CTkFont(size=14, weight="bold"), corner_radius=8,
            command=self._do_login
        )
        self.btn_login.pack(fill="x")

        # Error label
        self.lbl_error = ctk.CTkLabel(self, text="", font=ctk.CTkFont(size=12), text_color=CLR_RED)
        self.lbl_error.pack(pady=(12, 0))

        self.entry_user.focus()

    def _do_login(self):
        now = time.time()
        if now < self._locked_until:
            remaining = int(self._locked_until - now)
            self.lbl_error.configure(text=f"⏳ Tài khoản bị khóa. Thử lại sau {remaining}s...")
            return

        username = self.entry_user.get().strip()
        password = self.entry_pass.get().strip()

        if username == LOGIN_USERNAME and password == LOGIN_PASSWORD:
            self.authenticated = True
            self.destroy()   # Thoát mainloop login một cách sạch — không có 'zero window' vì đây là root
        else:
            self._attempts += 1
            remaining = MAX_LOGIN_ATTEMPTS - self._attempts
            if remaining > 0:
                self.lbl_error.configure(text=f"❌ Sai tên đăng nhập hoặc mật khẩu! (Còn {remaining} lần)")
            else:
                self._locked_until = now + LOGIN_LOCKOUT_SECONDS
                self._attempts = 0
                self.lbl_error.configure(text=f"🔒 Sai quá nhiều lần! Khóa {LOGIN_LOCKOUT_SECONDS}s...")
            self._shake_window()

    def _shake_window(self):
        """Micro-animation khi nhập sai."""
        orig_x = self.winfo_x()
        orig_y = self.winfo_y()
        offsets = [10, -10, 8, -8, 5, -5, 3, -3, 0]
        for i, dx in enumerate(offsets):
            self.after(i * 40, lambda d=dx: self.geometry(f"+{orig_x + d}+{orig_y}"))

    def _on_close(self):
        """User bấm X → không đăng nhập → đóng hoàn toàn."""
        self.authenticated = False
        self.destroy()


class CameraStream:
    """Quản lý một luồng camera (đọc frame độc lập, tracker độc lập)."""
    def __init__(self, source):
        self.source = source
        self.source_str = str(source)
        self.display_name = os.path.basename(self.source_str) if os.path.isfile(self.source_str) else self.source_str
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.tracker = FireTracker()
        
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_detections: list = []
        self.inference_latency_ms = 0.0
        self.last_inference_at = 0.0
        self.last_counts: dict = {}
        self.roi_points: list[tuple[int, int]] = []
        self.roi_enabled = False
        self.roi_drag_index: Optional[int] = None
        
        self.fps = 0.0
        self._fps_start = time.time()
        self._frame_count = 0
        self.frame_id = 0
        self.is_video_file = False
        self.is_paused = False
        self.ended = False
        self.capture_backend = "unknown"
        
        # Ring Buffer & Event Recording
        import collections
        self.frame_buffer = collections.deque(maxlen=RING_BUFFER_SIZE)
        self.is_recording = False
        self.record_frames_left = 0
        self.record_cooldown = 0

        # Manual Recording
        self.manual_recording = False
        self.manual_rec_buffer: list = []

        # Error & Reconnect
        self.reconnect_count = 0
        self.has_error = False
        self.error_message = ""

    def _is_local_video_source(self, src_val) -> bool:
        return isinstance(src_val, str) and not src_val.startswith(("http://", "https://", "rtsp://"))

    def _backend_label(self, cap, fallback: str) -> str:
        try:
            backend_name = cap.getBackendName()
            if backend_name:
                return backend_name
        except Exception:
            pass
        return fallback

    def _open_capture(self, src_val):
        self.is_video_file = self._is_local_video_source(src_val)

        attempts = []
        if self.is_video_file and os.name == "nt":
            ffmpeg_backend = getattr(cv2, "CAP_FFMPEG", None)
            if ffmpeg_backend is not None:
                attempts.append((ffmpeg_backend, "FFMPEG"))
        attempts.append((None, "DEFAULT"))

        for backend_id, backend_label in attempts:
            try:
                cap = cv2.VideoCapture(src_val) if backend_id is None else cv2.VideoCapture(src_val, backend_id)
            except Exception as exc:
                logger.warning("⚠️ Không thể khởi tạo backend %s cho [%s]: %s", backend_label, self.source, exc)
                continue

            if cap is not None and cap.isOpened():
                self.capture_backend = self._backend_label(cap, backend_label)
                logger.info("🎞️ Nguồn [%s] đang dùng backend: %s", self.source, self.capture_backend)
                return cap

            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass

        self.capture_backend = "unopened"
        return None

    def start(self):
        src_val = int(self.source) if self.source.isdigit() else self.source.strip('"').strip("'")
        self.cap = self._open_capture(src_val)
        if self.cap is None:
            logger.error("Không thể mở nguồn: %s", self.source)
            self.cap = None
            return False

        self.video_fps = RECORD_FPS
        if self.is_video_file:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                self.video_fps = fps
        self.frame_delay = 1.0 / self.video_fps
        if Config.FRAME_WIDTH > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)

        self.is_running = True
        self.reconnect_count = 0
        self.has_error = False
        self.error_message = ""
        self.ended = False
        self.tracker.reset()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        logger.info("✅ Kết nối thành công: %s", self.source)
        return True

    def _capture_loop(self):
        last_read_time = time.time()
        while self.is_running and self.cap is not None:
            if self.is_paused:
                time.sleep(0.05)
                if not self.is_video_file:
                    self.cap.grab()
                continue
                
            ret, frame = self.cap.read()
            if not ret:
                if self.is_video_file:
                    logger.info("🎬 Phát hết video [%s].", self.source)
                    self.ended = True
                    self.is_running = False
                    break

                # Reconnect với retry limit
                self.reconnect_count += 1
                if self.reconnect_count > MAX_RECONNECT_RETRIES:
                    self.has_error = True
                    self.error_message = f"Mất kết nối sau {MAX_RECONNECT_RETRIES} lần thử"
                    logger.error("❌ [%s] Ngắt kết nối vĩnh viễn sau %d lần retry.", self.source, MAX_RECONNECT_RETRIES)
                    self.is_running = False
                    break

                self.has_error = True
                self.error_message = f"Đang thử kết nối lại ({self.reconnect_count}/{MAX_RECONNECT_RETRIES})..."
                logger.warning("⚠️ [%s] Mất kết nối. Thử lại %d/%d sau %ds...", self.source, self.reconnect_count, MAX_RECONNECT_RETRIES, RECONNECT_DELAY_SEC)
                time.sleep(RECONNECT_DELAY_SEC)
                if self.cap:
                    self.cap.release()
                src_val = int(self.source) if self.source.isdigit() else self.source.strip('"').strip("'")
                self.cap = self._open_capture(src_val)
                continue

            # Reset error state khi đọc frame thành công
            if self.has_error:
                self.has_error = False
                self.error_message = ""
                self.reconnect_count = 0
                logger.info("✅ [%s] Kết nối lại thành công!", self.source)

            # Giới hạn kích thước frame để tránh tràn RAM (OOM)
            h, w = frame.shape[:2]
            MAX_W = 1280
            if w > MAX_W:
                scale = MAX_W / w
                frame = cv2.resize(frame, (MAX_W, int(h * scale)))

            self.latest_frame = frame
            self.frame_id += 1
            if not self.is_paused:
                self.frame_buffer.append(frame.copy())
            
            # Tính FPS camera
            self._frame_count += 1
            now = time.time()
            if now - self._fps_start >= 1.0:
                self.fps = self._frame_count / (now - self._fps_start)
                self._frame_count = 0
                self._fps_start = now
                
            if self.is_video_file:
                elapsed = time.time() - last_read_time
                if elapsed < self.frame_delay:
                    time.sleep(self.frame_delay - elapsed)
                last_read_time = time.time()
            else:
                time.sleep(0.01)

    def stop(self):
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None
        self.thread = None


class SecurityApp(ctk.CTk):

    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.configure(fg_color=CLR_BG)

        self.title("🔥 Camera AI – Hệ Thống Giám Sát")
        self.geometry("1400x820")

        # Đặt icon taskbar / tiêu đề cửa sổ chính
        _ico_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.ico")
        if os.path.exists(_ico_path):
            try:
                self.iconbitmap(_ico_path)
            except Exception:
                pass
        self.minsize(1100, 680)

        # ── State ──
        self.camera_sources = [str(Config.CAMERA_INDEX)]
        # Load from history
        self._history_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_history.json")
        if os.path.exists(self._history_file):
            try:
                with open(self._history_file, 'r', encoding='utf-8') as f:
                    saved_sources = json.load(f)
                    for src in saved_sources:
                        if src not in self.camera_sources:
                            self.camera_sources.append(src)
            except Exception as e:
                logger.warning(f"Không thể tải lịch sử camera: {e}")
                
        self.active_sources_vars: Dict[str, ctk.BooleanVar] = {}

        self.streams: Dict[str, CameraStream] = {}
        self.video_labels: Dict[str, ctk.CTkLabel] = {}
        self._record_buttons: Dict[str, ctk.CTkButton] = {}
        
        self._video_frames: Dict[str, ctk.CTkFrame] = {}
        self._grid_configs: Dict[str, dict] = {}
        self._zoomed_src: Optional[str] = None
        self._retained_video_caps: list = []
        self._last_zoom_image = None  # Giữ reference CTkImage tránh GC sớm gây check_dpi_scaling crash
        self._ui_log_queue: "queue.SimpleQueue[str]" = queue.SimpleQueue()
        self._resource_metrics_queue: "queue.Queue[ResourceMetrics]" = queue.Queue(maxsize=1)
        self._display_maps: Dict[str, dict] = {}
        self._count_history: list[dict] = []
        self._last_count_sample = 0.0

        self.class_colors: dict = {}
        self.class_visibility: dict = {}
        self._filter_widgets: dict = {}
        
        self.camera_configs: dict = {
            "Tất cả Camera": {
                "fire_conf": Config.FIRE_CONFIDENCE_THRESHOLD,
                "def_conf": Config.CONFIDENCE_THRESHOLD,
                "iou": Config.YOLO_IOU,
                "class_visibility": {}
            }
        }
        self.config_target_var = ctk.StringVar(value="Tất cả Camera")
        self.roi_mode_var = ctk.BooleanVar(value=False)

        self.is_inferencing = False
        self._inference_thread: Optional[threading.Thread] = None
        self._cached_allowed_ids: Optional[set] = None
        self.event_store = EventStore(Config.EVENT_DB_PATH, Config.EVENT_SNAPSHOT_DIR)
        self.mqtt_publisher = AsyncMQTTPublisher(
            host=Config.MQTT_HOST,
            port=Config.MQTT_PORT,
            topic=Config.MQTT_TOPIC,
            client_id=Config.MQTT_CLIENT_ID,
            enabled=Config.MQTT_ENABLED,
        )
        self.system_monitor_agent: Optional[SystemMonitorAgent] = None
        if Config.RESOURCE_MONITOR_ENABLED:
            self.system_monitor_agent = SystemMonitorAgent(
                output_queue=self._resource_metrics_queue,
                poll_interval=Config.RESOURCE_MONITOR_INTERVAL_SECONDS,
                gpu_index=Config.RESOURCE_MONITOR_GPU_INDEX,
            )

        # ── Layout ──
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0)

        self._build_header()
        self._build_left_panel()
        self._build_video_panel()
        self._build_right_panel()
        self._build_status_bar()

        # Gắn Handler giao diện để xem log trên app (tạo Log Box)
        # Các stream Logging chuẩn đã được xử lý bởi setup_logging gốc
        handler = _TextboxHandler(self._ui_log_queue)
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
        logging.getLogger().addHandler(handler)
        
        # Vẫn dùng để catch sys.stdout/stderr lên UI
        sys.stdout = _StdoutRedirect(self._ui_log_queue)
        sys.stderr = _StdoutRedirect(self._ui_log_queue)
        self.after(100, self._drain_log_queue)
        self.after(500, self._drain_resource_metrics_queue)

        logger.info("=" * 50)
        logger.info("🔐 Đăng nhập thành công: %s @ %s", LOGIN_USERNAME, datetime.now().strftime("%H:%M:%S %d/%m/%Y"))
        logger.info("✅ Giao diện Camera AI đã khởi động")
        if self.system_monitor_agent:
            self.system_monitor_agent.start()
            logger.info("SystemMonitorAgent running in background, interval %.1fs", Config.RESOURCE_MONITOR_INTERVAL_SECONDS)
        logger.info("⏳ Đang nạp model YOLO...")
        self.detector = ObjectDetector()
        self.notifier = TelegramNotifier()
        logger.info("✅ Model AI đã sẵn sàng!")
        logger.info("=" * 50)

        self._refresh_camera_list()
        self._tick_clock()

        for class_name in Config.CLASS_REGISTRY:
            self._register_class(class_name)

        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Bắt đầu vòng lặp render chính (defer để window kịp vẽ xong)
        self.after(100, self._render_loop)

    # ════════════════════════════════════════════════════════
    #  BUILD UI
    # ════════════════════════════════════════════════════════

    def _build_header(self):
        hdr = ctk.CTkFrame(self, fg_color=CLR_PANEL, height=56, corner_radius=0)
        hdr.grid(row=0, column=0, columnspan=3, sticky="ew")
        hdr.grid_columnconfigure(1, weight=1)
        hdr.grid_propagate(False)

        logo = ctk.CTkFrame(hdr, fg_color="transparent")
        logo.grid(row=0, column=0, padx=(20, 0), pady=8, sticky="w")

        # Hiển thị logo công ty ở header
        _ico_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.ico")
        _header_logo_shown = False
        if os.path.exists(_ico_path):
            try:
                _pil_img = Image.open(_ico_path).convert("RGBA").resize((36, 36), Image.LANCZOS)
                self._header_logo_img = ctk.CTkImage(light_image=_pil_img, dark_image=_pil_img, size=(36, 36))
                ctk.CTkLabel(logo, image=self._header_logo_img, text="").pack(side="left", padx=(0, 8))
                _header_logo_shown = True
            except Exception:
                pass
        if not _header_logo_shown:
            ctk.CTkLabel(logo, text="⬤", font=ctk.CTkFont(size=18), text_color=CLR_RED).pack(side="left", padx=(0, 6))

        ctk.CTkLabel(logo, text="Camera AI", font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"), text_color=CLR_TEXT).pack(side="left")
        ctk.CTkLabel(logo, text=" · Hệ Thống Giám Sát Đa Màn Hình", font=ctk.CTkFont(family="Segoe UI", size=13), text_color=CLR_TEXT_DIM).pack(side="left")

        # Thêm khung cho các nút mở thư mục
        btn_frame = ctk.CTkFrame(hdr, fg_color="transparent")
        btn_frame.grid(row=0, column=1, padx=20, sticky="e")
        
        btn_events = ctk.CTkButton(btn_frame, text="📁 Events (Video)", width=120, height=28, font=ctk.CTkFont(size=12, weight="bold"), fg_color=CLR_PANEL2, hover_color="#3a3d4e", command=self._open_events_folder)
        btn_events.pack(side="right", padx=5)
        
        btn_logs = ctk.CTkButton(btn_frame, text="📁 Logs (Lịch sử)", width=120, height=28, font=ctk.CTkFont(size=12, weight="bold"), fg_color=CLR_PANEL2, hover_color="#3a3d4e", command=self._open_logs_folder)
        btn_logs.pack(side="right", padx=5)

        btn_export_csv = ctk.CTkButton(btn_frame, text="CSV", width=54, height=28, font=ctk.CTkFont(size=12, weight="bold"), fg_color=CLR_PANEL2, hover_color="#3a3d4e", command=self._export_counts_csv)
        btn_export_csv.pack(side="right", padx=5)

        btn_export_xlsx = ctk.CTkButton(btn_frame, text="XLSX", width=58, height=28, font=ctk.CTkFont(size=12, weight="bold"), fg_color=CLR_PANEL2, hover_color="#3a3d4e", command=self._export_counts_excel)
        btn_export_xlsx.pack(side="right", padx=5)

        self.clock_label = ctk.CTkLabel(hdr, text="", font=ctk.CTkFont(family="Segoe UI Mono", size=14), text_color=CLR_TEXT_DIM)
        self.clock_label.grid(row=0, column=2, padx=20, pady=8, sticky="e")

    def _build_left_panel(self):
        panel = ctk.CTkFrame(self, fg_color=CLR_PANEL, width=250, corner_radius=12)
        panel.grid(row=1, column=0, sticky="nsew", padx=(12, 6), pady=(10, 6))
        panel.grid_propagate(False)
        panel.grid_rowconfigure(3, weight=1)
        panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(panel, text="📷  NGUỒN CAMERA", font=ctk.CTkFont(size=11, weight="bold"), text_color=CLR_TEXT_DIM).grid(row=0, column=0, padx=14, pady=(14, 6), sticky="w")

        add_frame = ctk.CTkFrame(panel, fg_color=CLR_PANEL2, corner_radius=8)
        add_frame.grid(row=1, column=0, padx=12, pady=(0, 6), sticky="ew")
        add_frame.grid_columnconfigure(0, weight=1)

        self.add_cam_entry = ctk.CTkEntry(add_frame, placeholder_text="ID / File / RTSP...", fg_color="#2a2d3e", border_color=CLR_BORDER, text_color=CLR_TEXT, font=ctk.CTkFont(size=12))
        self.add_cam_entry.grid(row=0, column=0, padx=8, pady=(8, 4), sticky="ew")

        self.btn_add_cam = ctk.CTkButton(add_frame, text="+ Thêm Camera", height=30, fg_color=CLR_ACCENT, hover_color=CLR_ACCENT_DARK, font=ctk.CTkFont(size=12, weight="bold"), corner_radius=6, command=self._add_camera)
        self.btn_add_cam.grid(row=1, column=0, padx=8, pady=(0, 8), sticky="ew")

        self.camera_list_frame = ctk.CTkScrollableFrame(panel, fg_color="transparent", corner_radius=8)
        self.camera_list_frame.grid(row=3, column=0, padx=8, pady=4, sticky="nsew")
        self.camera_list_frame.grid_columnconfigure(0, weight=1)

        ctrl = ctk.CTkFrame(panel, fg_color="transparent")
        ctrl.grid(row=4, column=0, padx=12, pady=(4, 16), sticky="ew")
        ctrl.grid_columnconfigure(0, weight=1)
        ctrl.grid_columnconfigure(1, weight=1)
        ctrl.grid_columnconfigure(2, weight=1)

        self.btn_start = ctk.CTkButton(ctrl, text="▶ Bắt Đầu", height=38, fg_color=CLR_GREEN, hover_color="#0ca678", font=ctk.CTkFont(size=12, weight="bold"), corner_radius=8, command=self.start_cameras)
        self.btn_start.grid(row=0, column=0, padx=(0, 2), sticky="ew")

        self.btn_pause = ctk.CTkButton(ctrl, text="⏸ Tạm Dừng", height=38, fg_color=CLR_YELLOW, text_color="black", hover_color="#f0a500", font=ctk.CTkFont(size=12, weight="bold"), corner_radius=8, state="disabled", command=self.toggle_pause)
        self.btn_pause.grid(row=0, column=1, padx=(2, 2), sticky="ew")

        self.btn_stop = ctk.CTkButton(ctrl, text="■ Dừng", height=38, fg_color=CLR_RED, hover_color="#e03131", font=ctk.CTkFont(size=12, weight="bold"), corner_radius=8, state="disabled", command=self.stop_all_cameras)
        self.btn_stop.grid(row=0, column=2, padx=(2, 0), sticky="ew")

        # ── Cấu Hình Zalo OA API ──
        ctk.CTkLabel(panel, text="📱  ZALO OA API", font=ctk.CTkFont(size=11, weight="bold"), text_color=CLR_TEXT_DIM).grid(row=5, column=0, padx=14, pady=(8, 4), sticky="w")

        api_frame = ctk.CTkFrame(panel, fg_color=CLR_PANEL2, corner_radius=8)
        api_frame.grid(row=6, column=0, padx=12, pady=(0, 14), sticky="ew")
        api_frame.grid_columnconfigure(0, weight=1)

        # ── OA Access Token ──
        lbl_token = ctk.CTkFrame(api_frame, fg_color="transparent")
        lbl_token.grid(row=0, column=0, padx=8, pady=(8, 0), sticky="ew")
        ctk.CTkLabel(lbl_token, text="OA Access Token", font=ctk.CTkFont(size=10, weight="bold"), text_color=CLR_TEXT_DIM, anchor="w").pack(side="left")

        entry_row_token = ctk.CTkFrame(api_frame, fg_color="transparent")
        entry_row_token.grid(row=1, column=0, padx=8, pady=(2, 4), sticky="ew")
        entry_row_token.grid_columnconfigure(0, weight=1)
        self.zalo_token_entry = ctk.CTkEntry(
            entry_row_token, placeholder_text="Nhập OA Access Token...",
            fg_color="#2a2d3e", border_color=CLR_BORDER,
            text_color=CLR_TEXT, font=ctk.CTkFont(size=11), show="•"
        )
        self.zalo_token_entry.grid(row=0, column=0, sticky="ew")
        ctk.CTkButton(
            entry_row_token, text="✕", width=28, height=28,
            fg_color="#3a1a1a", hover_color=CLR_RED,
            text_color=CLR_TEXT_DIM, font=ctk.CTkFont(size=12),
            command=lambda: self._clear_api_field(self.zalo_token_entry, "ZALO_OA_TOKEN")
        ).grid(row=0, column=1, padx=(4, 0))

        # ── Zalo User ID ──
        lbl_uid = ctk.CTkFrame(api_frame, fg_color="transparent")
        lbl_uid.grid(row=2, column=0, padx=8, pady=(4, 0), sticky="ew")
        ctk.CTkLabel(lbl_uid, text="User ID (người nhận)", font=ctk.CTkFont(size=10, weight="bold"), text_color=CLR_TEXT_DIM, anchor="w").pack(side="left")

        entry_row_uid = ctk.CTkFrame(api_frame, fg_color="transparent")
        entry_row_uid.grid(row=3, column=0, padx=8, pady=(2, 4), sticky="ew")
        entry_row_uid.grid_columnconfigure(0, weight=1)
        self.zalo_uid_entry = ctk.CTkEntry(
            entry_row_uid, placeholder_text="Nhập Zalo User ID...",
            fg_color="#2a2d3e", border_color=CLR_BORDER,
            text_color=CLR_TEXT, font=ctk.CTkFont(size=11)
        )
        self.zalo_uid_entry.grid(row=0, column=0, sticky="ew")
        ctk.CTkButton(
            entry_row_uid, text="✕", width=28, height=28,
            fg_color="#3a1a1a", hover_color=CLR_RED,
            text_color=CLR_TEXT_DIM, font=ctk.CTkFont(size=12),
            command=lambda: self._clear_api_field(self.zalo_uid_entry, "ZALO_USER_ID")
        ).grid(row=0, column=1, padx=(4, 0))

        # ── Nút Lưu / Xóa tất cả ──
        btn_row = ctk.CTkFrame(api_frame, fg_color="transparent")
        btn_row.grid(row=4, column=0, padx=8, pady=(0, 8), sticky="ew")
        btn_row.grid_columnconfigure(0, weight=1)
        btn_row.grid_columnconfigure(1, weight=1)

        self.btn_save_api = ctk.CTkButton(
            btn_row, text="💾 Lưu API", height=30,
            fg_color="#1864ab", hover_color="#1c7ed6",
            font=ctk.CTkFont(size=11, weight="bold"), corner_radius=6,
            command=self._save_api_config
        )
        self.btn_save_api.grid(row=0, column=0, padx=(0, 3), sticky="ew")

        ctk.CTkButton(
            btn_row, text="🗑 Xóa API", height=30,
            fg_color="#3a1a1a", hover_color=CLR_RED,
            font=ctk.CTkFont(size=11, weight="bold"), corner_radius=6,
            command=self._clear_all_api
        ).grid(row=0, column=1, padx=(3, 0), sticky="ew")

        # Load giá trị hiện tại từ Config vào các ô nhập
        if getattr(Config, "ZALO_OA_TOKEN", ""):
            self.zalo_token_entry.insert(0, Config.ZALO_OA_TOKEN)
        if getattr(Config, "ZALO_USER_ID", ""):
            self.zalo_uid_entry.insert(0, Config.ZALO_USER_ID)

    def _save_api_config(self):
        """Lưu Zalo OA Token + User ID vào file cấu hình bền vững và nạp lại runtime."""
        new_token = self.zalo_token_entry.get().strip()
        new_uid   = self.zalo_uid_entry.get().strip()

        # Cập nhật runtime
        Config.ZALO_OA_TOKEN = new_token
        Config.ZALO_USER_ID  = new_uid

        # Ghi vào persistent env để lần mở app sau vẫn còn
        env_path = get_persistent_env_path()
        save_env_value(env_path, "ZALO_OA_TOKEN", new_token)
        save_env_value(env_path, "ZALO_USER_ID", new_uid)

        # Reload notifier credentials
        if hasattr(self, 'notifier') and self.notifier:
            self.notifier.reload_credentials()

        token_preview = f"{new_token[:8]}..." if len(new_token) > 8 else (new_token or "<rỗng>")
        logger.info("✅ Đã lưu Zalo OA API — Token: %s | User ID: %s", token_preview, new_uid or "<rỗng>")
        self.btn_save_api.configure(text="✔ Đã Lưu!", fg_color=CLR_GREEN)
        self.after(2000, lambda: self.btn_save_api.configure(text="💾 Lưu API", fg_color="#1864ab"))

    def _clear_api_field(self, entry: ctk.CTkEntry, env_key: str):
        """Xóa nội dung 1 ô nhập và ghi đè giá trị rỗng vào cấu hình bền vững."""
        entry.delete(0, "end")
        if env_key == "ZALO_OA_TOKEN":
            Config.ZALO_OA_TOKEN = ""
        elif env_key == "ZALO_USER_ID":
            Config.ZALO_USER_ID = ""
        env_path = get_persistent_env_path()
        save_env_value(env_path, env_key, "")
        if hasattr(self, 'notifier') and self.notifier:
            self.notifier.reload_credentials()
        logger.info("🗑 Đã xóa %s khỏi cấu hình.", env_key)

    def _clear_all_api(self):
        """Xóa toàn bộ Zalo API (token + user ID)."""
        self._clear_api_field(self.zalo_token_entry, "ZALO_OA_TOKEN")
        self._clear_api_field(self.zalo_uid_entry,   "ZALO_USER_ID")
        logger.info("🗑 Đã xóa toàn bộ Zalo API.")


    def _build_video_panel(self):
        self.video_panel_container = ctk.CTkFrame(self, fg_color=CLR_PANEL, corner_radius=12)
        self.video_panel_container.grid(row=1, column=1, sticky="nsew", padx=6, pady=(10, 6))
        self.video_panel_container.grid_rowconfigure(1, weight=1)
        self.video_panel_container.grid_columnconfigure(0, weight=1)

        vhdr = ctk.CTkFrame(self.video_panel_container, fg_color=CLR_PANEL2, height=36, corner_radius=0)
        vhdr.grid(row=0, column=0, sticky="ew")
        vhdr.grid_columnconfigure(1, weight=1)
        vhdr.grid_propagate(False)

        ctk.CTkLabel(vhdr, text="  🎥  HỆ THỐNG MÀN HÌNH", font=ctk.CTkFont(size=12, weight="bold"), text_color=CLR_TEXT_DIM).grid(row=0, column=0, padx=8, sticky="w")
        
        self.feed_count_label = ctk.CTkLabel(vhdr, text="0 camera đang bật", font=ctk.CTkFont(size=11), text_color=CLR_TEXT_DIM)
        self.feed_count_label.grid(row=0, column=1, padx=8, sticky="e")

        # Scrollable grid container cho nhiều camera
        self.video_grid = ctk.CTkScrollableFrame(self.video_panel_container, fg_color="black", corner_radius=0)
        self.video_grid.grid(row=1, column=0, sticky="nsew")

        # Khung Overlay dùng cho chức năng phóng to (Zoom)
        self.zoom_overlay = ctk.CTkFrame(self.video_panel_container, fg_color="black", corner_radius=0)
        self.zoom_overlay.grid_rowconfigure(1, weight=1)
        self.zoom_overlay.grid_columnconfigure(0, weight=1)
        
        self.zoom_hdr = ctk.CTkFrame(self.zoom_overlay, fg_color="transparent", height=24)
        self.zoom_hdr.grid(row=0, column=0, sticky="ew")
        self.zoom_hdr.grid_columnconfigure(0, weight=1)
        
        self.zoom_lbl_title = ctk.CTkLabel(self.zoom_hdr, text="🔍 Chế độ phóng to", font=ctk.CTkFont(size=12, weight="bold"), text_color=CLR_TEXT_DIM, cursor="hand2")
        self.zoom_lbl_title.grid(row=0, column=0, sticky="w", padx=8)
        self.zoom_lbl_title.bind("<Button-1>", lambda e: self._toggle_zoom(self._zoomed_src) if self._zoomed_src else None)
        
        btn_close_zoom = ctk.CTkButton(self.zoom_hdr, text="❌ Đóng", width=60, height=20, fg_color=CLR_RED, hover_color="#e03131", command=lambda: self._toggle_zoom(self._zoomed_src) if self._zoomed_src else None)
        btn_close_zoom.grid(row=0, column=1, padx=8)
        
        self.zoom_lbl_video = ctk.CTkLabel(self.zoom_overlay, text="")
        self.zoom_lbl_video.grid(row=1, column=0, sticky="nsew")
        self.zoom_lbl_video.bind("<Button-1>", lambda e: self._on_video_left_click(e, self._zoomed_src) if self._zoomed_src else None)
        self.zoom_lbl_video.bind("<B1-Motion>", lambda e: self._on_video_drag(e, self._zoomed_src) if self._zoomed_src else None)
        self.zoom_lbl_video.bind("<ButtonRelease-1>", lambda e: self._on_video_release(e, self._zoomed_src) if self._zoomed_src else None)
        self.zoom_lbl_video.bind("<Button-3>", lambda e: self._on_video_right_click(e, self._zoomed_src) if self._zoomed_src else None)

    def _drain_log_queue(self):
        try:
            processed = 0
            while processed < 200:
                msg = self._ui_log_queue.get_nowait()
                self.log_textbox.insert("end", msg)
                self.log_textbox.see("end")
                processed += 1
        except queue.Empty:
            pass
        except Exception:
            pass

        try:
            if self.winfo_exists():
                self.after(100, self._drain_log_queue)
        except Exception:
            pass

    def _drain_resource_metrics_queue(self):
        latest = None
        try:
            while True:
                latest = self._resource_metrics_queue.get_nowait()
        except queue.Empty:
            pass
        except Exception:
            latest = None

        if latest is not None:
            self._update_resource_stats(latest)

        try:
            if self.winfo_exists():
                self.after(500, self._drain_resource_metrics_queue)
        except Exception:
            pass

    def _build_right_panel(self):
        panel = ctk.CTkFrame(self, fg_color=CLR_PANEL, width=256, corner_radius=12)
        panel.grid(row=1, column=2, sticky="nsew", padx=(6, 12), pady=(10, 6))
        panel.grid_propagate(False)
        panel.grid_rowconfigure(2, weight=1)
        panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(panel, text="📊  TỔNG THỂ", font=ctk.CTkFont(size=11, weight="bold"), text_color=CLR_TEXT_DIM).grid(row=0, column=0, padx=14, pady=(14, 6), sticky="w")

        sf = ctk.CTkFrame(panel, fg_color=CLR_PANEL2, corner_radius=8)
        sf.grid(row=1, column=0, padx=12, pady=(0, 8), sticky="ew")
        sf.grid_columnconfigure(0, weight=1)
        sf.grid_columnconfigure(1, weight=1)

        self.stat_cam   = self._stat_card(sf, "SỐ CAM",      "0", CLR_TEXT,   0, 0)
        self.stat_obj   = self._stat_card(sf, "TỔNG VẬT THỂ","0", CLR_TEXT,   0, 1)
        self.stat_fire  = self._stat_card(sf, "LỬA/KHÓI",    "0", CLR_YELLOW, 1, 0)
        self.stat_state = self._stat_card(sf, "TRẠNG THÁI",  "OK", CLR_GREEN,  1, 1)
        self.stat_cpu   = self._stat_card(sf, "CPU",          "--", CLR_TEXT,   2, 0)
        self.stat_ram   = self._stat_card(sf, "RAM",          "--", CLR_TEXT,   2, 1)
        self.stat_gpu   = self._stat_card(sf, "GPU",          "N/A", CLR_TEXT_DIM, 3, 0)
        self.stat_vram  = self._stat_card(sf, "VRAM",         "N/A", CLR_TEXT_DIM, 3, 1)

        ctk.CTkLabel(panel, text="🎯  BỘ LỌC HIỂN THỊ", font=ctk.CTkFont(size=11, weight="bold"), text_color=CLR_TEXT_DIM).grid(row=2, column=0, padx=14, pady=(8, 6), sticky="w")

        filt = ctk.CTkFrame(panel, fg_color=CLR_PANEL2, corner_radius=8)
        filt.grid(row=3, column=0, padx=12, pady=(0, 4), sticky="ew")
        filt.grid_columnconfigure(0, weight=1)

        self.filter_scroll = ctk.CTkScrollableFrame(filt, fg_color="transparent", height=120, corner_radius=6)
        self.filter_scroll.grid(row=1, column=0, padx=4, pady=(6, 6), sticky="ew")
        self.filter_scroll.grid_columnconfigure(0, weight=1)

        # 🎯 NGƯỠNG NHẬN DIỆN (THRESHOLDS)
        ctk.CTkLabel(panel, text="⚙️  CẤU HÌNH NHẬN DIỆN", font=ctk.CTkFont(size=11, weight="bold"), text_color=CLR_TEXT_DIM).grid(row=4, column=0, padx=14, pady=(8, 2), sticky="w")
        
        self.config_target_menu = ctk.CTkOptionMenu(
            panel, variable=self.config_target_var,
            values=["Tất cả Camera"], 
            command=self._on_config_target_change,
            fg_color=CLR_PANEL2, button_color=CLR_PANEL2, button_hover_color=CLR_CAM_IDLE,
            text_color=CLR_TEXT, font=ctk.CTkFont(size=12, weight="bold")
        )
        self.config_target_menu.grid(row=5, column=0, padx=12, pady=(0, 6), sticky="ew")
        
        thresh_frame = ctk.CTkFrame(panel, fg_color=CLR_PANEL2, corner_radius=8)
        thresh_frame.grid(row=6, column=0, padx=12, pady=(0, 4), sticky="ew")
        thresh_frame.grid_columnconfigure(0, weight=1)
        
        # Fire Slider
        hdr_f = ctk.CTkFrame(thresh_frame, fg_color="transparent")
        hdr_f.grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 0))
        ctk.CTkLabel(hdr_f, text="Lửa & Khói", font=ctk.CTkFont(size=10)).pack(side="left")
        self.lbl_val_fire = ctk.CTkLabel(hdr_f, text="25%", font=ctk.CTkFont(size=10, weight="bold"), text_color=CLR_RED)
        self.lbl_val_fire.pack(side="right")
        
        self.slider_fire = ctk.CTkSlider(thresh_frame, from_=0.01, to=0.75, number_of_steps=74, height=12, button_color=CLR_RED, progress_color=CLR_RED, command=self._on_threshold_change)
        self.slider_fire.set(Config.FIRE_CONFIDENCE_THRESHOLD)
        self.slider_fire.grid(row=1, column=0, padx=8, pady=(0, 6), sticky="ew")
        self.lbl_val_fire.configure(text=f"{int(self.slider_fire.get()*100)}%")

        # Default Object Slider
        hdr_d = ctk.CTkFrame(thresh_frame, fg_color="transparent")
        hdr_d.grid(row=2, column=0, sticky="ew", padx=8, pady=(2, 0))
        ctk.CTkLabel(hdr_d, text="Vật thể chung", font=ctk.CTkFont(size=10)).pack(side="left")
        self.lbl_val_def = ctk.CTkLabel(hdr_d, text="25%", font=ctk.CTkFont(size=10, weight="bold"), text_color=CLR_ACCENT)
        self.lbl_val_def.pack(side="right")

        self.slider_def = ctk.CTkSlider(thresh_frame, from_=0.15, to=0.85, number_of_steps=70, height=12, button_color=CLR_ACCENT, progress_color=CLR_ACCENT, command=self._on_threshold_change)
        self.slider_def.set(Config.CONFIDENCE_THRESHOLD)
        self.slider_def.grid(row=3, column=0, padx=8, pady=(0, 8), sticky="ew")
        self.lbl_val_def.configure(text=f"{int(self.slider_def.get()*100)}%")

        hdr_iou = ctk.CTkFrame(thresh_frame, fg_color="transparent")
        hdr_iou.grid(row=4, column=0, sticky="ew", padx=8, pady=(2, 0))
        ctk.CTkLabel(hdr_iou, text="NMS IoU", font=ctk.CTkFont(size=10)).pack(side="left")
        self.lbl_val_iou = ctk.CTkLabel(hdr_iou, text="85%", font=ctk.CTkFont(size=10, weight="bold"), text_color=CLR_GREEN)
        self.lbl_val_iou.pack(side="right")

        self.slider_iou = ctk.CTkSlider(thresh_frame, from_=0.30, to=0.95, number_of_steps=65, height=12, button_color=CLR_GREEN, progress_color=CLR_GREEN, command=self._on_threshold_change)
        self.slider_iou.set(Config.YOLO_IOU)
        self.slider_iou.grid(row=5, column=0, padx=8, pady=(0, 8), sticky="ew")
        self.lbl_val_iou.configure(text=f"{int(self.slider_iou.get()*100)}%")

        roi_row = ctk.CTkFrame(thresh_frame, fg_color="transparent")
        roi_row.grid(row=6, column=0, sticky="ew", padx=8, pady=(0, 8))
        roi_row.grid_columnconfigure(0, weight=1)
        ctk.CTkSwitch(
            roi_row,
            text="ROI",
            variable=self.roi_mode_var,
            font=ctk.CTkFont(size=10, weight="bold"),
            command=self._on_roi_mode_change,
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkButton(
            roi_row,
            text="Clear",
            width=54,
            height=24,
            fg_color=CLR_PANEL2,
            hover_color="#3a3d4e",
            font=ctk.CTkFont(size=10, weight="bold"),
            command=self._clear_roi_for_target,
        ).grid(row=0, column=1, sticky="e")


        ctk.CTkLabel(panel, text="📋  NHẬT KÝ SỰ KIỆN", font=ctk.CTkFont(size=11, weight="bold"), text_color=CLR_TEXT_DIM).grid(row=7, column=0, padx=14, pady=(8, 6), sticky="w")

        self.log_textbox = ctk.CTkTextbox(panel, wrap="word", fg_color=CLR_PANEL2, text_color=CLR_TEXT, font=ctk.CTkFont(family="Consolas", size=11), border_color=CLR_BORDER, border_width=1, corner_radius=8)
        self.log_textbox.grid(row=8, column=0, padx=12, pady=(0, 12), sticky="nsew")
        panel.grid_rowconfigure(8, weight=1)

    def _build_status_bar(self):
        bar = ctk.CTkFrame(self, fg_color=CLR_PANEL, height=30, corner_radius=0)
        bar.grid(row=2, column=0, columnspan=3, sticky="ew")
        bar.grid_columnconfigure(1, weight=1)
        bar.grid_propagate(False)

        self.statusbar_left = ctk.CTkLabel(bar, text="  🟢  Hệ thống sẵn sàng", font=ctk.CTkFont(size=11), text_color=CLR_TEXT_DIM)
        self.statusbar_left.grid(row=0, column=0, padx=10, sticky="w")
        ctk.CTkLabel(bar, text="Camera AI v2.1   ", font=ctk.CTkFont(size=11), text_color=CLR_TEXT_DIM).grid(row=0, column=2, padx=10, sticky="e")

    # ════════════════════════════════════════════════════════
    #  UI HELPERS & ACTIONS
    # ════════════════════════════════════════════════════════

    def _open_events_folder(self):
        folder = os.path.abspath(Config.EVENT_DIR)
        if not os.path.exists(folder):
            os.makedirs(folder)
        try:
            os.startfile(folder)
        except Exception as e:
            logger.error("Không thể mở thư mục Events: %s", e)

    def _open_logs_folder(self):
        folder = os.path.abspath("logs")
        if not os.path.exists(folder):
            os.makedirs(folder)
        try:
            os.startfile(folder)
        except Exception as e:
            logger.error("Không thể mở thư mục Logs: %s", e)

    def _stat_card(self, parent, title, value, color, row, col):
        card = ctk.CTkFrame(parent, fg_color="#1e2133", corner_radius=6)
        card.grid(row=row, column=col, padx=6, pady=6, sticky="ew")
        card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=9, weight="bold"), text_color=CLR_TEXT_DIM).grid(row=0, column=0, pady=(6, 0))
        value_size = 15 if title in {"RAM", "GPU", "VRAM"} else 20
        lbl = ctk.CTkLabel(card, text=value, font=ctk.CTkFont(size=value_size, weight="bold"), text_color=color)
        lbl.grid(row=1, column=0, pady=(0, 6))
        return lbl

    def _update_resource_stats(self, metrics: ResourceMetrics):
        system = metrics.system
        if system.error:
            self.stat_cpu.configure(text="N/A", text_color=CLR_TEXT_DIM)
            self.stat_ram.configure(text="N/A", text_color=CLR_TEXT_DIM)
        else:
            self.stat_cpu.configure(
                text=f"{system.cpu_percent:.0f}%",
                text_color=self._resource_color(system.cpu_percent),
            )
            ram_total_gb = system.ram_total_mb / 1024.0
            ram_used_gb = system.ram_used_mb / 1024.0
            self.stat_ram.configure(
                text=f"{ram_used_gb:.1f}/{ram_total_gb:.0f}G\n{system.ram_percent:.0f}%",
                text_color=self._resource_color(system.ram_percent),
            )

        gpu = metrics.gpu
        if not gpu.available:
            self.stat_gpu.configure(text="N/A", text_color=CLR_TEXT_DIM)
            self.stat_vram.configure(text="N/A", text_color=CLR_TEXT_DIM)
            return

        self.stat_gpu.configure(
            text=f"{gpu.utilization_percent:.0f}%\n{gpu.temperature_c:.0f}°C",
            text_color=self._resource_color(gpu.utilization_percent),
        )
        self.stat_vram.configure(
            text=f"{gpu.vram_used_mb / 1024.0:.1f}/{gpu.vram_total_mb / 1024.0:.0f}G\n{gpu.vram_percent:.0f}%",
            text_color=self._resource_color(gpu.vram_percent),
        )

    def _resource_color(self, percent: float) -> str:
        if percent >= 90.0:
            return CLR_RED
        if percent >= 75.0:
            return CLR_YELLOW
        return CLR_GREEN

    def _tick_clock(self):
        self.clock_label.configure(text=datetime.now().strftime("%H:%M:%S  —  %d/%m/%Y"))
        self.after(1000, self._tick_clock)

    def _set_status(self, msg: str, color: str, icon: str = "⚪"):
        self.statusbar_left.configure(text=f"  {icon}  {msg}", text_color=color)

    def _on_config_target_change(self, target: str):
        if target not in self.camera_configs:
            return
        cfg = self.camera_configs[target]
        self.slider_fire.set(cfg["fire_conf"])
        self.slider_def.set(cfg["def_conf"])
        self.slider_iou.set(cfg.get("iou", Config.YOLO_IOU))
        self.lbl_val_fire.configure(text=f"{int(cfg['fire_conf']*100)}%")
        self.lbl_val_def.configure(text=f"{int(cfg['def_conf']*100)}%")
        self.lbl_val_iou.configure(text=f"{int(cfg.get('iou', Config.YOLO_IOU)*100)}%")
        
        for cls_name, visible in cfg["class_visibility"].items():
            if cls_name in self.class_visibility:
                self.class_visibility[cls_name].set(visible)

    def _on_threshold_change(self, value=None):
        if not hasattr(self, 'camera_configs'): return
        
        fire_conf = self.slider_fire.get()
        def_conf = self.slider_def.get()
        iou = self.slider_iou.get() if hasattr(self, "slider_iou") else Config.YOLO_IOU
        self.lbl_val_fire.configure(text=f"{int(fire_conf*100)}%")
        self.lbl_val_def.configure(text=f"{int(def_conf*100)}%")
        if hasattr(self, "lbl_val_iou"):
            self.lbl_val_iou.configure(text=f"{int(iou*100)}%")
        Config.YOLO_IOU = iou
        
        target = self.config_target_var.get()
        if target in self.camera_configs:
            self.camera_configs[target]["fire_conf"] = fire_conf
            self.camera_configs[target]["def_conf"] = def_conf
            self.camera_configs[target]["iou"] = iou
            
        if target == "Tất cả Camera":
            for src, cfg in self.camera_configs.items():
                if src != "Tất cả Camera":
                    cfg["fire_conf"] = fire_conf
                    cfg["def_conf"] = def_conf
                    cfg["iou"] = iou

    def _on_roi_mode_change(self):
        if self.roi_mode_var.get():
            logger.info("ROI edit mode enabled. Left-click video to add/move points; right-click to remove.")
        else:
            for stream in self.streams.values():
                stream.roi_drag_index = None
            logger.info("ROI edit mode disabled.")

    def _clear_roi_for_target(self):
        target = self.config_target_var.get()
        targets = self.streams.values() if target == "Tất cả Camera" else [self.streams.get(target)]
        cleared = 0
        for stream in targets:
            if not stream:
                continue
            stream.roi_points.clear()
            stream.roi_enabled = False
            stream.roi_drag_index = None
            cleared += 1
        logger.info("Cleared ROI for %d stream(s).", cleared)

    # ════════════════════════════════════════════════════════
    #  CLASS FILTER 
    # ════════════════════════════════════════════════════════

    def _on_class_visibility_change(self, class_name: str, var: ctk.BooleanVar):
        target = self.config_target_var.get()
        if target in self.camera_configs:
            self.camera_configs[target]["class_visibility"][class_name] = var.get()
            
        if target == "Tất cả Camera":
            for src, cfg in self.camera_configs.items():
                cfg["class_visibility"][class_name] = var.get()

    def _register_class(self, class_name: str):
        if class_name in self.class_visibility: return
        var = ctk.BooleanVar(value=True)
        self.class_visibility[class_name] = var
        
        for cfg in self.camera_configs.values():
            if class_name not in cfg["class_visibility"]:
                cfg["class_visibility"][class_name] = True

        meta = Config.CLASS_REGISTRY.get(class_name.lower(), {})
        icon = meta.get("icon", "■")
        dot_color = meta.get("color", None)
        if dot_color is None:
            if class_name not in self.class_colors:
                self.class_colors[class_name] = _OBJECT_PALETTE[len(self.class_colors) % len(_OBJECT_PALETTE)]
            bgr = self.class_colors[class_name]
            dot_color = "#{:02x}{:02x}{:02x}".format(bgr[2], bgr[1], bgr[0])

        row_f = ctk.CTkFrame(self.filter_scroll, fg_color="#1e2133", corner_radius=6)
        row_f.pack(fill="x", pady=2, padx=2)
        row_f.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(row_f, text=icon, text_color=dot_color, font=ctk.CTkFont(size=13)).grid(row=0, column=0, padx=(8, 4), pady=4)
        ctk.CTkLabel(row_f, text=class_name.upper(), font=ctk.CTkFont(size=11, weight="bold"), text_color=CLR_TEXT, anchor="w").grid(row=0, column=1, padx=2, pady=4, sticky="ew")
        cb = ctk.CTkCheckBox(row_f, text="", variable=var, width=30, checkbox_width=18, checkbox_height=18, fg_color=CLR_ACCENT, hover_color=CLR_ACCENT_DARK, border_color=CLR_BORDER, corner_radius=4, command=lambda c=class_name, v=var: self._on_class_visibility_change(c, v))
        cb.grid(row=0, column=2, padx=(4, 8), pady=4)

    def _is_class_visible(self, class_name: str) -> bool:
        return self.class_visibility.get(class_name, ctk.BooleanVar(value=True)).get()

    def _get_class_color(self, class_name: str) -> tuple:
        if class_name not in self.class_colors:
            self.class_colors[class_name] = _OBJECT_PALETTE[len(self.class_colors) % len(_OBJECT_PALETTE)]
        return self.class_colors[class_name]

    # _compute_allowed_ids removed because filtering is now done post-inference per camera

    # ════════════════════════════════════════════════════════
    #  CAMERA LIST MANAGEMENT
    # ════════════════════════════════════════════════════════

    def _refresh_camera_list(self):
        if not hasattr(self, '_camera_list_widgets'):
            self._camera_list_widgets = []
        for w in self._camera_list_widgets:
            w.destroy()
        self._camera_list_widgets.clear()

        for src in self.camera_sources:
            if src not in self.active_sources_vars:
                self.active_sources_vars[src] = ctk.BooleanVar(value=False)
            
            is_sel = self.active_sources_vars[src].get()
            
            if src.isdigit():
                icon, label = "📷", f"Webcam {src}"
            elif src.startswith(("rtsp://", "http://", "https://")):
                icon, label = "📡", "Stream"
            elif os.path.isfile(src):
                icon, label = "🎬", os.path.basename(src)
            else:
                icon, label = "🔗", src

            row_f = ctk.CTkFrame(self.camera_list_frame, fg_color=CLR_CAM_SEL if is_sel else CLR_CAM_IDLE, corner_radius=8)
            row_f.pack(fill="x", pady=3, padx=2)
            row_f.grid_columnconfigure(1, weight=1)
            self._camera_list_widgets.append(row_f)

            ctk.CTkLabel(row_f, text=icon, font=ctk.CTkFont(size=14)).grid(row=0, column=0, padx=(8, 4), pady=6)
            
            cb = ctk.CTkCheckBox(row_f, text=label, variable=self.active_sources_vars[src], font=ctk.CTkFont(size=12, weight="bold" if is_sel else "normal"), text_color=CLR_TEXT if is_sel else CLR_TEXT_DIM, command=self._refresh_camera_list)
            cb.grid(row=0, column=1, padx=4, pady=6, sticky="w")
            
            # Badge trạng thái
            stream = self.streams.get(src)
            if stream:
                if stream.has_error:
                    ctk.CTkLabel(row_f, text="⚠ LỖI", font=ctk.CTkFont(size=10, weight="bold"), text_color=CLR_RED).grid(row=0, column=2, padx=(4, 8), pady=6)
                elif stream.is_running:
                    ctk.CTkLabel(row_f, text="● LIVE", font=ctk.CTkFont(size=10, weight="bold"), text_color=CLR_GREEN).grid(row=0, column=2, padx=(4, 8), pady=6)

            # Nút xóa
            btn_del = ctk.CTkButton(row_f, text="❌", width=24, height=24, fg_color="transparent", hover_color=CLR_RED, 
                                    text_color=CLR_TEXT_DIM, font=ctk.CTkFont(size=10),
                                    command=lambda s=src: self._remove_camera(s))
            btn_del.grid(row=0, column=3, padx=(0, 6), pady=6)

    def _remove_camera(self, src: str):
        if src in self.camera_sources:
            self.camera_sources.remove(src)
            if src in self.active_sources_vars:
                del self.active_sources_vars[src]
            
            # Xóa khỏi history
            try:
                with open(self._history_file, 'w', encoding='utf-8') as f:
                    json.dump(self.camera_sources, f, ensure_ascii=False, indent=4)
            except Exception as e:
                logger.error(f"Lỗi khi lưu lịch sử camera: {e}")
                
            self._refresh_camera_list()
            logger.info("❌ Đã xóa camera: %s", src)

    def _add_camera(self):
        src = self.add_cam_entry.get().strip()
        if src and src not in self.camera_sources:
            self.camera_sources.append(src)
            self.active_sources_vars[src] = ctk.BooleanVar(value=True)
            self._refresh_camera_list()
            self.add_cam_entry.delete(0, "end")
            logger.info("➕ Đã thêm camera: %s", src)
            
            # Save to history
            try:
                with open(self._history_file, 'w', encoding='utf-8') as f:
                    json.dump(self.camera_sources, f, ensure_ascii=False, indent=4)
            except Exception as e:
                logger.error(f"Lỗi khi lưu lịch sử camera: {e}")
        else:
            logger.warning("Camera đã tồn tại hoặc bỏ trống!")

    # ════════════════════════════════════════════════════════
    #  MANUAL RECORDING
    # ════════════════════════════════════════════════════════

    def _toggle_manual_record(self, src: str):
        """Bật/tắt ghi hình thủ công cho camera cụ thể."""
        stream = self.streams.get(src)
        if not stream or not stream.is_running:
            return

        if stream.manual_recording:
            # Dừng ghi → export video
            stream.manual_recording = False
            frames_copy = stream.manual_rec_buffer.copy()
            stream.manual_rec_buffer.clear()

            btn = self._record_buttons.get(src)
            if btn:
                btn.configure(text="⏺", fg_color=CLR_PANEL2, hover_color="#3a3d4e")

            if frames_copy:
                threading.Thread(
                    target=self._save_event_video,
                    args=(frames_copy, f"MANUAL_{stream.display_name}"),
                    daemon=True
                ).start()
                logger.info("⏹ Dừng ghi hình thủ công [%s] — %d frames", stream.display_name, len(frames_copy))
        else:
            # Bắt đầu ghi
            stream.manual_recording = True
            stream.manual_rec_buffer = list(stream.frame_buffer)

            btn = self._record_buttons.get(src)
            if btn:
                btn.configure(text="⏹", fg_color=CLR_RED, hover_color="#e03131")

            logger.info("🔴 Bắt đầu ghi hình thủ công [%s]", stream.display_name)

    # ════════════════════════════════════════════════════════
    #  ZOOM CAMERA
    # ════════════════════════════════════════════════════════

    def _clear_zoom_overlay(self, reason: str = ""):
        """Ẩn overlay phóng to an toàn."""
        self._zoomed_src = None
        try:
            self.zoom_overlay.place_forget()
            self.zoom_lbl_title.configure(text="🔍 Chế độ phóng to")
            # KHÔNG XÓA image hay reference ở đây để tránh Tkinter Tcl Access Violation
            # khi render loop đang vẽ dở. place_forget là đủ để ẩn.
        except Exception:
            pass
        if reason:
            logger.info("🔍 Thu nhỏ zoom: %s", reason)

    def _toggle_zoom(self, src: str):
        if not self._video_frames or src not in self.streams:
            return
        stream = self.streams.get(src)
        if stream and stream.ended:
            return

        if self._zoomed_src == src:
            # Unzoom
            self._clear_zoom_overlay("người dùng thu nhỏ")
        else:
            # Zoom
            if self._zoomed_src:
                self._toggle_zoom(self._zoomed_src) # Unzoom camera cũ
                
            self._zoomed_src = src
            stream = self.streams.get(src)
            if stream:
                self.zoom_lbl_title.configure(text=f"  🔍 Đang phóng to: {stream.display_name} (Click để thu nhỏ)")
            
            # Che toàn bộ container (gồm header Hệ Thống Màn Hình và video grid)
            self.zoom_overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
            self.zoom_overlay.lift()
            logger.info("🔍 Phóng to video toàn khung hình bằng Overlay")

    # ════════════════════════════════════════════════════════
    #  MULTI-CAMERA INFERENCE & RENDER
    # ════════════════════════════════════════════════════════

    def _on_video_left_click(self, event, src: str):
        if not src:
            return
        if not self.roi_mode_var.get():
            self._toggle_zoom(src)
            return

        stream = self.streams.get(src)
        point = self._event_to_frame_point(event, src)
        if not stream or point is None:
            return

        frame_w = stream.latest_frame.shape[1] if stream.latest_frame is not None else 640
        idx = nearest_vertex(point, stream.roi_points, max_distance=max(18.0, frame_w * 0.015))
        if idx is None:
            stream.roi_points.append(point)
            stream.roi_drag_index = len(stream.roi_points) - 1
        else:
            stream.roi_points[idx] = point
            stream.roi_drag_index = idx
        stream.roi_enabled = len(stream.roi_points) >= 3

    def _on_video_drag(self, event, src: str):
        if not src or not self.roi_mode_var.get():
            return
        stream = self.streams.get(src)
        point = self._event_to_frame_point(event, src)
        if not stream or point is None or stream.roi_drag_index is None:
            return
        if 0 <= stream.roi_drag_index < len(stream.roi_points):
            stream.roi_points[stream.roi_drag_index] = point
            stream.roi_enabled = len(stream.roi_points) >= 3

    def _on_video_release(self, event, src: str):
        stream = self.streams.get(src) if src else None
        if stream:
            stream.roi_drag_index = None

    def _on_video_right_click(self, event, src: str):
        if not src or not self.roi_mode_var.get():
            return
        stream = self.streams.get(src)
        point = self._event_to_frame_point(event, src)
        if not stream or point is None or not stream.roi_points:
            return

        idx = nearest_vertex(point, stream.roi_points, max_distance=40.0)
        if idx is None:
            idx = len(stream.roi_points) - 1
        del stream.roi_points[idx]
        stream.roi_enabled = len(stream.roi_points) >= 3
        stream.roi_drag_index = None

    def _event_to_frame_point(self, event, src: str) -> Optional[tuple[int, int]]:
        display = self._display_maps.get(src)
        if not display:
            return None
        return frame_point_from_widget_event(event, display, display.get("widget"))

    def _should_defer_capture_release(self, stream: CameraStream) -> bool:
        if os.name != "nt" or not stream.is_video_file:
            return False
        backend_name = (getattr(stream, "capture_backend", "") or "").upper()
        return "FFMPEG" not in backend_name

    def _retain_capture_for_process_exit(self, cap, stream: CameraStream):
        if cap is None:
            return
        self._retained_video_caps.append(cap)
        logger.warning(
            "âš ï¸ Táº¡m giá»¯ handle video [%s] Ä‘áº¿n lÃºc app thoÃ¡t Ä‘á»ƒ trÃ¡nh treo cap.release() (backend=%s).",
            stream.display_name,
            getattr(stream, "capture_backend", "unknown"),
        )

    def _shutdown_stream_capture(self, stream: CameraStream):
        capture_thread = stream.thread
        if capture_thread and capture_thread.is_alive() and capture_thread is not threading.current_thread():
            logger.info("[DIAG-BG] â± Chá» capture thread thoÃ¡t [%s]...", stream.display_name)
            capture_thread.join(timeout=2.0)
            logger.info("[DIAG-BG] â± Capture thread alive=%s [%s]", capture_thread.is_alive(), stream.display_name)

        cap = stream.cap
        stream.cap = None
        stream.thread = None

        if cap is None:
            return

        if self._should_defer_capture_release(stream):
            self._retain_capture_for_process_exit(cap, stream)
            return

        logger.info("[DIAG-BG] â± Báº¯t Ä‘áº§u cap.release()... [%s]", stream.display_name)
        cap.release()
        logger.info("[DIAG-BG] â± cap.release() OK [%s]", stream.display_name)

    def start_cameras(self):
        selected = [src for src, var in self.active_sources_vars.items() if var.get()]
        if not selected:
            logger.error("Chưa chọn camera nào để bắt đầu!")
            return

        # Dọn dẹp trước
        self.stop_all_cameras()
        
        self.video_labels.clear()
        self._record_buttons.clear()
        for fb in list(self._video_frames.values()):
            try:
                fb.destroy()
            except Exception:
                pass
        self._video_frames.clear()
        self._grid_configs.clear()
        self._clear_zoom_overlay()

        # Tính toán Layout (Chia lưới tự động)
        n = len(selected)
        cols = 1 if n == 1 else (2 if n <= 4 else (3 if n <= 9 else 4))
        
        self.video_grid.update_idletasks()
        grid_w = self.video_grid.winfo_width()
        
        if grid_w < 100: 
            grid_w = self.winfo_width() - SIDE_PANEL_TOTAL_WIDTH
            if grid_w < 100: grid_w = 900
        
        cell_w = max(MIN_CELL_WIDTH, (grid_w - 30) // cols)
        cell_h = int(cell_w * VIDEO_ASPECT_RATIO) + CAM_TILE_HEADER_HEIGHT
        
        for idx in range(n):
            self.video_grid.grid_columnconfigure(idx % cols, weight=1)
            self.video_grid.grid_rowconfigure(idx // cols, weight=1)

        for idx, src in enumerate(selected):
            # Init config if missing
            if src not in self.camera_configs:
                self.camera_configs[src] = {
                    "fire_conf": Config.FIRE_CONFIDENCE_THRESHOLD,
                    "def_conf": Config.CONFIDENCE_THRESHOLD,
                    "iou": Config.YOLO_IOU,
                    "class_visibility": {k: v.get() for k, v in self.class_visibility.items()}
                }
                
            stream = CameraStream(src)
            if stream.start():
                self.streams[src] = stream
                
                frame_box = ctk.CTkFrame(self.video_grid, fg_color="#181a25", border_width=2, border_color=CLR_BORDER, width=cell_w, height=cell_h)
                frame_box.grid_propagate(False)
                r, c = idx // cols, idx % cols
                frame_box.grid(row=r, column=c, padx=4, pady=4, sticky="nsew")
                frame_box.grid_rowconfigure(1, weight=1)
                frame_box.grid_columnconfigure(0, weight=1)
                
                self._video_frames[src] = frame_box
                self._grid_configs[src] = {"row": r, "column": c}

                # Header cam (với nút Record)
                hdr = ctk.CTkFrame(frame_box, fg_color="transparent", height=24)
                hdr.grid(row=0, column=0, sticky="ew")
                hdr.grid_columnconfigure(0, weight=1)
                lbl_title = ctk.CTkLabel(hdr, text=f"  {stream.display_name}", font=ctk.CTkFont(size=11, weight="bold"), text_color=CLR_TEXT_DIM, cursor="hand2")
                lbl_title.grid(row=0, column=0, sticky="w")
                
                # Nút Record thủ công
                rec_btn = ctk.CTkButton(
                    hdr, text="⏺", width=28, height=22,
                    fg_color=CLR_PANEL2, hover_color="#3a3d4e",
                    font=ctk.CTkFont(size=14), corner_radius=4,
                    command=lambda s=src: self._toggle_manual_record(s)
                )
                rec_btn.grid(row=0, column=1, padx=(4, 4), sticky="e")
                self._record_buttons[src] = rec_btn
                
                # Video label
                lbl = ctk.CTkLabel(frame_box, text="Đang tải...", text_color=CLR_TEXT_DIM, cursor="hand2")
                lbl.grid(row=1, column=0, sticky="nsew")
                self.video_labels[src] = lbl

                # Bind click events for zooming / ROI editing
                lbl.bind("<Button-1>", lambda e, s=src: self._on_video_left_click(e, s))
                lbl.bind("<B1-Motion>", lambda e, s=src: self._on_video_drag(e, s))
                lbl.bind("<ButtonRelease-1>", lambda e, s=src: self._on_video_release(e, s))
                lbl.bind("<Button-3>", lambda e, s=src: self._on_video_right_click(e, s))
                lbl_title.bind("<Button-1>", lambda e, s=src: self._toggle_zoom(s))
                hdr.bind("<Button-1>", lambda e, s=src: self._toggle_zoom(s))
                frame_box.bind("<Button-1>", lambda e, s=src: self._toggle_zoom(s))

        if not self.streams:
            self._set_status("LỖI: Chẳng có camera nào kết nối được", CLR_RED, "🔴")
            return

        # Update Config Target Menu
        menu_values = ["Tất cả Camera"] + list(self.streams.keys())
        self.config_target_menu.configure(values=menu_values)
        if self.config_target_var.get() not in menu_values:
            self.config_target_var.set("Tất cả Camera")
            self._on_config_target_change("Tất cả Camera")

        self.feed_count_label.configure(text=f"{len(self.streams)} camera đang bật")
        self.btn_start.configure(state="disabled")
        self.btn_pause.configure(state="normal", text="⏸ Tạm Dừng", fg_color=CLR_YELLOW, text_color="black")
        self.btn_stop.configure(state="normal")
        self.add_cam_entry.configure(state="disabled")
        self.btn_add_cam.configure(state="disabled")
        self._set_status("ĐANG THEO DÕI", CLR_GREEN, "🟢")
        self._refresh_camera_list()

        # Khởi động thread inference
        self.is_inferencing = True
        self._inference_thread = threading.Thread(target=self._multi_inference_loop, daemon=True)
        self._inference_thread.start()

    def _multi_inference_loop(self):
        """Vòng lặp AI quét qua tất cả các luồng camera đang chạy để detect theo lô (batch)."""
        last_processed_frames = {}
        
        while self.is_inferencing:
            try:
                active_streams = list(self.streams.items())
            except RuntimeError:
                time.sleep(0.01)
                continue
            
            batch_frames = []
            batch_streams = []
            
            for src, stream in active_streams:
                if not stream.is_running:
                    continue
                
                frame = stream.latest_frame
                if frame is not None and stream.frame_id != last_processed_frames.get(src):
                    batch_frames.append(frame)
                    batch_streams.append(stream)
                    last_processed_frames[src] = stream.frame_id
                    
            if batch_frames:
                started = time.perf_counter()
                batch_detections = self.detector.detect_batch(batch_frames, allowed_class_ids=None)
                latency_ms = (time.perf_counter() - started) * 1000.0
                per_stream_latency = latency_ms / max(1, len(batch_streams))
                
                for stream, detections in zip(batch_streams, batch_detections):
                    stream.latest_detections = detections
                    stream.inference_latency_ms = per_stream_latency
                    stream.last_inference_at = time.time()
                
                time.sleep(0.005)
            else:
                time.sleep(0.01)

    def _render_loop(self):
        """Vòng lặp render chính — được bảo vệ bởi try/except."""
        try:
            self._render_loop_inner()
        except Exception as e:
            logger.error("❌ Lỗi nghiêm trọng trong render loop: %s", e, exc_info=True)

        # Luôn schedule lần render tiếp theo dù có lỗi
        interval = RENDER_INTERVAL_MS if self.is_inferencing else IDLE_RENDER_INTERVAL_MS
        self.after(interval, self._render_loop)

    def _render_loop_inner(self):
        """Logic render thực tế, tách ra để try/except bao bọc."""
        if not self.is_inferencing:
            return
        total_obj = 0
        total_fire = 0
        fire_alerted = False
        has_error_any = False

        active_streams = list(self.streams.values())
        
        for stream in active_streams:
            if not stream.is_running:
                continue

            # Hiển thị error state trên video label
            if stream.has_error:
                has_error_any = True
                lbl = self.video_labels.get(stream.source)
                if lbl:
                    lbl.configure(text=f"⚠️ {stream.error_message}", text_color=CLR_RED)
                continue

            frame = stream.latest_frame
            detections = stream.latest_detections
            
            if frame is not None:
                # Đăng ký class mới từ nhận diện gốc
                for det in detections:
                    if det.class_name not in self.class_visibility:
                        self._register_class(det.class_name)
                        
                # ── Post-Filtering theo cấu hình của từng camera ──
                filtered_detections = []
                cfg = self.camera_configs.get(stream.source)
                if cfg:
                    for det in detections:
                        cls_name = det.class_name
                        cname_lower = cls_name.lower()
                        
                        # 1. Lọc theo visibility (Checkboxes)
                        if not cfg["class_visibility"].get(cls_name, True):
                            continue
                            
                        # 2. Lọc theo threshold riêng
                        if cname_lower == Config.FIRE_CLASS_NAME.lower() or cname_lower == Config.SMOKE_CLASS_NAME.lower():
                            if det.confidence < cfg.get("fire_conf", 0.0):
                                continue
                        else:
                            if det.confidence < cfg.get("def_conf", 0.0):
                                continue
                                
                        filtered_detections.append(det)
                    detections = filtered_detections
                if stream.roi_enabled and len(stream.roi_points) >= 3:
                    detections = filter_detections_by_roi(detections, stream.roi_points)
                
                # Cập nhật thông số thống kê
                stream.last_counts = class_counts(detections)
                should_sample_counts = time.time() - self._last_count_sample >= 1.0
                if should_sample_counts:
                    self._append_count_sample(stream, detections)
                
                if not stream.is_paused:
                    fire_confirmed = stream.tracker.update(detections)
                else:
                    fire_confirmed = False
                    
                display_frame = self._draw_overlay(frame, detections, stream)
                
                # Cảnh báo Telegram Đa Kênh
                if fire_confirmed:
                    fire_dets = stream.tracker.get_fire_detections(detections)
                    logger.info("🔥 ĐÁM CHÁY XÁC NHẬN TẠI %s!", stream.display_name)
                    threading.Thread(
                        target=self.notifier.send_fire_alert,
                        args=(display_frame.copy(), len(fire_dets), stream.display_name),
                        daemon=True,
                    ).start()
                    self._dispatch_fire_event(stream, display_frame.copy(), fire_dets)
                    
                    # Event Recording Trigger
                    if not stream.is_recording and time.time() > stream.record_cooldown:
                        stream.is_recording = True
                        stream.record_frames_left = RECORD_FUTURE_FRAMES
                        stream.rec_buffer = list(stream.frame_buffer)
                        logger.info("[Record] 🔴 Bắt đầu ghi hình sự cố cho %s (%d frames buffer)...", stream.display_name, len(stream.rec_buffer))

                # Ghi luồng Video tương lai (event recording)
                if stream.is_recording and not stream.is_paused:
                    stream.rec_buffer.append(display_frame.copy())
                    stream.record_frames_left -= 1
                    
                    if stream.record_frames_left <= 0:
                        stream.is_recording = False
                        stream.record_cooldown = time.time() + Config.ALERT_COOLDOWN_SECONDS
                        logger.info("[DIAG] ⏱ Bắt đầu copy rec_buffer (%d frames)...", len(stream.rec_buffer))
                        frames_ref = stream.rec_buffer  # Chỉ chuyển reference, KHÔNG copy
                        stream.rec_buffer = []           # Gán list mới cho stream
                        logger.info("[DIAG] ⏱ Đã chuyển rec_buffer sang thread lưu video")
                        threading.Thread(
                            target=self._save_event_video,
                            args=(frames_ref, stream.display_name),
                            daemon=True
                        ).start()

                # Ghi hình thủ công
                if stream.manual_recording and not stream.is_paused:
                    stream.manual_rec_buffer.append(display_frame.copy())

                # Render ra grid label
                lbl = self.video_labels.get(stream.source)
                if lbl:
                    if self._zoomed_src is None:
                        # Render bình thường trong grid
                        self._display_on_label(display_frame, lbl, stream.source)
                    elif self._zoomed_src == stream.source:
                        # Chỉ render lên khung Zoom (overlay)
                        self._display_on_label(display_frame, self.zoom_lbl_video, stream.source)

                # Thống kê
                visible = [d for d in detections if self._is_class_visible(d.class_name)]
                n_fire = len([d for d in visible if d.is_fire])
                total_obj += len(visible)
                total_fire += n_fire
                if stream.tracker.is_tracking and stream.tracker.elapsed >= Config.FIRE_CONFIRM_SECONDS:
                    fire_alerted = True

        if active_streams and time.time() - self._last_count_sample >= 1.0:
            self._last_count_sample = time.time()

        self._update_global_stats(total_obj, total_fire, fire_alerted, active_streams)

        # Hiển thị error banner nếu có camera lỗi
        if has_error_any and not fire_alerted:
            self._set_status("⚠️ MỘT SỐ CAMERA GẶP LỖI KẾT NỐI", CLR_YELLOW, "🟡")

        # Dọn dẹp các stream đã chết
        dead_keys = [k for k, v in self.streams.items() if not v.is_running]
        if dead_keys:
            logger.info("[DIAG] 🔍 Phát hiện %d stream chết: %s", len(dead_keys), dead_keys)
        for k in dead_keys:
            dead_stream = self.streams[k]
            logger.info("[DIAG] 🔍 Cleanup stream [%s] — is_recording=%s, rec_buffer=%d, manual_rec=%s",
                        k, dead_stream.is_recording,
                        len(getattr(dead_stream, 'rec_buffer', []) or []),
                        dead_stream.manual_recording)
            if dead_stream.has_error:
                logger.error("❌ Camera [%s] đã ngắt vĩnh viễn: %s", k, dead_stream.error_message)

            # Tự động thu nhỏ zoom khi camera kết thúc
            if self._zoomed_src == k:
                logger.info("[DIAG] 🔍 Thu nhỏ zoom overlay...")
                self._clear_zoom_overlay(f"video [{k}] đã kết thúc — tự động thu nhỏ")
                logger.info("[DIAG] 🔍 Đã thu nhỏ zoom overlay OK")

            # Chuyển TOÀN BỘ tác vụ nặng sang background thread
            def _cleanup_dead_stream(s, display_name):
                try:
                    s.is_running = False
                    if s.is_recording and hasattr(s, 'rec_buffer') and s.rec_buffer:
                        logger.info("[DIAG-BG] ⏱ Bắt đầu lưu rec_buffer (%d frames) cho %s", len(s.rec_buffer), display_name)
                        self._save_event_video(s.rec_buffer, display_name)
                        s.is_recording = False
                        s.rec_buffer = []
                        logger.info("[DIAG-BG] ⏱ Đã lưu rec_buffer OK")

                    if s.manual_recording and s.manual_rec_buffer:
                        logger.info("[DIAG-BG] ⏱ Bắt đầu lưu manual_rec_buffer...")
                        self._save_event_video(s.manual_rec_buffer, f"MANUAL_{display_name}")
                        s.manual_recording = False
                        s.manual_rec_buffer = []
                        logger.info("[DIAG-BG] ⏱ Đã lưu manual_rec_buffer OK")

                    logger.info("[DIAG-BG] ⏱ Bắt đầu cap.release()...")
                    self._shutdown_stream_capture(s)
                    logger.info("[DIAG-BG] ⏱ cap.release() OK")
                except Exception as e:
                    logger.error("❌ Lỗi cleanup stream %s: %s", display_name, e, exc_info=True)

            threading.Thread(
                target=_cleanup_dead_stream,
                args=(dead_stream, dead_stream.display_name),
                daemon=True
            ).start()
            del self.streams[k]
            logger.info("[DIAG] 🔍 Đã del stream [%s] khỏi dict — main thread không bị block", k)

        if self.is_inferencing and dead_keys:
            logger.info("[DIAG] 🔍 Bắt đầu _refresh_camera_list...")
            self._refresh_camera_list()
            logger.info("[DIAG] 🔍 _refresh_camera_list OK")
        if self.is_inferencing and not self.streams:
            logger.info("⌛ Tất cả camera đã ngắt, dừng hệ thống.")
            self.after(0, self.stop_all_cameras)

    def _append_count_sample(self, stream: CameraStream, detections: list):
        counts = class_counts(detections)
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "source": stream.display_name,
            "total": len(detections),
            "fire": sum(1 for det in detections if det.class_name.lower() == Config.FIRE_CLASS_NAME.lower()),
            "smoke": sum(1 for det in detections if det.class_name.lower() == Config.SMOKE_CLASS_NAME.lower()),
        }
        row.update(counts)
        self._count_history.append(row)
        if len(self._count_history) > 10000:
            del self._count_history[:1000]

    def _export_counts_csv(self):
        self._export_counts("csv")

    def _export_counts_excel(self):
        self._export_counts("xlsx")

    def _export_counts(self, fmt: str):
        rows = list(self._count_history)
        if not rows:
            logger.warning("No count data to export yet.")
            return

        def _run_export():
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                os.makedirs(Config.EXPORT_DIR, exist_ok=True)
                if fmt == "xlsx":
                    path = export_count_rows_excel(rows, os.path.join(Config.EXPORT_DIR, f"counts_{timestamp}.xlsx"))
                else:
                    path = export_count_rows_csv(rows, os.path.join(Config.EXPORT_DIR, f"counts_{timestamp}.csv"))
                logger.info("Exported %d count rows to %s", len(rows), path)
            except Exception as exc:
                logger.error("Export failed: %s", exc, exc_info=True)

        threading.Thread(target=_run_export, daemon=True).start()

    def _dispatch_fire_event(self, stream: CameraStream, frame: np.ndarray, fire_detections: list):
        metrics = {
            "fps": stream.fps,
            "latency_ms": stream.inference_latency_ms,
            "counts": dict(stream.last_counts),
            "roi_enabled": stream.roi_enabled,
        }

        def _log_and_publish():
            try:
                record = self.event_store.log_event(
                    event_type="fire_confirmed",
                    source_name=stream.display_name,
                    frame=frame,
                    detections=fire_detections,
                    metrics=metrics,
                )
                logger.info("Event logged #%d with snapshot %s", record.event_id, record.snapshot_path)
                self.mqtt_publisher.publish_alert(
                    {
                        "event_id": record.event_id,
                        "timestamp": record.timestamp,
                        "event_type": record.event_type,
                        "source": record.source_name,
                        "detection_count": record.detection_count,
                        "snapshot_path": record.snapshot_path,
                        "metrics": metrics,
                    }
                )
            except Exception as exc:
                logger.error("Event logging failed: %s", exc, exc_info=True)

        threading.Thread(target=_log_and_publish, daemon=True).start()

    def _display_on_label(self, frame: np.ndarray, lbl: ctk.CTkLabel, src: Optional[str] = None):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # parent frame size
        parent = lbl.master
        lw = parent.winfo_width()
        lh = parent.winfo_height() - 24 # - header height
        if lw > 10 and lh > 10:
            ih, iw = rgb.shape[:2]
            scale = min(lw / iw, lh / ih)
            nw, nh = int(iw * scale), int(ih * scale)
            if src is not None:
                self._display_maps[src] = {
                    "widget": lbl,
                    "frame_w": iw,
                    "frame_h": ih,
                    "image_w": nw,
                    "image_h": nh,
                    "offset_x": max(0, (lw - nw) // 2),
                    "offset_y": max(0, (lh - nh) // 2),
                }
            rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
            pil = Image.fromarray(rgb)
            tk_img = ImageTk.PhotoImage(image=pil)
            lbl.configure(image=tk_img, text="")
            lbl.image = tk_img
            # Giữ reference để tránh GC sớm
            if lbl is self.zoom_lbl_video:
                self._last_zoom_image = tk_img

    def _update_global_stats(self, n_obj, n_fire, fire_alerted, active_streams):
        self.stat_cam.configure(text=str(len(active_streams)))
        self.stat_obj.configure(text=str(n_obj))
        self.stat_fire.configure(text=str(n_fire), text_color=CLR_RED if n_fire > 0 else CLR_YELLOW)

        if fire_alerted:
            self.stat_state.configure(text="CHÁY!", text_color=CLR_RED)
            self._set_status("🔥 CÓ ĐÁM CHÁY TRONG HỆ THỐNG", CLR_RED, "🔴")
        elif n_fire > 0:
            self.stat_state.configure(text="CẢNH BÁO", text_color=CLR_YELLOW)
            self._set_status("⚠️  PHÁT HIỆN LỬA / KHÓI", CLR_YELLOW, "🟡")
        else:
            self.stat_state.configure(text="OK", text_color=CLR_GREEN)
            self._set_status("ĐANG THEO DÕI", CLR_GREEN, "🟢")

    def stop_all_cameras(self):
        # Guard: tránh gọi trùng từ nhiều nguồn (render loop + user click)
        if getattr(self, '_is_stopping', False):
            return
        self._is_stopping = True
        
        logger.info("⏹  Đang dừng toàn bộ camera...")
        self.is_inferencing = False

        # KHÔNG dùng thread.join() trên main thread — sẽ block Tkinter event loop và gây treo UI.
        old_thread = self._inference_thread
        self._inference_thread = None

        # Chuyển TOÀN BỘ tác vụ nặng sang background thread:
        # - copy rec_buffer (hàng trăm MB)
        # - save event video (ghi file disk)
        # - cap.release() + thread.join()
        streams_to_stop = list(self.streams.values())
        self.streams.clear()
        
        def _stop_all_bg(streams, old_t):
            for s in streams:
                try:
                    # Lưu video sự cố đang ghi dở
                    if s.manual_recording and s.manual_rec_buffer:
                        s.manual_recording = False
                        self._save_event_video(s.manual_rec_buffer, f"MANUAL_{s.display_name}")
                        s.manual_rec_buffer = []
                    if s.is_recording and hasattr(s, 'rec_buffer') and s.rec_buffer:
                        s.is_recording = False
                        logger.info("[Record] 🔴 Lưu video sự cố đang ghi dở do hệ thống dừng.")
                        self._save_event_video(s.rec_buffer, s.display_name)
                        s.rec_buffer = []
                    # Release OpenCV capture
                    s.is_running = False
                    self._shutdown_stream_capture(s)
                except Exception:
                    pass
            if old_t and old_t.is_alive():
                old_t.join(timeout=3.0)
        threading.Thread(target=_stop_all_bg, args=(streams_to_stop, old_thread), daemon=True).start()

        # Xóa hoàn toàn GRID UI thay vì ẩn
        # (Lỗi check_dpi_scaling trước đây là do CTkImage, nay đổi sang ImageTk nên destroy() an toàn)
        for fb in list(self._video_frames.values()):
            try:
                fb.destroy()
            except Exception:
                pass
        self.video_labels.clear()
        self._record_buttons.clear()
        self._video_frames.clear()
        self._grid_configs.clear()
        self._clear_zoom_overlay()

        self.btn_start.configure(state="normal")
        self.btn_pause.configure(state="disabled", text="⏸ Tạm Dừng", fg_color=CLR_YELLOW, text_color="black")
        self.btn_stop.configure(state="disabled")
        self.add_cam_entry.configure(state="normal")
        self.btn_add_cam.configure(state="normal")
        
        self.feed_count_label.configure(text="0 camera đang bật")
        self._set_status("ĐÃ DỪNG", CLR_TEXT_DIM, "⚪")
        
        self.stat_cam.configure(text="0")
        self.stat_obj.configure(text="0")
        self.stat_fire.configure(text="0", text_color=CLR_YELLOW)
        self.stat_state.configure(text="—", text_color=CLR_TEXT_DIM)
        
        self._refresh_camera_list()
        self._is_stopping = False

    def toggle_pause(self):
        if not self.streams:
            return

        is_currently_paused = any(s.is_paused for s in self.streams.values())
        new_pause_state = not is_currently_paused

        for stream in self.streams.values():
            stream.is_paused = new_pause_state
            if new_pause_state:
                stream.tracker.pause()
            else:
                stream.tracker.resume()

        if new_pause_state:
            self.btn_pause.configure(text="▶ Tiếp Tục", fg_color=CLR_GREEN, text_color="white", hover_color="#0ca678")
            self._set_status("TẠM DỪNG", CLR_YELLOW, "⏸")
        else:
            self.btn_pause.configure(text="⏸ Tạm Dừng", fg_color=CLR_YELLOW, text_color="black", hover_color="#f0a500")
            self._set_status("ĐANG THEO DÕI", CLR_GREEN, "🟢")

    # ════════════════════════════════════════════════════════
    #  OVERLAY
    # ════════════════════════════════════════════════════════

    def _draw_roi_overlay(self, frame: np.ndarray, stream: CameraStream):
        if not stream.roi_points:
            return

        points = np.array(stream.roi_points, dtype=np.int32)
        color = (0, 220, 120) if stream.roi_enabled else (0, 180, 255)
        if len(points) >= 3:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [points], color)
            cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
            cv2.polylines(frame, [points], True, color, 2)
        elif len(points) >= 2:
            cv2.polylines(frame, [points], False, color, 2)

        for index, (x, y) in enumerate(stream.roi_points, start=1):
            cv2.circle(frame, (int(x), int(y)), 6, color, -1)
            cv2.circle(frame, (int(x), int(y)), 8, (255, 255, 255), 1)
            cv2.putText(frame, str(index), (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    def _draw_overlay(self, frame: np.ndarray, detections: list, stream: CameraStream) -> np.ndarray:
        frame = frame.copy()
        h, w = frame.shape[:2]
        fire_dets = []
        is_confirmed = (stream.tracker.is_tracking and stream.tracker.elapsed >= Config.FIRE_CONFIRM_SECONDS)
        self._draw_roi_overlay(frame, stream)

        for det in detections:
            if det.is_fire:
                fire_dets.append(det)
                if not self._is_class_visible(det.class_name): continue
                is_smoke = det.class_name.lower() == Config.SMOKE_CLASS_NAME.lower()
                color = Config.COLOR_FIRE if is_confirmed else (Config.COLOR_SMOKE if is_smoke else Config.COLOR_FIRE_UNCONFIRMED)
                label = f"{'SMOKE' if is_smoke else 'FIRE'} {det.confidence:.0%}"
            else:
                if not self._is_class_visible(det.class_name): continue
                color = self._get_class_color(det.class_name)
                label = f"{det.class_name} {det.confidence:.0%}"

            cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), color, 3)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
            cv2.rectangle(frame, (det.x1, det.y1 - th - 8), (det.x1 + tw + 6, det.y1), color, -1)
            cv2.putText(frame, label, (det.x1 + 3, det.y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)

        # Countdown bar
        if stream.tracker.is_tracking:
            bar_h, bar_y = 28, h - 38
            bx1, bx2 = 10, w - 10
            cv2.rectangle(frame, (bx1, bar_y), (bx2, bar_y + bar_h), (30, 30, 30), -1)
            filled = int((bx2 - bx1) * stream.tracker.progress)
            if filled > 0:
                bc = (0, 0, 220) if is_confirmed else Config.COLOR_DANGER
                cv2.rectangle(frame, (bx1, bar_y), (bx1 + filled, bar_y + bar_h), bc, -1)
            cv2.rectangle(frame, (bx1, bar_y), (bx2, bar_y + bar_h), (180, 180, 180), 1)
            cv2.putText(frame, f"Xac nhan chay: {stream.tracker.remaining:.1f}s con lai", (bx1 + 6, bar_y + bar_h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

            if is_confirmed:
                ov = frame.copy()
                bnh = 56
                cv2.rectangle(ov, (0, h // 2 - bnh // 2), (w, h // 2 + bnh // 2), (0, 0, 180), -1)
                cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
                alert = "!!! DAM CHAY DA DUOC XAC NHAN !!!"
                (atw, _), _ = cv2.getTextSize(alert, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
                cv2.putText(frame, alert, ((w - atw) // 2, h // 2 + 10), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

        # Info panel mờ góc trái
        pw, ph_ = 330, 112
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (pw, ph_), (15, 17, 28), -1)
        cv2.addWeighted(ov, 0.80, frame, 0.20, 0, frame)
        cv2.rectangle(frame, (0, 0), (pw, ph_), (50, 60, 100), 1)

        vis = [d for d in detections if self._is_class_visible(d.class_name)]
        vis_fire = [d for d in fire_dets if self._is_class_visible(d.class_name)]
        nf = len(vis_fire)
        fc = (50, 50, 255) if nf > 0 else (200, 200, 200)
        
        cv2.putText(frame, f"FPS: {stream.fps:.1f}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 220, 255), 1)
        cv2.putText(frame, f"Latency: {stream.inference_latency_ms:.1f} ms", (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 220, 255), 1)
        cv2.putText(frame, f"Count: {len(vis)} | Fire/Smoke: {nf}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.52, fc, 1)
        roi_state = "ROI ON" if stream.roi_enabled else ("ROI EDIT" if stream.roi_points else "ROI OFF")
        cv2.putText(frame, f"{roi_state} | {'ALERT' if nf > 0 else 'Normal'}", (10, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.52, fc, 1)

        # Badge Recording nếu đang ghi hình thủ công
        if stream.manual_recording:
            cv2.putText(frame, "● REC", (w - 80, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        return frame
        
    def _save_event_video(self, frames: list, source_name: str):
        if not frames: return
        try:
            logger.info("[DIAG-SAVE] ⏱ Bắt đầu _save_event_video: %s (%d frames)", source_name, len(frames))
            event_dir = getattr(Config, "EVENT_DIR", "events")
            if not os.path.exists(event_dir):
                os.makedirs(event_dir)
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = unicodedata.normalize('NFKD', source_name).encode('ascii', 'ignore').decode('ascii')
            safe_name = "".join([c if c.isalnum() else "_" for c in safe_name])
            safe_name = safe_name.replace("__", "_").strip("_")
            
            filepath = os.path.join(event_dir, f"FIRE_{safe_name}_{timestamp}.avi")
            
            h, w = frames[0].shape[:2]
            
            # Đảm bảo width và height là số CHẴN (Bắt buộc với một số codec để tránh crash C++)
            if w % 2 != 0: w -= 1
            if h % 2 != 0: h -= 1
            
            logger.info("[DIAG-SAVE] ⏱ Tạo VideoWriter: %s (%dx%d)", filepath, w, h)
            import sys
            sys.stdout.flush()
            
            # Sử dụng XVID và .avi thay vì mp4v để tránh lỗi GIL Deadlock / Freeze trên Windows Media Foundation
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(filepath, fourcc, RECORD_FPS, (w, h))
            if not out.isOpened():
                logger.error("❌ VideoWriter KHÔNG thể mở: %s", filepath)
                return
            
            logger.info("[DIAG-SAVE] ⏱ Bắt đầu ghi %d frames...", len(frames))
            sys.stdout.flush()
            for i, f in enumerate(frames):
                # Resize frame nếu bị lẻ kích thước
                if f.shape[1] != w or f.shape[0] != h:
                    f = cv2.resize(f, (w, h))
                out.write(f)
            out.release()
            logger.info("[DIAG-SAVE] ✅ Đã lưu video tại: %s (%d frames)", filepath, len(frames))
        except Exception as e:
            logger.error("❌ Lỗi ghi video sự cố: %s", e, exc_info=True)

    def on_closing(self):
        self.stop_all_cameras()
        if self.system_monitor_agent:
            self.system_monitor_agent.stop(timeout=1.0)
        if hasattr(self, "mqtt_publisher"):
            self.mqtt_publisher.stop()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        self.destroy()


if __name__ == "__main__":
    setup_logging()

    # Phase 1: Login — chạy root Tk riêng, mainloop độc lập
    login_app = LoginApp()
    login_app.mainloop()

    # Phase 2: Nếu đăng nhập thành công → khởi động app chính
    if login_app.authenticated:
        app = SecurityApp()
        app.mainloop()
