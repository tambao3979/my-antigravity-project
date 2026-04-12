"""
gui.py - Camera AI – Hệ Thống Giám Sát Đa Màn Hình
"""

import logging
import os
import sys
import threading
import time
from datetime import datetime
from typing import Optional, Dict

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image

from config import Config
from detector import ObjectDetector
from fire_tracker import FireTracker
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

_OBJECT_PALETTE = [
    (255, 180,   0), (  0, 220, 100), (200,   0, 255),
    (  0, 180, 255), (120,   0, 200), (  0, 255, 180),
    (255,  80, 180), ( 30, 200, 255), (200, 120,   0),
    (  0,  80, 255),
]


class _TextboxHandler(logging.Handler):
    def __init__(self, textbox: ctk.CTkTextbox):
        super().__init__()
        self.textbox = textbox

    def emit(self, record: logging.LogRecord):
        msg = self.format(record) + "\n"
        def _append():
            try:
                self.textbox.insert("end", msg)
                self.textbox.see("end")
            except Exception:
                pass
        # Phải dùng after() để đưa lệnh update UI về main thread, tránh crash/garble log
        self.textbox.after(0, _append)

class _StdoutRedirect:
    def __init__(self, textbox: ctk.CTkTextbox):
        self.textbox = textbox

    def write(self, s: str):
        if not s.strip(): return
        def _append():
            try:
                self.textbox.insert("end", s)
                self.textbox.see("end")
            except Exception:
                pass
        self.textbox.after(0, _append)

    def flush(self): pass


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
        
        self.fps = 0.0
        self._fps_start = time.time()
        self._frame_count = 0
        self.frame_id = 0 # biến đếm an toàn thay cho id(frame)
        self.is_video_file = False
        self.is_paused = False
        
        # Phiên bản 2.0: Ring Buffer & Recording
        import collections
        self.frame_buffer = collections.deque(maxlen=150)
        self.is_recording = False
        self.record_frames_left = 0
        self.record_cooldown = 0

    def start(self):
        src_val = int(self.source) if self.source.isdigit() else self.source.strip('"').strip("'")
        self.cap = cv2.VideoCapture(src_val)
        if not self.cap.isOpened():
            logger.error("Không thể mở nguồn: %s", self.source)
            self.cap = None
            return False

        self.is_video_file = isinstance(src_val, str) and not src_val.startswith(("http://", "https://", "rtsp://"))
        
        self.video_fps = 30.0 # Mặc định
        if self.is_video_file:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                self.video_fps = fps
        self.frame_delay = 1.0 / self.video_fps
        if Config.FRAME_WIDTH > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)

        self.is_running = True
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
                    self.is_running = False
                    break
                logger.warning(f"[Camera] ⚠️ Mất kết nối từ {self.source}. Đang thử lại sau 3s...")
                time.sleep(3)
                if self.cap:
                    self.cap.release()
                src_val = int(self.source) if self.source.isdigit() else self.source.strip('"').strip("'")
                self.cap = cv2.VideoCapture(src_val)
                continue
            
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
                # Ép khung hình chờ theo đúng chuẩn thời gian thực (ví dụ 30fps = chờ 0.033s)
                elapsed = time.time() - last_read_time
                if elapsed < self.frame_delay:
                    time.sleep(self.frame_delay - elapsed)
                last_read_time = time.time()
            else:
                time.sleep(0.01) # Giảm tải CPU cho camera thật

    def stop(self):
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None


class SecurityApp(ctk.CTk):

    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.configure(fg_color=CLR_BG)

        self.title("🔥 Camera AI – Hệ Thống Giám Sát")
        self.geometry("1400x820")
        self.minsize(1100, 680)

        # ── State ──
        self.camera_sources = [str(Config.CAMERA_INDEX)]
        self.active_sources_vars: Dict[str, ctk.BooleanVar] = {}
        
        self.streams: Dict[str, CameraStream] = {}
        self.video_labels: Dict[str, ctk.CTkLabel] = {}
        
        self.class_colors: dict = {}
        self.class_visibility: dict = {}
        self._filter_widgets: dict = {}

        self.is_inferencing = False
        self._inference_thread: Optional[threading.Thread] = None
        self._cached_allowed_ids: Optional[set] = None

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

        # Logging setup
        if not os.path.exists("logs"):
            os.makedirs("logs")
        log_filename = datetime.now().strftime("logs/audit_%Y_%m.log")
        file_handler = logging.FileHandler(log_filename, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
        logging.getLogger().addHandler(file_handler)

        handler = _TextboxHandler(self.log_textbox)
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
        logging.getLogger().addHandler(handler)
        
        # Vẫn dùng để catch sys.stdout/stderr lên UI, và nó sẽ in ra file log nữa do logger
        sys.stdout = _StdoutRedirect(self.log_textbox)
        sys.stderr = _StdoutRedirect(self.log_textbox)

        logger.info("=" * 50)
        logger.info("✅ Giao diện Camera AI đã khởi động")
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
        ctk.CTkLabel(logo, text="⬤", font=ctk.CTkFont(size=18), text_color=CLR_RED).pack(side="left", padx=(0, 6))
        ctk.CTkLabel(logo, text="Camera AI", font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"), text_color=CLR_TEXT).pack(side="left")
        ctk.CTkLabel(logo, text=" · Hệ Thống Giám Sát Đa Màn Hình", font=ctk.CTkFont(family="Segoe UI", size=13), text_color=CLR_TEXT_DIM).pack(side="left")

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

        # ── Cấu Hình API (Webhook) ──
        ctk.CTkLabel(panel, text="🌐  CẤU HÌNH API", font=ctk.CTkFont(size=11, weight="bold"), text_color=CLR_TEXT_DIM).grid(row=5, column=0, padx=14, pady=(8, 6), sticky="w")
        
        api_frame = ctk.CTkFrame(panel, fg_color=CLR_PANEL2, corner_radius=8)
        api_frame.grid(row=6, column=0, padx=12, pady=(0, 16), sticky="ew")
        api_frame.grid_columnconfigure(0, weight=1)
        
        self.api_entry = ctk.CTkEntry(api_frame, placeholder_text="Nhập URL API Webhook...", fg_color="#2a2d3e", border_color=CLR_BORDER, text_color=CLR_TEXT, font=ctk.CTkFont(size=11))
        self.api_entry.grid(row=0, column=0, padx=8, pady=(8, 4), sticky="ew")
        # Load current API from config
        if hasattr(Config, "WEBHOOK_URL") and Config.WEBHOOK_URL:
            self.api_entry.insert(0, Config.WEBHOOK_URL)
            
        self.btn_save_api = ctk.CTkButton(api_frame, text="Lưu API", height=28, fg_color="#1864ab", hover_color="#1864ab", font=ctk.CTkFont(size=11, weight="bold"), corner_radius=6, command=self._save_api_config)
        self.btn_save_api.grid(row=1, column=0, padx=8, pady=(0, 8), sticky="ew")

    def _save_api_config(self):
        new_api = self.api_entry.get().strip()
        Config.WEBHOOK_URL = new_api
        
        # Save to .env
        env_path = '.env'
        lines = []
        if os.path.exists(env_path):
            with open(env_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
        found = False
        with open(env_path, 'w', encoding='utf-8') as f:
            for line in lines:
                if line.startswith('WEBHOOK_URL='):
                    f.write(f'WEBHOOK_URL={new_api}\n')
                    found = True
                else:
                    f.write(line)
            if not found:
                f.write(f'WEBHOOK_URL={new_api}\n')
                
        logger.info(f"✅ Đã lưu API Webhook thành công: {new_api if new_api else 'Rỗng'}")
        self.btn_save_api.configure(text="Đã Lưu ✔️", fg_color=CLR_GREEN)
        self.after(2000, lambda: self.btn_save_api.configure(text="Lưu API", fg_color="#1864ab"))

    def _build_video_panel(self):
        outer = ctk.CTkFrame(self, fg_color=CLR_PANEL, corner_radius=12)
        outer.grid(row=1, column=1, sticky="nsew", padx=6, pady=(10, 6))
        outer.grid_rowconfigure(1, weight=1)
        outer.grid_columnconfigure(0, weight=1)

        vhdr = ctk.CTkFrame(outer, fg_color=CLR_PANEL2, height=36, corner_radius=0)
        vhdr.grid(row=0, column=0, sticky="ew")
        vhdr.grid_columnconfigure(1, weight=1)
        vhdr.grid_propagate(False)

        ctk.CTkLabel(vhdr, text="  🎥  HỆ THỐNG MÀN HÌNH", font=ctk.CTkFont(size=12, weight="bold"), text_color=CLR_TEXT_DIM).grid(row=0, column=0, padx=8, sticky="w")
        
        self.feed_count_label = ctk.CTkLabel(vhdr, text="0 camera đang bật", font=ctk.CTkFont(size=11), text_color=CLR_TEXT_DIM)
        self.feed_count_label.grid(row=0, column=1, padx=8, sticky="e")

        # Scrollable grid container cho nhiều camera
        self.video_grid = ctk.CTkScrollableFrame(outer, fg_color="black", corner_radius=0)
        self.video_grid.grid(row=1, column=0, sticky="nsew")

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

        ctk.CTkLabel(panel, text="🎯  BỘ LỌC HIỂN THỊ", font=ctk.CTkFont(size=11, weight="bold"), text_color=CLR_TEXT_DIM).grid(row=2, column=0, padx=14, pady=(8, 6), sticky="w")

        filt = ctk.CTkFrame(panel, fg_color=CLR_PANEL2, corner_radius=8)
        filt.grid(row=3, column=0, padx=12, pady=(0, 4), sticky="ew")
        filt.grid_columnconfigure(0, weight=1)

        self.filter_scroll = ctk.CTkScrollableFrame(filt, fg_color="transparent", height=120, corner_radius=6)
        self.filter_scroll.grid(row=1, column=0, padx=4, pady=(6, 6), sticky="ew")
        self.filter_scroll.grid_columnconfigure(0, weight=1)

        # 🎯 NGƯỠNG NHẬN DIỆN (THRESHOLDS)
        ctk.CTkLabel(panel, text="⚙️  NGƯỠNG NHẬN DIỆN (CONFIDENCE)", font=ctk.CTkFont(size=11, weight="bold"), text_color=CLR_TEXT_DIM).grid(row=4, column=0, padx=14, pady=(8, 2), sticky="w")
        
        thresh_frame = ctk.CTkFrame(panel, fg_color=CLR_PANEL2, corner_radius=8)
        thresh_frame.grid(row=5, column=0, padx=12, pady=(0, 4), sticky="ew")
        thresh_frame.grid_columnconfigure(0, weight=1)
        
        # Fire Slider
        hdr_f = ctk.CTkFrame(thresh_frame, fg_color="transparent")
        hdr_f.grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 0))
        ctk.CTkLabel(hdr_f, text="Lửa & Khói", font=ctk.CTkFont(size=10)).pack(side="left")
        self.lbl_val_fire = ctk.CTkLabel(hdr_f, text="25%", font=ctk.CTkFont(size=10, weight="bold"), text_color=CLR_RED)
        self.lbl_val_fire.pack(side="right")
        
        self.slider_fire = ctk.CTkSlider(thresh_frame, from_=0.05, to=0.75, number_of_steps=70, height=12, button_color=CLR_RED, progress_color=CLR_RED, command=self._on_threshold_change)
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


        ctk.CTkLabel(panel, text="📋  NHẬT KÝ SỰ KIỆN", font=ctk.CTkFont(size=11, weight="bold"), text_color=CLR_TEXT_DIM).grid(row=6, column=0, padx=14, pady=(8, 6), sticky="w")

        self.log_textbox = ctk.CTkTextbox(panel, wrap="word", fg_color=CLR_PANEL2, text_color=CLR_TEXT, font=ctk.CTkFont(family="Consolas", size=11), border_color=CLR_BORDER, border_width=1, corner_radius=8)
        self.log_textbox.grid(row=7, column=0, padx=12, pady=(0, 12), sticky="nsew")
        panel.grid_rowconfigure(7, weight=1)

    def _build_status_bar(self):
        bar = ctk.CTkFrame(self, fg_color=CLR_PANEL, height=30, corner_radius=0)
        bar.grid(row=2, column=0, columnspan=3, sticky="ew")
        bar.grid_columnconfigure(1, weight=1)
        bar.grid_propagate(False)

        self.statusbar_left = ctk.CTkLabel(bar, text="  🟢  Hệ thống sẵn sàng", font=ctk.CTkFont(size=11), text_color=CLR_TEXT_DIM)
        self.statusbar_left.grid(row=0, column=0, padx=10, sticky="w")
        ctk.CTkLabel(bar, text="Camera AI v2.0   ", font=ctk.CTkFont(size=11), text_color=CLR_TEXT_DIM).grid(row=0, column=2, padx=10, sticky="e")

    # ════════════════════════════════════════════════════════
    #  UI HELPERS
    # ════════════════════════════════════════════════════════

    def _stat_card(self, parent, title, value, color, row, col):
        card = ctk.CTkFrame(parent, fg_color="#1e2133", corner_radius=6)
        card.grid(row=row, column=col, padx=6, pady=6, sticky="ew")
        card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=9, weight="bold"), text_color=CLR_TEXT_DIM).grid(row=0, column=0, pady=(6, 0))
        lbl = ctk.CTkLabel(card, text=value, font=ctk.CTkFont(size=20, weight="bold"), text_color=color)
        lbl.grid(row=1, column=0, pady=(0, 6))
        return lbl

    def _tick_clock(self):
        self.clock_label.configure(text=datetime.now().strftime("%H:%M:%S  —  %d/%m/%Y"))
        self.after(1000, self._tick_clock)

    def _set_status(self, msg: str, color: str, icon: str = "⚪"):
        self.statusbar_left.configure(text=f"  {icon}  {msg}", text_color=color)

    def _on_threshold_change(self, value=None):
        if hasattr(self, 'detector'):
            fire_conf = self.slider_fire.get()
            def_conf = self.slider_def.get()
            self.lbl_val_fire.configure(text=f"{int(fire_conf*100)}%")
            self.lbl_val_def.configure(text=f"{int(def_conf*100)}%")
            # Khói dùng chung mốc lưới với Lửa
            self.detector.update_thresholds(fire_conf=fire_conf, smoke_conf=fire_conf, default_conf=def_conf)

    # ════════════════════════════════════════════════════════
    #  CLASS FILTER 
    # ════════════════════════════════════════════════════════

    def _register_class(self, class_name: str):
        if class_name in self.class_visibility: return
        var = ctk.BooleanVar(value=True)
        self.class_visibility[class_name] = var

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
        ctk.CTkCheckBox(row_f, text="", variable=var, width=30, checkbox_width=18, checkbox_height=18, fg_color=CLR_ACCENT, hover_color=CLR_ACCENT_DARK, border_color=CLR_BORDER, corner_radius=4).grid(row=0, column=2, padx=(4, 8), pady=4)

    def _is_class_visible(self, class_name: str) -> bool:
        return self.class_visibility.get(class_name, ctk.BooleanVar(value=True)).get()

    def _get_class_color(self, class_name: str) -> tuple:
        if class_name not in self.class_colors:
            self.class_colors[class_name] = _OBJECT_PALETTE[len(self.class_colors) % len(_OBJECT_PALETTE)]
        return self.class_colors[class_name]

    def _compute_allowed_ids(self) -> Optional[set]:
        enabled = {n for n, v in self.class_visibility.items() if v.get()}
        if len(enabled) == len(self.class_visibility): return None
        ids = set()
        for name in enabled:
            cid = self.detector.get_class_id(name)
            if cid >= 0: ids.add(cid)
        return ids if ids else None

    # ════════════════════════════════════════════════════════
    #  CAMERA LIST MANAGEMENT
    # ════════════════════════════════════════════════════════

    def _refresh_camera_list(self):
        for w in self.camera_list_frame.winfo_children(): w.destroy()

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

            ctk.CTkLabel(row_f, text=icon, font=ctk.CTkFont(size=14)).grid(row=0, column=0, padx=(8, 4), pady=6)
            
            cb = ctk.CTkCheckBox(row_f, text=label, variable=self.active_sources_vars[src], font=ctk.CTkFont(size=12, weight="bold" if is_sel else "normal"), text_color=CLR_TEXT if is_sel else CLR_TEXT_DIM, command=self._refresh_camera_list)
            cb.grid(row=0, column=1, padx=4, pady=6, sticky="w")
            
            # Badge trạng thái nếu đang chạy stream
            msg = "● LIVE" if src in self.streams and self.streams[src].is_running else ""
            if msg:
                ctk.CTkLabel(row_f, text=msg, font=ctk.CTkFont(size=10, weight="bold"), text_color=CLR_GREEN).grid(row=0, column=2, padx=(4, 8), pady=6)

    def _add_camera(self):
        src = self.add_cam_entry.get().strip()
        if src and src not in self.camera_sources:
            self.camera_sources.append(src)
            self.active_sources_vars[src] = ctk.BooleanVar(value=True) # tự động check khi thêm mới
            self._refresh_camera_list()
            self.add_cam_entry.delete(0, "end")
            logger.info("➕ Đã thêm camera: %s", src)
        else:
            logger.warning("Camera đã tồn tại hoặc bỏ trống!")

    # ════════════════════════════════════════════════════════
    #  MULTI-CAMERA INFERENCE & RENDER
    # ════════════════════════════════════════════════════════

    def start_cameras(self):
        selected = [src for src, var in self.active_sources_vars.items() if var.get()]
        if not selected:
            logger.error("Chưa chọn camera nào để bắt đầu!")
            return

        # Dọn dẹp trước
        self.stop_all_cameras()
        
        # Grid clear
        for widget in self.video_grid.winfo_children():
            widget.destroy()
        self.video_labels.clear()
        
        # Tính toán Layout (Chia lưới tự động) và kích thước ô vuông cố định
        n = len(selected)
        # Nếu có 1 camera: ưu tiên 1 cột to. 2-4 cam: 2 cột ngang. >=5 cam: 3 cột ngang
        cols = 1 if n == 1 else (2 if n <= 4 else (3 if n <= 9 else 4))
        
        # Bắt buộc UI tính toán xong layout để lấy chiều rộng chính xác thay vì lấy nhầm 1px
        self.video_grid.update_idletasks()
        grid_w = self.video_grid.winfo_width()
        
        if grid_w < 100: 
            grid_w = self.winfo_width() - 550 # Lấy tổng chiều rộng trừ đi 2 menu 2 bên (~550px)
            if grid_w < 100: grid_w = 900
        
        cell_w = max(200, (grid_w - 30) // cols) # hạ min xuống 200px để không bị cắt mép ngang
        cell_h = int(cell_w * 9 / 16) + 30       # Giữ tỷ lệ 16:9 và cộng thêm 30px cho Header
        
        for idx in range(n):
            self.video_grid.grid_columnconfigure(idx % cols, weight=1)
            self.video_grid.grid_rowconfigure(idx // cols, weight=1)

        # Kết nối
        for idx, src in enumerate(selected):
            stream = CameraStream(src)
            if stream.start():
                self.streams[src] = stream
                
                # Tạo khung chứa riêng cho camera này kèm kích thước cố định để video không bị phóng to từ từ
                frame_box = ctk.CTkFrame(self.video_grid, fg_color="#181a25", border_width=2, border_color=CLR_BORDER, width=cell_w, height=cell_h)
                frame_box.grid_propagate(False)
                r, c = idx // cols, idx % cols
                frame_box.grid(row=r, column=c, padx=4, pady=4, sticky="nsew")
                frame_box.grid_rowconfigure(1, weight=1)
                frame_box.grid_columnconfigure(0, weight=1)
                
                # Header cam
                hdr = ctk.CTkFrame(frame_box, fg_color="transparent", height=24)
                hdr.grid(row=0, column=0, sticky="ew")
                ctk.CTkLabel(hdr, text=f"  {stream.display_name}", font=ctk.CTkFont(size=11, weight="bold"), text_color=CLR_TEXT_DIM).pack(side="left")
                
                # Video label
                lbl = ctk.CTkLabel(frame_box, text="Đang tải...", text_color=CLR_TEXT_DIM)
                lbl.grid(row=1, column=0, sticky="nsew")
                self.video_labels[src] = lbl

        if not self.streams:
            self._set_status("LỖI: Chẳng có camera nào kết nối được", CLR_RED, "🔴")
            return

        self.feed_count_label.configure(text=f"{len(self.streams)} camera đang bật")
        self.btn_start.configure(state="disabled")
        self.btn_pause.configure(state="normal", text="⏸ Tạm Dừng", fg_color=CLR_YELLOW, text_color="black")
        self.btn_stop.configure(state="normal")
        self.add_cam_entry.configure(state="disabled")
        self.btn_add_cam.configure(state="disabled")
        self._set_status("ĐANG THEO DÕI", CLR_GREEN, "🟢")
        self._refresh_camera_list()

        # Khởi động thread inference chạy tuần tự cho tất cả camera
        self.is_inferencing = True
        self._inference_thread = threading.Thread(target=self._multi_inference_loop, daemon=True)
        self._inference_thread.start()

    def _multi_inference_loop(self):
        """Vòng lặp AI quét qua tất cả các luồng camera đang chạy để detect theo lô (batch)."""
        last_processed_frames = {} # dict lưu id frame cuối cùng đã chạy AI
        
        while self.is_inferencing:
            allowed_ids = self._cached_allowed_ids
            
            # Để tránh lỗi RuntimeError do dictionary size changes, catch ngoại lệ
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
                
                # Lấy frame mới nhất
                frame = stream.latest_frame
                # Chỉ xử lý nếu có frame và KHÁC với frame đã chạy lần trước
                if frame is not None and stream.frame_id != last_processed_frames.get(src):
                    batch_frames.append(frame)
                    batch_streams.append(stream)
                    last_processed_frames[src] = stream.frame_id
                    
            if batch_frames:
                # Chạy AI inference cho toàn bộ batch cùng 1 lúc
                batch_detections = self.detector.detect_batch(batch_frames, allowed_class_ids=allowed_ids)
                
                # Cập nhật kết quả về lại cho từng stream
                for stream, detections in zip(batch_streams, batch_detections):
                    stream.latest_detections = detections
                
                time.sleep(0.005) # Delay cực nhỏ để nhường CPU
            else:
                time.sleep(0.01) # Nhường CPU lâu hơn nếu không có frame mới

    def _render_loop(self):
        if not self.is_inferencing:
            self.after(50, self._render_loop)
            return

        self._cached_allowed_ids = self._compute_allowed_ids()
        
        total_obj = 0
        total_fire = 0
        fire_alerted = False

        active_streams = list(self.streams.values())
        
        for stream in active_streams:
            if not stream.is_running:
                continue
                
            frame = stream.latest_frame
            detections = stream.latest_detections
            
            if frame is not None:
                # Đăng ký class mới
                for det in detections:
                    if det.class_name not in self.class_visibility:
                        self._register_class(det.class_name)
                
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
                    
                    # 2.0 Event Recording Trigger
                    if not stream.is_recording and time.time() > stream.record_cooldown:
                        stream.is_recording = True
                        stream.record_frames_left = 150 # 5s tương lai ở 30fps
                        stream.rec_buffer = list(stream.frame_buffer) # ~5s quá khứ
                        logger.info(f"[Record] 🔴 Bắt đầu ghi hình sự cố cho {stream.display_name}...")

                # Ghi luồng Video tương lai
                if stream.is_recording and not stream.is_paused:
                    stream.rec_buffer.append(display_frame.copy())
                    stream.record_frames_left -= 1
                    
                    if stream.record_frames_left <= 0:
                        stream.is_recording = False
                        stream.record_cooldown = time.time() + Config.ALERT_COOLDOWN_SECONDS
                        frames_copy = stream.rec_buffer.copy()
                        stream.rec_buffer.clear()
                        threading.Thread(
                            target=self._save_event_video,
                            args=(frames_copy, stream.display_name),
                            daemon=True
                        ).start()

                # Render ra grid label
                lbl = self.video_labels.get(stream.source)
                if lbl:
                    self._display_on_label(display_frame, lbl)

                # Thống kê
                visible = [d for d in detections if self._is_class_visible(d.class_name)]
                n_fire = len([d for d in visible if d.is_fire])
                total_obj += len(visible)
                total_fire += n_fire
                if stream.tracker.is_tracking and stream.tracker.elapsed >= Config.FIRE_CONFIRM_SECONDS:
                    fire_alerted = True

        self._update_global_stats(total_obj, total_fire, fire_alerted, active_streams)
        
        # Dọn dẹp các stream đã chết
        dead_keys = [k for k, v in self.streams.items() if not v.is_running]
        for k in dead_keys:
            del self.streams[k]

        # Chỉ tự dừng khi đang inference và tất cả stream đã ngắt hết - không dừng khi chưa bắt đầu
        if self.is_inferencing and dead_keys:
            self._refresh_camera_list()
        if self.is_inferencing and not self.streams:
            logger.info("⌛ Tất cả camera đã ngắt, dừng hệ thống.")
            self.stop_all_cameras()
            return  # render loop sẽ được kết nối lại sau khi stop_all gọi after()

        self.after(30, self._render_loop)

    def _display_on_label(self, frame: np.ndarray, lbl: ctk.CTkLabel):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # parent frame size
        parent = lbl.master
        lw = parent.winfo_width()
        lh = parent.winfo_height() - 24 # - header height
        if lw > 10 and lh > 10:
            ih, iw = rgb.shape[:2]
            scale = min(lw / iw, lh / ih)
            nw, nh = int(iw * scale), int(ih * scale)
            rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
            pil = Image.fromarray(rgb)
            ctk_img = ctk.CTkImage(light_image=pil, dark_image=pil, size=(nw, nh))
            lbl.configure(image=ctk_img, text="")
            lbl.image = ctk_img

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
        logger.info("⏹  Đang dừng toàn bộ camera...")
        self.is_inferencing = False
        
        # Chờ inference thread tắt
        if self._inference_thread and self._inference_thread.is_alive():
            self._inference_thread.join(timeout=2.0)
        self._inference_thread = None

        # Dừng từng luồng
        for stream in self.streams.values():
            stream.stop()
        self.streams.clear()

        # Xóa Grid UI
        for widget in self.video_grid.winfo_children():
            widget.destroy()
        self.video_labels.clear()

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
        # Tiếp tục render loop ở idle mode (không dừng vòng lặp)
        self.after(200, self._render_loop)

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

    def _draw_overlay(self, frame: np.ndarray, detections: list, stream: CameraStream) -> np.ndarray:
        frame = frame.copy()
        h, w = frame.shape[:2]
        fire_dets = []
        is_confirmed = (stream.tracker.is_tracking and stream.tracker.elapsed >= Config.FIRE_CONFIRM_SECONDS)

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
        pw, ph_ = 280, 80
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (pw, ph_), (15, 17, 28), -1)
        cv2.addWeighted(ov, 0.80, frame, 0.20, 0, frame)
        cv2.rectangle(frame, (0, 0), (pw, ph_), (50, 60, 100), 1)

        vis = [d for d in detections if self._is_class_visible(d.class_name)]
        vis_fire = [d for d in fire_dets if self._is_class_visible(d.class_name)]
        nf = len(vis_fire)
        fc = (50, 50, 255) if nf > 0 else (200, 200, 200)
        
        cv2.putText(frame, f"FPS: {stream.fps:.1f}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 220, 255), 1)
        cv2.putText(frame, f"Vat the: {len(vis)} | Lua/Khoi: {nf}", (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.52, fc, 1)
        cv2.putText(frame, f"{'!!! CANH BAO CHAY !!!' if nf > 0 else 'Binh thuong'}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.52, fc, 1)

        return frame
        
    def _save_event_video(self, frames: list, source_name: str):
        if not frames: return
        try:
            event_dir = getattr(Config, "EVENT_DIR", "events")
            if not os.path.exists(event_dir):
                os.makedirs(event_dir)
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = source_name.replace(":", "").replace("/", "_").replace("\\", "_")
            filepath = os.path.join(event_dir, f"FIRE_{safe_name}_{timestamp}.mp4")
            
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filepath, fourcc, 30.0, (w, h))
            
            for f in frames:
                out.write(f)
            out.release()
            logger.info(f"✅ Đã lưu Video bằng chứng Sự cố (10s) tại: {filepath}")
        except Exception as e:
            logger.error(f"❌ Lỗi ghi video sự cố: {str(e)}")

    def on_closing(self):
        self.stop_all_cameras()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        self.destroy()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.__stdout__)])
    app = SecurityApp()
    app.mainloop()
