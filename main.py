"""
main.py - Entry point cho hệ thống camera AI (chế độ terminal / headless)
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)

from config import Config
from camera import CameraSystem


def check_env():
    """Kiểm tra cấu hình cơ bản trước khi chạy"""
    logger.info("=" * 60)
    logger.info("   🔥 AI Camera Fire Detection System")
    logger.info("=" * 60)

    # Kiểm tra model
    if not os.path.exists(Config.MODEL_PATH):
        logger.info(f"[Config] ❌ Không tìm thấy model tại: {Config.MODEL_PATH}")
        logger.info("         Kiểm tra lại MODEL_PATH trong file .env")
        sys.exit(1)
    else:
        logger.info(f"[Config] ✅ Model: {Config.MODEL_PATH}")

    # Kiểm tra Telegram
    if not Config.TELEGRAM_BOT_TOKEN or Config.TELEGRAM_BOT_TOKEN == "your_bot_token_here":
        logger.info("[Config] ⚠️  Telegram chưa cấu hình - thông báo SẼ KHÔNG được gửi")
        logger.info("         Hãy điền TELEGRAM_BOT_TOKEN và TELEGRAM_CHAT_ID vào file .env")
    else:
        logger.info(f"[Config] ✅ Telegram: Bot đã cấu hình (Chat ID: {Config.TELEGRAM_CHAT_ID})")

    logger.info(f"[Config] 🎯 Nhãn lửa: '{Config.FIRE_CLASS_NAME}'")
    logger.info(f"[Config] ⏱  Thời gian xác nhận: {Config.FIRE_CONFIRM_SECONDS}s")
    logger.info(f"[Config] 🔔 Cooldown thông báo: {Config.ALERT_COOLDOWN_SECONDS}s")
    logger.info(f"[Config] 📷 Camera index: {Config.CAMERA_INDEX}")
    logger.info("=" * 60)
    logger.info("")


def select_video_source():
    """Hiển thị menu và trả về nguồn video được chọn"""
    print("\n" + "=" * 60)
    print("   CHỌN NGUỒN VIDEO ĐỂ DỰ ĐOÁN")
    print("=" * 60)
    print("1. Camera (Webcam/USB)")
    print("2. File Video (.mp4, .avi, ...)")
    print("3. RTSP / Stream URL")
    print("=" * 60)
    
    while True:
        choice = input("Lựa chọn của bạn (1-3) [mặc định: 1]: ").strip()
        if not choice or choice == "1":
            try:
                cam_id = input(f"Nhập ID camera [mặc định: {Config.CAMERA_INDEX}]: ").strip()
                return int(cam_id) if cam_id else Config.CAMERA_INDEX
            except ValueError:
                print("❌ ID camera phải là số nguyên!")
                continue
        elif choice == "2":
            path = input("Nhập đường dẫn file video: ").strip().strip('"').strip("'")
            if path and os.path.exists(path):
                return path
            else:
                print("❌ File không tồn tại, vui lòng nhập lại!")
                continue
        elif choice == "3":
            url = input("Nhập địa chỉ RTSP/Stream URL: ").strip()
            if url:
                return url
            else:
                print("❌ URL không được để trống!")
                continue
        else:
            print("❌ Lựa chọn không hợp lệ, vui lòng chọn 1, 2, hoặc 3.")


def main():
    check_env()

    # Chọn nguồn video
    source = select_video_source()

    # Khởi chạy hệ thống camera
    system = CameraSystem()
    system.run(source)


if __name__ == "__main__":
    if not os.path.exists("logs"):
        os.makedirs("logs")
    
    import time
    log_filename = time.strftime("logs/audit_%Y_%m.log")
    
    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    
    stream_handler = logging.StreamHandler(sys.__stdout__)
    stream_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%H:%M:%S"))

    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, stream_handler]
    )
    main()
