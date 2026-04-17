import logging
import os
import sys
from logging.handlers import RotatingFileHandler

def setup_logging():
    """
    Thiết lập logging toàn cục (Root Logger) của toàn hệ thống.
    """
    root_logger = logging.getLogger()
    
    # Nếu root logger đã được cấu hình các Handler này thì bỏ qua (tránh lặp log)
    if hasattr(root_logger, '_custom_configured'):
        return
        
    root_logger.setLevel(logging.INFO)
    
    # Dọn dẹp config cũ nếu có
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)

    # Đảm bảo thư mục logs tồn tại
    if not os.path.exists("logs"):
        os.makedirs("logs")

    # 1. Console Handler (Standard Output)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    # Định dạng Console: [TT:TT:TT] [MỨC] [MODULE] Nội dung
    console_formatter = logging.Formatter(
        "[%(_asctime)s] [%(levelname)s] [%(name)s] %(message)s", 
        datefmt="%H:%M:%S"
    )
    # Monkey patch để xài asctime cho thân thiện nếu cần, ở đây ta dùng chuẩn
    console_formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)

    # 2. System Log Handler (Chứa đủ mọi thứ từ INFO trở lên, luân phiên file)
    system_handler = RotatingFileHandler(
        "logs/system.log", 
        maxBytes=10 * 1024 * 1024, # 10 MB
        backupCount=5, 
        encoding="utf-8"
    )
    system_handler.setLevel(logging.INFO)
    system_formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    system_handler.setFormatter(system_formatter)

    # 3. Error Log Handler (Chỉ chứa WARNING, ERROR, CRITICAL + Traceback)
    error_handler = RotatingFileHandler(
        "logs/error.log",
        maxBytes=10 * 1024 * 1024, # 10 MB
        backupCount=5,
        encoding="utf-8"
    )
    error_handler.setLevel(logging.WARNING)
    # Định dạng chi tiết để debug lỗi
    error_formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(threadName)s] [%(name)s:%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    error_handler.setFormatter(error_formatter)

    # Gắn handler vào root
    root_logger.addHandler(console_handler)
    root_logger.addHandler(system_handler)
    root_logger.addHandler(error_handler)
    
    root_logger._custom_configured = True
    
    # Log 1 câu để biết đã thiết lập xong
    root_logger.info("✅ Hệ thống Logging chuyên nghiệp đã được khởi tạo.")
