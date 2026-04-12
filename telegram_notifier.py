"""
telegram_notifier.py - Gửi thông báo cảnh báo cháy qua Telegram Bot API
Có cooldown 60 giây giữa các lần gửi để tránh spam
"""

import io
import logging
import time

import cv2
import numpy as np
import requests
from config import Config

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Gửi thông báo + ảnh qua Telegram khi phát hiện đám cháy.

    Đảm bảo khoảng cách tối thiểu giữa các lần gửi = ALERT_COOLDOWN_SECONDS.
    """

    def __init__(self):
        self._last_sent_times: dict[str, float] = {}  # {source_name: timestamp}
        self._token = Config.TELEGRAM_BOT_TOKEN
        self._chat_id = Config.TELEGRAM_CHAT_ID
        self._base_url = f"https://api.telegram.org/bot{self._token}"

        if not self._token or not self._chat_id:
            logger.warning(
                "TELEGRAM_BOT_TOKEN hoặc TELEGRAM_CHAT_ID chưa được cấu hình. "
                "Thông báo sẽ không được gửi!"
            )
        else:
            logger.info("Telegram Bot sẵn sàng. Chat ID: %s", self._chat_id)

    def cooldown_remaining(self, source_name: str = "default") -> float:
        """Số giây còn lại trong cooldown cho camera cụ thể (0 nếu có thể gửi ngay)"""
        last_sent = self._last_sent_times.get(source_name, 0.0)
        elapsed = time.time() - last_sent
        return max(0.0, Config.ALERT_COOLDOWN_SECONDS - elapsed)

    def can_send(self, source_name: str = "default") -> bool:
        """True nếu có thể gửi thông báo (cooldown đã hết)"""
        return self.cooldown_remaining(source_name) == 0.0

    def send_fire_alert(self, frame: np.ndarray, num_detections: int = 1, source_name: str = "Không rõ") -> bool:
        """
        Gửi cảnh báo cháy kèm ảnh chụp màn hình camera.

        Args:
            frame: Frame OpenCV tại thời điểm phát hiện cháy
            num_detections: Số lượng vùng lửa được phát hiện

        Returns:
            True nếu gửi thành công, False nếu thất bại hoặc còn trong cooldown
        """
        if not self.can_send(source_name):
            logger.info("Cooldown cho [%s] còn %.1fs, bỏ qua.", source_name, self.cooldown_remaining(source_name))
            return False

        webhook_url = getattr(Config, "WEBHOOK_URL", "")
        if not self._token and not webhook_url:
            logger.warning("Cả Telegram Bot và Webhook API đều trống, bỏ qua cảnh báo.")
            return False

        now_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        message = (
            f"🔥 CẢNH BÁO CHÁY!\n\n"
            f"📹 Nguồn: <b>{source_name}</b>\n"
            f"📅 Thời gian: {now_str}\n"
            f"🔍 Phát hiện: {num_detections} vùng lửa\n"
            f"⏱ Đã xác nhận: liên tục {Config.FIRE_CONFIRM_SECONDS:.0f} giây\n\n"
            f"⚠️ Hãy kiểm tra ngay lập tức!"
        )

        success = False
        
        # Gửi sang Telegram nếu cấu hình
        if self._token and self._chat_id:
            success = self._send_photo_with_caption(frame, message)
            if success:
                logger.info("✅ Cảnh báo Telegram cho [%s] đã gửi thành công", source_name)
            else:
                logger.error("❌ Gửi Telegram thất bại.")
                
        # Gửi sang Webhook API (Third-party App)
        if webhook_url:
            webhook_success = self._send_webhook(webhook_url, frame, source_name, num_detections)
            success = success or webhook_success

        if success:
            self._last_sent_times[source_name] = time.time()

        return success

    def _send_photo_with_caption(self, frame: np.ndarray, caption: str) -> bool:
        """Gửi ảnh kèm caption qua sendPhoto API"""
        try:
            # Encode frame thành JPEG bytes
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            image_bytes = io.BytesIO(buffer.tobytes())
            image_bytes.name = "fire_alert.jpg"

            response = requests.post(
                url=f"{self._base_url}/sendPhoto",
                data={
                    "chat_id": self._chat_id,
                    "caption": caption,
                    "parse_mode": "HTML"
                },
                files={"photo": ("fire_alert.jpg", image_bytes, "image/jpeg")},
                timeout=10
            )

            if response.status_code == 200:
                return True
            else:
                logger.error("API lỗi %d: %s", response.status_code, response.text[:200])
                return False

        except requests.exceptions.ConnectionError:
            logger.error("Lỗi kết nối. Kiểm tra internet.")
            return False
        except requests.exceptions.Timeout:
            logger.error("Request timeout.")
            return False
        except Exception as e:
            logger.error("Lỗi không xác định: %s", e)
            return False

    def _send_webhook(self, url: str, frame: np.ndarray, source_name: str, num_detections: int) -> bool:
        """Gửi Multipart request kèm JSON data và ảnh về API URL của User"""
        try:
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            data = {
                "source": source_name,
                "timestamp": int(time.time()),
                "detections": num_detections,
                "message": "CẢNH BÁO CHÁY"
            }
            
            files = {
                "image": ("alert.jpg", buffer.tobytes(), "image/jpeg")
            }
            
            response = requests.post(url, data=data, files=files, timeout=5)
            if response.status_code in (200, 201):
                logger.info("✅ Đã bắn Webhook API tới ứng dụng User thành công.")
                return True
            else:
                logger.error(f"❌ Cảnh báo: Webhook API trả về mã lỗi {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Lỗi gửi Webhook API: {e}")
            return False
        """Gửi tin nhắn văn bản thuần (dùng để test)"""
        try:
            response = requests.post(
                url=f"{self._base_url}/sendMessage",
                data={"chat_id": self._chat_id, "text": message},
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            logger.error("Lỗi gửi text: %s", e)
            return False
