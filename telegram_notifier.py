"""
telegram_notifier.py - Gửi thông báo cảnh báo cháy qua:
  1. Telegram Bot API
  2. Zalo OA (Official Account) API
  3. Webhook tuỳ chỉnh (Third-party)

Có cooldown 60 giây giữa các lần gửi để tránh spam.
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
    Gửi thông báo + ảnh qua Telegram, Zalo OA và Webhook khi phát hiện đám cháy.

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
                "Thông báo Telegram sẽ không được gửi!"
            )
        else:
            logger.info("Telegram Bot sẵn sàng. Chat ID: %s", self._chat_id)

        # Log Zalo status
        _zalo_token = getattr(Config, "ZALO_OA_TOKEN", "")
        _zalo_uid   = getattr(Config, "ZALO_USER_ID", "")
        if _zalo_token and _zalo_uid:
            logger.info("Zalo OA sẵn sàng. User ID: %s", _zalo_uid)
        else:
            logger.warning(
                "ZALO_OA_TOKEN hoặc ZALO_USER_ID chưa được cấu hình. "
                "Thông báo Zalo sẽ không được gửi!"
            )

    def reload_credentials(self):
        """Reload token/id từ Config sau khi user lưu mới qua GUI."""
        self._token   = Config.TELEGRAM_BOT_TOKEN
        self._chat_id = Config.TELEGRAM_CHAT_ID
        self._base_url = f"https://api.telegram.org/bot{self._token}"

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
            source_name: Tên nguồn camera

        Returns:
            True nếu ít nhất 1 kênh gửi thành công
        """
        if not self.can_send(source_name):
            logger.info("Cooldown cho [%s] còn %.1fs, bỏ qua.", source_name, self.cooldown_remaining(source_name))
            return False

        # Kiểm tra có ít nhất 1 kênh được cấu hình
        zalo_token = getattr(Config, "ZALO_OA_TOKEN", "")
        zalo_uid   = getattr(Config, "ZALO_USER_ID", "")
        webhook_url = getattr(Config, "WEBHOOK_URL", "")
        has_telegram = bool(self._token and self._chat_id)
        has_zalo     = bool(zalo_token and zalo_uid)
        has_webhook  = bool(webhook_url)

        if not has_telegram and not has_zalo and not has_webhook:
            logger.warning("Không có kênh thông báo nào được cấu hình, bỏ qua cảnh báo.")
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

        # ── Kênh 1: Telegram ──
        if has_telegram:
            ok = self._send_photo_with_caption(frame, message)
            if ok:
                logger.info("✅ Cảnh báo Telegram cho [%s] đã gửi thành công", source_name)
                success = True
            else:
                logger.error("❌ Gửi Telegram thất bại.")

        # ── Kênh 2: Zalo OA ──
        if has_zalo:
            plain_msg = (
                f"🔥 CẢNH BÁO CHÁY!\n"
                f"📹 Nguồn: {source_name}\n"
                f"📅 Thời gian: {now_str}\n"
                f"🔍 Phát hiện: {num_detections} vùng lửa\n"
                f"⚠️ Hãy kiểm tra ngay lập tức!"
            )
            ok = self._send_zalo(zalo_token, zalo_uid, frame, plain_msg)
            success = success or ok

        # ── Kênh 3: Webhook tuỳ chỉnh ──
        if has_webhook:
            ok = self._send_webhook(webhook_url, frame, source_name, num_detections)
            success = success or ok

        if success:
            self._last_sent_times[source_name] = time.time()

        return success

    # ─────────────────────────────────────────────────────────────
    #  Telegram
    # ─────────────────────────────────────────────────────────────

    def _send_photo_with_caption(self, frame: np.ndarray, caption: str) -> bool:
        """Gửi ảnh kèm caption qua sendPhoto API"""
        try:
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
                logger.error("Telegram API lỗi %d: %s", response.status_code, response.text[:200])
                return False

        except requests.exceptions.ConnectionError:
            logger.error("Telegram: Lỗi kết nối. Kiểm tra internet.")
            return False
        except requests.exceptions.Timeout:
            logger.error("Telegram: Request timeout.")
            return False
        except Exception as e:
            logger.error("Telegram: Lỗi không xác định: %s", e)
            return False

    # ─────────────────────────────────────────────────────────────
    #  Zalo OA
    # ─────────────────────────────────────────────────────────────

    def _send_zalo(self, token: str, user_id: str, frame: np.ndarray, text: str) -> bool:
        """
        Gửi tin nhắn kèm ảnh qua Zalo OA API.
        Bước 1: Upload ảnh → lấy attachment_id
        Bước 2: Gửi tin nhắn có ảnh đính kèm
        """
        headers = {"access_token": token}

        # Bước 1: Upload ảnh
        try:
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            img_bytes = io.BytesIO(buffer.tobytes())
            img_bytes.name = "fire_alert.jpg"

            upload_resp = requests.post(
                url="https://openapi.zalo.me/v2.0/oa/upload/image",
                headers=headers,
                files={"file": ("fire_alert.jpg", img_bytes, "image/jpeg")},
                timeout=15
            )
            upload_data = upload_resp.json()
            attachment_id = upload_data.get("data", {}).get("attachment_id", "")
            if not attachment_id:
                logger.error("❌ Zalo: Upload ảnh thất bại — %s", upload_data)
                # Fallback: chỉ gửi text
                return self._send_zalo_text(token, user_id, text)
        except Exception as e:
            logger.error("❌ Zalo: Lỗi upload ảnh: %s", e)
            return self._send_zalo_text(token, user_id, text)

        # Bước 2: Gửi tin nhắn có ảnh
        try:
            payload = {
                "recipient": {"user_id": user_id},
                "message": {
                    "attachment": {
                        "type": "image",
                        "payload": {"attachment_id": attachment_id}
                    },
                    "text": text
                }
            }
            resp = requests.post(
                url="https://openapi.zalo.me/v2.0/oa/message",
                headers={**headers, "Content-Type": "application/json"},
                json=payload,
                timeout=10
            )
            data = resp.json()
            if data.get("error") == 0:
                logger.info("✅ Cảnh báo Zalo đã gửi thành công.")
                return True
            else:
                logger.error("❌ Zalo: Gửi tin nhắn thất bại — %s", data)
                return False
        except Exception as e:
            logger.error("❌ Zalo: Lỗi gửi tin nhắn: %s", e)
            return False

    def _send_zalo_text(self, token: str, user_id: str, text: str) -> bool:
        """Fallback: gửi chỉ text khi upload ảnh thất bại."""
        try:
            payload = {
                "recipient": {"user_id": user_id},
                "message": {"text": text}
            }
            resp = requests.post(
                url="https://openapi.zalo.me/v2.0/oa/message",
                headers={"access_token": token, "Content-Type": "application/json"},
                json=payload,
                timeout=10
            )
            data = resp.json()
            if data.get("error") == 0:
                logger.info("✅ Zalo (text-only) gửi thành công.")
                return True
            else:
                logger.error("❌ Zalo text: %s", data)
                return False
        except Exception as e:
            logger.error("❌ Zalo text: %s", e)
            return False

    # ─────────────────────────────────────────────────────────────
    #  Webhook tuỳ chỉnh
    # ─────────────────────────────────────────────────────────────

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
                logger.error(f"❌ Webhook API trả về mã lỗi {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Lỗi gửi Webhook API: {e}")
            return False
