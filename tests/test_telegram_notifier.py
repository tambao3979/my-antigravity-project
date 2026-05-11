import unittest
from unittest.mock import patch

import numpy as np
import requests

from config import Config
from telegram_notifier import TelegramNotifier


class _Response:
    def __init__(self, status_code=200, text="ok"):
        self.status_code = status_code
        self.text = text


class TelegramNotifierTests(unittest.TestCase):
    def setUp(self):
        self._send_retries = getattr(Config, "ALERT_SEND_RETRIES", None)
        self._retry_backoff = getattr(Config, "ALERT_RETRY_BACKOFF_SECONDS", None)
        Config.ALERT_SEND_RETRIES = 2
        Config.ALERT_RETRY_BACKOFF_SECONDS = 0.0

    def tearDown(self):
        if self._send_retries is None:
            delattr(Config, "ALERT_SEND_RETRIES")
        else:
            Config.ALERT_SEND_RETRIES = self._send_retries
        if self._retry_backoff is None:
            delattr(Config, "ALERT_RETRY_BACKOFF_SECONDS")
        else:
            Config.ALERT_RETRY_BACKOFF_SECONDS = self._retry_backoff

    def test_send_photo_retries_transient_request_failure(self):
        notifier = object.__new__(TelegramNotifier)
        notifier._base_url = "https://api.telegram.org/bottest"
        notifier._chat_id = "chat-1"
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        encoded = np.array([1, 2, 3], dtype=np.uint8)

        with (
            patch("telegram_notifier.cv2.imencode", return_value=(True, encoded)),
            patch(
                "telegram_notifier.requests.post",
                side_effect=[requests.exceptions.ConnectionError(), _Response(200)],
            ) as post,
            patch("telegram_notifier.time.sleep"),
        ):
            self.assertTrue(notifier._send_photo_with_caption(frame, "fire"))

        self.assertEqual(post.call_count, 2)


if __name__ == "__main__":
    unittest.main()
