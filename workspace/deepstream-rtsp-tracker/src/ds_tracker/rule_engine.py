"""Rule engine: áp luật trên action_label theo track_id → cảnh báo.

Theo CLAUDE.md giai đoạn 5. Đầu vào là kết quả đọc được TỪ probe (osd_probe.py),
KHÔNG tự đọc metadata GStreamer — tách rời để test luật độc lập, không cần
pipeline thật (unit test thuần Python).

action_label là "using_phone" THẬT (không phải placeholder) — luật phone-usage
chạy trong yolo_probe.py (xem config.py PHONE_*), KHÔNG qua SGIE riêng.

(Đã bỏ fusion với gaze DIRECTION — SGIE gaze compass gây nhầm lẫn với gaze
TARGET, xem pipeline.py. Gaze TARGET (gaze_target_probe.py) ra điểm pixel,
chưa có nhãn ngữ nghĩa để fusion vào đây — thêm lại khi có bước định nghĩa
zone theo camera.)"""

from __future__ import annotations

import time

from .logging_utils import log

# Nhãn action coi là cảnh báo ngay lập tức.
# "using_phone" = luật phone-usage thật (yolo_probe.py) — người đang dùng điện
# thoại trong khung giám sát → cảnh báo ngay.
IMMEDIATE_ALERT_ACTIONS = {"using_phone"}

# Chống spam: không bắn lại cùng 1 alert cho cùng track_id trong khoảng này.
ALERT_COOLDOWN_SEC = 10.0


class RuleEngine:
    """Áp luật theo track_id. Giữ state nhỏ (thời điểm alert gần nhất mỗi
    track) để không log lặp mỗi frame."""

    def __init__(self):
        self._last_alert: dict[tuple[int, str], float] = {}

    def _fire(self, track_id: int, cam: str, reason: str) -> str | None:
        key = (track_id, reason)
        now = time.time()
        last = self._last_alert.get(key, 0.0)
        if now - last < ALERT_COOLDOWN_SEC:
            return None
        self._last_alert[key] = now
        msg = f"ALERT [{cam}] track_id={track_id}: {reason}"
        log.warning(msg)
        return msg

    def evaluate(self, track_id: int, cam: str,
                 action_label: str | None = None) -> str | None:
        """Áp luật cho 1 object/frame. Trả về message alert nếu có (đã log),
        None nếu không match luật nào hoặc đang trong cooldown."""
        if action_label is None:
            return None

        if action_label in IMMEDIATE_ALERT_ACTIONS:
            return self._fire(track_id, cam, f"action={action_label}")

        return None
