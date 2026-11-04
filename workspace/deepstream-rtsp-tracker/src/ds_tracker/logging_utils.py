"""Cấu hình logging: logger chung của pipeline + logger ghi file riêng per-cam."""

from __future__ import annotations

import logging
import os

logging.basicConfig(
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

# Logger chung dùng khắp package (import từ đây thay vì tạo lại mỗi module).
log = logging.getLogger("ds-pipeline")


def make_cam_logger(cam: str, log_dir: str) -> logging.Logger:
    """Tạo logger ghi file riêng cho từng cam: <log_dir>/<cam>.log.

    Mỗi cam một FileHandler riêng → N cam ghi song song vào N file độc lập
    (logging của Python thread-safe). propagate=False để không lẫn lên root."""
    logger = logging.getLogger(f"cam.{cam}")
    if logger.handlers:                       # đã khởi tạo trước đó
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fh = logging.FileHandler(os.path.join(log_dir, f"{cam}.log"), mode="w")
    fh.setFormatter(logging.Formatter(
        "[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(fh)
    return logger
