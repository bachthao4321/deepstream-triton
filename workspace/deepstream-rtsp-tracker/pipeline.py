#!/usr/bin/env python3
"""Entrypoint: `python3 pipeline.py ...` và run.sh chạy như cũ.

Thêm src/ vào sys.path, parse argument, override config rồi gọi build_and_run.
Code dựng graph GStreamer nằm trong package src/ds_tracker/ (config, yolo_probe,
osd_probe, source_bin, gst_utils, pipeline, runtime)."""

import argparse
import logging
import os
import sys

# Tắt log ồn của GStreamer/plugin C. PHẢI set TRƯỚC khi import ds_tracker (kéo
# theo Gst). Cho phép env ghi đè khi cần debug: GST_DEBUG=3 python pipeline.py
os.environ.setdefault("GST_DEBUG", "0")


def _silence_native_stdout():
    """Nuốt stdout mức C (fd 1) → /dev/null để chặn các dòng printf của plugin
    DeepStream ('Opening in BLOCKING MODE', 'Protocol set: 0x4', 'max_fps_dur'...)
    mà GST_DEBUG không tắt được. Log Python đi qua stderr (logging.basicConfig)
    nên vẫn hiện. Bỏ qua khi có --debug để soi đầy đủ."""
    if "--debug" in sys.argv:
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)          # fd 1 (stdout) → /dev/null
    os.close(devnull)


_silence_native_stdout()

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from ds_tracker.config import load_config   # noqa: E402
from ds_tracker.logging_utils import log     # noqa: E402
from ds_tracker.runtime import build_and_run  # noqa: E402

DEFAULT_CONFIG = "/workspace/deepstream-rtsp-tracker/configs/config.yaml"
DEFAULT_GRAPH_DIR = "/workspace/deepstream-rtsp-tracker"


def main():
    parser = argparse.ArgumentParser(
        description="DeepStream RTSP + YOLOv11 + Tracker → Tiled MP4")
    parser.add_argument("--config", "-c", default=DEFAULT_CONFIG)
    parser.add_argument("--rtsp", "-r", nargs="*", help="Ghi đè RTSP source URLs")
    parser.add_argument("--duration", type=int,
                        help="Số giây ghi (0 = tới Ctrl+C)")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dump-graph", nargs="?", const=DEFAULT_GRAPH_DIR,
                        metavar="DIR", default=None,
                        help="Vẽ graph pipeline ra DIR/pipeline_graph.dot (+ .png "
                             f"nếu có graphviz). Mặc định DIR={DEFAULT_GRAPH_DIR}")
    args = parser.parse_args()

    if args.debug:
        log.setLevel(logging.DEBUG)

    cfg = load_config(args.config)

    if args.rtsp:
        cfg["sources"] = [
            {"source_id": i, "uri": u, "camera_name": f"cam{i}", "enabled": True}
            for i, u in enumerate(args.rtsp)
        ]
    if args.duration is not None:
        cfg.setdefault("record", {})["duration_sec"] = args.duration

    build_and_run(cfg, dump_graph_dir=args.dump_graph)


if __name__ == "__main__":
    main()
