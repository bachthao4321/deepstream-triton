"""Runtime: chạy pipeline bằng GLib.MainLoop + bus watch, xử lý EOS/shutdown,
remux MKV→MP4 khi kết thúc."""

from __future__ import annotations

import os
import signal
import sys
import threading
import time

from gi.repository import GLib, Gst

from .config import EOS_GRACE_SEC, parse_sources
from .graph import dump_pipeline_graph, enable_dot_dump
from .logging_utils import log
from .pipeline import build_pipeline


def bus_call(bus, message, loop):
    """Bus watch (theo common/bus_call.py của repo)."""
    t = message.type
    if t == Gst.MessageType.EOS:
        log.info("Bus: End-Of-Stream")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        # Hạ xuống debug: RTSP live rớt/reconnect liên tục sinh warning ồn
        # ("Could not send/write to resource"). Xem chi tiết khi chạy --debug.
        log.debug("Bus WARNING %s: %s | %s", message.src.name, err, debug)
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        log.error("Bus ERROR %s: %s | %s", message.src.name, err, debug)
        loop.quit()
    return True


def _remux_mkv_to_mp4(mkv_path: str, mp4_path: str):
    """Remux MKV → MP4 bằng ffmpeg -c copy (không re-encode, giữ nguyên chất
    lượng). MKV luôn hợp lệ nên remux nhanh; xóa MKV nếu thành công."""
    if not os.path.exists(mkv_path) or os.path.getsize(mkv_path) == 0:
        log.warning("File MKV không tồn tại/rỗng: %s", mkv_path)
        return
    ret = os.system(
        f"ffmpeg -y -i {mkv_path} -c copy -movflags +faststart {mp4_path} 2>/dev/null"
    )
    if ret == 0 and os.path.exists(mp4_path) and os.path.getsize(mp4_path) > 0:
        os.remove(mkv_path)
        log.info("Remux → %s (%.1f MB)", mp4_path, os.path.getsize(mp4_path) / 1e6)
    else:
        log.warning("Remux thất bại, giữ MKV: %s", mkv_path)


def build_and_run(cfg: dict, dump_graph_dir: str | None = None):
    uris, cam_names = parse_sources(cfg["sources"])
    if not uris:
        log.error("Không có nguồn RTSP nào được bật.")
        sys.exit(1)

    infer_cfg = cfg["inference"]
    mode = infer_cfg.get("mode", "nvinfer")
    if mode not in ("nvinfer", "triton"):
        raise ValueError(
            f"inference.mode chỉ hỗ trợ 'nvinfer' hoặc 'triton' (nhận '{mode}')")
    pgie_config_path = infer_cfg.get("nvinfer_config")
    sgie_config_path = infer_cfg.get("sgie_config") or None
    # "Action recognition" (CLAUDE.md giai đoạn 3) = luật phone-usage, chạy
    # qua PGIE#2 riêng (model tự train FPI-Det, phone+face) — xem
    # pipeline.py PhoneFaceProbe. Để trống = tắt.
    phone_face_config_path = cfg.get("phone_rule", {}).get("pgie2_config") or None
    # Gaze TARGET ("nhìn vào đâu trong khung hình", khác gaze DIRECTION ở
    # trên) — model Gazelle-DINOv3, ONNXRuntime độc lập (không qua nvinfer).
    # Tắt mặc định (onnx_path trống), bật bằng cách set đường dẫn model.
    gt_cfg = cfg.get("gaze_target", {}) or {}
    gaze_target_onnx_path = gt_cfg.get("onnx_path") or None
    gaze_target_inout_threshold = float(gt_cfg.get("inout_threshold", 0.3))
    triton_cfg = infer_cfg.get("triton") or {}
    if mode == "nvinfer" and not pgie_config_path:
        raise ValueError("inference.nvinfer_config bắt buộc khi mode=nvinfer")
    if mode == "triton" and not triton_cfg.get("yolo_model"):
        raise ValueError("inference.triton.yolo_model bắt buộc khi mode=triton")

    rec_cfg = cfg.get("record", {})
    rec_seconds = int(rec_cfg.get("duration_sec", 0))   # 0 = chạy tới Ctrl+C
    rec_dir = rec_cfg.get("output_dir", "/workspace/recordings")
    rec_file = rec_cfg.get("filename", "tiled.mp4")
    # Pipeline ghi MKV (an toàn khi hard-exit); cuối cùng remux sang MP4.
    base = rec_file.rsplit(".", 1)[0] if "." in rec_file else rec_file
    mkv_path = os.path.join(rec_dir, base + ".mkv")
    mp4_path = os.path.join(rec_dir, base + ".mp4")

    os.makedirs(rec_dir, exist_ok=True)

    # Thư mục log detection per-cam: <rec_dir>/cam_log/<cam_name>.log
    cam_log_dir = os.path.join(rec_dir, "cam_log")
    os.makedirs(cam_log_dir, exist_ok=True)

    log.info("Sources: %d %s | inference=%s", len(uris),
             list(cam_names.values()), mode)
    log.info("Log detection per-cam → %s/<cam>.log", cam_log_dir)

    # Bật dump .dot TRƯỚC Gst.init() (GStreamer đọc env lúc init).
    if dump_graph_dir:
        enable_dot_dump(dump_graph_dir)

    # Khởi tạo GStreamer
    Gst.init(None)

    if mode == "nvinfer" and sgie_config_path:
        log.info("SGIE classifier: %s", sgie_config_path)
    if mode == "nvinfer" and phone_face_config_path:
        log.info("PGIE#2 phone+face (action recognition): %s", phone_face_config_path)
    if gaze_target_onnx_path:
        log.info("Gaze TARGET (nhìn vào đâu trong khung hình): %s (inout>=%.2f)",
                 gaze_target_onnx_path, gaze_target_inout_threshold)
    if mode == "triton":
        log.info("Triton: url=%s yolo_model=%s sgie_model=%s",
                 triton_cfg.get("url"), triton_cfg.get("yolo_model"),
                 triton_cfg.get("sgie_model") or "(tắt)")
    pipeline, refs = build_pipeline(cfg, uris, cam_names, pgie_config_path,
                                    mkv_path, cam_log_dir,
                                    sgie_config_path=sgie_config_path,
                                    phone_face_config_path=phone_face_config_path,
                                    gaze_target_onnx_path=gaze_target_onnx_path,
                                    gaze_target_inout_threshold=gaze_target_inout_threshold,
                                    mode=mode, triton_cfg=triton_cfg)

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    t0 = time.time()
    _stopping = {"done": False}

    def _do_stop():
        """Gửi EOS để muxer finalize, chờ EOS lan tới sink rồi quit loop.

        Nguồn RTSP live không tự kết thúc → phải chủ động send_event(EOS).
        matroskamux ghi liên tục nên file MKV đã hợp lệ; sau khi loop.quit()
        ta remux MKV→MP4."""
        if _stopping["done"]:
            return
        _stopping["done"] = True
        log.info("Gửi EOS, chờ tối đa %ds flush...", EOS_GRACE_SEC)
        pipeline.send_event(Gst.Event.new_eos())
        # Fallback: nếu EOS không lan tới trong EOS_GRACE_SEC thì tự quit.
        GLib.timeout_add_seconds(EOS_GRACE_SEC, lambda: (loop.quit(), False)[1])
        return False

    def _shutdown(_sig, _frame):
        # Lần đầu: lên lịch _do_stop trên main loop (an toàn với GLib).
        # Lần hai (đã đang dừng): người dùng nhấn Ctrl+C lần nữa → thoát ngay.
        if _stopping["done"]:
            log.warning("Nhận tín hiệu lần hai — thoát ngay.")
            os._exit(1)
        GLib.idle_add(_do_stop)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Timer dừng sau rec_seconds (0 = vô hạn)
    if rec_seconds:
        def _on_timeout():
            log.info("Đủ %ds, gửi EOS rồi stop...", rec_seconds)
            _do_stop()
            return False
        GLib.timeout_add_seconds(rec_seconds, _on_timeout)

        def _tick(state={"n": 0}):
            state["n"] += 1
            if state["n"] % 10 == 0:
                log.info("%d/%ds", state["n"], rec_seconds)
            return not _stopping["done"] and state["n"] < rec_seconds
        GLib.timeout_add_seconds(1, _tick)

    dur = f"{rec_seconds}s" if rec_seconds else "∞"
    log.info("Ghi tiled %d-màn (%dx%d) → %s (run=%s)",
             len(uris), refs["cols"], refs["rows"], mp4_path, dur)
    log.info("Ctrl+C để dừng")

    pipeline.set_state(Gst.State.PLAYING)

    # Dump graph sau khi pipeline đã PLAYING + kịp negotiate caps (2s). Chờ để
    # graph có đủ pad/caps thực tế thay vì chỉ khung tĩnh chưa link.
    if dump_graph_dir:
        def _dump():
            dump_pipeline_graph(pipeline, dump_graph_dir)
            return False   # chạy 1 lần
        GLib.timeout_add_seconds(2, _dump)

    try:
        loop.run()
    except KeyboardInterrupt:
        _do_stop()
        loop.run()

    # Teardown. QUAN TRỌNG: remux MKV→MP4 TRƯỚC khi đụng set_state(NULL).
    # matroskamux ghi index liên tục nên MKV đã hợp lệ ngay khi loop dừng.
    # set_state(NULL) trên rtspsrc live đang reconnect có thể BLOCK vô hạn HOẶC
    # segfault trong code C (Python try/except không cứu được) → nếu remux đặt
    # sau nó thì MP4 không bao giờ được tạo khi crash. Đảo thứ tự để MP4 luôn có.
    log.info("Dừng pipeline...")
    sz = os.path.getsize(mkv_path) / 1e6 if os.path.exists(mkv_path) else 0
    log.info("Ghi xong MKV → %s (%.1f MB, %.1fs)", mkv_path, sz, time.time() - t0)
    _remux_mkv_to_mp4(mkv_path, mp4_path)

    # set_state(NULL) chạy trong thread có timeout; nếu treo/crash cũng đã có MP4.
    null_done = threading.Event()

    def _set_null():
        try:
            pipeline.set_state(Gst.State.NULL)
        except Exception as e:
            log.warning("set_state(NULL) lỗi: %s", e)
        null_done.set()

    threading.Thread(target=_set_null, daemon=True).start()
    if not null_done.wait(timeout=EOS_GRACE_SEC):
        log.warning("set_state(NULL) treo >%ds (rtspsrc reconnect) — bỏ qua teardown.",
                    EOS_GRACE_SEC)

    # Nguồn RTSP live: os._exit() để không kẹt ở atexit/GC của rtspsrc đang reconnect.
    os._exit(0)
