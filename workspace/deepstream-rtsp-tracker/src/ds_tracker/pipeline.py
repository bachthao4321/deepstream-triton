"""Dựng pipeline GStreamer (build_pipeline).

Luồng (theo pattern deepstream-test3, có queue giữa các stage):
    nvurisrcbin (×N) → nvstreammux → nvinfer → [probe: YoloTensorParser
                                                 (+ luật phone-usage)]
    → nvtracker → [SGIE staff/customer?] → [RGBA convert cho gaze target?]
    → [probe: GazeTargetProbe?] → [probe: TrackerMetadataHandler]
    → nvmultistreamtiler → nvvideoconvert → nvdsosd → nvvideoconvert
    → nvv4l2h264enc → h264parse → matroskamux → filesink
    (remux MKV→MP4 khi kết thúc)

SGIE sau tracker chạy TUẦN TỰ trong graph GStreamer (không phải nhánh song
song thật — NvDsBatchMeta gắn theo buffer nên tách nhánh bằng `tee` rồi merge
lại tốn kém và không cần thiết). Về mặt logic mỗi SGIE vẫn độc lập với nhau:
cùng đọc object 'person' do PGIE sinh (operate-on-gie-id=1), mỗi SGIE tự gắn
NvDsClassifierMeta riêng (phân biệt bằng gie-unique-id — xem config.py
GIE_ID_*) — coi như "song song" theo nghĩa CLAUDE.md mô tả (không phụ thuộc
kết quả của nhau), chỉ khác thứ tự thực thi vật lý trên GPU.

"Action recognition" (CLAUDE.md giai đoạn 3) cho project này = kiểm tra có
đang dùng điện thoại hay không. Chạy qua PGIE#2 riêng (pgie2_phone_face_config.yml,
model YOLO11n tự train trên FPI-Det — github.com/KvCgRv/FPI-Det, 2 class
phone/face, ảnh góc camera giám sát thật) — chạy TUẦN TỰ sau PGIE#1 person
detector, cùng buffer. PhoneFaceProbe (phone_face_probe.py) decode tensor
phone+face, áp luật "phone GẦN face" rồi gắn nhãn 'using_phone' vào đúng
object person (đã tạo bởi PGIE#1) chứa face đó. Bật qua `phone_face_config_path`.

Gaze TARGET — "người này đang nhìn vào ĐÂU trong khung hình" — chạy qua
GazeTargetProbe (gaze_target_probe.py, model Gazelle-DINOv3, ONNXRuntime độc
lập, KHÔNG qua nvinfer vì model cần 2 input ảnh+bbox). Gắn trên src pad của
nhánh RGBA (chèn ngay trong main chain, KHÔNG dùng `tee` — xem comment ở
gt_conv/gt_caps), chỉ bật khi gaze_target_onnx_path được set. Cần
nvbuf-memory-type=3 trên streammux để đọc pixel CPU.

(Đã bỏ SGIE gaze DIRECTION — model MobileGaze chỉ ra hướng nhìn tương đối so
với camera, gây nhầm lẫn với gaze TARGET ở trên. Gaze TARGET là feature gaze
duy nhất còn lại trong pipeline.)"""

from __future__ import annotations

import math

from gi.repository import Gst

from .gaze_target_probe import GazeTargetProbe
from .gst_utils import link_chain, make, set_if_has
from .logging_utils import log
from .osd_probe import TrackerMetadataHandler
from .phone_face_probe import PhoneFaceProbe
from .rule_engine import RuleEngine
from .source_bin import create_source_bin
from .triton_infer_probe import TritonYoloSgieProbe
from .yolo_probe import YoloTensorParser


def _tiler_layout(n: int) -> tuple[int, int]:
    """Số cam → (cols, rows). Layout cố định cho số phổ biến, sqrt cho số lẻ."""
    if n <= 1:
        return 1, 1
    if n <= 2:
        return 2, 1
    if n <= 4:
        return 2, 2
    if n <= 6:
        return 3, 2
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return cols, rows


def build_pipeline(cfg: dict, uris: list[str], cam_names: dict[int, str],
                   pgie_config_path: str | None, mkv_path: str,
                   cam_log_dir: str | None = None,
                   sgie_config_path: str | None = None,
                   phone_face_config_path: str | None = None,
                   gaze_target_onnx_path: str | None = None,
                   gaze_target_inout_threshold: float = 0.3,
                   mode: str = "nvinfer",
                   triton_cfg: dict | None = None) -> tuple[Gst.Pipeline, dict]:
    """Dựng pipeline Gst-Python. Trả về (pipeline, refs) với refs giữ các probe
    object để không bị GC và các element cần thao tác sau."""
    n = len(uris)

    gpu_id = cfg.get("gpu_id", 0)
    width = int(cfg.get("native_width", 1920))
    height = int(cfg.get("native_height", 1080))
    # Kích thước để scale bbox YOLO — tách riêng khỏi mux/tiler size. Bỏ trống
    # trong config → fallback về native_width/height (behavior cũ).
    bbox_w = int(cfg.get("bbox_scale_width", width))
    bbox_h = int(cfg.get("bbox_scale_height", height))
    cols, rows = _tiler_layout(n)

    pipeline = Gst.Pipeline.new("rtsp-ai-tracker")
    if not pipeline:
        raise RuntimeError("Không tạo được Gst.Pipeline")

    # ── nvstreammux ──────────────────────────────────────────────────────────
    # Set property có điều kiện qua find_property(): new nvstreammux
    # (USE_NEW_NVSTREAMMUX=yes) KHÔNG có width/height/live-source/gpu-id
    # (kích thước lấy từ source, cấu hình qua config-file-path); old mux thì có.
    # YoloTensorParser scale bbox theo net_width/net_height CỐ ĐỊNH = mux
    # width/height, nên old mux (set width/height tường minh) giữ đúng behavior.
    streammux = make("nvstreammux", "mux")
    streammux.set_property("batch-size", n)
    have_wh = (set_if_has(streammux, "width", width)
               and set_if_has(streammux, "height", height))
    set_if_has(streammux, "live-source", 1)
    set_if_has(streammux, "gpu-id", gpu_id)
    # 40ms ~ 1 frame @25fps. (Thử 150ms → detection rớt thêm; nút thắt là
    # throughput decode nhiều luồng HEVC 1440p, không phải đồng bộ batch.)
    set_if_has(streammux, "batched-push-timeout", 40000)
    set_if_has(streammux, "attach-sys-ts", True)
    # căn frame các cam theo timestamp trước khi gom batch → ô tile cùng mốc
    set_if_has(streammux, "sync-inputs", 1)
    set_if_has(streammux, "max-latency", 100000000)
    if mode == "triton" or gaze_target_onnx_path:
        # TritonYoloSgieProbe / GazeTargetProbe đều đọc pixel qua
        # get_nvds_buf_surface() ở Python → cần CUDA Unified memory để CPU
        # truy cập trực tiếp (x86 dGPU).
        set_if_has(streammux, "nvbuf-memory-type", 3)
    if not have_wh:
        # New streammux: không set được width/height → cảnh báo vì scale bbox
        # trong YoloTensorParser dựa trên net_width/net_height. Chạy old mux
        # (bỏ USE_NEW_NVSTREAMMUX) để có width/height ổn định.
        log.warning("nvstreammux không có property width/height (new mux). "
                    "Scale bbox dùng native %dx%d — bỏ USE_NEW_NVSTREAMMUX "
                    "để dùng old mux nếu box lệch.", width, height)
    pipeline.add(streammux)

    # ── source bins → streammux.sink_%u ──────────────────────────────────────
    for i, uri in enumerate(uris):
        source_bin = create_source_bin(i, uri)
        pipeline.add(source_bin)
        sinkpad = streammux.request_pad_simple(f"sink_{i}")
        if not sinkpad:
            raise RuntimeError(f"Không request được sink_{i} trên streammux")
        srcpad = source_bin.get_static_pad("src")
        if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
            raise RuntimeError(f"Không link được source bin {i} vào streammux")

    # ── PGIE: nvinfer (TensorRT cục bộ) hoặc conv RGBA + probe Triton ──────────
    pgie = None            # element nvinfer, chỉ tồn tại ở mode=nvinfer
    triton_conv = None     # nvvideoconvert→RGBA, chỉ tồn tại ở mode=triton
    triton_caps = None
    pgie2 = None            # PGIE#2 phone+face, chỉ tồn tại khi mode=nvinfer + bật
    if mode == "nvinfer":
        pgie = make("nvinfer", "primary-inference")
        pgie.set_property("config-file-path", pgie_config_path)
        pgie.set_property("batch-size", n)
        # PGIE#2 (phone+face) — process-mode=1, toàn khung hình, chạy TUẦN TỰ
        # ngay sau PGIE#1 (đọc CÙNG buffer chưa đổi pixel). CHƯA hỗ trợ
        # mode=triton (giống sgie_config_path/gaze_target).
        if phone_face_config_path:
            pgie2 = make("nvinfer", "primary-phone-face")
            pgie2.set_property("config-file-path", phone_face_config_path)
            pgie2.set_property("batch-size", n)
    else:
        # nvinferserver (client Triton "chính chủ" của DeepStream) cần
        # libtritonserver.so cục bộ — không có trong image build tay này (kể cả
        # ở chế độ gRPC remote, xem README). Thay vào đó dùng TritonYoloSgieProbe:
        # tự đọc pixel RGBA, gọi Triton qua tritonclient.grpc, tự attach
        # NvDsObjectMeta/NvDsClassifierMeta — downstream (tracker/tiler/osd)
        # không cần biết khác biệt gì so với mode=nvinfer.
        triton_conv = make("nvvideoconvert", "triton-conv-rgba")
        set_if_has(triton_conv, "gpu-id", gpu_id)
        set_if_has(triton_conv, "nvbuf-memory-type", 3)
        triton_caps = make("capsfilter", "triton-caps-rgba")
        triton_caps.set_property(
            "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA"))
        if phone_face_config_path:
            log.warning("phone_face_config chưa hỗ trợ ở mode=triton — bỏ qua.")

    # ── nvtracker ─────────────────────────────────────────────────────────────
    tracker = make("nvtracker", "tracker")
    tracker.set_property("ll-lib-file", cfg["tracker"]["lib_file"])
    tracker.set_property("ll-config-file", cfg["tracker"]["config_file"])
    tracker.set_property("tracker-width", 640)
    tracker.set_property("tracker-height", 384)
    set_if_has(tracker, "gpu-id", gpu_id)

    # ── SGIE classifier staff/customer ──────────────────────────────────────────
    # mode=nvinfer: process-mode=2, tự crop mỗi object person do PGIE/tracker
    #   sinh ra, classify. Chỉ tạo khi có sgie_config_path — cho phép tắt độc
    #   lập (test SGIE mới riêng lẻ trước khi ghép, theo quy ước CLAUDE.md mục 6).
    # mode=triton: KHÔNG có element riêng — TritonYoloSgieProbe (trên) đã gọi
    #   Triton classify inline. Phone-usage rule chạy trong YoloTensorParser
    #   (mode=nvinfer) — chưa áp dụng ở mode=triton.
    sgie_stages: list[tuple[str, Gst.Element]] = []
    if mode == "nvinfer" and sgie_config_path:
        sgie = make("nvinfer", "secondary-classifier")
        sgie.set_property("config-file-path", sgie_config_path)
        sgie_stages.append(("staff_customer", sgie))

    # ── nvmultistreamtiler ─────────────────────────────────────────────────────
    tiler = make("nvmultistreamtiler", "tiler")
    tiler.set_property("rows", rows)
    tiler.set_property("columns", cols)
    tiler.set_property("width", width)
    tiler.set_property("height", height)
    set_if_has(tiler, "gpu-id", gpu_id)

    conv1 = make("nvvideoconvert", "conv1")
    set_if_has(conv1, "gpu-id", gpu_id)

    # ── nvdsosd ────────────────────────────────────────────────────────────────
    nvosd = make("nvdsosd", "onscreendisplay")
    nvosd.set_property("process-mode", 1)   # GPU
    nvosd.set_property("display-text", 1)

    conv2 = make("nvvideoconvert", "conv2")
    set_if_has(conv2, "gpu-id", gpu_id)

    # ── encoder → h264parse → matroskamux → filesink ──────────────────────────
    # nvv4l2h264enc: encoder phần cứng của DeepStream (thay nvh264enc cũ)
    encoder = make("nvv4l2h264enc", "encoder")
    set_if_has(encoder, "bitrate", 8000000)   # 8 Mbps (đơn vị bps)
    set_if_has(encoder, "iframeinterval", 25)

    parser = make("h264parse", "h264-parser")

    # matroskamux ghi index liên tục, không cần finalize → an toàn khi hard-exit.
    # mp4mux/qtmux ghi moov ở cuối nên dễ hỏng khi hard-kill nguồn RTSP live.
    muxout = make("matroskamux", "mux-out")

    sink = make("filesink", "filesink")
    sink.set_property("location", mkv_path)
    sink.set_property("sync", False)
    sink.set_property("async", False)

    # ── convert RGBA cho GazeTargetProbe (chỉ tạo khi bật) ──────────────────────
    # GazeTargetProbe cần đọc pixel qua get_nvds_buf_surface() — hàm này CHỈ hỗ
    # trợ RGBA/RGB (thử trên buffer NV12 mặc định sau tracker → RuntimeError
    # "only support RGBA/RGB"). Chèn nvvideoconvert(RGBA) NGAY TRONG main chain
    # (KHÔNG dùng `tee`) — vì `tee` nhân bản buffer thành 2 nhánh ĐỘC LẬP, ghi
    # display_meta ở nhánh phụ sẽ KHÔNG xuất hiện ở video nhánh chính (đã tự
    # kiểm chứng hiểu nhầm này trước khi sửa). Chèn SAU sgie_stages (SGIE
    # staff/customer vẫn ăn NV12 như cũ, không đụng gì) — chỉ đoạn tiler/osd/
    # encoder chạy RGBA, đúng pattern mode=triton đã chạy production (ở đó
    # head=triton_caps RGBA chảy thẳng qua toàn bộ chuỗi này).
    gt_conv = gt_caps = None
    if gaze_target_onnx_path:
        gt_conv = make("nvvideoconvert", "gt-conv-rgba")
        set_if_has(gt_conv, "gpu-id", gpu_id)
        set_if_has(gt_conv, "nvbuf-memory-type", 3)
        gt_caps = make("capsfilter", "gt-caps-rgba")
        gt_caps.set_property(
            "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA"))

    # queue giữa các stage (theo deepstream-test3)
    queues = [make("queue", f"queue{i}") for i in range(1, 6)]
    q1, q2, q3, q4, q5 = queues

    # Thêm mọi element (trừ streammux đã add) vào pipeline
    for el in (tracker, tiler, conv1, nvosd, conv2,
               encoder, parser, muxout, sink, *queues):
        pipeline.add(el)
    if pgie is not None:
        pipeline.add(pgie)
    if pgie2 is not None:
        pipeline.add(pgie2)
    for _, el in sgie_stages:
        pipeline.add(el)
    if triton_conv is not None:
        pipeline.add(triton_conv)
        pipeline.add(triton_caps)
    if gt_conv is not None:
        pipeline.add(gt_conv)
        pipeline.add(gt_caps)

    # ── link chuỗi xử lý (có queue giữa các stage) ────────────────────────────
    if mode == "nvinfer":
        if not streammux.link(pgie):
            raise RuntimeError("link mux→pgie thất bại")
        if pgie2 is not None:
            if not pgie.link(pgie2):
                raise RuntimeError("link pgie→pgie2 (phone_face) thất bại")
            head = pgie2
        else:
            head = pgie
    else:
        link_chain([streammux, triton_conv, triton_caps])
        head = triton_caps
    # SGIE (nếu có, chỉ ở mode=nvinfer) chèn giữa tracker và tiler. gt_conv/
    # gt_caps (nếu bật) chèn ngay sau sgie_stages, trước q2/tiler.
    tail = [q3, conv1, q4, nvosd, q5, conv2, encoder, parser, muxout, sink]
    mid = [*[el for _, el in sgie_stages], *([gt_conv, gt_caps] if gt_conv else [])]
    link_chain([head, q1, tracker, *mid, q2, tiler, *tail])

    # ── gắn probe ──────────────────────────────────────────────────────────────
    if mode == "nvinfer":
        # YoloTensorParser trên src pad của nvinfer (tensor meta có sẵn ở đây).
        yolo_probe = YoloTensorParser(cam_names, cam_log_dir,
                                      net_width=bbox_w, net_height=bbox_h)
        pgie.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, yolo_probe, 0)
    else:
        # TritonYoloSgieProbe: detect + classify qua Triton, attach thẳng
        # NvDsObjectMeta/NvDsClassifierMeta — trên src pad của capsfilter RGBA
        # (trước tracker), thay hoàn toàn cho pgie/sgie nvinfer.
        yolo_probe = TritonYoloSgieProbe(
            triton_url=triton_cfg["url"],
            yolo_model=triton_cfg["yolo_model"],
            sgie_model=triton_cfg.get("sgie_model") or None,
            sgie_labels_path=triton_cfg.get("sgie_labels"),
            cam_names=cam_names, cam_log_dir=cam_log_dir,
            yolo_max_batch=int(triton_cfg.get("yolo_max_batch", 4)),
            sgie_max_batch=int(triton_cfg.get("sgie_max_batch", 16)))
        triton_caps.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, yolo_probe, 0)

    # PhoneFaceProbe: "action recognition" = phone-usage, gắn trên src pad
    # PGIE#2 (object person của PGIE#1 đã có sẵn trong CÙNG buffer tới đây).
    phone_face_probe = None
    if pgie2 is not None:
        phone_face_probe = PhoneFaceProbe(net_width=bbox_w, net_height=bbox_h)
        pgie2.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, phone_face_probe, 0)

    # GazeTargetProbe: "nhìn vào ĐÂU trong khung hình" — không qua nvinfer
    # (model 2 input ảnh+bbox). Gắn trên src pad của gt_caps (RGBA thật — xem
    # comment ở gt_conv/gt_caps phía trên; object meta/bbox person vẫn còn
    # nguyên qua nvvideoconvert).
    gaze_target_probe = None
    if gaze_target_onnx_path:
        gaze_target_probe = GazeTargetProbe(
            gaze_target_onnx_path, inout_threshold=gaze_target_inout_threshold)
        gt_caps.get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER, gaze_target_probe, 0)

    # TrackerMetadataHandler đọc tracking + classifier meta. Gắn trên src pad
    # của SGIE CUỐI CÙNG trong chuỗi nếu có (mọi classifier meta đã attach vào
    # object tới đó), ngược lại trên src pad của tracker (mode=triton: classifier
    # đã attach TRƯỚC tracker bởi TritonYoloSgieProbe, meta vẫn còn nguyên khi
    # qua tới đây).
    # RuleEngine chỉ tạo khi có phone_face (PGIE#2) bật — tránh eval fusion vô
    # ích khi tắt.
    rule_engine = RuleEngine() if phone_face_config_path else None
    tracker_probe = TrackerMetadataHandler(cam_names, rule_engine=rule_engine,
                                           phone_face_probe=phone_face_probe)
    meta_src = sgie_stages[-1][1] if sgie_stages else tracker
    meta_src.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, tracker_probe, 0)

    # Giữ ref các probe để không bị GC (add_probe không giữ ref Python object).
    refs = {"yolo_probe": yolo_probe, "tracker_probe": tracker_probe,
            "phone_face_probe": phone_face_probe,
            "gaze_target_probe": gaze_target_probe,
            "streammux": streammux, "rows": rows, "cols": cols}
    return pipeline, refs
