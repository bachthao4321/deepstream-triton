# DeepStream 9 — RTSP AI Tracker Pipeline

Pipeline xử lý nhiều luồng RTSP, chạy AI detector (ResNet18 TrafficCamNet) và NvDCF tracker.

## Cấu trúc thư mục

```
deepstream-rtsp-tracker/
├── pipeline.py              # Entrypoint: argparse + main() → build_and_run
├── run.sh                   # Wrapper (unset USE_NEW_NVSTREAMMUX rồi chạy)
├── src/ds_tracker/          # Package code chính
│   ├── __init__.py          # require_version("Gst")
│   ├── config.py            # LABELS, ngưỡng, parse_sources, load_config
│   ├── logging_utils.py     # logger chung + logger file per-cam
│   ├── gst_utils.py         # helper: make / set_if_has / link_chain
│   ├── source_bin.py        # nvurisrcbin + ghost pad (create_source_bin)
│   ├── yolo_probe.py        # YoloTensorParser (tensor → NvDsObjectMeta)
│   ├── osd_probe.py         # TrackerMetadataHandler (log + vẽ OSD text)
│   ├── pipeline.py          # build_pipeline (dựng graph GStreamer)
│   └── runtime.py           # build_and_run, bus_call, remux MKV→MP4
├── configs/
│   ├── config.yaml          # Cấu hình runtime (RTSP URL, output mode...)
│   ├── pgie_config.yml      # Cấu hình PGIE (detector)
│   └── tracker_NvDCF.yml    # Cấu hình NvDCF tracker
└── models/                  # (tuỳ chọn) đặt model tùy chỉnh ở đây
```

Chạy: `python3 pipeline.py` hoặc `./run.sh` (tương đương nhau).

## Cài đặt phụ thuộc

```bash
pip3 install pyyaml
# pyservicemaker được cài sẵn cùng DeepStream 9
```

## Cấu hình

Chỉnh file `configs/config.yaml`:

```yaml
sources:
  - rtsp://user:pass@192.168.1.100:554/stream1
  - rtsp://user:pass@192.168.1.101:554/stream1

output:
  mode: headless   # headless | display | file | rtsp
```

## Chạy

```bash
# Chạy với config mặc định (headless)
python3 pipeline.py

# Chỉ định config khác
python3 pipeline.py --config /path/to/config.yaml

# Ghi đè RTSP URL trực tiếp
python3 pipeline.py --rtsp rtsp://cam1 rtsp://cam2

# Hiển thị màn hình (cần X11/EGL)
python3 pipeline.py --output display

# Ghi ra file
python3 pipeline.py --output file

# Debug chi tiết (in từng object/frame)
python3 pipeline.py --debug

# Vẽ graph pipeline ra pipeline_graph.dot (+ .png nếu có graphviz)
python3 pipeline.py --dump-graph

# Vẽ graph vào thư mục khác
python3 pipeline.py --dump-graph /tmp/graphs
```

> `--dump-graph` set `GST_DEBUG_DUMP_DOT_DIR`, chờ pipeline PLAYING ~2s (negotiate
> caps) rồi `Gst.debug_bin_to_dot_file()` → `pipeline_graph.dot`, tự convert sang
> `.png` bằng `dot` (cài graphviz: `apt-get install graphviz`). Mỗi source-bin RTSP
> được gộp thành 1 node để graph chỉ nêu bật các stage chính, không xổ ruột từng cam.

## Kiến trúc pipeline

Dựng bằng **Gst-Python + pyds** (theo repo `NVIDIA-AI-IOT/deepstream_python_apps`,
mẫu `deepstream-test2` / `deepstream-test3`). Mỗi RTSP là một source-bin
(`nvurisrcbin` + ghost pad), link động vào `nvstreammux` qua
`request_pad_simple("sink_%u")`. Điều khiển bằng `GLib.MainLoop` + bus watch.

```
nvurisrcbin x N   (source-bin, TCP, reconnect ∞, drop-on-latency=False)
     │  request_pad_simple("sink_%u")
nvstreammux  (batch size = số luồng, sync-inputs=1)
     │
nvinfer (PGIE#1)  ── pgie_config.yml  (YOLOv11m no-NMS, output-tensor-meta=1)
     │
[Probe: YoloTensorParser]  (src pad PGIE#1)
  - đọc raw tensor output0 [84,8400] từ NvDsInferTensorMeta
  - decode + NMS (torch/torchvision, GPU) → NvDsObjectMeta (person)
  - ghi log detection per-cam ra <output_dir>/cam_log/<cam>.log
     │
nvinfer (PGIE#2)  ── pgie2_phone_face_config.yml  (tuỳ chọn, gie-unique-id=4,
     │                model tự train FPI-Det: phone+face, output0 [6,8400])
[Probe: PhoneFaceProbe]  (src pad PGIE#2, chỉ khi phone_rule.pgie2_config bật)
  - decode tensor phone+face (dùng chung tensor_utils.py với PGIE#1)
  - luật "phone GẦN face" → tìm object person (PGIE#1) chứa face đó
  - gắn nhãn 'using_phone' (NvDsClassifierMeta, unique_component_id=GIE_ID_SGIE_ACTION)
     │
nvtracker ─── tracker_NvDCF.yml + libnvds_nvmultiobjecttracker.so
     │
[SGIE staff/customer]  ── sgie_staff_customer.yml       (tuỳ chọn, gie-unique-id=2)
     │
[nvvideoconvert RGBA]  ── chỉ chèn khi gaze_target bật (xem gt_conv/gt_caps
     │                    trong pipeline.py)
[Probe: GazeTargetProbe]  ── chỉ khi gaze_target.onnx_path được set
     │                       (model Gazelle-DINOv3, ONNXRuntime độc lập —
     │                        xem gaze_target_probe.py)
[Probe: TrackerMetadataHandler]  (src pad của SGIE cuối cùng đang bật, hoặc
                                   src pad nvtracker nếu không SGIE nào bật)
  - đọc track_id, class, bbox, confidence
  - đọc classifier meta của MỌI SGIE/rule đang bật (tách theo gie-unique-id)
  - vẽ text overlay (tên cam + count) lên nvdsosd display meta
  - action_label (phone-usage) → RuleEngine (rule_engine.py) → alert theo track_id
  - in thống kê mỗi 30 frame
     │
nvmultistreamtiler  (ghép lưới rows×cols)
     │
nvvideoconvert → nvdsosd (bbox + label + chấm gaze target nếu có) → nvvideoconvert
     │
nvv4l2h264enc → h264parse → matroskamux → filesink (.mkv)
     │
(khi dừng) ffmpeg -c copy  →  .mp4
```

> Ghi MKV trong lúc chạy (matroskamux ghi index liên tục → an toàn khi hard-exit
> nguồn RTSP live), khi nhận EOS/Ctrl+C thì remux MKV→MP4 bằng `ffmpeg -c copy`.

## Thay đổi tracker

Sửa `tracker.config_file` trong `config.yaml`:

| Tracker       | Config file |
|---------------|-------------|
| NvDCF (mặc định) | `configs/tracker_NvDCF.yml` |
| NvSORT        | `/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvSORT.yml` |
| NvDeepSORT    | `/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDeepSORT.yml` |
| IOU           | `/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml` |

## Thay đổi model detector

Sửa `configs/pgie_config.yml`:
- `onnx-file`: đường dẫn ONNX model mới
- `model-engine-file`: TensorRT engine (sẽ tự build lần đầu chạy)
- `labelfile-path`: file nhãn tương ứng
- `num-detected-classes`: số lớp của model

## Action recognition = phone-usage detection (WatchFlow giai đoạn 3)

**Đã có model thật, train xong, wire vào pipeline** — không còn khung sườn.

Kiến trúc: `configs/pgie2_phone_face_config.yml` (`gie-unique-id=4`) là
**PGIE#2** — model YOLO11n tự train (`training/fpi_det/`) trên dataset
[FPI-Det](https://github.com/KvCgRv/FPI-Det) (MIT, 22,879 ảnh góc camera giám
sát thật, 2 class `phone`/`face`). Chạy TUẦN TỰ sau PGIE#1 (person detector)
trong graph, process-mode=1 (toàn khung hình — model cần thấy cả phone lẫn
face cùng lúc để áp luật khoảng cách).

Kết quả train (val, 1730 ảnh): mAP50 **phone=0.902**, face=0.550 (xem
`training/fpi_det/runs/phone_face_yolo11n/results.csv`).

`src/ds_tracker/phone_face_probe.py` (`PhoneFaceProbe`, gắn trên src pad
PGIE#2) decode tensor phone+face (dùng chung `tensor_utils.py` với PGIE#1),
áp luật **"phone GẦN face"** (tâm bbox face nằm trong bbox phone đã phóng to
×2.5 — xem `config.py` `PHONE_FACE_*`) — chính xác hơn hẳn cách cũ ("phone
chồng lên toàn thân người", dễ nhầm khi phone để bàn). Khi khớp, tự tìm object
person (đã tạo bởi PGIE#1 trong CÙNG buffer) chứa face đó và gắn nhãn
`using_phone` (`NvDsClassifierMeta`, `unique_component_id=GIE_ID_SGIE_ACTION`)
— `TrackerMetadataHandler`/`RuleEngine` đọc được y hệt classifier thật.

**Đã test thật** trong container `thaokb_ds9`: engine TensorRT build OK, chạy
hết video không crash. Luật hoạt động đúng — 1 detection "phone" false-positive
(vật nhỏ trên quầy) trong video test KHÔNG kích hoạt alert vì không có face
gần đó, đúng như thiết kế (khác bản cũ, dễ báo nhầm hơn).

Để bật: set `phone_rule.pgie2_config` trong `config.yaml` (đã bật sẵn theo
mặc định). Fix format label trước khi train: `scripts/fix_fpi_det_labels.py`
(dataset gốc có ~1% dòng label bị dính 2 box do bug export — đã tự phát hiện
và sửa, xem comment trong file). Train lại: `training/fpi_det/train.py`.

> Toolchain MMAction2 (train video action recognition qua UCF101/NTU RGB+D,
> `training/mmaction2/`) vẫn còn giữ làm hạ tầng tham khảo — đã verify chạy
> được (TSN-ResNet18, 77% acc trên UCF101 sau 3 epoch) nhưng KHÔNG phải luồng
> chính hiện tại (phone+face detector đơn giản và chính xác hơn cho nhu cầu
> thực tế). Dùng lại nếu sau này cần nhận diện hành vi phức tạp hơn (ngã,
> loitering...) mà object detection thuần không giải quyết được.

## Gaze TARGET — "đang nhìn vào ĐÂU trong khung hình" (WatchFlow giai đoạn 4)

> Trước đây có thêm một hướng gaze DIRECTION (model MobileGaze/yakhyo, chỉ ra
> hướng nhìn tương đối so với camera — trái/phải/lên/xuống dạng la bàn 9
> hướng). Đã **bỏ hẳn** vì gây nhầm lẫn với gaze TARGET dưới đây (2 khái niệm
> khác nhau, cùng tên "gaze") — chỉ giữ lại gaze TARGET vì hữu ích hơn hẳn (ra
> điểm nhìn thật, không chỉ hướng chung chung).

Model **Gazelle-DINOv3** (bản distill nhẹ HGNetV2-Atto, ~12MB) từ
[PINTO0309/gazelle-dinov3](https://github.com/PINTO0309/gazelle-dinov3)
— input là CẢ khung hình + bbox đầu người (ước lượng thô từ bbox person, CHƯA
có face/head detector riêng), output là 1 **điểm pixel THẬT trong chính khung
hình đó** (heatmap 32×32 lấy argmax) + điểm inout (0-1: xác suất điểm nhìn còn
nằm trong khung — thấp = đang nhìn ra ngoài/về phía camera).

Model có 2 input (ảnh + bbox) → KHÔNG hợp pattern crop-per-object của SGIE
nvinfer thông thường. Chạy qua **ONNXRuntime độc lập** (CPU, không có CUDA EP
trong container hiện tại — model rất nhẹ nên vẫn chấp nhận được) trong
`src/ds_tracker/gaze_target_probe.py` (`GazeTargetProbe`). Đọc pixel qua
`get_nvds_buf_surface()` — hàm này CHỈ hỗ trợ RGBA/RGB, nên pipeline tự chèn
`nvvideoconvert(RGBA)` NGAY TRONG main chain (không dùng `tee` — tee làm buffer
tách nhánh độc lập, overlay vẽ ở nhánh phụ sẽ KHÔNG lên video nhánh chính, đã
tự vấp lỗi này khi build). Chèn sau SGIE staff/customer (vẫn ăn NV12 như cũ,
không bị ảnh hưởng) — chỉ đoạn tiler/osd/encoder chạy RGBA (cùng pattern
`inference.mode: triton` vốn đã chạy production).

Đã test THẬT: chấm đỏ + đường vàng nối từ đầu người → điểm nhìn hiện đúng trên
video (VD: nhân viên pha chế nhìn xuống máy pha cà phê trước mặt — hợp lý về
mặt thực tế, không chỉ là hướng chung chung).

Để bật: set `gaze_target.onnx_path` trong `config.yaml`:
```yaml
gaze_target:
  onnx_path: /workspace/deepstream-rtsp-tracker/models/gaze_target/gazelle_atto_inout.onnx
  inout_threshold: 0.3
```
Độc lập với `inference.mode` (chạy được cả `nvinfer` lẫn `triton`, không qua nvinfer).

**Hạn chế cần biết:**
- Bbox đầu người là ước lượng thô (top 25% bbox person) — kém chính xác hơn
  head/face detector thật.
- Chỉ ra được ĐIỂM trong ảnh, KHÔNG tự gắn nhãn ngữ nghĩa ("đang nhìn vào kệ
  hàng A") — muốn vậy cần thêm bước định nghĩa vùng quan tâm (zone) theo toạ
  độ mỗi camera rồi kiểm tra điểm nhìn có rơi vào zone nào, CHƯA làm.
- CPU-only ONNXRuntime trong container hiện tại (không có CUDA EP) — nếu cần
  nhanh hơn, build lại onnxruntime-gpu với CUDA libs khớp trong container.

## Rule engine (phone-usage)

Kết quả `action_label` (phone-usage rule) được `TrackerMetadataHandler`
(`osd_probe.py`) đọc theo `track_id` (tách theo `gie-unique-id`, xem
`config.py` `GIE_ID_*`) và đưa vào `src/ds_tracker/rule_engine.py`
(`RuleEngine`) để áp luật, log `ALERT [...]` (có cooldown chống spam theo
`track_id`). Luật hiện tại: `action="using_phone"` → alert ngay
(`IMMEDIATE_ALERT_ACTIONS`). Sửa biến này trong `rule_engine.py` khi cần luật
khác. Gaze TARGET (mục trên) ra điểm pixel, chưa có nhãn ngữ nghĩa để fusion
vào đây — thêm lại khi có bước định nghĩa zone theo camera.

Phone-usage rule chỉ hỗ trợ ở `inference.mode: nvinfer`; chưa hỗ trợ ở
`mode: triton`. Gaze TARGET (mục trên) độc lập với `mode`.

## Debug DeepStream trong container

DeepStream (`pyds`, GStreamer, TensorRT) chạy trong container `thaokb_ds9`
(`docker exec -it thaokb_ds9 bash`, rồi `source /workspace/venv/bin/activate`)
— KHÔNG có trên host trực tiếp. `/workspace` trong container map thẳng ra
thư mục `workspace/` của repo này trên host, nên sửa file trên host là container
thấy ngay, không cần rebuild/copy.
