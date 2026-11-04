# DeepStream RTSP client + Triton

Pipeline Python nhận video RTSP, gọi model qua Triton gRPC, theo dõi người bằng
NvDCF, ghi log và video có bounding box/nhãn. Máy chạy DeepStream vẫn cần GPU
NVIDIA và môi trường DeepStream; không cần màn hình khi chạy `headless`.

```text
RTSP → DeepStream: decode / preprocess → Triton: YOLO + classifier
                 ← kết quả inference ←
       DeepStream: tracker / overlay / ghi video
```

## Chuẩn bị cấu hình riêng

Tại thư mục gốc dự án:

```bash
cp workspace/deepstream-rtsp-tracker/configs/config.triton.example.yaml \
   workspace/deepstream-rtsp-tracker/configs/config.local.yaml
```

Sửa `config.local.yaml`: URL camera và `inference.triton.url` thành địa chỉ
`IP_SERVER:8101` (không có tiền tố giao thức). File này đã được ignore.
Nếu camera/password chứa ký tự đặc biệt, URL-encode giá trị tương ứng.
Không đưa tài khoản thật vào file example.

Triton server phải có hai model đúng giao diện hiện tại:

| Model | Input FP32 | Output FP32 | Max batch |
| --- | --- | --- | --- |
| `yolo11m_nonms` | `images`: `[B,3,640,640]` | `output0`: `[B,84,N]` | 4 |
| `staff_customer` | `input`: `[B,3,224,224]` | `output`: `[B,2]`, đã softmax | 16 |

Giữ tên tensor, preprocessing và thứ tự nhãn khớp với model trên server.
File labels hiện tại là `customer;staff`. Có thể tắt classifier bằng cách đặt
`sgie_model: ""`. Model weights/engines nằm trên server Triton; client chỉ cần
mã nguồn, labels và cấu hình tracker. Các file `config.pbtxt` được giữ trong Git.
Engine TensorRT cần được build phù hợp với GPU và runtime của server.

## Chạy với môi trường hiện có

Trong container DeepStream đã cài đủ thư viện:

```bash
cd /workspace/deepstream-rtsp-tracker
bash run.sh --config configs/config.local.yaml
```

Hoặc từ host đang có container `thaokb_ds9`:

```bash
docker exec -it thaokb_ds9 bash /workspace/deepstream-rtsp-tracker/run.sh \
  --config /workspace/deepstream-rtsp-tracker/configs/config.local.yaml
```

Chạy client riêng mà không khởi động Triton tại máy client:

```bash
docker compose up -d --no-deps ds9-test
```

Lệnh trên cần image `deepstream:9.0-triton` đã có hoặc build được. Dockerfile hiện
tại là nền môi trường; không đóng gói venv. Clone repo trên máy mới cần tự chuẩn
bị bộ cài `.deb` được `COPY` trong Dockerfile và cài Python bindings/dependencies
phù hợp (Gst-Python/PyGObject, `pyds`, PyYAML, torch, torchvision, OpenCV, numpy,
`tritonclient[grpc]`; có `ffmpeg` để remux video). Không thể chỉ clone repo rồi
chạy ngay bằng Python thông thường. Venv hiện có ở `workspace/venv` không upload.

Máy DeepStream cần truy cập được camera và cổng gRPC của server Triton. Dùng
mạng riêng/VPN cho kết nối giữa hai máy; client hiện gọi gRPC không bật TLS/auth.
`localhost:8101` chỉ đúng khi Triton ở cùng máy. Docker Compose dùng host network
và đường dẫn hiện tại dành cho Linux.

Chế độ Triton hiện hỗ trợ YOLO11m + Staff/Customer. Phone/face rule chưa hỗ trợ
qua Triton. Gaze chạy ONNX cục bộ nếu bật, nên mẫu này tắt cả phone rule và gaze.
Probe gọi inference đồng bộ; tốc độ phụ thuộc cả độ trễ và băng thông mạng.

## Đưa lên GitHub

Giữ mã `.py`, `run.sh`, Dockerfile, Compose, cấu hình model/tracker, labels và
config mẫu. `.gitignore` loại bộ cài SDK, model/checkpoint, dataset/archive,
venv/cache, video/log, kết quả train và cấu hình camera thật `configs/config.yaml`.
`.dockerignore` chỉ gửi Dockerfile và bộ cài DeepStream vào build context.

Thư mục gốc chưa có Git tại thời điểm chuẩn bị. Kiểm tra danh sách trước khi commit:

```bash
git init
git add .
git diff --cached --stat
git diff --cached
```

Chỉ commit/push sau khi xem nội dung staged và xác nhận không còn thông tin riêng.
Nếu áp dụng vào repo đã từng track file nhạy cảm/model, `.gitignore` không gỡ
file đã track hoặc xóa lịch sử Git; cần xử lý riêng trước khi public.

Xem thêm tài liệu pipeline trong `workspace/deepstream-rtsp-tracker/README.md`.
