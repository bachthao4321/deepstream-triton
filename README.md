# DeepStream RTSP + Triton

Nhận video từ camera RTSP, gọi model qua Triton gRPC để phát hiện người và
phân loại nhân viên/khách hàng, sau đó theo dõi bằng NvDCF và ghi video có nhãn.
Máy client cần GPU NVIDIA và môi trường DeepStream/Python đã cài đủ phụ thuộc.
Model được triển khai trên server Triton, không nằm trong repo.

## Cấu hình RTSP và Triton

Từ thư mục gốc repo:

```bash
cp workspace/deepstream-rtsp-tracker/configs/config.triton.example.yaml \
   workspace/deepstream-rtsp-tracker/configs/config.local.yaml
```

Sửa các giá trị sau trong `config.local.yaml` (giữ các mục còn lại của file mẫu):

```yaml
sources:
  - uri: "rtsp://USERNAME:PASSWORD@CAMERA_IP:554/stream1"
    camera_name: camera_1
    enabled: true

inference:
  mode: triton
  triton:
    url: "TRITON_SERVER_IP:8101"
    yolo_model: yolo11m_nonms
    yolo_max_batch: 4
    sgie_model: staff_customer
    sgie_labels: /workspace/deepstream-rtsp-tracker/models/staff_customer_labels.txt
    sgie_max_batch: 16
```

Thêm camera bằng cách thêm phần tử vào `sources`. Triton phải có model
`yolo11m_nonms` và `staff_customer` khớp cấu hình trong `workspace/triton_model_repo/`.
Địa chỉ Triton dùng dạng `host:port`, không thêm `http://`; client cần truy cập
được camera và cổng gRPC `8101`. Đặt `sgie_model: ""` nếu chỉ cần phát hiện người.
File `config.local.yaml` được Git ignore.

## Chạy

Trong container DeepStream đã chuẩn bị môi trường:

```bash
cd /workspace/deepstream-rtsp-tracker
bash run.sh --config configs/config.local.yaml
```

Hoặc ghi đè nguồn RTSP khi chạy:

```bash
bash run.sh --config configs/config.local.yaml \
  --rtsp "rtsp://USERNAME:PASSWORD@CAMERA_IP:554/stream1"
```

Mẫu chạy `headless`, không cần màn hình; video/log lưu ở `/workspace/recordings`
(trên host là `workspace/recordings`). Mặc định chạy 60 giây; thêm `--duration 0`
để chạy đến khi nhấn Ctrl+C. Chế độ Triton hiện chưa hỗ trợ luật dùng điện thoại.
