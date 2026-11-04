"""ds_tracker — DeepStream RTSP + YOLOv11 + Tracker → annotated tiled MP4.

Pipeline dựng bằng Gst-Python + pyds (theo repo NVIDIA-AI-IOT/
deepstream_python_apps: apps/deepstream-test2, deepstream-test3). Mỗi nguồn RTSP
là một source-bin (nvurisrcbin) gắn ghost pad, link động vào nvstreammux. nvinfer
chạy với output-tensor-meta=1; YoloTensorParser (pad probe) parse tensor output0
→ tạo NvDsObjectMeta cho tracker.
"""

# Phải gọi require_version TRƯỚC khi bất kỳ module nào import Gst từ
# gi.repository, nếu không PyGI cảnh báo và có thể load nhầm version.
import gi

gi.require_version("Gst", "1.0")
