"""Helper dùng chung để đọc/decode tensor thô YOLO (output0 [C, 8400], C = 4 +
số class) từ NvDsInferTensorMeta — tách ra từ yolo_probe.py để PhoneFaceProbe
(pgie2, phone_face_probe.py) dùng lại, không copy-paste logic decode."""

from __future__ import annotations

import ctypes

import pyds
import torch
from torchvision.ops import nms as tv_nms

from .config import NMS_IOU


def tensor_from_user_meta(user_meta, layer_name: str = "output0",
                          expected_unique_id: int | None = None):
    """Lấy layer `layer_name` từ NvDsInferTensorMeta → torch tensor.

    QUAN TRỌNG: khi có ≥2 nvinfer chạy TUẦN TỰ trên CÙNG buffer/frame_meta
    (VD: PGIE#1 person detector → PGIE#2 phone+face detector), user_meta ở
    frame_meta.frame_user_meta_list có thể chứa tensor của CẢ HAI (mỗi
    nvinfer tự attach tensor của nó, không xoá tensor của nvinfer trước) — cả
    2 đều đặt tên layer output là "output0" theo convention export chuẩn, nên
    lọc CHỈ theo layer_name có thể vô tình đọc nhầm tensor của nvinfer KHÁC
    (đã tự vấp bug này: PhoneFaceProbe đọc nhầm tensor 84-kênh COCO của PGIE#1
    thay vì tensor 6-kênh thật của PGIE#2 → luôn ra 0 phone/face dù model
    đúng). expected_unique_id (khớp gie-unique-id trong config nvinfer, xem
    config.py GIE_ID_*) lọc đúng tensor của ĐÚNG nvinfer — bắt buộc truyền khi
    có nhiều nvinfer output-tensor-meta trong cùng pipeline."""
    tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
    if expected_unique_id is not None and tensor_meta.unique_id != expected_unique_id:
        return None
    for i in range(tensor_meta.num_output_layers):
        layer = pyds.get_nvds_LayerInfo(tensor_meta, i)
        if layer.layerName != layer_name:
            continue
        dims = layer.inferDims
        shape = [dims.d[j] for j in range(dims.numDims)]
        n = 1
        for s in shape:
            n *= int(s)
        ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
        arr = torch.frombuffer(
            (ctypes.c_float * n).from_address(ctypes.addressof(ptr.contents)),
            dtype=torch.float32,
        ).clone().reshape(shape)
        return arr
    return None


def decode_yolo_tensor(raw, class_id: int, conf_threshold: float,
                       nms_iou: float = NMS_IOU) -> list[tuple[float, float, float, float, float]]:
    """Decode output thô YOLO [C, 8400] (hoặc [1,C,8400]) của 1 frame → list
    bbox (x1,y1,x2,y2,score) ở scale 640 cho MỘT class, sau NMS.

    C = 4 (cx,cy,w,h) + số class (80 với YOLO11m COCO, 2 với phone_face_yolo11n)
    — hàm này KHÔNG quan tâm C cụ thể, chỉ cần class_id hợp lệ trong C-4 class.
    Toàn bộ chạy trên GPU (torch/torchvision) rồi mới về CPU."""
    t = raw
    if t.dim() == 3:
        t = t[0]                       # [C, 8400]
    t = t.float().t()                  # [8400, C]

    cls_score = t[:, 4 + class_id]
    keep = cls_score >= conf_threshold
    if not bool(keep.any()):
        return []

    boxes_c = t[keep, :4]
    scores = cls_score[keep]

    cx, cy, w, h = boxes_c[:, 0], boxes_c[:, 1], boxes_c[:, 2], boxes_c[:, 3]
    x1 = cx - w * 0.5
    y1 = cy - h * 0.5
    x2 = cx + w * 0.5
    y2 = cy + h * 0.5
    xyxy = torch.stack([x1, y1, x2, y2], dim=1)

    idx = tv_nms(xyxy, scores, nms_iou)
    xyxy = xyxy[idx].cpu().numpy()
    scores = scores[idx].cpu().numpy()
    return [(b[0], b[1], b[2], b[3], s) for b, s in zip(xyxy, scores)]


class PadScaleCache:
    """(sx, sy) = kích_thước_buffer_thật / 640, đọc từ caps của 1 pad nvinfer,
    cache lại vì caps cố định suốt phiên. Tách thành class nhỏ để nhiều probe
    (mỗi cái theo dõi 1 pad khác nhau) dùng độc lập, không lẫn cache."""

    def __init__(self, net_w: int = 1920, net_h: int = 1080, log_prefix: str = ""):
        self._net_w = net_w
        self._net_h = net_h
        self._log_prefix = log_prefix
        self._cached_wh: tuple[int, int] | None = None

    def get(self, pad) -> tuple[float, float]:
        from .logging_utils import log
        if self._cached_wh is None:
            w = h = None
            caps = pad.get_current_caps()
            if caps is not None and caps.get_size() > 0:
                s = caps.get_structure(0)
                ok_w, w = s.get_int("width")
                ok_h, h = s.get_int("height")
                if not (ok_w and ok_h):
                    w = h = None
            if w and h:
                self._cached_wh = (w, h)
                log.info("%s: scale bbox theo buffer nvinfer %dx%d (đọc từ pad caps)",
                         self._log_prefix, w, h)
            else:
                self._cached_wh = (self._net_w, self._net_h)
                log.warning("%s: không đọc được pad caps → fallback %dx%d (config).",
                           self._log_prefix, self._net_w, self._net_h)
        w, h = self._cached_wh
        return w / 640.0, h / 640.0
