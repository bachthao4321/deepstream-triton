"""Probe: parse tensor YOLOv11 (output0 no-NMS) → NvDsObjectMeta cho tracker.

Gắn làm pad-probe trên src pad của nvinfer (PGIE#1 — person detector). Một lần
probe xử lý cả batch N cam; mỗi frame tự decode + NMS độc lập rồi ghi log
detection ra file riêng per-cam.

"Action recognition" (phone-usage) KHÔNG còn nằm ở đây — xem
phone_face_probe.py (PGIE#2, model tự train FPI-Det: phone+face, chạy sau
PGIE#1 trong graph, đọc lại object person đã tạo ở đây để gán nhãn
'using_phone' theo track)."""

from __future__ import annotations

import logging

import pyds
from gi.repository import Gst

from .config import CONF_THRESHOLD, GIE_ID_PGIE, LABELS, PERSON_CLASS
from .logging_utils import make_cam_logger
from .tensor_utils import PadScaleCache, decode_yolo_tensor, tensor_from_user_meta


class YoloTensorParser:
    """Parse output0 [84, 8400] từ tensor_meta → tạo NvDsObjectMeta cho tracker.

    Dùng làm pad-probe callback trên src pad của nvinfer. Đồng thời ghi log
    detection per-frame ra file riêng cho từng cam (nếu cam_log_dir != None).
    Một lần probe xử lý cả batch N cam nên N file được ghi song song."""

    def __init__(self, cam_names: dict[int, str] | None = None,
                 cam_log_dir: str | None = None,
                 net_width: int = 1920, net_height: int = 1080):
        self._cam_names = cam_names or {}
        self._cam_log_dir = cam_log_dir
        self._cam_loggers: dict[int, logging.Logger] = {}
        self._scale = PadScaleCache(net_width, net_height, "YoloTensorParser")

    def _cam(self, pad_index: int) -> str:
        return self._cam_names.get(pad_index, f"cam{pad_index}")

    def _cam_logger(self, pad_index: int) -> logging.Logger | None:
        if not self._cam_log_dir:
            return None
        lg = self._cam_loggers.get(pad_index)
        if lg is None:
            lg = make_cam_logger(self._cam(pad_index), self._cam_log_dir)
            self._cam_loggers[pad_index] = lg
        return lg

    def __call__(self, pad, info, u_data):
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        # Scale (sx, sy) đọc từ caps của pad nvinfer = kích thước buffer THẬT
        # / 640. Đúng cho cả old mux (1920×1080) lẫn new mux (source 2560×1440)
        # nên KHÔNG cần chỉnh bbox_scale_* trong config theo từng camera.
        sx, sy = self._scale.get(pad)

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            pad_index = frame_meta.pad_index

            # Đọc tensor thô output0 [84, 8400] (KHÔNG NMS) của RIÊNG frame này.
            # Model no-NMS output batch-aware; ta tự decode + NMS từng frame nên
            # mọi cam trong batch được xử lý độc lập, detection đều đặn.
            raw = None
            l_user = frame_meta.frame_user_meta_list
            while l_user is not None:
                try:
                    user_meta = pyds.NvDsUserMeta.cast(l_user.data)
                except StopIteration:
                    break
                if user_meta.base_meta.meta_type == \
                        pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                    raw = tensor_from_user_meta(user_meta, expected_unique_id=GIE_ID_PGIE)
                    if raw is not None:
                        break
                try:
                    l_user = l_user.next
                except StopIteration:
                    break

            if raw is None:
                try:
                    l_frame = l_frame.next
                except StopIteration:
                    break
                continue

            dets = decode_yolo_tensor(raw, PERSON_CLASS, CONF_THRESHOLD)
            cam_lg = self._cam_logger(pad_index)
            n_det = 0                            # số person detect trên frame này

            for x1, y1, x2, y2, score in dets:
                obj_meta = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
                obj_meta.class_id = PERSON_CLASS
                obj_meta.confidence = float(score)
                obj_meta.obj_label = LABELS[0]
                obj_meta.unique_component_id = 1

                rect = obj_meta.rect_params
                rect.left = float(x1) * sx
                rect.top = float(y1) * sy
                rect.width = float(x2 - x1) * sx
                rect.height = float(y2 - y1) * sy
                rect.border_width = 2
                rect.border_color.set(0.0, 1.0, 0.0, 1.0)   # xanh lá
                rect.has_bg_color = 0

                # tracker cần detector_bbox_info để bám theo
                dinfo = obj_meta.detector_bbox_info.org_bbox_coords
                dinfo.left = rect.left
                dinfo.top = rect.top
                dinfo.width = rect.width
                dinfo.height = rect.height

                pyds.nvds_add_obj_meta_to_frame(frame_meta, obj_meta, None)

                if cam_lg is not None:
                    n_det += 1
                    cam_lg.info(
                        "Frame%-6d det#%-2d %-8s conf=%.2f bbox=(%.0f,%.0f,%.0f,%.0f)",
                        frame_meta.frame_num, n_det, LABELS[0], float(score),
                        rect.left, rect.top, rect.width, rect.height)

            # Dòng tổng kết mỗi frame — luôn ghi (kể cả 0 detection) để thấy nhịp
            # frame của từng cam song song trong file log riêng.
            if cam_lg is not None:
                cam_lg.info("Frame%-6d --- total person=%d",
                            frame_meta.frame_num, n_det)

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK
