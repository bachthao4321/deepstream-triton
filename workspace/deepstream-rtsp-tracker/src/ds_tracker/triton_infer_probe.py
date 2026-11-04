"""Probe: PGIE detect (YOLOv11) + SGIE classify (staff/customer) qua Triton
Inference Server (gRPC), thay thế hoàn toàn 2 element nvinfer.

Vì DeepStream 9.0 nvinferserver cần libtritonserver.so cục bộ (không có trong
image build tay, kể cả ở chế độ gRPC remote — xem ghi chú trong README), probe
này tự đọc pixel từ NvBufSurface (RGBA), tiền xử lý bằng OpenCV/numpy, gọi
Triton qua tritonclient.grpc, rồi tự attach NvDsObjectMeta + NvDsClassifierMeta
— tạo ra ĐÚNG cấu trúc meta mà TrackerMetadataHandler (osd_probe.py) đã kỳ vọng
sẵn, nên phần downstream (tracker, tiler, osd) không cần sửa gì.

Gắn làm pad-probe trên src pad của element ngay SAU streammux (đã convert
RGBA) — TRƯỚC nvtracker. Một lần probe xử lý cả batch N cam."""

from __future__ import annotations

import logging

import cv2
import numpy as np
import pyds
import torch
import tritonclient.grpc as grpcclient
from gi.repository import Gst
from torchvision.ops import nms as tv_nms

from .config import CONF_THRESHOLD, LABELS, NMS_IOU, PERSON_CLASS
from .logging_utils import make_cam_logger, log

YOLO_NET = 640
SGIE_NET = 224
# Chuẩn hoá ImageNet cho staff_customer (ViT) — khớp sgie_staff_customer.yml
_SGIE_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_SGIE_SCALE = 0.01735


class TritonYoloSgieProbe:
    """Detect (Triton yolo11m_nonms) + classify (Triton staff_customer),
    gộp làm một probe để chỉ cần 1 lần đọc pixel/frame."""

    def __init__(self, triton_url: str, yolo_model: str, sgie_model: str | None,
                 sgie_labels_path: str | None, cam_names: dict[int, str] | None = None,
                 cam_log_dir: str | None = None,
                 yolo_max_batch: int = 4, sgie_max_batch: int = 16):
        self._client = grpcclient.InferenceServerClient(url=triton_url)
        self._yolo_model = yolo_model
        self._sgie_model = sgie_model
        self._yolo_max_batch = yolo_max_batch
        self._sgie_max_batch = sgie_max_batch
        self._sgie_labels = self._load_labels(sgie_labels_path) if sgie_model else None

        self._cam_names = cam_names or {}
        self._cam_log_dir = cam_log_dir
        self._cam_loggers: dict[int, logging.Logger] = {}

        log.info("TritonYoloSgieProbe: url=%s yolo_model=%s sgie_model=%s",
                 triton_url, yolo_model, sgie_model or "(tắt)")

    @staticmethod
    def _load_labels(path: str) -> list[str]:
        with open(path) as f:
            line = f.readline().strip()
        return line.split(";")

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

    # ── Triton calls (tự chunk theo max_batch_size của model) ────────────────
    def _infer_batched(self, model_name: str, input_name: str, output_name: str,
                        batch: np.ndarray, max_batch: int) -> np.ndarray:
        outs = []
        for i in range(0, len(batch), max_batch):
            chunk = batch[i:i + max_batch]
            inp = grpcclient.InferInput(input_name, chunk.shape, "FP32")
            inp.set_data_from_numpy(chunk)
            res = self._client.infer(model_name=model_name, inputs=[inp],
                                      outputs=[grpcclient.InferRequestedOutput(output_name)])
            outs.append(res.as_numpy(output_name))
        return np.concatenate(outs, axis=0)

    # ── decode YOLOv11 no-NMS output (giống YoloTensorParser._decode) ────────
    @staticmethod
    def _decode(raw_np: np.ndarray):
        """raw_np: [84, N] (1 frame) → list (x1,y1,x2,y2,score) scale 640, sau NMS."""
        t = torch.from_numpy(raw_np).float().t()          # [N, 84]
        cls_person = t[:, 4 + PERSON_CLASS]
        keep = cls_person >= CONF_THRESHOLD
        if not bool(keep.any()):
            return []
        boxes_c = t[keep, :4]
        scores = cls_person[keep]
        cx, cy, w, h = boxes_c[:, 0], boxes_c[:, 1], boxes_c[:, 2], boxes_c[:, 3]
        xyxy = torch.stack([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5], dim=1)
        idx = tv_nms(xyxy, scores, NMS_IOU)
        xyxy = xyxy[idx].numpy()
        scores = scores[idx].numpy()
        return [(b[0], b[1], b[2], b[3], s) for b, s in zip(xyxy, scores)]

    @staticmethod
    def _attach_obj_meta(batch_meta, frame_meta, x1, y1, x2, y2, score):
        obj_meta = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
        obj_meta.class_id = PERSON_CLASS
        obj_meta.confidence = float(score)
        obj_meta.obj_label = LABELS[0]
        obj_meta.unique_component_id = 1

        rect = obj_meta.rect_params
        rect.left, rect.top = float(x1), float(y1)
        rect.width, rect.height = float(x2 - x1), float(y2 - y1)
        rect.border_width = 2
        rect.border_color.set(0.0, 1.0, 0.0, 1.0)
        rect.has_bg_color = 0

        dinfo = obj_meta.detector_bbox_info.org_bbox_coords
        dinfo.left, dinfo.top = rect.left, rect.top
        dinfo.width, dinfo.height = rect.width, rect.height

        pyds.nvds_add_obj_meta_to_frame(frame_meta, obj_meta, None)
        return obj_meta

    @staticmethod
    def _attach_classifier_meta(batch_meta, obj_meta, label: str, class_id: int, prob: float):
        cls_meta = pyds.nvds_acquire_classifier_meta_from_pool(batch_meta)
        cls_meta.unique_component_id = 2
        label_info = pyds.nvds_acquire_label_info_meta_from_pool(batch_meta)
        label_info.result_label = label
        label_info.result_class_id = class_id
        label_info.result_prob = float(prob)
        pyds.nvds_add_label_info_meta_to_classifier(cls_meta, label_info)
        pyds.nvds_add_classifier_meta_to_object(obj_meta, cls_meta)

    def __call__(self, pad, info, u_data):
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

        # ── Pass 1: đọc pixel + tiền xử lý toàn batch cho YOLO ────────────────
        frame_metas, yolo_batch, frame_rgbs = [], [], []
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break
            try:
                n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                rgba = np.array(n_frame, copy=True, order="C")
            except Exception:
                log.exception("TritonYoloSgieProbe: không đọc được buf surface pad=%d",
                              frame_meta.pad_index)
                try:
                    l_frame = l_frame.next
                except StopIteration:
                    break
                continue
            rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
            frame_rgbs.append(rgb)
            frame_metas.append(frame_meta)
            r = cv2.resize(rgb, (YOLO_NET, YOLO_NET)).astype(np.float32) / 255.0
            yolo_batch.append(np.transpose(r, (2, 0, 1)))
            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        if not frame_metas:
            return Gst.PadProbeReturn.OK

        yolo_out = self._infer_batched(
            self._yolo_model, "images", "output0",
            np.stack(yolo_batch, axis=0), self._yolo_max_batch)   # [B,84,N]

        # ── Pass 2: decode mỗi frame, attach obj_meta, gom crop cho SGIE ──────
        sgie_batch, sgie_targets = [], []   # sgie_targets: list[(obj_meta, batch_meta)]
        for i, frame_meta in enumerate(frame_metas):
            H, W = frame_rgbs[i].shape[:2]
            sx, sy = W / YOLO_NET, H / YOLO_NET
            dets = self._decode(yolo_out[i])
            cam_lg = self._cam_logger(frame_meta.pad_index)
            n_det = 0

            for x1, y1, x2, y2, score in dets:
                x1s, y1s, x2s, y2s = x1 * sx, y1 * sy, x2 * sx, y2 * sy
                obj_meta = self._attach_obj_meta(batch_meta, frame_meta,
                                                  x1s, y1s, x2s, y2s, score)
                n_det += 1
                if cam_lg is not None:
                    cam_lg.info(
                        "Frame%-6d det#%-2d %-8s conf=%.2f bbox=(%.0f,%.0f,%.0f,%.0f)",
                        frame_meta.frame_num, n_det, LABELS[0], float(score),
                        x1s, y1s, x2s - x1s, y2s - y1s)

                if self._sgie_model is not None:
                    xi1, yi1 = max(0, int(x1s)), max(0, int(y1s))
                    xi2, yi2 = min(W, int(x2s)), min(H, int(y2s))
                    if xi2 - xi1 >= 8 and yi2 - yi1 >= 8:
                        crop = frame_rgbs[i][yi1:yi2, xi1:xi2]
                        cr = cv2.resize(crop, (SGIE_NET, SGIE_NET)).astype(np.float32)
                        cr = (cr - _SGIE_MEAN) * _SGIE_SCALE
                        sgie_batch.append(np.transpose(cr, (2, 0, 1)))
                        sgie_targets.append(obj_meta)

            if cam_lg is not None:
                cam_lg.info("Frame%-6d --- total person=%d",
                            frame_meta.frame_num, n_det)

        # ── SGIE classify (1 lần cho toàn bộ object trong batch) ──────────────
        if self._sgie_model is not None and sgie_batch:
            sgie_out = self._infer_batched(
                self._sgie_model, "input", "output",
                np.stack(sgie_batch, axis=0), self._sgie_max_batch)   # [K,2]
            for obj_meta, probs in zip(sgie_targets, sgie_out):
                cid = int(np.argmax(probs))
                self._attach_classifier_meta(
                    batch_meta, obj_meta, self._sgie_labels[cid], cid, probs[cid])

        return Gst.PadProbeReturn.OK
