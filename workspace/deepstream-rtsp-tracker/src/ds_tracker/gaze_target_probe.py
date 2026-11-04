"""Probe: gaze TARGET estimation — trả lời thẳng câu hỏi "người này đang nhìn
vào ĐÂU trong khung hình", không chỉ "đang nhìn về hướng nào so với camera"
(đó là gaze DIRECTION — model MobileGaze, đã bỏ vì gây nhầm lẫn với feature này).

Model: Gazelle-DINOv3 (bản distill nhẹ, HGNetV2-Atto) từ
https://github.com/PINTO0309/gazelle-dinov3 — input là CẢ khung hình + bbox
đầu người (không phải crop đơn như SGIE thông thường), output là heatmap xác
suất trên chính khung hình đó → lấy điểm cực đại làm "điểm nhìn tới" (pixel
thật trong ảnh), cộng điểm inout (0-1: xác suất điểm nhìn còn nằm TRONG khung
hình, thấp = đang nhìn ra ngoài khung/về phía camera).

KHÔNG chạy qua nvinfer (model 2 input ảnh+bbox, nvinfer secondary chỉ hỗ trợ
1 input dạng crop-per-object) — tự đọc pixel qua get_nvds_buf_surface() và
gọi ONNXRuntime trực tiếp trong Python, cùng pattern với TritonYoloSgieProbe
(triton_infer_probe.py) đã dùng trong repo này.

Head bbox KHÔNG có detector riêng — ước lượng thô từ bbox 'person' (phần trên
cùng, xem HEAD_HEIGHT_RATIO) vì pipeline hiện chưa có face/head detector. Kém
chính xác hơn head detector thật, tự tinh chỉnh khi có video thật."""

from __future__ import annotations

import cv2
import numpy as np
import pyds
from gi.repository import Gst

from .config import DETECT_CLASSES
from .logging_utils import log

# Tỉ lệ chiều cao bbox person tính từ đỉnh dùng làm bbox đầu (ước lượng thô,
# KHÔNG phải head detector thật).
HEAD_HEIGHT_RATIO = 0.25
HEAD_WIDTH_RATIO = 1.0   # bề rộng head = bề rộng bbox person (an toàn, hơi rộng)

NET_SIZE = 320   # khớp input model gazelle_hgnetv2_atto (1x3x320x320)


class GazeTargetProbe:
    """Ước lượng điểm nhìn tới (gaze target) trên chính khung hình, vẽ chấm đỏ
    + đường nối từ đầu người → điểm nhìn, bằng model Gazelle-DINOv3 (ONNXRuntime,
    chạy CPU — model rất nhẹ ~12MB, xem README nếu cần GPU EP sau này).

    Dùng làm pad-probe callback trên src pad của nvtracker (cần bbox person đã
    có sẵn từ PGIE; không cần track_id nên không bắt buộc chạy SAU tracker,
    nhưng gắn ở đây cho tiện — cùng điểm TrackerMetadataHandler đọc)."""

    def __init__(self, onnx_path: str, inout_threshold: float = 0.3,
                 head_height_ratio: float = HEAD_HEIGHT_RATIO):
        import onnxruntime as ort
        self._session = ort.InferenceSession(
            onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self._input_names = [i.name for i in self._session.get_inputs()]
        self._output_names = [o.name for o in self._session.get_outputs()]
        log.info("GazeTargetProbe: model=%s inputs=%s outputs=%s providers=%s",
                 onnx_path, self._input_names, self._output_names,
                 self._session.get_providers())
        self._inout_threshold = inout_threshold
        self._head_height_ratio = head_height_ratio
        self._ticks = 0
        self._n_drawn = 0

    def _infer(self, image_bgr: np.ndarray, head_boxes_px: list[tuple[float, float, float, float]]
               ) -> tuple[np.ndarray, np.ndarray] | None:
        """head_boxes_px: list (x1,y1,x2,y2) pixel trong ẢNH GỐC (image_bgr).
        Trả về (heatmaps [N,h,w], inout [N]) hoặc None nếu không có head nào."""
        if not head_boxes_px:
            return None
        h, w = image_bgr.shape[:2]
        resized = cv2.resize(image_bgr, (NET_SIZE, NET_SIZE), interpolation=cv2.INTER_LINEAR)
        # Preprocess theo đúng export gốc: BGR, CHW, float32, KHÔNG /255, KHÔNG
        # trừ mean/std (xem training/gazelle-dinov3/demo_*.py _preprocess()).
        chw = resized.transpose(2, 0, 1).astype(np.float32)
        image_input = chw[None, ...]

        boxes_norm = np.array(
            [[x1 / w, y1 / h, x2 / w, y2 / h] for x1, y1, x2, y2 in head_boxes_px],
            dtype=np.float32)
        bbox_input = boxes_norm[None, ...]   # [1, N, 4]

        outputs = self._session.run(
            self._output_names,
            {self._input_names[0]: image_input, self._input_names[1]: bbox_input})
        heatmap_i = self._output_names.index("heatmap")
        inout_i = self._output_names.index("inout")
        return outputs[heatmap_i], outputs[inout_i]

    def _draw_target(self, batch_meta, frame_meta, head_cx: float, head_cy: float,
                     gaze_x: float, gaze_y: float, inout: float) -> None:
        """Chấm đỏ tại điểm nhìn + đường vàng nối từ đầu người → điểm đó."""
        dm = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        dm.num_circles = 1
        circ = dm.circle_params[0]
        circ.xc, circ.yc = int(gaze_x), int(gaze_y)
        circ.radius = 10
        circ.circle_color.set(1.0, 0.0, 0.0, 0.9)   # đỏ
        circ.has_bg_color = 1
        circ.bg_color.set(1.0, 0.0, 0.0, 0.9)

        dm.num_lines = 1
        line = dm.line_params[0]
        line.x1, line.y1 = int(head_cx), int(head_cy)
        line.x2, line.y2 = int(gaze_x), int(gaze_y)
        line.line_width = 2
        line.line_color.set(1.0, 1.0, 0.0, 0.8)     # vàng

        dm.num_labels = 1
        txt = dm.text_params[0]
        txt.display_text = f"inout={inout:.2f}"
        txt.x_offset = max(0, int(gaze_x) + 14)
        txt.y_offset = max(0, int(gaze_y) - 10)
        txt.font_params.font_name = "Serif"
        txt.font_params.font_size = 11
        txt.font_params.font_color.set(1.0, 0.3, 0.3, 1.0)
        txt.set_bg_clr = 1
        txt.text_bg_clr.set(0.0, 0.0, 0.0, 0.6)

        pyds.nvds_add_display_meta_to_frame(frame_meta, dm)

    def __call__(self, pad, info, u_data):
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            # Ảnh RGBA của CHÍNH frame này (cần nvbuf-memory-type=3 trên
            # streammux để CPU đọc trực tiếp — xem pipeline.py).
            n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            bgr = cv2.cvtColor(n_frame, cv2.COLOR_RGBA2BGR)
            h, w = bgr.shape[:2]

            # Gom bbox đầu (ước lượng) của mọi person trong frame này.
            head_boxes: list[tuple[float, float, float, float]] = []
            centers: list[tuple[float, float]] = []
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break
                if obj_meta.class_id in DETECT_CLASSES:
                    rect = obj_meta.rect_params
                    hx1 = rect.left
                    hy1 = rect.top
                    hx2 = min(w, rect.left + rect.width * HEAD_WIDTH_RATIO)
                    hy2 = min(h, rect.top + rect.height * self._head_height_ratio)
                    if hx2 > hx1 and hy2 > hy1:
                        head_boxes.append((hx1, hy1, hx2, hy2))
                        centers.append(((hx1 + hx2) / 2, (hy1 + hy2) / 2))
                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            result = self._infer(bgr, head_boxes)
            if result is not None:
                heatmaps, inouts = result
                for i, (hm, inout) in enumerate(zip(heatmaps, inouts)):
                    inout_f = float(inout)
                    if inout_f < self._inout_threshold:
                        continue   # nhìn ra ngoài khung / không đủ tin cậy
                    row, col = np.unravel_index(int(np.argmax(hm)), hm.shape)
                    gaze_x = (col + 0.5) / hm.shape[1] * w
                    gaze_y = (row + 0.5) / hm.shape[0] * h
                    cx, cy = centers[i]
                    self._draw_target(batch_meta, frame_meta, cx, cy, gaze_x, gaze_y, inout_f)
                    self._n_drawn += 1

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        self._ticks += 1
        if self._ticks % 60 == 0:
            log.info("GazeTargetProbe: đã vẽ %d điểm nhìn (tích luỹ)", self._n_drawn)

        return Gst.PadProbeReturn.OK
