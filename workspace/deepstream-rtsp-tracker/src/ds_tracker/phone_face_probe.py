"""Probe: parse tensor phone+face (PGIE#2, model tự train YOLO11n fine-tune
trên FPI-Det — github.com/KvCgRv/FPI-Det, MIT) → luật "đang dùng điện thoại"
= phone GẦN face (không phải phone chồng lên toàn thân người như bản cũ).

Gắn trên src pad của PGIE#2 (pgie2_phone_face_config.yml, chạy TUẦN TỰ sau
PGIE#1 person detector trong graph — cùng buffer, pixel chưa đổi). Tại thời
điểm này object 'person' đã được YoloTensorParser (PGIE#1) tạo sẵn trong
frame_meta.obj_meta_list — probe này CHỈ cần gắn nhãn 'using_phone' vào đúng
object person chứa cặp (phone, face) gần nhau, KHÔNG tự tạo object mới."""

from __future__ import annotations

import pyds
from gi.repository import Gst

from .config import (
    DETECT_CLASSES,
    GIE_ID_PGIE2_PHONE_FACE,
    GIE_ID_SGIE_ACTION,
    PERSON_CLASS,
    PHONE_FACE_CLASS_FACE,
    PHONE_FACE_CLASS_PHONE,
    PHONE_FACE_CONF_THRESHOLD,
    PHONE_FACE_PROXIMITY_EXPAND,
)
from .logging_utils import log
from .tensor_utils import PadScaleCache, decode_yolo_tensor, tensor_from_user_meta


def _expand(box: tuple[float, float, float, float], factor: float
           ) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh = (x2 - x1) / 2 * factor, (y2 - y1) / 2 * factor
    return cx - hw, cy - hh, cx + hw, cy + hh


def _point_in_box(px: float, py: float, box: tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


class PhoneFaceProbe:
    """Decode tensor phone+face từ PGIE#2 → tìm cặp (phone, face) đủ gần nhau
    → gán nhãn 'using_phone' vào object person (tạo bởi PGIE#1) chứa face đó.

    Dùng làm pad-probe callback trên src pad của PGIE#2 (process-mode=1, toàn
    khung hình — model cần thấy cả phone lẫn face cùng lúc)."""

    def __init__(self, net_width: int = 1920, net_height: int = 1080):
        self._scale = PadScaleCache(net_width, net_height, "PhoneFaceProbe")
        self._ticks = 0
        self._n_alerts = 0
        # Đếm cộng dồn để log heartbeat — xác nhận probe THẬT SỰ đang chạy mỗi
        # frame, không chỉ im lặng khi chưa có ai dùng điện thoại (dễ tưởng
        # nhầm là không hoạt động nếu chỉ log lúc có alert).
        self._n_frames_checked = 0
        self._n_phone_seen = 0
        self._n_face_seen = 0
        # Face box (đã scale ra pixel thật) CHƯA vẽ ngay ở đây — nvtracker
        # (chạy NGAY SAU PGIE#2 trong graph) xoá sạch display_meta gắn TRƯỚC
        # nó (đã tự kiểm chứng bug này: vẽ display_meta rect ở đây thì mất
        # tăm trên video dù log vẫn báo face thấy được). Object 'person'
        # (obj_meta, KHÔNG phải display_meta) thì sống sót qua tracker bình
        # thường — nên chỉ stash box vào đây, để TrackerMetadataHandler
        # (osd_probe.py, chạy SAU tracker) đọc + vẽ thật, giống hệt cách
        # 'using_phone' text/gaze target dot đã vẽ thành công ở đó.
        # key = (pad_index, frame_num) → list (x1,y1,x2,y2,is_hit).
        self.pending_face_boxes: dict[tuple[int, int], list[tuple[float, float, float, float, bool]]] = {}
        # Tương tự cho phone — is_hit = True nếu phone này đủ gần 1 face (tức
        # đang tính là "using_phone"), vẽ màu khác để phân biệt phone thường
        # (VD điện thoại để trên bàn, không cầm gần mặt) với phone đang dùng.
        self.pending_phone_boxes: dict[tuple[int, int], list[tuple[float, float, float, float, bool]]] = {}

    def _find_using_phone_faces(self, raw):
        """Trả về (phones, faces, face_hit_idx, phone_hit_idx) — phones/faces:
        list bbox (x1,y1,x2,y2,score) ở scale 640 (TẤT CẢ detect được, để vẽ
        bbox và log heartbeat); face_hit_idx: index (trong faces) của face có
        phone đủ gần (tâm face nằm trong bbox phone đã phóng to); phone_hit_idx:
        index (trong phones) của phone gây ra ít nhất 1 hit đó — dùng cả để gán
        'using_phone' lẫn tô màu bbox khác cho face/phone "đang dùng điện thoại"."""
        phones = decode_yolo_tensor(raw, PHONE_FACE_CLASS_PHONE, PHONE_FACE_CONF_THRESHOLD)
        faces = decode_yolo_tensor(raw, PHONE_FACE_CLASS_FACE, PHONE_FACE_CONF_THRESHOLD)
        if not phones or not faces:
            return phones, faces, set(), set()

        face_hit_idx: set[int] = set()
        phone_hit_idx: set[int] = set()
        for p, (px1, py1, px2, py2, _pscore) in enumerate(phones):
            phone_expanded = _expand((px1, py1, px2, py2), PHONE_FACE_PROXIMITY_EXPAND)
            for i, (fx1, fy1, fx2, fy2, _fscore) in enumerate(faces):
                fcx, fcy = _center((fx1, fy1, fx2, fy2))
                if _point_in_box(fcx, fcy, phone_expanded):
                    face_hit_idx.add(i)
                    phone_hit_idx.add(p)
        return phones, faces, face_hit_idx, phone_hit_idx

    def _attach_using_phone(self, batch_meta, obj_meta) -> None:
        """Gắn NvDsClassifierMeta (unique_component_id=GIE_ID_SGIE_ACTION)
        label='using_phone' vào object person — để osd_probe.py/rule_engine.py
        đọc được y hệt classifier SGIE thật, không cần sửa gì thêm ở đó."""
        cls_meta = pyds.nvds_acquire_classifier_meta_from_pool(batch_meta)
        cls_meta.unique_component_id = GIE_ID_SGIE_ACTION

        label_info = pyds.nvds_acquire_label_info_meta_from_pool(batch_meta)
        label_info.result_label = "using_phone"
        label_info.result_class_id = 0
        label_info.result_prob = 1.0

        pyds.nvds_add_label_info_meta_to_classifier(cls_meta, label_info)
        pyds.nvds_add_classifier_meta_to_object(obj_meta, cls_meta)

    def __call__(self, pad, info, u_data):
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        sx, sy = self._scale.get(pad)

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            raw = None
            l_user = frame_meta.frame_user_meta_list
            while l_user is not None:
                try:
                    user_meta = pyds.NvDsUserMeta.cast(l_user.data)
                except StopIteration:
                    break
                if user_meta.base_meta.meta_type == \
                        pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                    raw = tensor_from_user_meta(
                        user_meta, expected_unique_id=GIE_ID_PGIE2_PHONE_FACE)
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

            # faces/phones: TẤT CẢ bbox detect được (scale 640). face_hit_idx/
            # phone_hit_idx: index của face/phone thuộc 1 cặp đủ gần → vừa để
            # vẽ bbox màu khác vừa để gán 'using_phone' vào đúng object person.
            phones, faces, face_hit_idx, phone_hit_idx = self._find_using_phone_faces(raw)
            self._n_frames_checked += 1
            self._n_phone_seen += len(phones)
            self._n_face_seen += len(faces)

            # Stash bbox quanh MỌI face/phone (quy đổi scale 640 → buffer thật
            # bằng sx,sy) — cam nếu đang dùng điện thoại, xanh lá/vàng nếu
            # không. VẼ thật ở TrackerMetadataHandler (osd_probe.py, sau
            # tracker) — xem comment ở __init__.
            # float() BẮT BUỘC — fx1/fy1/... là np.float32 (từ decode_yolo_tensor,
            # qua numpy array), gán thẳng np.float32 vào rect_params (pyds
            # property C-struct) ÂM THẦM không lỗi nhưng KHÔNG vẽ được gì
            # (đã tự vấp bug này: log xác nhận có gọi vẽ, tọa độ hợp lệ,
            # nhưng video không hiện bbox nào — nvds_add_display_meta_to_frame
            # vẫn chạy "thành công"). obj_meta.rect_params (yolo_probe.py)
            # tránh được vì đã có sẵn float(...) quanh mọi phép gán.
            key = (frame_meta.pad_index, frame_meta.frame_num)
            if faces:
                self.pending_face_boxes[key] = [
                    (float(fx1 * sx), float(fy1 * sy), float(fx2 * sx), float(fy2 * sy),
                     i in face_hit_idx)
                    for i, (fx1, fy1, fx2, fy2, _fscore) in enumerate(faces)
                ]
            if phones:
                self.pending_phone_boxes[key] = [
                    (float(px1 * sx), float(py1 * sy), float(px2 * sx), float(py2 * sy),
                     p in phone_hit_idx)
                    for p, (px1, py1, px2, py2, _pscore) in enumerate(phones)
                ]

            hit_centers = [(cx * sx, cy * sy) for cx, cy in
                          (_center((faces[i][0], faces[i][1], faces[i][2], faces[i][3]))
                           for i in face_hit_idx)]

            if hit_centers:
                l_obj = frame_meta.obj_meta_list
                while l_obj is not None:
                    try:
                        obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                    except StopIteration:
                        break
                    if obj_meta.class_id in DETECT_CLASSES:
                        rect = obj_meta.rect_params
                        pbox = (rect.left, rect.top,
                               rect.left + rect.width, rect.top + rect.height)
                        for cx, cy in hit_centers:
                            if _point_in_box(cx, cy, pbox):
                                self._attach_using_phone(batch_meta, obj_meta)
                                self._n_alerts += 1
                                break
                    try:
                        l_obj = l_obj.next
                    except StopIteration:
                        break

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        self._ticks += 1
        if self._ticks % 60 == 0:
            # Heartbeat — LUÔN in ra, kể cả khi chưa phát hiện gì, để biết
            # chắc probe đang chạy thật mỗi frame (không phải đứng im).
            log.info("PhoneFaceProbe: đang check — %d frame, phone thấy %d lần, "
                     "face thấy %d lần, using_phone gắn %d lần (tích luỹ)",
                     self._n_frames_checked, self._n_phone_seen,
                     self._n_face_seen, self._n_alerts)

        return Gst.PadProbeReturn.OK
