"""Probe: đọc tracking result sau nvtracker → log + vẽ OSD text (tên cam + count).

Gắn làm pad-probe trên src pad của nvtracker (trước tiler). Text dùng offset
tuyệt đối trong khung source; tiler transform sang ô tile khi composite."""

from __future__ import annotations

import pyds
from gi.repository import Gst

from .config import DETECT_CLASSES, GIE_ID_SGIE_ACTION, LABELS
from .logging_utils import log
from .rule_engine import RuleEngine


class TrackerMetadataHandler:
    """Đọc tracking result sau tracker → log + vẽ OSD text. Dùng làm pad-probe
    callback trên src pad của nvtracker (hoặc src pad của SGIE cuối cùng trong
    chuỗi, nếu có SGIE — xem pipeline.py: meta_src)."""

    def __init__(self, cam_names: dict[int, str], rule_engine: RuleEngine | None = None,
                 phone_face_probe=None):
        self._cam_names = cam_names
        self._seen_pads: set[int] = set()
        # Đếm số lần mỗi (unique_component_id, class_id, label) được classifier
        # gán — dùng để DÒ thứ tự class: sau khi chạy, in ra để biết index 0/1
        # ứng nhãn nào thực tế, tách riêng theo staff/customer vs phone-usage rule.
        self._class_tally: dict[tuple[int, int, str], int] = {}
        # None nếu phone-usage rule tắt (pipeline.py chỉ tạo RuleEngine khi
        # bật) → bỏ qua fusion, không tốn CPU.
        self._rule_engine = rule_engine
        # PhoneFaceProbe (chạy TRƯỚC tracker) stash bbox face vào
        # phone_face_probe.pending_face_boxes thay vì tự vẽ — display_meta
        # gắn trước tracker bị nvtracker xoá mất (đã tự vấp bug này). Vẽ THẬT
        # ở đây (SAU tracker, cùng chỗ đã vẽ 'using_phone' text thành công).
        self._phone_face_probe = phone_face_probe

    def _cam(self, pad_index: int) -> str:
        return self._cam_names.get(pad_index, f"cam{pad_index}")

    def _read_classifiers(self, obj_meta) -> dict[int, tuple[str, int, float]]:
        """Đọc TẤT CẢ kết quả SGIE classifier gắn vào object, theo
        unique_component_id (2=staff/customer, 3=phone-usage rule — xem
        config.py GIE_ID_*).

        Trả về {unique_component_id: (result_label, result_class_id, result_prob)}
        — object nào chưa được SGIE nào đó classify (crop quá nhỏ / SGIE đó
        chưa bật) thì đơn giản là thiếu key tương ứng."""
        out: dict[int, tuple[str, int, float]] = {}
        l_cls = obj_meta.classifier_meta_list
        while l_cls is not None:
            try:
                cls_meta = pyds.NvDsClassifierMeta.cast(l_cls.data)
            except StopIteration:
                break
            comp_id = cls_meta.unique_component_id
            l_lab = cls_meta.label_info_list
            while l_lab is not None:
                try:
                    lab = pyds.NvDsLabelInfo.cast(l_lab.data)
                except StopIteration:
                    break
                key = (comp_id, lab.result_class_id, lab.result_label)
                self._class_tally[key] = self._class_tally.get(key, 0) + 1
                out[comp_id] = (lab.result_label, lab.result_class_id, lab.result_prob)
                break   # 1 label/classifier là đủ (top-1)
            try:
                l_cls = l_cls.next
            except StopIteration:
                break
        return out

    def _draw_boxes(self, batch_meta, frame_meta,
                    boxes: list[tuple[float, float, float, float, bool]],
                    normal_color: tuple[float, float, float, float]) -> None:
        """Vẽ 1 display_meta rect/box, dùng chung cho face lẫn phone (chỉ khác
        màu 'bình thường' — hit (đang dùng điện thoại) LUÔN tô cam để 2 loại
        box cùng nổi bật nhất quán khi có alert)."""
        for x1, y1, x2, y2, is_hit in boxes:
            dm = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            dm.num_rects = 1
            r = dm.rect_params[0]
            r.left, r.top = x1, y1
            r.width, r.height = x2 - x1, y2 - y1
            r.border_width = 2
            if is_hit:
                r.border_color.set(0.9, 0.35, 0.0, 1.0)   # cam — đang dùng phone
            else:
                r.border_color.set(*normal_color)
            r.has_bg_color = 0
            pyds.nvds_add_display_meta_to_frame(frame_meta, dm)

    def __call__(self, pad, info, u_data):
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

        # Phát hiện cam mới lên online
        active: set[int] = set()
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                fm = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break
            active.add(fm.pad_index)
            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        new = active - self._seen_pads
        if new:
            for idx in sorted(new):
                log.info("CAM ONLINE: pad=%d → %s", idx, self._cam(idx))
            self._seen_pads |= new
            log.info("Tổng cam online: %d %s",
                     len(self._seen_pads),
                     [self._cam(i) for i in sorted(self._seen_pads)])

        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            counts: dict[str, int] = {}
            cam = self._cam(frame_meta.pad_index)

            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break
                if obj_meta.class_id in DETECT_CLASSES:
                    label = LABELS.get(obj_meta.class_id, f"cls{obj_meta.class_id}")
                    counts[label] = counts.get(label, 0) + 1

                    # Kết quả TẤT CẢ SGIE/rule classifier đang bật
                    # (staff/customer, phone-usage), tách theo unique_component_id.
                    cls_by_gie = self._read_classifiers(obj_meta)
                    # obj_meta.text_params CHỈ có 1 slot text/object (nvdsosd) —
                    # dùng cho classifier "chính" (staff/customer nếu có, không
                    # thì bất kỳ classifier nào tìm thấy đầu tiên).
                    primary = cls_by_gie.get(2) or next(iter(cls_by_gie.values()), None)
                    rect = obj_meta.rect_params
                    if primary is not None:
                        clabel, ccid, cprob = primary
                        ckey = f"{clabel}"
                        counts[ckey] = counts.get(ckey, 0) + 1
                        txt = obj_meta.text_params
                        txt.display_text = f"{clabel} {cprob:.2f}"
                        # Vị trí ngay trên box (nvosd không tự suy ra từ rect_params
                        # — phải set x_offset/y_offset tường minh mới hiện được).
                        txt.x_offset = max(0, int(rect.left))
                        txt.y_offset = max(0, int(rect.top) - 20)
                        txt.font_params.font_name = "Serif"
                        txt.font_params.font_size = 12
                        txt.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
                        txt.set_bg_clr = 1
                        # staff (nhãn chứa 'staff') tô đỏ, còn lại xanh dương
                        if "staff" in clabel.lower():
                            txt.text_bg_clr.set(0.8, 0.0, 0.0, 0.6)
                        else:
                            txt.text_bg_clr.set(0.0, 0.3, 0.8, 0.6)

                    # using_phone (GIE_ID_SGIE_ACTION): obj_meta.text_params đã
                    # bị "primary" chiếm (thường là staff/customer) khi cả 2
                    # cùng bật → vẽ RIÊNG bằng display_meta (không giới hạn 1
                    # text/object như text_params), cùng cách đã làm cho gaze
                    # target — để using_phone THẬT SỰ hiện trên video thay vì
                    # chỉ có trong log ALERT.
                    action = cls_by_gie.get(GIE_ID_SGIE_ACTION)
                    if action is not None and (primary is None or primary is not action):
                        alabel, acid, aprob = action
                        counts[alabel] = counts.get(alabel, 0) + 1
                        action_dm = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
                        action_dm.num_labels = 1
                        atxt = action_dm.text_params[0]
                        atxt.display_text = f"! {alabel}"
                        atxt.x_offset = max(0, int(rect.left))
                        # dòng thứ 2 — dưới dòng primary (hoặc ngay trên box
                        # nếu không có primary nào để chồng lên).
                        y_line2 = int(rect.top) - 20 + 16 if primary is not None else int(rect.top) - 20
                        atxt.y_offset = max(0, y_line2)
                        atxt.font_params.font_name = "Serif"
                        atxt.font_params.font_size = 12
                        atxt.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
                        atxt.set_bg_clr = 1
                        atxt.text_bg_clr.set(0.9, 0.35, 0.0, 0.75)   # cam — cảnh báo
                        pyds.nvds_add_display_meta_to_frame(frame_meta, action_dm)

                    # Rule engine (chỉ chạy khi phone-usage rule thực sự bật —
                    # pipeline.py mới tạo RuleEngine trong trường hợp đó, xem
                    # build_pipeline()).
                    if self._rule_engine is not None:
                        self._rule_engine.evaluate(
                            track_id=obj_meta.object_id, cam=cam,
                            action_label=action[0] if action else None)
                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            # Vẽ bbox quanh mọi face/phone (stash bởi PhoneFaceProbe TRƯỚC
            # tracker — xem comment ở PhoneFaceProbe.__init__). Cam = đang
            # dùng điện thoại (face gần phone / phone gần face); face thường
            # = xanh dương (tách hẳn khỏi xanh lá của box person/staff to bên
            # ngoài — trước dùng chung xanh lá nên rất dễ lẫn, khó nhận ra),
            # phone thường (VD để trên bàn, không cầm gần mặt) = vàng — để
            # phân biệt 2 loại box. pop() luôn, kể cả khi phone_face_probe tắt
            # (None-safe) hoặc frame này không có box nào.
            if self._phone_face_probe is not None:
                key = (frame_meta.pad_index, frame_meta.frame_num)
                face_boxes = self._phone_face_probe.pending_face_boxes.pop(key, None)
                phone_boxes = self._phone_face_probe.pending_phone_boxes.pop(key, None)
                if face_boxes:
                    self._draw_boxes(batch_meta, frame_meta, face_boxes,
                                     normal_color=(0.1, 0.4, 1.0, 1.0))    # xanh dương
                if phone_boxes:
                    self._draw_boxes(batch_meta, frame_meta, phone_boxes,
                                     normal_color=(1.0, 0.85, 0.0, 1.0))   # vàng

            # Vẽ tên cam + count lên từng frame (trước tiler — sau tiler chỉ còn
            # 1 frame tổng). nvmultistreamtiler transform tọa độ obj_meta của mỗi
            # source sang ô tile khi composite, nên bbox nằm đúng ô; text ở đây
            # dùng offset tuyệt đối trong khung source.
            # Gộp tên cam + count vào MỘT label duy nhất để tránh 2 dòng text
            # chồng lên nhau ở góc trái (sau khi tiler thu nhỏ tile).
            summary = " | ".join(f"{k}:{v}" for k, v in counts.items())
            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            display_meta.num_labels = 1
            txt0 = display_meta.text_params[0]
            txt0.display_text = f"{cam}  {summary}" if summary else cam
            txt0.x_offset = 8
            txt0.y_offset = 8
            txt0.font_params.font_name = "Serif"
            txt0.font_params.font_size = 14
            txt0.font_params.font_color.set(1.0, 1.0, 0.0, 1.0)   # vàng
            txt0.set_bg_clr = 1
            txt0.text_bg_clr.set(0.0, 0.0, 0.0, 0.7)

            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        # Dò thứ tự class: in tally classifier định kỳ. result_class_id là index
        # model trả (0/1), result_label lấy từ labelfile theo index đó. Nếu nhãn
        # thấy trên người mặc đồng phục KHÔNG khớp 'staff' → hoán đổi 2 dòng trong
        # models/staff_customer_labels.txt (đổi 'customer;staff' ↔ 'staff;customer').
        self._tally_ticks = getattr(self, "_tally_ticks", 0) + 1
        if self._class_tally and self._tally_ticks % 60 == 0:
            summary = ", ".join(
                f"gie{gid}/idx{cid}={lab}:{n}"
                for (gid, cid, lab), n in sorted(self._class_tally.items()))
            log.info("CLASSIFIER tally (dò thứ tự class): %s", summary)

        return Gst.PadProbeReturn.OK
