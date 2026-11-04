"""Hằng số toàn cục + đọc/parse config YAML.

Tách từ pipeline.py: LABELS (COCO), ngưỡng detection/NMS, và các hàm đọc
config.yaml → danh sách URI + tên cam."""

from __future__ import annotations

import yaml

# ── COCO labels ──────────────────────────────────────────────────────────────
LABELS = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat",
    35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket",
    39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife",
    44: "spoon", 45: "bowl", 46: "banana", 47: "apple", 48: "sandwich",
    49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
    54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "potted plant",
    59: "bed", 60: "dining table", 61: "toilet", 62: "tv", 63: "laptop",
    64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
    69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
    74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier",
    79: "toothbrush",
}

# ── Ngưỡng detection ─────────────────────────────────────────────────────────
DETECT_CLASSES = {0}    # person only (đếm/vẽ overlay)
PERSON_CLASS = 0        # index class person trong 80 class COCO
CONF_THRESHOLD = 0.15
NMS_IOU = 0.45          # IoU threshold cho NMS thủ công (output thô no-NMS)

# "Action recognition" cho project này = kiểm tra CÓ ĐANG DÙNG ĐIỆN THOẠI hay
# không. BẢN CŨ dùng class 'cell phone' (COCO index 67) từ chính PGIE + luật
# "phone chồng lên toàn thân người" — thô, dễ nhầm khi phone để bàn. BẢN GIỜ
# dùng model tự train riêng (YOLO11n fine-tune trên FPI-Det, xem
# training/fpi_det/) — 2 class 'phone'/'face', ảnh góc camera giám sát thật
# (không phải COCO đời thường) — chạy như PGIE#2 (pgie2_phone_face_config.yml),
# xem PhoneFaceProbe trong phone_face_probe.py. Luật: phone GẦN face (không
# phải chồng lên toàn thân) — chính xác hơn hẳn bản cũ.
PHONE_FACE_CLASS_PHONE = 0    # index 'phone' trong model phone_face_yolo11n
PHONE_FACE_CLASS_FACE = 1     # index 'face'
PHONE_FACE_CONF_THRESHOLD = 0.4    # classifier-threshold khi train (mAP50 phone=0.90 tại threshold mặc định)
# "Gần" = tâm bbox face nằm trong bbox phone đã phóng to theo hệ số này (phone
# thường nhỏ hơn face nhiều — phóng to để bắt được cả 2 khi tay cầm phone kề mặt).
PHONE_FACE_PROXIMITY_EXPAND = 2.5

# ── Runtime ──────────────────────────────────────────────────────────────────
EOS_GRACE_SEC = 3       # chờ EOS lan tới muxer finalize file trước khi thoát

# ── SGIE/rule unique-component-id (khớp gie-unique-id khai báo trong config
# YAML dưới configs/, hoặc id tự gán cho rule không qua SGIE thật). PGIE luôn
# =1. Dùng làm key phân biệt kết quả classifier khi nhiều nguồn cùng gắn vào 1
# object (staff/customer, phone-usage, gaze...).
GIE_ID_PGIE = 1
GIE_ID_SGIE_STAFF_CUSTOMER = 2
# id=3: nhãn "using_phone" CUỐI CÙNG (kết quả áp luật phone-gần-face) — do
# PhoneFaceProbe (phone_face_probe.py, chạy trên PGIE#2) tự gắn, KHÁC với id=4
# (nhãn thô phone/face của chính PGIE#2). Giữ tên GIE_ID_SGIE_ACTION để
# osd_probe.py/rule_engine.py không cần sửa gì thêm.
GIE_ID_SGIE_ACTION = 3
# id=4: PGIE#2 phone+face detector thật (pgie2_phone_face_config.yml) — khác
# GIE_ID_SGIE_ACTION(=3, nhãn "using_phone" cuối cùng do PhoneFaceProbe TỰ gắn
# sau khi áp luật phone-gần-face, không phải nhãn thô của PGIE#2 này).
GIE_ID_PGIE2_PHONE_FACE = 4


def parse_sources(raw: list) -> tuple[list[str], dict[int, str]]:
    """config['sources'] → (danh sách URI đã bật, map pad_index→tên cam).

    Chấp nhận cả string thuần lẫn dict {uri, camera_name, enabled}. Với dict,
    ép RTSP về TCP (protocols=4) để tránh 5s UDP timeout."""
    uris: list[str] = []
    cam_names: dict[int, str] = {}
    for entry in raw:
        if isinstance(entry, str):
            cam_names[len(uris)] = f"cam{len(uris)}"
            uris.append(entry)
        elif isinstance(entry, dict):
            if not entry.get("enabled", True):
                continue
            uri = entry["uri"]
            # Force TCP transport để tránh 5s UDP timeout
            if "?" not in uri:
                uri += "?protocols=4"
            elif "protocols=" not in uri:
                uri += "&protocols=4"
            name = entry.get("camera_name", f"cam{entry.get('source_id', len(uris))}")
            cam_names[len(uris)] = name
            uris.append(uri)
    return uris, cam_names


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
