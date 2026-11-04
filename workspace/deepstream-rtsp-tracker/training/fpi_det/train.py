"""Train YOLO11n (nano — nhẹ, khớp quy ước SGIE nhẹ của project) trên FPI-Det
(phone+face, dataset đã fix format ở fix_fpi_det_labels.py).

Dùng: conda activate watchflow && python3 train.py
"""
from ultralytics import YOLO

model = YOLO("yolo11n.pt")   # pretrained COCO, fine-tune sang 2 class phone/face

model.train(
    data="/ssd1/thaokb/deepstream/workspace/deepstream-rtsp-tracker/training/fpi_det/data.yaml",
    epochs=50,
    imgsz=640,
    batch=32,
    device=0,
    patience=15,             # early stop nếu val không cải thiện sau 15 epoch
    project="/ssd1/thaokb/deepstream/workspace/deepstream-rtsp-tracker/training/fpi_det/runs",
    name="phone_face_yolo11n",
    exist_ok=True,
)
