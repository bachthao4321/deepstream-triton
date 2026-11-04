#!/usr/bin/env python3
"""Fix label FPI-Det (phone+face, YOLO format) trước khi train.

2 lỗi tìm thấy khi rà soát (~1.06% dòng, không phải lỗi từ phía mình — bug
trong script export gốc của dataset):
  1. Dòng dính 2 box làm 1 (thiếu ký tự xuống dòng): "0 x y w h 0 x y w h"
     (10 số thay vì 5) → cắt thành từng cụm 5 số, mỗi cụm 1 box riêng.
  2. Toạ độ âm/ngoài [0,1] hoặc w/h<=0 (2 dòng trong ~39.5k) → bỏ dòng đó.

KHÔNG sửa trực tiếp trong reorganized_dataset/ (giữ nguyên bản gốc để đối
chiếu) — ghi ra thư mục mới cùng cấp, symlink ảnh (không copy, tiết kiệm
dung lượng — ảnh không đổi) + label đã fix.

Dùng:
    python3 fix_fpi_det_labels.py
"""
from __future__ import annotations

import os

SRC_ROOT = "/ssd1/thaokb/deepstream/workspace/datasets/fpi_det/reorganized_dataset"
DST_ROOT = "/ssd1/thaokb/deepstream/workspace/datasets/fpi_det_clean"
SPLITS = ["train", "val", "test"]
VALID_CLASSES = {0, 1}   # 0=phone, 1=face (theo Phone.yaml gốc)


def fix_label_lines(lines: list[str]) -> tuple[list[str], int, int]:
    """Trả về (list dòng đã fix, số box tách được từ dòng dính, số dòng bỏ do lỗi)."""
    fixed: list[str] = []
    n_split = 0
    n_dropped = 0
    for raw in lines:
        tokens = raw.split()
        if not tokens:
            continue
        if len(tokens) % 5 != 0:
            # Không chia hết cho 5 → không cứu được, bỏ nguyên dòng.
            n_dropped += 1
            continue
        n_boxes_in_line = len(tokens) // 5
        if n_boxes_in_line > 1:
            n_split += n_boxes_in_line - 1
        for i in range(n_boxes_in_line):
            chunk = tokens[i * 5:(i + 1) * 5]
            try:
                cls = int(chunk[0])
                x, y, w, h = (float(v) for v in chunk[1:])
            except ValueError:
                n_dropped += 1
                continue
            if cls not in VALID_CLASSES:
                n_dropped += 1
                continue
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                n_dropped += 1
                continue
            if w <= 0 or h <= 0 or w > 1.0 or h > 1.0:
                n_dropped += 1
                continue
            fixed.append(f"{cls} {x} {y} {w} {h}")
    return fixed, n_split, n_dropped


def main():
    total_split = 0
    total_dropped = 0
    total_files = 0
    total_boxes = 0

    for split in SPLITS:
        src_img_dir = os.path.join(SRC_ROOT, split, "images")
        src_lbl_dir = os.path.join(SRC_ROOT, split, "labels")
        dst_img_dir = os.path.join(DST_ROOT, split, "images")
        dst_lbl_dir = os.path.join(DST_ROOT, split, "labels")
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)

        # Symlink toàn bộ ảnh (không đổi, không cần copy → tiết kiệm ~1.2GB nữa).
        for fname in os.listdir(src_img_dir):
            src = os.path.join(src_img_dir, fname)
            dst = os.path.join(dst_img_dir, fname)
            if not os.path.exists(dst):
                os.symlink(src, dst)

        for fname in os.listdir(src_lbl_dir):
            src_path = os.path.join(src_lbl_dir, fname)
            with open(src_path, "r", encoding="utf-8", errors="replace") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            fixed, n_split, n_dropped = fix_label_lines(lines)
            with open(os.path.join(dst_lbl_dir, fname), "w") as f:
                for ln in fixed:
                    f.write(ln + "\n")
            total_split += n_split
            total_dropped += n_dropped
            total_files += 1
            total_boxes += len(fixed)

        print(f"[{split}] {len(os.listdir(src_lbl_dir))} label file → "
              f"{dst_lbl_dir}")

    print(f"\nTổng: {total_files} label file, {total_boxes} box hợp lệ sau khi fix")
    print(f"  Tách được {total_split} box từ các dòng bị dính")
    print(f"  Bỏ {total_dropped} dòng lỗi (toạ độ âm/ngoài [0,1]/class lạ)")
    print(f"\nDataset đã fix: {DST_ROOT}")


if __name__ == "__main__":
    main()
