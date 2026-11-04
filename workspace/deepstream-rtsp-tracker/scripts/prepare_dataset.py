#!/usr/bin/env python3
"""Chuẩn bị annotation UCF101 theo format MMAction2 VideoDataset.

Không extract frame ra ảnh (tốn hàng trăm GB) — dùng thẳng .avi gốc qua
decord (MMAction2 VideoDataset đọc video trực tiếp). Chỉ cần sinh 2 file
annotation: "<ClassName>/<video>.avi <label_0based>" mỗi dòng.

UCF101 gốc: classInd.txt đánh số lớp 1-based; trainlist0X.txt có kèm label
(1-based); testlist0X.txt CHỈ có đường dẫn, không có label → suy ra từ tên
thư mục qua classInd.txt.

Dùng:
    python3 prepare_dataset.py [--root /ssd1/thaokb/deepstream/workspace/datasets/ucf101] [--split 1]
"""

from __future__ import annotations

import argparse
import os


def load_class_index(root: str) -> dict[str, int]:
    """classInd.txt: '1 ApplyEyeMakeup' (1-based) → {'ApplyEyeMakeup': 0} (0-based,
    khớp convention label MMAction2)."""
    path = os.path.join(root, "classInd.txt")
    class_to_idx: dict[str, int] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx_str, name = line.split()
            class_to_idx[name] = int(idx_str) - 1
    return class_to_idx


def write_train_list(root: str, split: int, class_to_idx: dict[str, int], out_path: str) -> int:
    src = os.path.join(root, f"trainlist{split:02d}.txt")
    n = 0
    with open(src) as f, open(out_path, "w") as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # "ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.avi 1"
            rel_path, _label = line.split()
            cls_name = rel_path.split("/")[0]
            label = class_to_idx[cls_name]
            out.write(f"{rel_path} {label}\n")
            n += 1
    return n


def write_val_list(root: str, split: int, class_to_idx: dict[str, int], out_path: str) -> int:
    src = os.path.join(root, f"testlist{split:02d}.txt")
    n = 0
    with open(src) as f, open(out_path, "w") as out:
        for line in f:
            rel_path = line.strip()
            if not rel_path:
                continue
            cls_name = rel_path.split("/")[0]
            label = class_to_idx[cls_name]
            out.write(f"{rel_path} {label}\n")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/ssd1/thaokb/deepstream/workspace/datasets/ucf101")
    ap.add_argument("--split", type=int, default=1, choices=[1, 2, 3])
    args = ap.parse_args()

    videos_dir = os.path.join(args.root, "videos")
    if not os.path.isdir(videos_dir):
        raise SystemExit(
            f"Không thấy {videos_dir} — chạy scripts/download_ucf101.sh trước.")

    class_to_idx = load_class_index(args.root)
    print(f"{len(class_to_idx)} class")

    train_out = os.path.join(args.root, "train_list.txt")
    val_out = os.path.join(args.root, "val_list.txt")
    n_train = write_train_list(args.root, args.split, class_to_idx, train_out)
    n_val = write_val_list(args.root, args.split, class_to_idx, val_out)

    labelmap_out = os.path.join(args.root, "label_map.txt")
    with open(labelmap_out, "w") as f:
        for name, idx in sorted(class_to_idx.items(), key=lambda kv: kv[1]):
            f.write(f"{name}\n")

    print(f"train: {n_train} video → {train_out}")
    print(f"val:   {n_val} video → {val_out}")
    print(f"label map (0-based, theo thứ tự): {labelmap_out}")
    print(f"\nAnnotation dùng path TƯƠNG ĐỐI so với {videos_dir}/ "
          f"(data_prefix trong config training trỏ vào đây).")


if __name__ == "__main__":
    main()
