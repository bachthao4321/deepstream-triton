#!/usr/bin/env bash
# Tải UCF101 (video gốc + split file chuẩn) trực tiếp từ CRCV UCF — không phụ
# thuộc YouTube, không cần đăng nhập. Theo CLAUDE.md mục 5 (dataset ưu tiên #1).
#
# Dùng: ./download_ucf101.sh [thư mục đích]
# Mặc định thư mục đích: /ssd1/thaokb/deepstream/workspace/datasets/ucf101
#
# LƯU Ý: server crcv.ucf.edu trả cert TLS không đầy đủ chain (thiếu intermediate
# CA) → curl mặc định từ chối bằng lỗi "unable to get local issuer certificate".
# Dùng -k (bỏ verify) — chấp nhận được vì đây là tải file public, không phải
# trao đổi thông tin nhạy cảm. (Tự kiểm chứng: curl -skI <url> để xem cert.)
set -euo pipefail

DEST="${1:-/ssd1/thaokb/deepstream/workspace/datasets/ucf101}"
mkdir -p "$DEST"
cd "$DEST"

BASE="https://www.crcv.ucf.edu/data/UCF101"
RAR="UCF101.rar"                                    # ~6.9 GB, 13320 video .avi
SPLITS="UCF101TrainTestSplits-RecognitionTask.zip"   # split 1/2/3 chuẩn

download_resumable() {
    # Server crcv.ucf.edu đóng kết nối giữa chừng (~mỗi 1GB) với file lớn —
    # --retry của curl không đủ (hết lượt trước khi tải xong 6.9GB). Vòng lặp
    # NGOÀI: cứ đứt là gọi lại curl -C - (resume theo byte), cho tới khi kích
    # thước local KHỚP Content-Length remote.
    local url="$1" out="$2"
    local remote_size local_size attempt=0
    remote_size=$(curl -ksI --max-time 20 "$url" | grep -i '^content-length:' | tail -1 | tr -d '\r' | awk '{print $2}')
    while true; do
        attempt=$((attempt + 1))
        local_size=$(stat -c%s "$out" 2>/dev/null || echo 0)
        if [ -n "$remote_size" ] && [ "$local_size" = "$remote_size" ]; then
            echo "  $out đã đủ $local_size byte, xong."
            break
        fi
        echo "  [lần $attempt] resume $out từ byte $local_size / ${remote_size:-?} ..."
        curl -k -L --retry 5 --retry-delay 5 --retry-all-errors -C - -o "$out" "$url" || true
        local_size=$(stat -c%s "$out" 2>/dev/null || echo 0)
        if [ -n "$remote_size" ] && [ "$local_size" = "$remote_size" ]; then
            echo "  $out đã đủ $local_size byte, xong."
            break
        fi
        if [ "$attempt" -ge 200 ]; then
            echo "LỖI: quá 200 lần thử vẫn chưa tải xong $out, dừng."
            return 1
        fi
        sleep 3
    done
}

echo "== Tải $RAR (~6.9GB, tự resume nếu dở dang) =="
download_resumable "$BASE/$RAR" "$RAR"

echo "== Tải $SPLITS =="
download_resumable "$BASE/$SPLITS" "$SPLITS"

echo "== Giải nén split (zip, không cần unrar) =="
unzip -o -q "$SPLITS" -d ucfTrainTestlist_raw
# Zip giải ra thư mục con ucfTrainTestlist/ chứa trainlist01.txt, classInd.txt...
find ucfTrainTestlist_raw -maxdepth 2 -type f -name "*.txt" -exec cp {} . \;
rm -rf ucfTrainTestlist_raw

echo "== Giải nén video (.rar — cần unrar hoặc 7z) =="
if command -v unrar >/dev/null 2>&1; then
    unrar x -o+ "$RAR" videos_raw/ >/dev/null
elif command -v 7z >/dev/null 2>&1; then
    7z x -y "$RAR" -ovideos_raw >/dev/null
else
    echo "LỖI: cần 'unrar' hoặc '7z' để giải nén $RAR."
    echo "  Ubuntu/Debian: sudo apt-get install unrar   (hoặc p7zip-full)"
    exit 1
fi
# Rar giải ra thư mục con UCF-101/<ClassName>/*.avi
if [ -d videos_raw/UCF-101 ]; then
    mv videos_raw/UCF-101 videos
    rmdir videos_raw
fi

n_classes=$(find videos -mindepth 1 -maxdepth 1 -type d | wc -l)
n_videos=$(find videos -name "*.avi" | wc -l)
echo "== Xong: $DEST =="
echo "  $n_classes class, $n_videos video .avi"
echo "  Split file: $(ls *.txt 2>/dev/null | tr '\n' ' ')"
echo ""
echo "Bước tiếp theo: chạy scripts/prepare_dataset.py để tách frame/annotation"
echo "  theo format MMAction2 (dùng tools/data/ucf101/ chuẩn của MMAction2)."
