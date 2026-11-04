#!/usr/bin/env bash
# Old nvstreammux (KHÔNG set USE_NEW_NVSTREAMMUX): có property width/height/
# live-source mà YoloTensorParser cần để scale bbox ổn định theo mux size cố định.
# New mux (USE_NEW_NVSTREAMMUX=yes) không có các property này.
unset USE_NEW_NVSTREAMMUX

# Tắt log GStreamer in thẳng ra stderr (vd nvvideoconvert "nvbuf-memory-type ...
# overridden"). Bus ERROR trong code (log.error) vẫn hoạt động độc lập nên lỗi
# nghiêm trọng vẫn thấy. Ghi đè khi cần debug: GST_DEBUG=3 ./run.sh
export GST_DEBUG="${GST_DEBUG:-0}"

# Activate venv (có pyds/torch/torchvision) nếu chưa active. Cho phép ghi đè
# đường dẫn venv qua biến môi trường VENV, mặc định /workspace/venv.
VENV="${VENV:-/workspace/venv}"
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f "$VENV/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "$VENV/bin/activate"
    else
        echo "CẢNH BÁO: không tìm thấy venv tại $VENV — chạy bằng python3 hệ thống." >&2
    fi
fi

exec python3 "$(dirname "$0")/pipeline.py" "$@"
