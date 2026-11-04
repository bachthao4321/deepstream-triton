"""Debug: vẽ graph pipeline GStreamer ra .dot (+ .png nếu có graphviz).

Dùng cơ chế chuẩn của GStreamer: set env GST_DEBUG_DUMP_DOT_DIR TRƯỚC Gst.init(),
sau khi pipeline PLAYING (đã negotiate caps) thì Gst.debug_bin_to_dot_file() dump
file .dot, rồi convert sang PNG bằng lệnh `dot` của graphviz.

Mặc định thu gọn: mỗi source-bin RTSP (nvurisrcbin + rtspsrc/depay/decoder bên
trong) được gộp thành MỘT node duy nhất, để graph chỉ nêu bật các stage chính
của pipeline thay vì xổ hết ruột từng cam."""

from __future__ import annotations

import os
import re
import shutil

from gi.repository import Gst

from .logging_utils import log

# Cluster của mỗi source-bin trong .dot: `subgraph cluster_source_bin_00_0x... {`
_SRCBIN_CLUSTER = re.compile(r"subgraph\s+(cluster_source_bin_(\d+)_0x[0-9a-fA-F]+)\s*\{")
# Khai báo node: `nodeid [ ... ];` (nodeid là token đầu dòng, không phải subgraph/edge)
_NODE_DECL = re.compile(r'^\s*([A-Za-z_][\w]*)\s*\[')
# Cạnh: `a -> b [ ... ]`
_EDGE = re.compile(r'^\s*([A-Za-z_][\w]*)\s*->\s*([A-Za-z_][\w]*)')


def enable_dot_dump(dump_dir: str) -> None:
    """Bật dump .dot của GStreamer. PHẢI gọi trước Gst.init()."""
    os.makedirs(dump_dir, exist_ok=True)
    os.environ["GST_DEBUG_DUMP_DOT_DIR"] = dump_dir
    log.info("Debug graph: GST_DEBUG_DUMP_DOT_DIR=%s", dump_dir)


def _collapse_source_bins(dot_text: str) -> str:
    """Gộp mỗi cluster source-bin trong .dot thành 1 node đại diện.

    - Xóa toàn bộ subgraph cluster_source_bin_XX_0x... (kể cả các node/subgraph
      lồng bên trong: rtspsrc, depay, nvv4l2decoder...).
    - Thay bằng 1 node `srcbin_XX` nhãn "source-bin-XX (RTSP)".
    - Mọi cạnh trỏ tới node bên trong bin → ánh xạ về node đại diện; cạnh nội bộ
      (cả 2 đầu cùng bin) bị loại bỏ."""
    lines = dot_text.splitlines(keepends=True)
    out: list[str] = []
    # node_id (không kèm hex nhánh) → node đại diện. Ta gom mọi node khai báo bên
    # trong span cluster, map về "srcbin_XX".
    node_to_bin: dict[str, str] = {}
    bins_seen: set[str] = set()

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        m = _SRCBIN_CLUSTER.search(line)
        if not m:
            out.append(line)
            i += 1
            continue

        idx = m.group(2)                      # "00", "01", ...
        rep = f"srcbin_{idx}"                  # node đại diện
        bins_seen.add(idx)

        # Tìm span cân bằng ngoặc của cluster này (bắt đầu từ '{' trên dòng m).
        depth = 0
        started = False
        j = i
        while j < n:
            for ch in lines[j]:
                if ch == "{":
                    depth += 1
                    started = True
                elif ch == "}":
                    depth -= 1
            # Gom mọi node khai báo trong span để ánh xạ cạnh về node đại diện.
            decl = _NODE_DECL.match(lines[j])
            if decl and not lines[j].lstrip().startswith("subgraph"):
                node_to_bin[decl.group(1)] = rep
            if started and depth == 0:
                break
            j += 1

        # Thay cả span bằng 1 node đại diện (thụt lề như cluster gốc).
        indent = line[:len(line) - len(line.lstrip())]
        out.append(
            f'{indent}{rep} [shape=box3d, style="filled", fillcolor="#e0f0ff", '
            f'label="source-bin-{idx}\\n(RTSP ×1)"];\n'
        )
        i = j + 1

    if not bins_seen:
        return dot_text   # không có source-bin → giữ nguyên

    # Lượt 2: viết lại các cạnh tham chiếu node đã bị gộp.
    rewritten: list[str] = []
    for line in out:
        e = _EDGE.match(line)
        if not e:
            rewritten.append(line)
            continue
        a, b = e.group(1), e.group(2)
        na, nb = node_to_bin.get(a), node_to_bin.get(b)
        ea, eb = na or a, nb or b
        if na and nb and na == nb:
            continue            # cạnh nội bộ trong cùng bin → bỏ
        # Thay 2 endpoint đầu tiên; giữ nguyên phần label/attr còn lại của dòng.
        new_line = re.sub(r'^(\s*)[A-Za-z_][\w]*(\s*->\s*)[A-Za-z_][\w]*',
                          lambda mm: f"{mm.group(1)}{ea}{mm.group(2)}{eb}", line, count=1)
        rewritten.append(new_line)

    return "".join(rewritten)


def _write_png(dot_path: str, png_path: str) -> bool:
    dot_bin = shutil.which("dot")
    if not dot_bin:
        log.info("Chưa cài graphviz (lệnh `dot`) → chỉ có .dot. "
                 "Cài: apt-get install graphviz, rồi: dot -Tpng %s -o %s",
                 dot_path, png_path)
        return False
    ret = os.system(f"{dot_bin} -Tpng {dot_path} -o {png_path} 2>/dev/null")
    if ret == 0 and os.path.exists(png_path):
        log.info("Debug graph .png → %s (%.1f KB)",
                 png_path, os.path.getsize(png_path) / 1024)
        return True
    log.warning("Convert .dot→.png thất bại (giữ .dot): %s", dot_path)
    return False


def dump_pipeline_graph(pipeline: Gst.Pipeline, dump_dir: str,
                        name: str = "pipeline_graph",
                        collapse_sources: bool = True) -> None:
    """Dump graph pipeline hiện tại ra <dump_dir>/<name>.dot và convert .png.

    Gọi SAU khi pipeline đã PLAYING để graph có đầy đủ pad/caps đã negotiate.
    collapse_sources=True (mặc định): gộp mỗi source-bin RTSP thành 1 node."""
    # debug_bin_to_dot_file ghi vào GST_DEBUG_DUMP_DOT_DIR; GStreamer tự thêm .dot
    Gst.debug_bin_to_dot_file(pipeline, Gst.DebugGraphDetails.ALL, name)
    dot_path = os.path.join(dump_dir, f"{name}.dot")
    if not os.path.exists(dot_path):
        log.warning("Không tạo được %s. GStreamer có thể build không kèm hỗ trợ "
                    "dump .dot (cần bản debug).", dot_path)
        return

    if collapse_sources:
        try:
            with open(dot_path, "r") as f:
                original = f.read()
            collapsed = _collapse_source_bins(original)
            with open(dot_path, "w") as f:
                f.write(collapsed)
            n_bins = len(set(re.findall(r"source-bin-(\d+)", collapsed)))
            log.info("Debug graph .dot → %s (gộp %d source-bin thành node đơn)",
                     dot_path, n_bins)
        except Exception as e:
            log.warning("Thu gọn source-bin thất bại (giữ .dot đầy đủ): %s", e)
    else:
        log.info("Debug graph .dot → %s", dot_path)

    _write_png(dot_path, os.path.join(dump_dir, f"{name}.png"))
