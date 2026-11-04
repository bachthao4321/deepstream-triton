"""Helper nhỏ để dựng pipeline GStreamer gọn hơn: tạo element, set property
có điều kiện, và link chuỗi element."""

from __future__ import annotations

from gi.repository import Gst


def make(factory: str, name: str) -> Gst.Element:
    """Tạo element, raise nếu thiếu plugin (thay vì trả None âm thầm)."""
    el = Gst.ElementFactory.make(factory, name)
    if not el:
        raise RuntimeError(f"Không tạo được element '{factory}' ({name}). "
                           f"Kiểm tra DeepStream plugin đã cài chưa.")
    return el


def set_if_has(element: Gst.Element, prop: str, value) -> bool:
    """Set property chỉ khi element có property đó. Trả True nếu đã set.

    Gộp pattern lặp `if el.find_property(x) is not None: el.set_property(x, v)`
    xuất hiện nhiều lần khi cấu hình streammux/tracker/tiler/conv/encoder."""
    if element.find_property(prop) is not None:
        element.set_property(prop, value)
        return True
    return False


def link_chain(elements: list[Gst.Element]) -> None:
    """Link tuần tự a→b→c... Raise ngay tại cặp đầu tiên link thất bại."""
    for a, b in zip(elements, elements[1:]):
        if not a.link(b):
            raise RuntimeError(f"link {a.name}→{b.name} thất bại")
