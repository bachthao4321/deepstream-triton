"""Source bin: bọc mỗi URI RTSP vào Gst.Bin(nvurisrcbin) + ghost pad src.

Theo pattern deepstream-test3: nvurisrcbin cấu hình TCP transport + reconnect
vô hạn, giữ mọi frame (không drop-on-latency) để detection liên tục."""

from __future__ import annotations

from gi.repository import Gst

from .logging_utils import log


def _cb_newpad(decodebin, decoder_src_pad, source_bin):
    """Khi decoder tạo video pad (memory:NVMM) → set target cho ghost pad."""
    caps = decoder_src_pad.get_current_caps() or decoder_src_pad.query_caps(None)
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    if gstname.find("video") == -1:
        return
    features = caps.get_features(0)
    if not features.contains("memory:NVMM"):
        log.warning("decoder src pad không phải NVMM — cần nvv4l2decoder trong nvurisrcbin")
        return
    bin_ghost_pad = source_bin.get_static_pad("src")
    if not bin_ghost_pad.set_target(decoder_src_pad):
        log.error("Không set được target cho ghost pad của source bin")


def _decodebin_child_added(child_proxy, obj, name, user_data):
    """Cấu hình reconnect/latency sâu bên trong nvurisrcbin (rtspsrc/decoder)."""
    if name.find("decodebin") != -1:
        obj.connect("child-added", _decodebin_child_added, user_data)
    if "source" in name and obj.find_property("drop-on-latency") is not None:
        # KHÔNG drop-on-latency: cam đông người (bitrate cao) trước đây bị vứt
        # frame inter khi đến trễ → detection ngắt quãng. Giữ mọi frame.
        obj.set_property("drop-on-latency", False)
    if "source" in name and obj.find_property("latency") is not None:
        obj.set_property("latency", 2000)


def create_source_bin(index: int, uri: str) -> Gst.Bin:
    """Bọc một URI RTSP vào Gst.Bin với nvurisrcbin + ghost pad src."""
    bin_name = f"source-bin-{index:02d}"
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        raise RuntimeError(f"Không tạo được {bin_name}")

    uri_decode_bin = Gst.ElementFactory.make("nvurisrcbin", f"uri-decode-bin-{index}")
    if not uri_decode_bin:
        raise RuntimeError("Không tạo được nvurisrcbin")

    uri_decode_bin.set_property("uri", uri)
    # latency cao + KHÔNG drop-on-latency (giữ frame → detection liên tục)
    uri_decode_bin.set_property("latency", 2000)
    if uri_decode_bin.find_property("drop-on-latency") is not None:
        uri_decode_bin.set_property("drop-on-latency", False)
    # TCP transport tránh 5s UDP timeout; reconnect vô hạn với nguồn live
    if uri_decode_bin.find_property("select-rtp-protocol") is not None:
        uri_decode_bin.set_property("select-rtp-protocol", 4)   # TCP
    if uri_decode_bin.find_property("rtsp-reconnect-interval") is not None:
        uri_decode_bin.set_property("rtsp-reconnect-interval", 2)
    if uri_decode_bin.find_property("rtsp-reconnect-attempts") is not None:
        uri_decode_bin.set_property("rtsp-reconnect-attempts", -1)

    uri_decode_bin.connect("pad-added", _cb_newpad, nbin)
    uri_decode_bin.connect("child-added", _decodebin_child_added, nbin)

    Gst.Bin.add(nbin, uri_decode_bin)
    # ghost pad chưa có target — sẽ set trong _cb_newpad khi decoder ra pad
    ghost = Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC)
    if not ghost:
        raise RuntimeError("Không tạo được ghost pad cho source bin")
    nbin.add_pad(ghost)
    return nbin
