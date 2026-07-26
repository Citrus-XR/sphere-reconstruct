"""insv.scan_boxes / insv.layout の単体テスト.

実サンプル INSV を持たないため, 合成した最小 MP4 バイナリでテストする.

構成:
- ftyp (16 bytes)
- moov (16 bytes)
- mdat (16 bytes)
- 独自フッタ (32 bytes, ダミー)

scan_boxes が 3 個の box を返し, layout の footer_offset が 48, footer_size が 32
になることを確認する.
"""

from __future__ import annotations

import io
import struct
from pathlib import Path

from sphere_reconstruct.insta360 import insv


def _make_box(box_type: bytes, payload_size: int) -> bytes:
    total = 8 + payload_size
    return struct.pack(">I", total) + box_type + b"\x00" * payload_size


def _make_test_insv() -> bytes:
    ftyp = _make_box(b"ftyp", 8)  # 16 bytes
    moov = _make_box(b"moov", 8)  # 16 bytes
    mdat = _make_box(b"mdat", 8)  # 16 bytes
    footer = b"INSTA360-FOOTER-DUMMY-BYTES-32B!"  # ちょうど 32 bytes
    assert len(footer) == 32
    return ftyp + moov + mdat + footer


def test_scan_boxes_returns_top_level():
    data = _make_test_insv()
    boxes = insv.scan_boxes(io.BytesIO(data), file_size=len(data))
    types = [b.box_type for b in boxes]
    assert types == [b"ftyp", b"moov", b"mdat"]
    assert boxes[0].offset == 0
    assert boxes[1].offset == 16
    assert boxes[2].offset == 32


def test_layout_detects_footer(tmp_path: Path):
    data = _make_test_insv()
    p = tmp_path / "sample.insv"
    p.write_bytes(data)
    layout = insv.layout(p)
    assert layout.file_size == len(data)
    assert len(layout.boxes) == 3
    assert layout.footer_offset == 48
    assert layout.footer_size == 32
    assert insv.has_footer(p) is True


def test_layout_no_footer_when_file_ends_exactly(tmp_path: Path):
    # 独自フッタが無い純粋な MP4 は footer_offset=None になる.
    data = _make_box(b"ftyp", 8) + _make_box(b"mdat", 8)
    p = tmp_path / "pure.mp4"
    p.write_bytes(data)
    layout = insv.layout(p)
    assert layout.footer_offset is None
    assert layout.footer_size is None
