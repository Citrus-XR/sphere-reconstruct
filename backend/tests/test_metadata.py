"""metadata.read_footer の単体テスト.

実 X5 INSV を用意せずとも, 合成した最小 inst box + トレイラで構造チェックできる.
"""

from __future__ import annotations

import struct
from pathlib import Path

from sphere_reconstruct.insta360 import insv, metadata


def _make_box(box_type: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", 8 + len(payload)) + box_type + payload


def _make_insv(inst_payload_size: int = 32, version: int = 3) -> bytes:
    ftyp = _make_box(b"ftyp", b"\x00" * 8)  # 16 bytes
    mdat = _make_box(b"mdat", b"\x00" * 8)  # 16 bytes
    moov = _make_box(b"moov", b"\x00" * 8)  # 16 bytes
    inst_data = bytes(range(inst_payload_size))
    inst = _make_box(b"inst", inst_data)  # 8 + N bytes

    # trailer: (protobuf/padding 適当) + 8 バイトヘッダ + 32 バイト署名
    trailer_body = b"\x00" * 24  # 適当な padding
    trailer_header = struct.pack("<II", len(inst_data), version)  # inst_data_size, version
    trailer = trailer_body + trailer_header + metadata.FOOTER_SIGNATURE

    return ftyp + mdat + moov + inst + trailer


def test_read_footer_full_structure(tmp_path: Path):
    data = _make_insv(inst_payload_size=64, version=3)
    p = tmp_path / "sample.insv"
    p.write_bytes(data)

    # まず insv.layout が footer を検出できるか.
    lay = insv.layout(p)
    assert lay.footer_offset is not None
    assert lay.footer_offset == 16 + 16 + 16  # ftyp+mdat+moov = 48

    footer = metadata.read_footer(p, lay.footer_offset)
    assert footer.file_size == len(data)
    assert footer.inst_box_offset == lay.footer_offset
    assert footer.inst_box_total_size == 8 + 64
    assert footer.inst_box_data_offset == lay.footer_offset + 8
    assert footer.inst_box_data_size == 64
    assert footer.trailer_offset == lay.footer_offset + 8 + 64
    # trailer は 適当 padding 24 + header 8 + sig 32 = 64 bytes
    assert footer.trailer_size == 24 + 8 + 32
    assert footer.reported_inst_data_size == 64
    assert footer.version == 3
    assert footer.signature_valid is True
    assert footer.signature_offset == len(data) - 32


def test_read_inst_box_bytes_roundtrip(tmp_path: Path):
    data = _make_insv(inst_payload_size=64)
    p = tmp_path / "sample.insv"
    p.write_bytes(data)
    lay = insv.layout(p)
    assert lay.footer_offset is not None
    footer = metadata.read_footer(p, lay.footer_offset)
    inst_bytes = metadata.read_inst_box_bytes(footer)
    assert len(inst_bytes) == 64
    assert inst_bytes == bytes(range(64))


def test_footer_not_found_on_pure_mp4(tmp_path: Path):
    ftyp = _make_box(b"ftyp", b"\x00" * 8)
    mdat = _make_box(b"mdat", b"\x00" * 8)
    p = tmp_path / "pure.mp4"
    p.write_bytes(ftyp + mdat)
    lay = insv.layout(p)
    assert lay.footer_offset is None


def _varint(value: int) -> bytes:
    encoded = bytearray()
    while value >= 0x80:
        encoded.append((value & 0x7F) | 0x80)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


def _protobuf_varint(field: int, value: int) -> bytes:
    return _varint(field << 3) + _varint(value)


def _protobuf_bytes(field: int, value: bytes) -> bytes:
    return _varint((field << 3) | 2) + _varint(len(value)) + value


def test_parse_extra_metadata_fields_needed_for_imu_timestamps():
    gyro_config = _protobuf_varint(1, 32) + _protobuf_varint(2, 2000)
    payload = b"".join(
        [
            _protobuf_bytes(2, b"Insta360 X5"),
            _protobuf_varint(24, 158_412_034_768),
            _varint((25 << 3) | 1) + struct.pack("<d", 21.244001388549805),
            _varint((28 << 3) | 1) + struct.pack("<d", 1.6),
            _protobuf_varint(29, 1),
            _protobuf_varint(62, 1),
            _protobuf_bytes(65, gyro_config),
        ]
    )
    parsed = metadata.parse_extra_metadata(payload)
    assert parsed.camera_type == "Insta360 X5"
    assert parsed.first_frame_timestamp == 158_412_034_768
    assert parsed.rolling_shutter_time_ms == 21.244001388549805
    assert parsed.gyro_timestamp == 1.6
    assert parsed.has_gyro_timestamp is True
    assert parsed.is_raw_gyro is True
    assert parsed.acc_range == 32
    assert parsed.gyro_range == 2000
