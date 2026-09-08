"""INSV の ``inst`` box、record directory、timing / crop metadata を解析する。"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

# 末尾トレイラの ASCII シグネチャ (32 バイト). バイナリ UUID ではない.
FOOTER_SIGNATURE = b"8db42d694ccc418790edff439fe026bf"
assert len(FOOTER_SIGNATURE) == 32

# トレイラヘッダ (シグネチャ直前 8 バイト) のフィールド構造.
_TRAILER_HEADER_STRUCT = struct.Struct("<II")  # inst_box_data_size, version
TRAILER_HEADER_SIZE = _TRAILER_HEADER_STRUCT.size  # 8
SIGNATURE_SIZE = len(FOOTER_SIGNATURE)  # 32

# インストールボックスヘッダ ("inst" プレフィックス) のサイズ (MP4 box 通常形式).
_INST_HEADER_STRUCT = struct.Struct(">I4s")  # box_size BE u32, box_type 4s
_INST_BOX_TYPE = b"inst"
INST_HEADER_SIZE = _INST_HEADER_STRUCT.size  # 8


class FooterNotFoundError(Exception):
    """Insta360 独自フッタが検出できなかった. 通常の MP4 かフォーマット違い."""


@dataclass(frozen=True)
class WindowCropInfo:
    source_width: int
    source_height: int
    cropped_width: int
    cropped_height: int

    def __post_init__(self) -> None:
        if min(self.source_width, self.source_height, self.cropped_width, self.cropped_height) <= 0:
            raise ValueError("window crop の size は正数でなければなりません")
        if self.cropped_width > self.source_width or self.cropped_height > self.source_height:
            raise ValueError("window crop の出力は入力 size を超えられません")


@dataclass(frozen=True)
class ExtraMetadata:
    camera_type: str
    first_frame_timestamp: int
    rolling_shutter_time_ms: float
    gyro_timestamp: float
    has_gyro_timestamp: bool
    is_raw_gyro: bool
    acc_range: int | None
    gyro_range: int | None
    window_crop: WindowCropInfo | None


@dataclass
class InsvFooter:
    """検出できたフッタ構造の要約. record 内部は保持しない (必要時に raw 参照)."""

    file_size: int

    # inst box (Insta360 メタデータボックス).
    inst_box_offset: int  # ボックスヘッダの開始 offset
    inst_box_total_size: int  # ヘッダ含めた全長 (BE u32)
    inst_box_data_offset: int  # ヘッダ 8 バイト後
    inst_box_data_size: int  # data 部の bytes (== box_total_size - 8)

    # 末尾トレイラ (512 バイトを実サンプルで観測).
    trailer_offset: int  # inst box の直後
    trailer_size: int  # file_size - trailer_offset

    # トレイラヘッダ (シグネチャ直前 8 バイト).
    trailer_header_offset: int
    reported_inst_data_size: int  # トレイラヘッダの u32 (inst_box_data_size と一致すべき)
    version: int

    # ASCII シグネチャ.
    signature_offset: int
    signature_valid: bool

    # 生バイトへのアクセスを促す open() 用フィールド.
    path: Path = field(default_factory=lambda: Path())


def read_footer(path: Path, footer_offset: int) -> InsvFooter:
    """`insv.layout` が返した footer_offset を渡す. 構造チェックを行い要約を返す.

    - inst box ヘッダ (BE u32 size + "inst") を確認.
    - 末尾トレイラを 512 バイト or 十分な長さ読み, シグネチャを探す.
    - トレイラヘッダの申告サイズと inst box data サイズが一致するか確認.
    """
    file_size = path.stat().st_size
    if footer_offset >= file_size:
        raise FooterNotFoundError("footer_offset is at or past EOF")

    with path.open("rb") as f:
        # 1) inst box ヘッダ.
        f.seek(footer_offset)
        header = f.read(INST_HEADER_SIZE)
        if len(header) < INST_HEADER_SIZE:
            raise FooterNotFoundError("cannot read inst box header")
        box_size, box_type = _INST_HEADER_STRUCT.unpack(header)
        if box_type != _INST_BOX_TYPE:
            raise FooterNotFoundError(f"expected inst box, got {box_type!r} at offset {footer_offset}")
        if box_size < INST_HEADER_SIZE:
            raise FooterNotFoundError(f"nonsensical inst box size {box_size}")

        inst_box_data_offset = footer_offset + INST_HEADER_SIZE
        inst_box_data_size = box_size - INST_HEADER_SIZE
        trailer_offset = footer_offset + box_size
        if trailer_offset > file_size:
            raise FooterNotFoundError(f"inst box overruns EOF (ends at {trailer_offset}, size={file_size})")
        trailer_size = file_size - trailer_offset

        # 2) 末尾シグネチャ. ファイル末尾に必ず 32 バイトあると仮定して直接読む.
        if file_size < SIGNATURE_SIZE + TRAILER_HEADER_SIZE:
            raise FooterNotFoundError("file too small to contain trailer")
        signature_offset = file_size - SIGNATURE_SIZE
        f.seek(signature_offset)
        sig_bytes = f.read(SIGNATURE_SIZE)
        signature_valid = sig_bytes == FOOTER_SIGNATURE

        # 3) トレイラヘッダ.
        trailer_header_offset = signature_offset - TRAILER_HEADER_SIZE
        f.seek(trailer_header_offset)
        reported_size, version = _TRAILER_HEADER_STRUCT.unpack(f.read(TRAILER_HEADER_SIZE))

    return InsvFooter(
        file_size=file_size,
        inst_box_offset=footer_offset,
        inst_box_total_size=box_size,
        inst_box_data_offset=inst_box_data_offset,
        inst_box_data_size=inst_box_data_size,
        trailer_offset=trailer_offset,
        trailer_size=trailer_size,
        trailer_header_offset=trailer_header_offset,
        reported_inst_data_size=reported_size,
        version=version,
        signature_offset=signature_offset,
        signature_valid=signature_valid,
        path=path,
    )


def read_inst_box_bytes(footer: InsvFooter) -> bytes:
    """inst box の data 部分をまるごと読む.

    現状はこの生バイト列を上位モジュール (imu.py, calibration.py) が
    パースするための入り口. 11.4 MB 程度あるので必要な時だけ呼ぶこと.
    """
    with footer.path.open("rb") as f:
        f.seek(footer.inst_box_data_offset)
        return f.read(footer.inst_box_data_size)


def read_trailer_bytes(footer: InsvFooter) -> bytes:
    """トレイラ 512 バイト (inst box 直後 〜 EOF) をまるごと読む.

    ここには record directory と固定 trailer が含まれる.
    """
    with footer.path.open("rb") as f:
        f.seek(footer.trailer_offset)
        return f.read(footer.trailer_size)


# トレイラ末尾の固定領域: padding(32) + size(4) + version(4) + magic(32) = 72 バイト.
# record ディレクトリはこの手前に詰まる. 参考: AdrianEddy/telemetry-parser insta360.
_RECORDS_TRAILER = 72
_DESC_SIZE = 6  # 各 record 末尾の [format:u8][id:u8][size:u32 LE].


def iter_trailer_records(footer: InsvFooter):
    """inst box 末尾の record 群を (id, format, data) で列挙する.

    レイアウト (telemetry-parser 準拠): ファイル末尾から
      [72B トレイラ][desc0][data0][desc1][data1]... (後方に詰まる)
    各 record は物理的に `[data][format:u8][id:u8][size:u32 LE]`. 末尾側先頭に
    Offsets(id=0) 索引があればそれで各 record の (offset,size) を引き, 無ければ線形に
    後方走査する. record データ領域は [file_size - extra_size, file_size].
    """
    extra_size = footer.reported_inst_data_size
    file_size = footer.file_size
    extra_start = file_size - extra_size
    if extra_start < 0 or extra_size <= _RECORDS_TRAILER:
        return
    with footer.path.open("rb") as f:
        f.seek(extra_start)
        tail = f.read(extra_size)
    n = len(tail)

    def _desc(offset_from_end: int):
        pos = n - offset_from_end
        if pos < 0 or pos + _DESC_SIZE > n:
            return None
        fmt = tail[pos]
        rid = tail[pos + 1]
        size = int.from_bytes(tail[pos + 2 : pos + 6], "little")
        return pos, fmt, rid, size

    # 1) Offsets 索引モード: 末尾側先頭 record が id==0 なら (id -> offset,size) を引く.
    first = _desc(_RECORDS_TRAILER + _DESC_SIZE)
    if first is not None and first[2] == 0:
        pos, _fmt, _rid, size = first
        data_start = pos - size
        if data_start >= 0:
            index_data = tail[data_start:pos]
            entries: dict[int, tuple[int, int, int]] = {}
            off = 0
            while off + 10 <= len(index_data):
                eid = index_data[off]
                efmt = index_data[off + 1]
                esize = int.from_bytes(index_data[off + 2 : off + 6], "little")
                eoff = int.from_bytes(index_data[off + 6 : off + 10], "little")
                off += 10
                if eid > 0:
                    entries[eid] = (eoff, esize, efmt)
            if entries:
                for rid, (eoff, esize, efmt) in entries.items():
                    if eoff >= 0 and eoff + esize <= n:
                        yield rid, efmt, tail[eoff : eoff + esize]
                return

    # 2) 線形後方走査 (索引 record が無い / 壊れている場合).
    offset = _RECORDS_TRAILER + _DESC_SIZE
    guard = 0
    while offset < extra_size and guard < 1_000_000:
        guard += 1
        d = _desc(offset)
        if d is None:
            break
        pos, fmt, rid, size = d
        data_start = pos - size
        if data_start < 0:
            break
        yield rid, fmt, tail[data_start:pos]
        if size == 0:
            break
        offset += size + _DESC_SIZE


def read_extra_metadata(footer: InsvFooter) -> ExtraMetadata | None:
    """Record id=1 から IMU timing、camera type、window crop を読む."""
    for record_id, record_format, data in iter_trailer_records(footer):
        if record_id == 1:
            if record_format != 1:
                raise ValueError(f"metadata record format が不正です: {record_format}")
            return parse_extra_metadata(data)
    return None


def parse_extra_metadata(data: bytes) -> ExtraMetadata:
    fields = {field: value for field, _wire, value in _iter_metadata_fields(data)}
    camera_type = bytes(fields.get(2, b"")).decode("utf-8", "replace")
    gyro_config = fields.get(65)
    acc_range = gyro_range = None
    if isinstance(gyro_config, bytes):
        config_fields = {field: value for field, _wire, value in _iter_metadata_fields(gyro_config)}
        acc_range = int(config_fields[1]) if 1 in config_fields else None
        gyro_range = int(config_fields[2]) if 2 in config_fields else None
    window_crop = fields.get(27)
    parsed_window_crop = None
    if isinstance(window_crop, bytes):
        crop_fields = {field: value for field, _wire, value in _iter_metadata_fields(window_crop)}
        missing = sorted({1, 2, 3, 4} - crop_fields.keys())
        if missing:
            raise ValueError(f"window crop metadata の field が不足しています: {missing}")
        parsed_window_crop = WindowCropInfo(
            source_width=int(crop_fields[1]),
            source_height=int(crop_fields[2]),
            cropped_width=int(crop_fields[3]),
            cropped_height=int(crop_fields[4]),
        )
    return ExtraMetadata(
        camera_type=camera_type,
        first_frame_timestamp=int(fields.get(24, 0)),
        rolling_shutter_time_ms=float(fields.get(25, 0.0)),
        gyro_timestamp=float(fields.get(28, 0.0)),
        has_gyro_timestamp=bool(fields.get(29, 0)),
        is_raw_gyro=bool(fields.get(62, 0)),
        acc_range=acc_range,
        gyro_range=gyro_range,
        window_crop=parsed_window_crop,
    )


def _iter_metadata_fields(data: bytes):
    offset = 0
    while offset < len(data):
        key, offset = _read_varint(data, offset)
        field = key >> 3
        wire = key & 0x07
        if wire == 0:
            value, offset = _read_varint(data, offset)
        elif wire == 1:
            if offset + 8 > len(data):
                raise ValueError("metadata fixed64 が途中で終了しました")
            value = struct.unpack_from("<d", data, offset)[0]
            offset += 8
        elif wire == 2:
            length, offset = _read_varint(data, offset)
            end = offset + length
            if end > len(data):
                raise ValueError("metadata bytes field が途中で終了しました")
            value = data[offset:end]
            offset = end
        elif wire == 5:
            if offset + 4 > len(data):
                raise ValueError("metadata fixed32 が途中で終了しました")
            value = struct.unpack_from("<f", data, offset)[0]
            offset += 4
        else:
            raise ValueError(f"未対応の metadata wire type です: {wire}")
        yield field, wire, value


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while offset < len(data) and shift < 70:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte & 0x80 == 0:
            return value, offset
        shift += 7
    raise ValueError("metadata varint が不正です")
