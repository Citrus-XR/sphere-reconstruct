"""Insta360 独自フッタの構造解析.

実 X5 INSV (VID_20260724_021825_00_001.insv, ~916 MB) を dump した結果:

- MP4 本体の直後に **"inst" box** が置かれる. これは通常の ISO BMFF box フォーマットで,
  先頭 4 バイトが BE u32 の box size, 次の 4 バイトが type ("inst"). box 全体で
  約 11.4 MB, EOF まで続く (トレイラは inst box 内部).
- inst box の **末尾 40 バイト** が固定レイアウトのフッタトレイラ:
    - 8 バイト: u32 LE `inst_box_data_size` + u32 LE `version` (X5 サンプルでは 3)
    - 32 バイト: ASCII 16 進シグネチャ (`8db42d694ccc418790edff439fe026bf`)
- そのすぐ手前には protobuf エンコードされた小さなメタデータ (mask ラベル名など
  "invisibleDive", "heat_bare", "heat_protector" が観測される) と 0 パディングが並ぶ.

例:
    footer_offset       = 904941928     (mdat + moov の直後)
    file_size           = 916377948
    inst_box_header     = 904941928..904941936     (8 bytes: size + "inst")
    inst_box_data       = 904941936..916377948     (11 436 012 bytes; EOF まで)
    trailer 内部        = 916377908..916377948     (末尾 40 バイト = 8B ヘッダ + 32B シグ)

各サブレコード (0x0300 IMU / offset_v3 / MEI キャリブ他) は inst box の内部に並んで
いると推測されるが, その正確な内部レイアウトは実サンプル 1 つでは確定できない.
そのため本モジュールは inst box を **byte 配列としてマウントするだけ** に留め,
record レベルの解析は後続の PR で個別に足していく (offset_v3 → imu → mei_pb 順).

参考: https://github.com/BenjaminHenriksson/insv-stitch (独立に再実装しているが結論は近い).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path


# 末尾トレイラの ASCII シグネチャ (32 バイト). バイナリ UUID ではない.
FOOTER_SIGNATURE = b"8db42d694ccc418790edff439fe026bf"
assert len(FOOTER_SIGNATURE) == 32

# トレイラヘッダ (シグネチャ直前 8 バイト) のフィールド構造.
_TRAILER_HEADER_STRUCT = struct.Struct("<II")   # inst_box_data_size, version
TRAILER_HEADER_SIZE = _TRAILER_HEADER_STRUCT.size  # 8
SIGNATURE_SIZE = len(FOOTER_SIGNATURE)             # 32

# インストールボックスヘッダ ("inst" プレフィックス) のサイズ (MP4 box 通常形式).
_INST_HEADER_STRUCT = struct.Struct(">I4s")     # box_size BE u32, box_type 4s
_INST_BOX_TYPE = b"inst"
INST_HEADER_SIZE = _INST_HEADER_STRUCT.size        # 8


class FooterNotFoundError(Exception):
    """Insta360 独自フッタが検出できなかった. 通常の MP4 かフォーマット違い."""


@dataclass
class InsvFooter:
    """検出できたフッタ構造の要約. record 内部は保持しない (必要時に raw 参照)."""

    file_size: int

    # inst box (Insta360 メタデータボックス).
    inst_box_offset: int      # ボックスヘッダの開始 offset
    inst_box_total_size: int  # ヘッダ含めた全長 (BE u32)
    inst_box_data_offset: int # ヘッダ 8 バイト後
    inst_box_data_size: int   # data 部の bytes (== box_total_size - 8)

    # 末尾トレイラ (512 バイトを実サンプルで観測).
    trailer_offset: int       # inst box の直後
    trailer_size: int         # file_size - trailer_offset

    # トレイラヘッダ (シグネチャ直前 8 バイト).
    trailer_header_offset: int
    reported_inst_data_size: int  # トレイラヘッダの u32 (inst_box_data_size と一致すべき)
    version: int                  # X5 サンプルでは 3

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
            raise FooterNotFoundError(
                f"expected inst box, got {box_type!r} at offset {footer_offset}"
            )
        if box_size < INST_HEADER_SIZE:
            raise FooterNotFoundError(f"nonsensical inst box size {box_size}")

        inst_box_data_offset = footer_offset + INST_HEADER_SIZE
        inst_box_data_size = box_size - INST_HEADER_SIZE
        trailer_offset = footer_offset + box_size
        if trailer_offset > file_size:
            raise FooterNotFoundError(
                f"inst box overruns EOF (ends at {trailer_offset}, size={file_size})"
            )
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
        reported_size, version = _TRAILER_HEADER_STRUCT.unpack(
            f.read(TRAILER_HEADER_SIZE)
        )

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

    現状はこの生バイト列を上位モジュール (imu.py, calibration.py, protobuf.py) が
    パースするための入り口. 11.4 MB 程度あるので必要な時だけ呼ぶこと.
    """
    with footer.path.open("rb") as f:
        f.seek(footer.inst_box_data_offset)
        return f.read(footer.inst_box_data_size)


def read_trailer_bytes(footer: InsvFooter) -> bytes:
    """トレイラ 512 バイト (inst box 直後 〜 EOF) をまるごと読む.

    ここには protobuf エンコードされた mask ラベル一覧が含まれる (実サンプル観測).
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
                    if 0 <= eoff and eoff + esize <= n:
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
