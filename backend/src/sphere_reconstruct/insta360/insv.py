"""INSV = MP4 (ISO BMFF) + 独自フッタ.

このモジュールは:
- MP4 のトップレベル box を安全に列挙する.
- Insta360 フッタの開始位置を検出する.
- 各ストリームの codec / 解像度 / 尺 / 時間ベース などの基本情報を返す.
  (詳細な IMU / offset_v3 パースは metadata.py と imu.py に分ける)

方針:
- ffmpeg / ffprobe を子プロセスで呼ぶ実装は別途 imaging/ に置く. ここは
  純粋にファイル構造の直接パースに徹する (依存ゼロで unit test 可能).
- 「fisheye 2 本 + 音声 1 本」を強く仮定しない. あくまで観測した stream 数と
  時間ベースを返し, 解釈は呼び出し側 (calibration.py) に任せる.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


# ISO BMFF box header は最短 8 bytes: uint32 size + 4 bytes type. size==1 なら
# その後 uint64 largesize が続く. size==0 なら EOF まで続く.
_BOX_HEADER_MIN = 8

# トップレベルで期待する MP4 box type. これ以外が出てきたら Insta360 フッタか
# 破損の可能性がある.
_KNOWN_TOP_BOXES = frozenset({b"ftyp", b"moov", b"mdat", b"free", b"skip", b"uuid", b"wide", b"pdin"})


@dataclass
class MP4Box:
    box_type: bytes  # 4 bytes
    offset: int      # ファイル先頭からの byte offset
    header_size: int # 8 か 16
    total_size: int  # header 含む byte 数 (0 なら EOF まで)


@dataclass
class InsvLayout:
    file_size: int
    boxes: list[MP4Box]
    footer_offset: int | None  # Insta360 フッタが始まる (と推定される) offset
    footer_size: int | None    # None なら EOF まで


def scan_boxes(fp: BinaryIO, file_size: int) -> list[MP4Box]:
    """MP4 のトップレベル box を列挙する.

    未知の box type が現れた時点でその位置を返して打ち切る. これによって
    「MP4 本体の直後に貼り付けられた Insta360 独自フッタ」の開始位置が分かる.
    """
    boxes: list[MP4Box] = []
    fp.seek(0)
    pos = 0
    while pos + _BOX_HEADER_MIN <= file_size:
        fp.seek(pos)
        header = fp.read(_BOX_HEADER_MIN)
        if len(header) < _BOX_HEADER_MIN:
            break
        size32 = struct.unpack(">I", header[:4])[0]
        box_type = header[4:8]

        if size32 == 1:
            ext = fp.read(8)
            if len(ext) < 8:
                break
            total = struct.unpack(">Q", ext)[0]
            header_size = 16
        elif size32 == 0:
            total = 0  # EOF まで
            header_size = 8
        else:
            total = size32
            header_size = 8

        # box_type が未知なら, Insta360 フッタ開始と判断して中断する.
        if box_type not in _KNOWN_TOP_BOXES:
            break

        boxes.append(
            MP4Box(box_type=box_type, offset=pos, header_size=header_size, total_size=total)
        )

        if total == 0:
            # mdat が 0 size で EOF までを意味する. これ以降 box は現れない前提で終了.
            break
        pos += total
    return boxes


def layout(path: Path) -> InsvLayout:
    """INSV のトップレベル layout を取得する.

    - MP4 部分の box リスト
    - Insta360 フッタ開始 offset (未検出なら None)
    """
    file_size = path.stat().st_size
    with path.open("rb") as fp:
        boxes = scan_boxes(fp, file_size)

    footer_offset: int | None = None
    footer_size: int | None = None
    if boxes:
        last = boxes[-1]
        # 最後の box が EOF まで (total_size=0) なら フッタは無し.
        if last.total_size != 0:
            after = last.offset + last.total_size
            if after < file_size:
                footer_offset = after
                footer_size = file_size - after
    else:
        # box 走査で 1 個も取れなかったら, フッタしか無いか, MP4 として壊れている.
        footer_offset = 0
        footer_size = file_size

    return InsvLayout(
        file_size=file_size,
        boxes=boxes,
        footer_offset=footer_offset,
        footer_size=footer_size,
    )


def has_footer(path: Path) -> bool:
    return layout(path).footer_offset is not None
