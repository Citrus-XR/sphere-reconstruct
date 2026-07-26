"""キャリブレーションモデル.

INSV / PB / 内蔵 profile / manual override のいずれかから得られた「デュアル魚眼
キャリブレーション」を統一データ構造で保持する.

- MeiLensCalibration: 1 レンズ. MEI (Mei-Rives) + 拡張 (radial k1-k3, tangential
  p1/p2) + 姿勢 (3 角) + オフセット (tx/ty/tz).
- DualLensCalibration: 前後 2 レンズ. どちらが front / back かの解釈は
  「観測を通じて決める」もので, 静的には決め打ちしない.

X5 サンプルで観測された offset_v3 の値 (実例):
    lens A: xi=2.0 fx=4278.3 fy=4277.33 cx=2694.63 cy=2681.84
            angles=(0.615, 0.016, 89.937) trans=(0,0,0)
            k=(0.184, 2.073, -3.280) p=(-5.3e-5, 6.5e-4)
            ref=10752x5376 tag=113
    lens B: xi=2.0 fx=4296.81 fy=4298.54 cx=8064.92 cy=2686.41
            angles=(-0.718, 0.211, 89.840) trans=(-4.8e-5, 1.3e-4, -0.032273)
            k=(0.183, 2.053, -3.267) p=(1.87e-3, 3.82e-4)
            ref=10752x5376 tag=113
    trailing id: 197632

lens B の tz が -32.3mm = X5 前後鏡頭の物理ベースライン (~30mm) と一致する.
lens B の cx が 8064.92 = 2686.41 + 5376.0 なので, 参照座標系は 「左右横並び
10752 x 5376」の合成画像で表現されている (共通イメージ座標).

`source_priority` の降級鎖:
  PB -> offset_v3 -> 機種内蔵 profile -> manual override -> エラー
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class CalibSource(StrEnum):
    PB = "pb"
    OFFSET_V3 = "offset_v3"
    BUILTIN_PROFILE = "builtin_profile"
    USER = "user"


@dataclass
class MeiLensCalibration:
    """MEI (Mei-Rives) 鱼眼モデル + 拡張畜れみ.

    投影過程:
      P = point in camera coords, normalized to ||P|| = 1
      x = P.x / (P.z + xi)
      y = P.y / (P.z + xi)
      r2 = x*x + y*y
      radial = 1 + k1*r2 + k2*r2^2 + k3*r2^3
      x' = x * radial + tangential(p1, p2, x, y)
      y' = y * radial + tangential(p1, p2, y, x)
      u = fx * x' + cx
      v = fy * y' + cy

    yaw/pitch/roll と tx/ty/tz は「参照座標系 -> このレンズ座標系」の姿勢.
    参照座標系のスケールは m (メートル) と観測.
    """

    xi: float
    fx: float
    fy: float
    cx: float
    cy: float
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    # 参照解像度 (offset_v3 の cx, cy はこの座標系に定義されている).
    ref_image_width: int = 0
    ref_image_height: int = 0
    lens_flags: int = 0  # X5 では 113 が観測される. 意味は要調査.

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__.keys()}


@dataclass
class DualLensCalibration:
    source: CalibSource
    lenses: list[MeiLensCalibration] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)  # 出所依存の生データ

    def is_valid(self) -> bool:
        return len(self.lenses) >= 2 and all(l.fx > 0 and l.fy > 0 for l in self.lenses)


# ---------- offset_v3 (ASCII underscore-separated) --------------------------------
# X5 では inst box の末尾付近に, 数字を "_" で連結した長い ASCII 文字列として
# offset_v3 が入っている. 実サンプルで確認したレイアウト:
#
#   <count=2>_<lens A 19 values>_<lens B 19 values>_<calibration_id>
#
# 各レンズブロック 19 items の並び:
#   xi, fx, fy, cx, cy,
#   angle1, angle2, angle3,     (yaw / pitch / roll 相当, 単位未確定. deg 想定)
#   tx, ty, tz,                 (m)
#   k1, k2, k3,                 (radial distortion, k4 なし)
#   p1, p2,                     (tangential)
#   ref_w, ref_h, lens_flags    (参照解像度 + フラグ 113)
#
# 全体で 1 + 19*2 + 1 = 40 items.


LENS_ITEMS = 19
TOTAL_ITEMS = 1 + LENS_ITEMS * 2 + 1

# 数字 6 個以上を "_" で連結した ASCII 塊を候補として拾う正規表現.
_ASCII_CALIB_RE = re.compile(rb"-?[0-9]+(?:\.[0-9]+)?(?:_-?[0-9]+(?:\.[0-9]+)?){5,}")


@dataclass
class OffsetV3Ascii:
    """inst box 内で発見された ASCII underscore-separated calibration 文字列."""

    inst_offset: int  # inst box data 内の byte offset
    text: str
    values: list[float]


def find_ascii_calibrations(inst_bytes: bytes) -> list[OffsetV3Ascii]:
    """inst box 内から ASCII underscore-separated calibration 候補を全部列挙する.

    候補には短いバージョン (5-6 items のみ) から長いバージョン (40 items) まで
    複数含まれる. 呼び出し側は `pick_offset_v3()` で最も情報量の多いものを選ぶ.
    """
    out: list[OffsetV3Ascii] = []
    for m in _ASCII_CALIB_RE.finditer(inst_bytes):
        text = m.group().decode("ascii", "replace")
        parts = text.split("_")
        try:
            values = [float(p) for p in parts]
        except ValueError:
            continue
        out.append(OffsetV3Ascii(inst_offset=m.start(), text=text, values=values))
    return out


def pick_offset_v3(candidates: list[OffsetV3Ascii]) -> OffsetV3Ascii | None:
    """40 items ちょうどの候補を優先. 見つからなければ最長のものを返す."""
    exact = [c for c in candidates if len(c.values) == TOTAL_ITEMS]
    if exact:
        # 複数あれば inst_offset が大きい (末尾側の) ものを取る.
        return max(exact, key=lambda c: c.inst_offset)
    return max(candidates, key=lambda c: len(c.values), default=None)


def parse_offset_v3_ascii(cand: OffsetV3Ascii) -> DualLensCalibration:
    """40 items 版の offset_v3 を DualLensCalibration にマップする."""
    v = cand.values
    if len(v) != TOTAL_ITEMS:
        return DualLensCalibration(
            source=CalibSource.OFFSET_V3,
            lenses=[],
            raw={
                "kind": "ascii_underscore",
                "text": cand.text,
                "count": len(v),
                "note": f"expected {TOTAL_ITEMS} items, got {len(v)}; not confidently mappable",
            },
        )

    count = int(v[0])  # 通常 2
    lens_a = _parse_lens(v[1 : 1 + LENS_ITEMS])
    lens_b = _parse_lens(v[1 + LENS_ITEMS : 1 + LENS_ITEMS * 2])
    calibration_id = int(v[-1])
    return DualLensCalibration(
        source=CalibSource.OFFSET_V3,
        lenses=[lens_a, lens_b],
        raw={
            "kind": "ascii_underscore",
            "count": count,
            "calibration_id": calibration_id,
            "text": cand.text,
            "inst_offset": cand.inst_offset,
        },
    )


def _parse_lens(items: list[float]) -> MeiLensCalibration:
    if len(items) != LENS_ITEMS:
        raise ValueError(f"lens block must have {LENS_ITEMS} items, got {len(items)}")
    (
        xi,
        fx,
        fy,
        cx,
        cy,
        a1,
        a2,
        a3,
        tx,
        ty,
        tz,
        k1,
        k2,
        k3,
        p1,
        p2,
        ref_w,
        ref_h,
        lens_flags,
    ) = items
    return MeiLensCalibration(
        xi=xi,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        yaw=a1,
        pitch=a2,
        roll=a3,
        tx=tx,
        ty=ty,
        tz=tz,
        k1=k1,
        k2=k2,
        k3=k3,
        p1=p1,
        p2=p2,
        ref_image_width=int(ref_w),
        ref_image_height=int(ref_h),
        lens_flags=int(lens_flags),
    )


# ---------- offset_v3 (bytes-level, legacy path) ---------------------------------
# 一部の古い INSV では inst box 内に float32 バイナリで保持されている可能性がある.
# 実サンプルでは今のところ観測できていないため, ここは維持だけしておく.


OFFSET_V3_RECORD_TYPE = 0x0101  # 未確認


@dataclass
class OffsetV3Raw:
    version: int
    values: list[float]
    payload_hex: str


def parse_offset_v3_bytes(payload: bytes) -> OffsetV3Raw:
    if len(payload) < 4:
        raise ValueError("offset_v3 payload too small")
    version = struct.unpack_from("<I", payload, 0)[0]
    floats_bytes = payload[4:]
    n = len(floats_bytes) // 4
    values = list(struct.unpack_from(f"<{n}f", floats_bytes, 0))
    return OffsetV3Raw(version=version, values=values, payload_hex=payload.hex())


# ---------- 選択ロジック --------------------------------------------------------


def choose(*candidates: DualLensCalibration | None) -> DualLensCalibration | None:
    """降級鎖に従って最初に is_valid() な候補を返す.

    呼び出し側は優先度順に None 可で並べる. 全部ダメなら None -> 呼び出し側は
    「静默にデフォルト焦距を差し込むのではなく」明示エラーを出す責務がある.
    """
    for c in candidates:
        if c is not None and c.is_valid():
            return c
    return None
