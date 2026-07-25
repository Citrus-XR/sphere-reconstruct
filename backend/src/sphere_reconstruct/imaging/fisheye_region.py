"""魚眼の有効領域 (円) の永続化.

X5 の各レンズ画像は円形の有効領域を持ち, 外周は黒縁 + レンズ端のケラレ/反射/汚れが
乗る. 自動半径推定は反射/汚れで不安定なため, ユーザが UI で円 (中心 + 半径) を手動調整
できるようにし, その結果を project 直下に保存する.

保存形式 (`<project>/fisheye_region.json`), 解像度非依存の正規化座標:
  {"lens0": {"cx": 0.5, "cy": 0.5, "r": 0.459}, "lens1": {...}}
cx/cy/r は画像幅 W に対する比 (魚眼は正方なので H==W 前提, cy も W 基準で扱う).
"""

from __future__ import annotations

import json
from pathlib import Path

FILENAME = "fisheye_region.json"
# 既定半径 (正規化). 実測でレンズ有効円がこの比に収まることが多い.
DEFAULT_R_NORM = 0.459
DEFAULT_LENS = {"cx": 0.5, "cy": 0.5, "r": DEFAULT_R_NORM}


def default_region() -> dict:
    return {"lens0": dict(DEFAULT_LENS), "lens1": dict(DEFAULT_LENS)}


def region_path(project_dir: Path) -> Path:
    return project_dir / FILENAME


def load_region(project_dir: Path) -> dict:
    """保存済み region を返す. 無ければ既定. 欠けたレンズは既定で補完する."""
    p = region_path(project_dir)
    if not p.exists():
        return default_region()
    data = json.loads(p.read_text())
    out = default_region()
    for lens in ("lens0", "lens1"):
        if isinstance(data.get(lens), dict):
            out[lens] = {**DEFAULT_LENS, **{k: float(data[lens][k]) for k in ("cx", "cy", "r") if k in data[lens]}}
    return out


def save_region(project_dir: Path, data: dict) -> dict:
    """region を検証して保存する. 返り値は正規化済みの保存内容."""
    out = default_region()
    for lens in ("lens0", "lens1"):
        d = data.get(lens)
        if isinstance(d, dict):
            out[lens] = {
                "cx": _clamp01(float(d.get("cx", 0.5))),
                "cy": _clamp01(float(d.get("cy", 0.5))),
                "r": max(0.01, min(0.75, float(d.get("r", DEFAULT_R_NORM)))),
            }
    region_path(project_dir).write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def circle_px(lens_region: dict, width: int, height: int) -> tuple[float, float, float]:
    """正規化 region を画素座標の (cx, cy, r) に変換する. 基準は幅 W."""
    return (
        lens_region["cx"] * width,
        lens_region["cy"] * height,
        lens_region["r"] * width,
    )


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))
