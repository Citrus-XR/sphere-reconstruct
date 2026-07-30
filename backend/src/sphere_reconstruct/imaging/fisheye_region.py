"""魚眼の有効領域 (円) の永続化.

円形 sensor image の外周にある黒縁・ケラレ・反射・汚れを除くため、画像から初期円を
推定し、UI で中心と半径を確認・調整した結果を project 直下に保存する。

保存形式 (`<project>/fisheye_regions.json`) は source ID ごとに分離する。
cx/cy/r は画像幅 W に対する比 (魚眼は正方なので H==W 前提, cy も W 基準で扱う).
"""

from __future__ import annotations

import json
from pathlib import Path

FILENAME = "fisheye_regions.json"
REGION_VERSION = 3
# 既定半径 (正規化). 実測でレンズ有効円がこの比に収まることが多い.
DEFAULT_R_NORM = 0.459
MAX_CUSTOM_OPERATIONS = 2048


def _default_lens() -> dict:
    return {"cx": 0.5, "cy": 0.5, "r": DEFAULT_R_NORM, "operations": []}


def default_region() -> dict:
    return {"lens0": _default_lens(), "lens1": _default_lens()}


def region_path(project_dir: Path) -> Path:
    return project_dir / FILENAME


def load_region(project_dir: Path, source_id: str) -> dict:
    """source の保存済み region を返す。無ければ画像から推定する。"""
    p = region_path(project_dir)
    if not p.exists():
        return detect_region(project_dir, source_id)
    data = json.loads(p.read_text())
    source_data = (data.get("sources") or {}).get(source_id)
    if not isinstance(source_data, dict):
        return detect_region(project_dir, source_id)
    out = default_region()
    for lens in ("lens0", "lens1"):
        if isinstance(source_data.get(lens), dict):
            out[lens] = {
                "cx": 0.5,
                "cy": 0.5,
                "r": max(
                    0.01,
                    min(0.75, float(source_data[lens].get("r", DEFAULT_R_NORM))),
                ),
                "operations": _normalize_operations(source_data[lens].get("operations", [])),
            }
    return out


def detect_region(project_dir: Path, source_id: str) -> dict:
    """先頭の前後レンズ画像から黒縁を検出し, カメラ機種非依存の初期円を返す."""
    manifest_path = project_dir / "extract_frames" / "manifest_frames.json"
    if not manifest_path.exists():
        return default_region()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = [frame for frame in manifest.get("frames", []) if frame.get("source_id") == source_id]
    if not frames:
        return default_region()
    first = frames[0]
    detected = default_region()
    for lens in (0, 1):
        key = f"lens{lens}"
        if key in first:
            detected[key] = detect_lens_region(project_dir / first[key])
    return detected


def detect_lens_region(image_path: Path) -> dict:
    import cv2  # noqa: PLC0415

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"cannot read fisheye image: {image_path}")
    height, width = image.shape[:2]
    scale = min(1.0, 512.0 / max(width, height))
    small = cv2.resize(
        image,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA,
    )
    binary = (small > 12).astype("uint8") * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _default_lens()
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < small.shape[0] * small.shape[1] * 0.2:
        return _default_lens()
    (_center_x, _center_y), radius = cv2.minEnclosingCircle(contour)
    radius_normalized = radius / small.shape[1]
    # 円が画像端で切れている camera は min-enclosing circle が色収差・反射を含む外周まで拾う。
    # 完全に見える円は 3%、切れた円は 10% 内側へ寄せ、UI で必要なら広げられる初期値にする。
    safety = 0.90 if radius_normalized > 0.5 else 0.97
    return {
        "cx": 0.5,
        "cy": 0.5,
        "r": max(0.3, min(0.52, radius_normalized * safety)),
        "operations": [],
    }


def save_region(project_dir: Path, source_id: str, data: dict) -> dict:
    """region を検証して保存する. 返り値は正規化済みの保存内容."""
    out = default_region()
    for lens in ("lens0", "lens1"):
        d = data.get(lens)
        if isinstance(d, dict):
            out[lens] = {
                "cx": 0.5,
                "cy": 0.5,
                "r": max(0.01, min(0.75, float(d.get("r", DEFAULT_R_NORM)))),
                "operations": _normalize_operations(d.get("operations", [])),
            }
    path = region_path(project_dir)
    document = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists()
        else {"version": REGION_VERSION, "sources": {}}
    )
    document["version"] = REGION_VERSION
    document.setdefault("sources", {})[source_id] = {
        **out,
        "_coordinate_version": REGION_VERSION,
    }
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
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


def _normalize_operations(value) -> list[dict]:
    if not isinstance(value, list):
        raise ValueError("custom region operations は array でなければなりません")
    if len(value) > MAX_CUSTOM_OPERATIONS:
        raise ValueError(
            f"custom region operations が上限を超えています: {len(value)} > {MAX_CUSTOM_OPERATIONS}"
        )
    operations = []
    for index, operation in enumerate(value):
        if not isinstance(operation, dict):
            raise ValueError(f"custom region operation {index} が object ではありません")
        mode = str(operation.get("mode", ""))
        if mode not in {"add", "subtract"}:
            raise ValueError(f"custom region operation {index} の mode が不正です: {mode}")
        radius = float(operation.get("r", 0.0))
        if not 0.002 <= radius <= 0.5:
            raise ValueError(
                f"custom region operation {index} の radius が範囲外です: {radius}"
            )
        operations.append(
            {
                "mode": mode,
                "x": _clamp01(float(operation.get("x", 0.5))),
                "y": _clamp01(float(operation.get("y", 0.5))),
                "r": radius,
            }
        )
    return operations
