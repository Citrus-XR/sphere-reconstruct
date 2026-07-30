"""LichtFeld Studio の Gaussian PLY を streaming memmap で要約する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _read_header(path: Path) -> tuple[int, int, list[str]]:
    with path.open("rb") as file:
        first = file.readline()
        if first != b"ply\n":
            raise ValueError(f"PLY header がありません: {path}")
        binary_little_endian = False
        vertex_count = None
        properties: list[str] = []
        reading_vertices = False
        while True:
            line = file.readline()
            if not line:
                raise ValueError("PLY header が途中で終わりました")
            text = line.decode("ascii").strip()
            if text == "format binary_little_endian 1.0":
                binary_little_endian = True
            elif text.startswith("element "):
                parts = text.split()
                reading_vertices = parts[1] == "vertex"
                if reading_vertices:
                    vertex_count = int(parts[2])
            elif text.startswith("property ") and reading_vertices:
                parts = text.split()
                if parts[1] != "float":
                    raise ValueError(f"未対応の vertex property です: {text}")
                properties.append(parts[2])
            elif text == "end_header":
                break
        if not binary_little_endian or vertex_count is None or not properties:
            raise ValueError("binary little-endian vertex PLY が必要です")
        return file.tell(), vertex_count, properties


def analyze(path: Path) -> dict:
    offset, count, properties = _read_header(path)
    expected_size = offset + count * len(properties) * 4
    if path.stat().st_size != expected_size:
        raise ValueError(
            f"PLY size が header と一致しません: {path.stat().st_size} != {expected_size}"
        )
    data = np.memmap(
        path,
        mode="r",
        dtype=np.dtype([(name, "<f4") for name in properties]),
        offset=offset,
        shape=(count,),
    )
    required = {"x", "y", "z", "scale_0", "scale_1", "scale_2", "opacity"}
    missing = sorted(required - set(properties))
    if missing:
        raise ValueError(f"必要な PLY property がありません: {missing}")
    xyz = np.column_stack((data["x"], data["y"], data["z"]))
    log_scales = np.column_stack((data["scale_0"], data["scale_1"], data["scale_2"]))
    finite = np.isfinite(xyz).all(axis=1) & np.isfinite(log_scales).all(axis=1)
    maximum_scale = np.exp(np.max(log_scales[finite], axis=1).astype(np.float64))
    centers = xyz[finite].astype(np.float64)
    center = np.median(centers, axis=0)
    radii = np.linalg.norm(centers - center, axis=1)
    scene_radius_p95 = float(np.quantile(radii, 0.95))
    if scene_radius_p95 <= 0.0:
        raise ValueError("Gaussian center の scene radius が 0 です")
    normalized_scale = maximum_scale / scene_radius_p95
    logits = data["opacity"][finite].astype(np.float64)
    opacity = np.empty_like(logits)
    positive = logits >= 0.0
    opacity[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exponential = np.exp(logits[~positive])
    opacity[~positive] = exponential / (1.0 + exponential)

    def quantiles(values: np.ndarray) -> dict[str, float]:
        return {
            name: float(value)
            for name, value in zip(
                ("median", "p95", "p99", "maximum"),
                np.quantile(values, (0.5, 0.95, 0.99, 1.0)),
                strict=True,
            )
        }

    return {
        "path": str(path),
        "vertices": count,
        "record_bytes": len(properties) * 4,
        "finite_vertices": int(finite.sum()),
        "scene_center_median": [float(value) for value in center],
        "scene_radius_p95": scene_radius_p95,
        "center_radius": quantiles(radii),
        "maximum_scale": quantiles(maximum_scale),
        "normalized_maximum_scale": quantiles(normalized_scale),
        "normalized_scale_over_0_2": int(np.count_nonzero(normalized_scale > 0.2)),
        "normalized_scale_over_0_5": int(np.count_nonzero(normalized_scale > 0.5)),
        "opacity": quantiles(opacity),
        "opacity_over_0_5": int(np.count_nonzero(opacity > 0.5)),
        "all_finite": bool(finite.all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ply", type=Path)
    args = parser.parse_args()
    result = analyze(args.ply.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
