"""Relate sparse-point support to local image texture and triangulation geometry."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend" / "src"))
from sphere_reconstruct.colmap.model import read_model


def _percentile(values: list[float], p: float) -> float | None:
    return float(np.percentile(values, p)) if values else None


def audit(model_dir: Path, spec_path: Path, image_root: Path) -> dict:
    reconstruction = read_model(model_dir)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    names = {str(item["name"]): item for item in spec["images"]}
    points = list(reconstruction.points3D.values())
    point_index = {point.point3D_id: index for index, point in enumerate(points)}
    texture_sum = np.zeros(len(points), dtype=np.float64)
    texture_count = np.zeros(len(points), dtype=np.int32)
    texture_min = np.full(len(points), np.inf, dtype=np.float64)
    capture_sets: list[set[int]] = [set() for _ in points]
    centers = {image.image_id: np.asarray(image.camera_center, dtype=np.float64) for image in reconstruction.images.values()}
    angles: list[float] = []
    ranges: list[float] = []
    tracks: list[int] = []
    errors: list[float] = []
    valid_indices: list[int] = []
    for index, point in enumerate(points):
        obs_centers = [centers[image_id] for image_id, _ in point.track if image_id in centers]
        if len(obs_centers) < 2:
            continue
        rays = np.asarray(point.xyz, dtype=np.float64) - np.asarray(obs_centers)
        distance = np.linalg.norm(rays, axis=1)
        valid = distance > 0
        if int(np.count_nonzero(valid)) < 2:
            continue
        directions = rays[valid] / distance[valid, None]
        minimum_dot = float(np.min(np.clip(directions @ directions.T, -1.0, 1.0)))
        angles.append(math.degrees(math.acos(minimum_dot)))
        ranges.append(float(np.min(distance[valid])))
        tracks.append(len(point.track))
        errors.append(float(point.error))
        valid_indices.append(index)
        for image_id, _ in point.track:
            record = names.get(reconstruction.images[image_id].name)
            if record and "capture_index" in record:
                capture_sets[index].add(int(record["capture_index"]))

    for image in reconstruction.images.values():
        path = image_root / image.name
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f"cannot read {path}")
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = cv2.boxFilter(cv2.magnitude(gx, gy), cv2.CV_32F, (9, 9), normalize=True)
        for point2d in image.points2D:
            index = point_index.get(point2d.point3D_id)
            if index is None:
                continue
            x = min(max(round(point2d.x), 0), gradient.shape[1] - 1)
            y = min(max(round(point2d.y), 0), gradient.shape[0] - 1)
            value = float(gradient[y, x])
            texture_sum[index] += value
            texture_count[index] += 1
            texture_min[index] = min(texture_min[index], value)

    records = []
    for index, angle, distance, track, error in zip(valid_indices, angles, ranges, tracks, errors, strict=True):
        if texture_count[index] <= 0:
            continue
        records.append({
            "texture": float(texture_sum[index] / texture_count[index]),
            "texture_min": float(texture_min[index]),
            "angle": angle,
            "range": distance,
            "track": track,
            "error": error,
            "captures": len(capture_sets[index]),
        })

    def summary(selected: list[dict]) -> dict:
        return {
            "points": len(selected),
            "texture_p10_p50_p90": [_percentile([r["texture"] for r in selected], p) for p in (10, 50, 90)],
            "texture_min_p10_p50_p90": [_percentile([r["texture_min"] for r in selected], p) for p in (10, 50, 90)],
            "angle_p10_p50_p90": [_percentile([r["angle"] for r in selected], p) for p in (10, 50, 90)],
            "range_p10_p50_p90": [_percentile([r["range"] for r in selected], p) for p in (10, 50, 90)],
            "track_p10_p50_p90": [_percentile([r["track"] for r in selected], p) for p in (10, 50, 90)],
            "error_p50_p95": [_percentile([r["error"] for r in selected], p) for p in (50, 95)],
        }

    texture_values = np.asarray([r["texture"] for r in records])
    low_cut = float(np.percentile(texture_values, 20)) if len(texture_values) else 0.0
    high_cut = float(np.percentile(texture_values, 80)) if len(texture_values) else 0.0
    low = [r for r in records if r["texture"] <= low_cut]
    high = [r for r in records if r["texture"] >= high_cut]
    weak_and_low_angle = [r for r in low if r["angle"] < 8.0]
    filter_counts = {}
    for texture_label, subset in (("all", records), ("weak20", low)):
        for minimum_track in (3, 4):
            for minimum_angle in (6.0, 8.0, 10.0):
                filter_counts[f"{texture_label}_track_lt_{minimum_track}_angle_lt_{minimum_angle:g}"] = sum(
                    r["track"] < minimum_track and r["angle"] < minimum_angle for r in subset
                )
    return {
        "model": {"images": len(reconstruction.images), "points": len(points), "analyzed_points": len(records)},
        "texture_cutoffs": {"p20": low_cut, "p80": high_cut},
        "all": summary(records),
        "weakest_texture_20pct": summary(low),
        "strongest_texture_20pct": summary(high),
        "weak_texture_and_angle_under_8deg": summary(weak_and_low_angle),
        "combined_filter_counts": filter_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("spec", type=Path)
    parser.add_argument("image_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.model_dir, args.spec, args.image_root), indent=2))


if __name__ == "__main__":
    main()
