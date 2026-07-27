"""重力整列済みの sparse points から広域な地面高度を推定する。"""

from __future__ import annotations

import numpy as np

from .model import Reconstruction


def estimate_ground_position(
    reconstruction: Reconstruction,
    *,
    min_camera_height_m: float = 0.5,
    max_camera_height_m: float = 4.0,
    histogram_bin_m: float = 0.05,
    inlier_band_m: float = 0.15,
    min_support_ratio: float = 0.005,
    min_horizontal_span_ratio: float = 0.25,
) -> dict:
    if not reconstruction.images or not reconstruction.points3D:
        return {"applied": False, "reason": "empty_reconstruction"}
    if not 0 < min_camera_height_m < max_camera_height_m:
        raise ValueError("ground camera-height range is invalid")
    if histogram_bin_m <= 0 or inlier_band_m <= 0:
        raise ValueError("ground histogram parameters must be positive")

    cameras = np.asarray([image.camera_center for image in reconstruction.images.values()])
    points = np.asarray([point.xyz for point in reconstruction.points3D.values()])
    camera_y = float(np.median(cameras[:, 1]))
    candidates = points[
        (points[:, 1] >= camera_y + min_camera_height_m)
        & (points[:, 1] <= camera_y + max_camera_height_m)
    ]
    minimum_support = max(1000, round(len(points) * min_support_ratio))
    if len(candidates) < minimum_support:
        return {
            "applied": False,
            "reason": "too_few_ground_candidates",
            "candidate_points": len(candidates),
            "required_points": minimum_support,
        }

    low = np.floor(float(candidates[:, 1].min()) / histogram_bin_m) * histogram_bin_m
    high = np.ceil(float(candidates[:, 1].max()) / histogram_bin_m) * histogram_bin_m
    edges = np.arange(low, high + histogram_bin_m * 1.5, histogram_bin_m)
    counts, _ = np.histogram(candidates[:, 1], edges)
    camera_horizontal_span = _horizontal_span(cameras)
    if camera_horizontal_span <= 1e-9:
        return {"applied": False, "reason": "collapsed_camera_trajectory"}

    hypotheses = []
    for index in np.argsort(counts)[-min(20, len(counts)) :]:
        center = float((edges[index] + edges[index + 1]) * 0.5)
        band = points[np.abs(points[:, 1] - center) <= inlier_band_m]
        if len(band) < minimum_support:
            continue
        span_ratio = _horizontal_span(band) / camera_horizontal_span
        if span_ratio < min_horizontal_span_ratio:
            continue
        ground_y = float(np.median(band[:, 1]))
        camera_height = ground_y - camera_y
        hypotheses.append(
            {
                "ground_y": ground_y,
                "camera_height_median_m": camera_height,
                "support_points": len(band),
                "support_ratio": len(band) / len(points),
                "horizontal_span_ratio": span_ratio,
                "score": len(band) * min(span_ratio, 1.0),
            }
        )
    if not hypotheses:
        return {
            "applied": False,
            "reason": "no_broad_horizontal_ground_mode",
            "candidate_points": len(candidates),
        }
    best = max(hypotheses, key=lambda item: item["score"])
    return {
        "applied": abs(best["ground_y"]) > 1e-9,
        "method": "gravity_constrained_point_mode",
        "translation": [0.0, -best["ground_y"], 0.0],
        "candidate_points": len(candidates),
        "hypotheses": len(hypotheses),
        **best,
    }


def _horizontal_span(points: np.ndarray) -> float:
    horizontal = points[:, [0, 2]]
    low, high = np.percentile(horizontal, [5.0, 95.0], axis=0)
    return float(np.linalg.norm(high - low))
