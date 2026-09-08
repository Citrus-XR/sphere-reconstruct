"""重力整列済み sparse points から撮影経路に沿った地面位置を推定する。"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .model import Reconstruction


def estimate_ground_position(
    reconstruction: Reconstruction,
    *,
    analysis_points: np.ndarray | None = None,
    reference_image_prefix: str | None = None,
    min_camera_height_m: float = 0.5,
    max_camera_height_m: float = 4.0,
    histogram_bin_m: float = 0.05,
    inlier_band_m: float = 0.15,
    max_path_distance_m: float = 4.0,
    min_support_ratio: float = 0.005,
    min_horizontal_span_ratio: float = 0.25,
) -> dict:
    if not reconstruction.images or not reconstruction.points3D:
        return {"applied": False, "reason": "empty_reconstruction"}
    if not 0 < min_camera_height_m < max_camera_height_m:
        raise ValueError("ground camera-height range is invalid")
    if histogram_bin_m <= 0 or inlier_band_m <= 0 or max_path_distance_m <= 0:
        raise ValueError("ground prediction distances must be positive")

    reference_images = [
        image
        for image in reconstruction.images.values()
        if reference_image_prefix is None or image.name.startswith(reference_image_prefix)
    ]
    if not reference_images:
        return {"applied": False, "reason": "reference_trajectory_unavailable"}
    cameras = np.asarray([image.camera_center for image in reference_images])
    points = analysis_points
    if points is None:
        points = sample_analysis_points(reconstruction)
    camera_horizontal_span = _horizontal_span(cameras)
    if camera_horizontal_span <= 1e-9:
        return {"applied": False, "reason": "collapsed_camera_trajectory"}

    distances, nearest_camera = _nearest_trajectory_samples(
        points[:, [0, 2]], cameras[:, [0, 2]], max_path_distance_m
    )
    has_nearby_camera = nearest_camera >= 0
    relative_height = np.full(len(points), np.nan)
    relative_height[has_nearby_camera] = (
        points[has_nearby_camera, 1] - cameras[nearest_camera[has_nearby_camera], 1]
    )
    candidates = (
        has_nearby_camera
        & (relative_height >= min_camera_height_m)
        & (relative_height <= max_camera_height_m)
    )
    minimum_support = max(1000, round(len(points) * min_support_ratio))
    candidate_count = int(np.count_nonzero(candidates))
    if candidate_count < minimum_support:
        return {
            "applied": False,
            "reason": "too_few_ground_candidates_near_trajectory",
            "candidate_points": candidate_count,
            "required_points": minimum_support,
        }

    low = np.floor(min_camera_height_m / histogram_bin_m) * histogram_bin_m
    high = np.ceil(max_camera_height_m / histogram_bin_m) * histogram_bin_m
    edges = np.arange(low, high + histogram_bin_m * 1.5, histogram_bin_m)
    counts, _ = np.histogram(relative_height[candidates], edges)
    hypotheses = []
    for index in np.argsort(counts)[-min(20, len(counts)) :]:
        center = float((edges[index] + edges[index + 1]) * 0.5)
        selected = candidates & (np.abs(relative_height - center) <= inlier_band_m)
        support = int(np.count_nonzero(selected))
        if support < minimum_support:
            continue
        selected_points = points[selected]
        span_ratio = _horizontal_span(selected_points) / camera_horizontal_span
        if span_ratio < min_horizontal_span_ratio:
            continue
        camera_indices = nearest_camera[selected]
        camera_ground_y, local_heights = _per_camera_ground(
            selected_points[:, 1], relative_height[selected], camera_indices
        )
        if not camera_ground_y:
            continue
        trajectory_coverage = len(camera_ground_y) / len(cameras)
        ground_values = np.asarray(camera_ground_y)
        ground_y = float(np.median(ground_values))
        hypotheses.append(
            {
                "ground_y": ground_y,
                "camera_height_median_m": float(np.median(local_heights)),
                "support_points": support,
                "support_ratio": support / len(points),
                "horizontal_span_ratio": span_ratio,
                "trajectory_samples": len(camera_ground_y),
                "trajectory_coverage_ratio": trajectory_coverage,
                "path_distance_median_m": float(np.median(distances[selected])),
                "ground_height_p10_m": float(np.percentile(ground_values, 10.0)),
                "ground_height_p90_m": float(np.percentile(ground_values, 90.0)),
                "score": support * min(span_ratio, 1.0) * min(trajectory_coverage * 2.0, 1.0),
            }
        )
    if not hypotheses:
        return {
            "applied": False,
            "reason": "no_trajectory_local_ground_mode",
            "candidate_points": candidate_count,
        }
    maximum_score = max(float(item["score"]) for item in hypotheses)
    dominant = [item for item in hypotheses if float(item["score"]) >= maximum_score * 0.9]
    best = min(dominant, key=lambda item: (item["camera_height_median_m"], -item["score"]))
    return {
        "applied": abs(best["ground_y"]) > 1e-9,
        "method": "trajectory_local_ground_mode",
        "translation": [0.0, -best["ground_y"], 0.0],
        "candidate_points": candidate_count,
        "hypotheses": len(hypotheses),
        "dominant_hypotheses": len(dominant),
        "plane_selection": "nearest_dominant_horizontal_plane",
        "reference_images": len(reference_images),
        "max_path_distance_m": max_path_distance_m,
        **best,
    }


def _nearest_trajectory_samples(
    point_horizontal: np.ndarray,
    camera_horizontal: np.ndarray,
    max_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """SciPy を必須依存にせず、2D spatial hash で最近傍経路 sample を求める。"""
    cell_size = max_distance
    camera_cells = np.floor(camera_horizontal / cell_size).astype(np.int64)
    camera_buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, cell in enumerate(camera_cells):
        camera_buckets[(int(cell[0]), int(cell[1]))].append(index)

    point_cells = np.floor(point_horizontal / cell_size).astype(np.int64)
    order = np.lexsort((point_cells[:, 1], point_cells[:, 0]))
    ordered_cells = point_cells[order]
    boundaries = np.flatnonzero(np.r_[True, np.any(ordered_cells[1:] != ordered_cells[:-1], axis=1), True])
    nearest = np.full(len(point_horizontal), -1, dtype=np.int64)
    distances = np.full(len(point_horizontal), np.inf)
    max_distance_squared = max_distance * max_distance

    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        cell = ordered_cells[start]
        candidates = [
            camera_index
            for x_offset in (-1, 0, 1)
            for z_offset in (-1, 0, 1)
            for camera_index in camera_buckets.get((int(cell[0] + x_offset), int(cell[1] + z_offset)), [])
        ]
        if not candidates:
            continue
        candidate_indices = np.asarray(candidates)
        candidate_points = camera_horizontal[candidate_indices]
        max_chunk_points = max(1, 2_000_000 // len(candidate_indices))
        for chunk_start in range(start, end, max_chunk_points):
            point_indices = order[chunk_start : min(end, chunk_start + max_chunk_points)]
            delta = point_horizontal[point_indices, np.newaxis, :] - candidate_points[np.newaxis, :, :]
            distance_squared = np.einsum("ijk,ijk->ij", delta, delta)
            local_nearest = np.argmin(distance_squared, axis=1)
            local_distance_squared = distance_squared[np.arange(len(point_indices)), local_nearest]
            accepted = local_distance_squared <= max_distance_squared
            accepted_points = point_indices[accepted]
            nearest[accepted_points] = candidate_indices[local_nearest[accepted]]
            distances[accepted_points] = np.sqrt(local_distance_squared[accepted])
    return distances, nearest


def _per_camera_ground(
    point_y: np.ndarray,
    relative_height: np.ndarray,
    camera_indices: np.ndarray,
    *,
    minimum_points: int = 3,
) -> tuple[list[float], list[float]]:
    order = np.argsort(camera_indices, kind="stable")
    labels = camera_indices[order]
    boundaries = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1], True])
    ground: list[float] = []
    heights: list[float] = []
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        if end - start < minimum_points:
            continue
        indices = order[start:end]
        ground.append(float(np.median(point_y[indices])))
        heights.append(float(np.median(relative_height[indices])))
    return ground, heights


def _horizontal_span(points: np.ndarray) -> float:
    horizontal = points[:, [0, 2]]
    low, high = np.percentile(horizontal, [5.0, 95.0], axis=0)
    return float(np.linalg.norm(high - low))


def sample_analysis_points(
    reconstruction: Reconstruction,
    *,
    maximum_points: int = 100_000,
) -> np.ndarray:
    """点数に関係なく幾何推定の計算量を固定する deterministic uniform sample。"""
    if maximum_points <= 0:
        raise ValueError("maximum analysis points must be positive")
    total = len(reconstruction.points3D)
    if total <= maximum_points:
        return np.asarray([point.xyz for point in reconstruction.points3D.values()], dtype=np.float64)

    targets = np.linspace(0, total - 1, maximum_points, dtype=np.int64)
    sampled = np.empty((maximum_points, 3), dtype=np.float64)
    target_index = 0
    for point_index, point in enumerate(reconstruction.points3D.values()):
        if point_index != targets[target_index]:
            continue
        sampled[target_index] = point.xyz
        target_index += 1
        if target_index == maximum_points:
            break
    return sampled
