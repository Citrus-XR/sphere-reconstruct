"""重力整列済み点群から Manhattan world の水平主軸を推定する。"""

from __future__ import annotations

import numpy as np


def estimate_scene_yaw(
    analysis_points: np.ndarray,
    *,
    ground_y: float | None,
    ground_band: float,
    maximum_points: int = 20_000,
    candidate_lines: int = 2_048,
) -> dict:
    points = _uniform_sample(analysis_points, maximum_points)
    if ground_y is not None:
        points = points[np.abs(points[:, 1] - ground_y) > ground_band * 1.5]
    if len(points) < 500:
        return {"applied": False, "reason": "too_few_non_ground_points", "analysis_points": len(points)}

    low, high = np.percentile(points, [5.0, 95.0], axis=0)
    horizontal_span = float(np.linalg.norm(high[[0, 2]] - low[[0, 2]]))
    vertical_span = float(high[1] - low[1])
    if horizontal_span <= 1e-9 or vertical_span <= 1e-9:
        return {"applied": False, "reason": "collapsed_scene_extent", "analysis_points": len(points)}

    horizontal = points[:, [0, 2]]
    distance_threshold = horizontal_span * 0.004
    rng = np.random.default_rng(20260801)
    pairs = rng.integers(0, len(points), size=(candidate_lines, 2))
    delta = horizontal[pairs[:, 1]] - horizontal[pairs[:, 0]]
    lengths = np.linalg.norm(delta, axis=1)
    valid = lengths >= horizontal_span * 0.08
    pairs, delta, lengths = pairs[valid], delta[valid], lengths[valid]
    if not len(pairs):
        return {"applied": False, "reason": "no_plane_candidates", "analysis_points": len(points)}

    normals = np.column_stack((-delta[:, 1], delta[:, 0])) / lengths[:, np.newaxis]
    offsets = -np.einsum("ij,ij->i", normals, horizontal[pairs[:, 0]])
    flip = (normals[:, 0] < 0) | ((np.abs(normals[:, 0]) <= 1e-12) & (normals[:, 1] < 0))
    normals[flip] *= -1
    offsets[flip] *= -1

    top_candidates: list[tuple[int, int]] = []
    for start in range(0, len(normals), 128):
        stop = min(start + 128, len(normals))
        distances = np.abs(horizontal @ normals[start:stop].T + offsets[np.newaxis, start:stop])
        counts = np.count_nonzero(distances <= distance_threshold, axis=0)
        keep = min(8, len(counts))
        for local_index in np.argpartition(counts, -keep)[-keep:]:
            top_candidates.append((int(counts[local_index]), start + int(local_index)))

    minimum_support = max(80, round(len(points) * 0.02))
    candidates = []
    for support, index in sorted(top_candidates, reverse=True)[:128]:
        selected = np.abs(horizontal @ normals[index] + offsets[index]) <= distance_threshold
        if support < minimum_support:
            continue
        tangent = np.array([-normals[index, 1], normals[index, 0]])
        along = horizontal[selected] @ tangent
        candidate_vertical_span = float(np.ptp(np.percentile(points[selected, 1], [10.0, 90.0])))
        candidate_horizontal_span = float(np.ptp(np.percentile(along, [10.0, 90.0])))
        if candidate_vertical_span < vertical_span * 0.3:
            continue
        if candidate_horizontal_span < horizontal_span * 0.12:
            continue
        score = (
            support
            * min(candidate_vertical_span / vertical_span, 1.0)
            * min(candidate_horizontal_span / horizontal_span, 1.0)
        )
        candidates.append(
            {
                "score": float(score),
                "support_points": support,
                "normal_angle_rad": float(np.arctan2(normals[index, 1], normals[index, 0]) % np.pi),
                "offset": float(offsets[index]),
                "vertical_span": candidate_vertical_span,
                "horizontal_span": candidate_horizontal_span,
            }
        )

    planes = _deduplicate_planes(candidates, distance_threshold)
    pair = _best_orthogonal_pair(planes, low[[0, 2]], high[[0, 2]])
    if pair is None:
        return {
            "applied": False,
            "reason": "orthogonal_vertical_planes_unavailable",
            "analysis_points": len(points),
            "candidate_planes": len(planes),
        }

    first, second, residual, intersection = pair
    angles = np.asarray([first["normal_angle_rad"], second["normal_angle_rad"]])
    weights = np.asarray([first["score"], second["score"]])
    axis_vector = np.sum(weights * np.exp(4j * angles))
    axis_angle = float(np.angle(axis_vector) / 4.0)
    yaw = float((axis_angle + np.pi / 4.0) % (np.pi / 2.0) - np.pi / 4.0)
    confidence = float(abs(axis_vector) / np.sum(weights))
    return {
        "applied": bool(abs(yaw) >= np.deg2rad(0.25)),
        "method": "orthogonal_vertical_planes",
        "reason": "already_axis_aligned" if abs(yaw) < np.deg2rad(0.25) else None,
        "yaw_deg": float(np.degrees(yaw)),
        "yaw_radians": yaw,
        "confidence": confidence,
        "orthogonality_residual_deg": float(np.degrees(residual)),
        "analysis_points": len(points),
        "candidate_planes": len(planes),
        "planes": [_public_plane(first), _public_plane(second)],
        "wall_intersection": intersection.tolist(),
    }


def yaw_rotation_matrix(yaw_radians: float) -> np.ndarray:
    cosine = float(np.cos(yaw_radians))
    sine = float(np.sin(yaw_radians))
    return np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )


def _uniform_sample(points: np.ndarray, maximum_points: int) -> np.ndarray:
    if maximum_points <= 0:
        raise ValueError("maximum orientation points must be positive")
    if len(points) <= maximum_points:
        return points
    return points[np.linspace(0, len(points) - 1, maximum_points, dtype=np.int64)]


def _angle_distance(first: float, second: float) -> float:
    return float(abs((first - second + np.pi / 2.0) % np.pi - np.pi / 2.0))


def _deduplicate_planes(candidates: list[dict], distance_threshold: float) -> list[dict]:
    planes = []
    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        duplicate = any(
            _angle_distance(candidate["normal_angle_rad"], plane["normal_angle_rad"])
            < np.deg2rad(4.0)
            and abs(candidate["offset"] - plane["offset"]) < distance_threshold * 3.0
            for plane in planes
        )
        if duplicate:
            continue
        planes.append(candidate)
        if len(planes) == 16:
            break
    return planes


def _best_orthogonal_pair(
    planes: list[dict],
    horizontal_low: np.ndarray,
    horizontal_high: np.ndarray,
) -> tuple[dict, dict, float, np.ndarray] | None:
    pairs = []
    for index, first in enumerate(planes):
        for second in planes[:index]:
            separation = _angle_distance(first["normal_angle_rad"], second["normal_angle_rad"])
            residual = abs(separation - np.pi / 2.0)
            if residual > np.deg2rad(10.0):
                continue
            matrix = np.asarray(
                [
                    [np.cos(first["normal_angle_rad"]), np.sin(first["normal_angle_rad"])],
                    [np.cos(second["normal_angle_rad"]), np.sin(second["normal_angle_rad"])],
                ]
            )
            intersection = np.linalg.solve(matrix, -np.asarray([first["offset"], second["offset"]]))
            margin = (horizontal_high - horizontal_low) * 0.25
            if np.any(intersection < horizontal_low - margin) or np.any(intersection > horizontal_high + margin):
                continue
            balanced_score = min(first["score"], second["score"]) * 2.0 + max(
                first["score"], second["score"]
            ) * 0.25
            pairs.append(
                (
                    balanced_score * (1.0 - residual / np.deg2rad(20.0)),
                    first,
                    second,
                    residual,
                    intersection,
                )
            )
    if not pairs:
        return None
    _, first, second, residual, intersection = max(pairs, key=lambda item: item[0])
    return first, second, residual, intersection


def _public_plane(plane: dict) -> dict:
    return {
        "normal_angle_deg": float(np.degrees(plane["normal_angle_rad"])),
        "support_points": plane["support_points"],
        "vertical_span": plane["vertical_span"],
        "horizontal_span": plane["horizontal_span"],
        "offset_model_units": plane["offset"],
    }
