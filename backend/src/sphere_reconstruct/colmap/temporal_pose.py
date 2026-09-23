"""Use local video motion to nominate poses for geometric re-localization."""

from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np


def local_pose_diagnostics(samples: list[dict], *, radius: int = 8) -> list[dict]:
    """Predictions are diagnostics only, never replacement camera poses."""
    if radius < 3:
        raise ValueError("trajectory radius must be at least three selected captures")
    if len(samples) < 7:
        return []
    ranks = np.asarray([sample["rank"] for sample in samples], dtype=int)
    times = np.asarray([sample["timestamp_sec"] for sample in samples], dtype=float)
    centers = np.asarray([sample["center"] for sample in samples], dtype=float)
    if (not np.isfinite(times).all() or not np.isfinite(centers).all()
            or np.any(np.diff(ranks) <= 0) or np.any(np.diff(times) <= 0)):
        raise ValueError("video samples require finite centers and increasing selected ranks/timestamps")
    consecutive = np.diff(ranks) == 1
    distances = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    if not consecutive.any():
        return []
    scene_step = float(np.median(distances[consecutive]))
    if scene_step <= 1e-12:
        return []
    results = []
    for index, sample in enumerate(samples):
        nearby = np.flatnonzero((abs(ranks - ranks[index]) <= radius) & (ranks != ranks[index]))
        if (len(nearby) < 6 or (ranks[nearby] < ranks[index]).sum() < 2
                or (ranks[nearby] > ranks[index]).sum() < 2):
            continue
        intervals = np.flatnonzero(
            consecutive & (ranks[:-1] >= ranks[index] - radius) & (ranks[1:] <= ranks[index] + radius)
        )
        if len(intervals) < 4:
            continue
        local_dt = float(np.median(np.diff(times)[intervals]))
        step = max(float(np.median(distances[intervals] / np.diff(times)[intervals])) * local_dt,
                   scene_step * 0.25)
        offsets = times[nearby] - times[index]
        xyz = centers[nearby]
        hypotheses = []
        for left, right in combinations(range(len(nearby)), 2):
            if offsets[left] >= 0 or offsets[right] <= 0:
                continue
            velocity = (xyz[right] - xyz[left]) / (offsets[right] - offsets[left])
            origin = xyz[left] - offsets[left] * velocity
            residual = np.linalg.norm(xyz - origin - offsets[:, None] * velocity, axis=1)
            hypotheses.append((float(np.median(residual)), origin, velocity, residual))
        median, _, _, residual = min(hypotheses, key=lambda item: item[0])
        inliers = residual <= max(3 * median, step * 0.5)
        design = np.column_stack([np.ones(inliers.sum()), offsets[inliers]])
        origin, velocity = np.linalg.lstsq(design, xyz[inliers], rcond=None)[0]
        competing = any(
            score <= max(median * 1.1, step * 0.1) and np.linalg.norm(other - origin) > 2 * step
            for score, other, _, _ in hypotheses
        )
        residual = np.linalg.norm(xyz - origin - offsets[:, None] * velocity, axis=1)
        scatter = float(np.median(residual))
        threshold = max(3 * step, 6 * scatter)
        innovation = float(np.linalg.norm(centers[index] - origin))
        trustworthy = nearby[residual <= max(step, 3 * scatter)]
        results.append({
            "image_id": sample["image_id"], "capture": sample["capture"], "rank": sample["rank"],
            "predicted_center": origin.tolist(), "innovation": innovation,
            "local_step": step, "fit_scatter": scatter, "threshold": threshold,
            "suspect": innovation > threshold,
            "ambiguous_motion": competing,
            "neighbor_ids": [samples[int(other)]["image_id"] for other in trustworthy],
        })
    return results


def unique_correspondences(votes: dict[tuple[int, int], set[int]]) -> list[tuple[int, int]]:
    """Resolve one query feature to one 3D point; ambiguous ties stay unused."""
    grouped: dict[int, list[tuple[int, int]]] = {}
    for (query_index, point_id), sources in votes.items():
        grouped.setdefault(query_index, []).append((len(sources), point_id))
    selected = []
    for query_index, candidates in sorted(grouped.items()):
        candidates.sort(reverse=True)
        if len(candidates) == 1 or candidates[0][0] > candidates[1][0]:
            selected.append((query_index, candidates[0][1]))
    by_point: dict[int, list[tuple[int, int]]] = {}
    for query_index, point_id in selected:
        by_point.setdefault(point_id, []).append((len(votes[query_index, point_id]), query_index))
    result = []
    for point_id, candidates in by_point.items():
        candidates.sort(reverse=True)
        if len(candidates) == 1 or candidates[0][0] > candidates[1][0]:
            result.append((candidates[0][1], point_id))
    return sorted(result)


def reprojection_errors(camera, pose, pixels: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    camera_xyz = pose * xyz
    projected = camera.img_from_cam(camera_xyz)
    errors = np.linalg.norm(projected - pixels, axis=1)
    errors[(~np.isfinite(errors)) | (camera_xyz[:, 2] <= 0)] = np.inf
    return errors


def validated_anchor_support(errors: np.ndarray, supporters: list[set[int]]) -> dict[int, int]:
    if len(errors) != len(supporters):
        raise ValueError("each correspondence requires its supporting anchor IDs")
    counts: Counter[int] = Counter()
    for error, image_ids in zip(errors, supporters, strict=True):
        if error <= 4.0:
            counts.update(image_ids)
    return {image_id: count for image_id, count in counts.items() if count >= 10}


def bidirectional_pose(camera, original, left: tuple, right: tuple, diagnostic: dict) -> tuple:
    """Accept only two-sided geometric evidence, with fixed camera intrinsics."""
    import pycolmap

    report = {"accepted": False, "left_correspondences": len(left[0]), "right_correspondences": len(right[0])}
    if min(len(left[0]), len(right[0])) < 30:
        return None, {**report, "reason": "insufficient_two_sided_correspondences"}
    if diagnostic["ambiguous_motion"]:
        return None, {**report, "reason": "ambiguous_local_motion"}
    estimation = {"estimate_focal_length": False, "ransac": {
        "max_error": 4.0, "min_inlier_ratio": 0.1, "min_num_trials": 200,
        "max_num_trials": 10000, "random_seed": 0,
    }}
    refinement = {"refine_focal_length": False, "refine_extra_params": False, "use_position_prior": False}
    solutions = [pycolmap.estimate_and_refine_absolute_pose(
        pixels, xyz, camera, estimation, refinement,
    ) for pixels, xyz in (left, right)]
    if any(solution is None for solution in solutions):
        return None, {**report, "reason": "one_sided_pnp_failed"}
    poses = [solution["cam_from_world"] for solution in solutions]

    def score(pose, correspondences):
        pixels, xyz = correspondences
        errors = reprojection_errors(camera, pose, pixels, xyz)
        mask = errors <= 4.0
        selected = pixels[mask]
        coverage = (np.ptp(selected, axis=0) / [camera.width, camera.height]).tolist() if len(selected) else [0, 0]
        return {"inliers": int(mask.sum()), "ratio": float(mask.mean()), "coverage": coverage,
                "clipped_error": float(np.minimum(errors, 4).mean())}

    def supported(value):
        return value["inliers"] >= 20 and value["ratio"] >= 0.25 and min(value["coverage"]) >= 0.15

    cross = [score(poses[0], right), score(poses[1], left)]
    report["cross_validation"] = cross
    if not all(supported(value) for value in cross):
        return None, {**report, "reason": "two_sided_cross_validation_failed"}
    centers = [pose.inverse().translation for pose in poses]
    distance = float(np.linalg.norm(centers[0] - centers[1]))
    relative_rotation = poses[0].rotation.matrix() @ poses[1].rotation.matrix().T
    angle = float(np.degrees(np.arccos(np.clip((np.trace(relative_rotation) - 1) / 2, -1, 1))))
    report.update(two_sided_center_disagreement=distance, two_sided_rotation_degrees=angle)
    if not np.isfinite(distance + angle) or distance > diagnostic["local_step"] or angle > 3:
        return None, {**report, "reason": "two_sided_poses_disagree"}
    # Merge shared query features before the final fit, so each feature votes once.
    pixels = np.concatenate([left[0], right[0]])
    xyz = np.concatenate([left[1], right[1]])
    unique_pixels, first, inverse = np.unique(pixels, axis=0, return_index=True, return_inverse=True)
    conflicting = np.any(xyz != xyz[first[inverse]], axis=1)
    usable = np.ones(len(unique_pixels), dtype=bool)
    usable[inverse[conflicting]] = False
    indices = first[usable]
    _, distinct_points = np.unique(xyz[indices], axis=0, return_index=True)
    indices = indices[distinct_points]
    if len(indices) < 30:
        return None, {**report, "reason": "conflicting_cross_side_correspondences"}
    pixels, xyz = pixels[indices], xyz[indices]
    consensus = (reprojection_errors(camera, poses[0], pixels, xyz) <= 4) & (
        reprojection_errors(camera, poses[1], pixels, xyz) <= 4
    )
    if int(consensus.sum()) < 30:
        return None, {**report, "reason": "insufficient_joint_consensus"}
    validated_rotations = [value.rotation.matrix().copy() for value in poses]
    solution = pycolmap.refine_absolute_pose(poses[0], pixels, xyz, consensus, camera, refinement)
    if solution is None:
        return None, {**report, "reason": "joint_pnp_failed"}
    pose = solution["cam_from_world"]
    for center, rotation in zip(centers, validated_rotations, strict=True):
        position_delta = float(np.linalg.norm(pose.inverse().translation - center))
        relative_rotation = pose.rotation.matrix() @ rotation.T
        rotation_delta = float(np.degrees(np.arccos(np.clip((np.trace(relative_rotation) - 1) / 2, -1, 1))))
        if (not np.isfinite(position_delta + rotation_delta)
                or position_delta > diagnostic["local_step"] or rotation_delta > 3):
            return None, {**report, "reason": "joint_refinement_left_validated_pose"}
    scores = [score(pose, correspondences) for correspondences in (left, right)]
    previous = [score(original, correspondences) for correspondences in (left, right)]
    report.update(candidate_scores=scores, original_scores=previous)
    if not all(supported(value) for value in scores):
        return None, {**report, "reason": "joint_pnp_lost_support"}
    new_loss = sum(value["clipped_error"] for value in scores)
    old_loss = sum(value["clipped_error"] for value in previous)
    if new_loss >= old_loss * 0.7:
        return None, {**report, "reason": "no_clear_geometric_improvement"}
    innovation = float(np.linalg.norm(pose.inverse().translation - diagnostic["predicted_center"]))
    report["candidate_innovation"] = innovation
    if (not np.isfinite(innovation) or innovation > diagnostic["innovation"] * 0.35
            or innovation > max(1.5 * diagnostic["local_step"], 3 * diagnostic["fit_scatter"])):
        return None, {**report, "reason": "candidate_outside_supported_motion"}
    return pose, {**report, "accepted": True, "reason": "two_sided_geometric_relocalization"}
