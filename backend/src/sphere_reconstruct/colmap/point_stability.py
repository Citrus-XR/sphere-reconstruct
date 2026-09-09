"""固定 camera の全 track / 分割 track による既存 sparse point の安定性評価。"""

from __future__ import annotations

import numpy as np

from ..imaging.camera_geometry import SUPPORTED_MODELS, camera_rays_to_pixels, pixels_to_camera_rays
from . import model


def triangulate_rays(centers, directions):
    centers, directions = np.asarray(centers), np.asarray(directions)
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(directions)):
        return None
    origin = centers.mean(axis=0)
    offsets = centers - origin
    projection = np.eye(3)[None] - directions[:, :, None] * directions[:, None, :]
    weights = np.ones(len(centers))
    for _ in range(4):
        matrix = np.einsum("n,nij->ij", weights, projection)
        eigenvalues = np.linalg.eigvalsh(matrix)
        if eigenvalues[0] <= eigenvalues[-1] * 1e-10:
            return None
        rhs = np.einsum("n,nij,nj->i", weights, projection, offsets)
        point = np.linalg.solve(matrix, rhs)
        distances = np.linalg.norm(point - offsets, axis=1)
        if np.any(distances <= 1e-12):
            return None
        weights = 1.0 / np.square(distances)
        weights /= weights.max()
    result = point + origin
    if np.any(np.sum((result - centers) * directions, axis=1) <= 0):
        return None
    return result


def geometry(reconstruction):
    unsupported = {camera.model for camera in reconstruction.cameras.values()} - SUPPORTED_MODELS
    if unsupported:
        raise ValueError(f"point stability does not support camera models: {sorted(unsupported)}")
    for camera in reconstruction.cameras.values():
        if not np.all(np.isfinite(camera.params)) or min(camera.width, camera.height) <= 0:
            raise ValueError(f"invalid camera calibration: {camera.camera_id}")
    return {
        image.image_id: {
            "image": image,
            "rotation": np.asarray(model.qvec_to_rotation(image.qvec)),
            "center": np.asarray(image.camera_center),
            "camera": reconstruction.cameras[image.camera_id],
        }
        for image in reconstruction.images.values()
    }


METRIC_COLUMNS = [
    "captures",
    "split_relative_difference",
    "original_relative_difference",
    "cross_p95_px",
    "cross_max_px",
    "conditional_radius95_relative",
    "original_p95_px",
    "original_max_px",
]
FULL_TRACK_COLUMNS = [
    "captures",
    "full_fit_relative_difference",
    "loo_relative_difference",
    "cross_p95_px",
    "cross_max_px",
    "conditional_radius95_relative",
    "original_p95_px",
    "original_max_px",
]


def project_and_jacobian(xyz, centers, rotations, cameras):
    local = np.einsum("nij,nj->ni", rotations, np.asarray(xyz, dtype=np.float64) - centers)
    if not np.all(np.isfinite(local)) or np.any(local[:, 2] <= 0):
        return None
    pixels = np.empty((len(local), 2))
    jacobian = np.empty((len(local), 2, 3))
    for camera_id in sorted({camera.camera_id for camera in cameras}):
        indices = np.array([i for i, camera in enumerate(cameras) if camera.camera_id == camera_id])
        camera = cameras[indices[0]]
        points = local[indices]
        step = np.linalg.norm(points, axis=1) * 1e-5
        if np.any(step <= 0):
            return None
        pixels[indices] = camera_rays_to_pixels(camera, points)
        for axis in range(3):
            offset = np.zeros_like(points)
            offset[:, axis] = step
            plus = camera_rays_to_pixels(camera, points + offset)
            minus = camera_rays_to_pixels(camera, points - offset)
            jacobian[indices, :, axis] = (plus - minus) / (2 * step[:, None])
    world_jacobian = np.einsum("nij,njk->nik", jacobian, rotations)
    if not np.all(np.isfinite(pixels)) or not np.all(np.isfinite(world_jacobian)):
        return None
    return pixels, world_jacobian


def relative_radius95(jacobian, sigma_px, distance):
    singular = np.linalg.svd(jacobian.reshape((-1, 3)), compute_uv=False)
    if singular[-1] <= singular[0] * 1e-10:
        return float("inf")
    # sqrt(chi2.ppf(0.95, 3)); 固定 camera・独立等方 pixel noise の線形近似。
    return float(2.7954834829151074 * sigma_px / singular[-1] / distance)


def assess_full_track(
    point,
    centers,
    rotations,
    cameras,
    pixels,
    directions,
    keys,
    captures,
    distance,
    original_projection,
    values,
    *,
    relative_budget,
    pixel_sigma,
    cross_limit,
):
    values[5] = relative_radius95(original_projection[1], pixel_sigma, distance)
    if values[5] > relative_budget:
        return "conditional_uncertainty", values
    fit = triangulate_rays(centers, directions)
    if fit is None:
        return "degenerate_full_track", values
    values[1] = np.linalg.norm(fit - point.xyz) / distance
    if values[1] > relative_budget:
        return "full_fit_disagreement", values
    errors, differences = [], []
    for capture in captures:
        heldout = np.array([key == capture for key in keys])
        fit = triangulate_rays(centers[~heldout], directions[~heldout])
        if fit is None:
            return "degenerate_leave_one_out", values
        differences.append(float(np.linalg.norm(fit - point.xyz) / distance))
        local = np.einsum("nij,nj->ni", rotations[heldout], fit - centers[heldout])
        if not np.all(np.isfinite(local)) or np.any(local[:, 2] <= 0):
            return "invalid_leave_one_out_projection", values
        selected_cameras = [camera for camera, selected in zip(cameras, heldout, strict=True) if selected]
        predicted = np.array(
            [
                camera_rays_to_pixels(camera, ray[None])[0]
                for camera, ray in zip(selected_cameras, local, strict=True)
            ]
        )
        errors.extend(np.linalg.norm(predicted - pixels[heldout], axis=1).tolist())
    values[2:5] = [max(differences), np.percentile(errors, 95), max(errors)]
    if not np.all(np.isfinite(values)):
        return "invalid_leave_one_out_projection", values
    if values[3] > cross_limit or values[4] > 2 * cross_limit:
        return "cross_reprojection", values
    return "keep", values


def assess_point(point, views, records, *, relative_budget, pixel_sigma, cross_limit, policy="split"):
    if policy not in {"split", "full_track"}:
        raise ValueError(f"unknown assessment policy: {policy}")
    values = np.full(len(METRIC_COLUMNS), np.nan)
    observations = sorted(
        point.track,
        key=lambda item: (
            records[views[item[0]]["image"].name]["source_id"],
            records[views[item[0]]["image"].name]["capture_index"],
        ),
    )
    keys = [
        (records[views[i]["image"].name]["source_id"], records[views[i]["image"].name]["capture_index"])
        for i, _ in observations
    ]
    captures = list(dict.fromkeys(keys))
    values[0] = len(captures)
    if len(captures) < (4 if policy == "split" else 3):
        return "insufficient_captures", values
    centers = np.array([views[i]["center"] for i, _ in observations])
    rotations = np.array([views[i]["rotation"] for i, _ in observations])
    cameras = [views[i]["camera"] for i, _ in observations]
    pixels = np.array(
        [[views[i]["image"].points2D[j].x, views[i]["image"].points2D[j].y] for i, j in observations]
    )
    directions = np.array(
        [
            pixels_to_camera_rays(camera, pixel[None])[0] @ rotation
            for camera, pixel, rotation in zip(cameras, pixels, rotations, strict=True)
        ]
    )
    distance = float(np.median(np.linalg.norm(np.asarray(point.xyz) - centers, axis=1)))
    if not np.isfinite(distance) or distance <= 0:
        return "invalid_geometry", values
    original_projection = project_and_jacobian(point.xyz, centers, rotations, cameras)
    if original_projection is None:
        return "invalid_original_projection", values
    original_errors = np.linalg.norm(original_projection[0] - pixels, axis=1)
    values[6:] = [np.percentile(original_errors, 95), original_errors.max()]
    if values[6] > cross_limit or values[7] > 2 * cross_limit:
        return "original_reprojection", values
    if policy == "full_track":
        return assess_full_track(
            point,
            centers,
            rotations,
            cameras,
            pixels,
            directions,
            keys,
            captures,
            distance,
            original_projection,
            values,
            relative_budget=relative_budget,
            pixel_sigma=pixel_sigma,
            cross_limit=cross_limit,
        )
    if len({key[0] for key in captures}) != 1:
        capture_centers = np.array(
            [centers[[key == capture for key in keys]].mean(axis=0) for capture in captures]
        )
        centered = capture_centers - capture_centers.mean(axis=0)
        axis = np.linalg.svd(centered, full_matrices=False)[2][0]
        captures = [captures[i] for i in np.argsort(centered @ axis, kind="stable")]
    capture_order = {key: i for i, key in enumerate(captures)}
    order = np.array([capture_order[key] for key in keys])
    splits = [order < len(captures) // 2, order % 2 == 0]
    split_difference = original_difference = uncertainty = 0.0
    cross_errors = []
    for half in splits:
        fits = [triangulate_rays(centers[mask], directions[mask]) for mask in (half, ~half)]
        if any(fit is None for fit in fits):
            return "degenerate_split", values
        split_difference = max(split_difference, float(np.linalg.norm(fits[0] - fits[1]) / distance))
        for fit, mask in zip(fits, (half, ~half), strict=True):
            original_difference = max(original_difference, float(np.linalg.norm(fit - point.xyz) / distance))
            projection = project_and_jacobian(fit, centers, rotations, cameras)
            if projection is None:
                return "invalid_projection", values
            predicted, jacobian = projection
            cross_errors.extend(np.linalg.norm(predicted[~mask] - pixels[~mask], axis=1).tolist())
            uncertainty = max(uncertainty, relative_radius95(jacobian[mask], pixel_sigma, distance))
    values[1:6] = [
        split_difference,
        original_difference,
        np.percentile(cross_errors, 95),
        max(cross_errors),
        uncertainty,
    ]
    if split_difference > relative_budget or original_difference > relative_budget:
        return "split_disagreement", values
    if values[3] > cross_limit or values[4] > 2 * cross_limit:
        return "cross_reprojection", values
    if uncertainty > relative_budget:
        return "conditional_uncertainty", values
    return "keep", values


def retain_points(reconstruction, retained):
    removed = set(reconstruction.points3D) - retained
    reconstruction.points3D = {
        pid: point for pid, point in reconstruction.points3D.items() if pid in retained
    }
    for image in reconstruction.images.values():
        for observation in image.points2D:
            if observation.point3D_id in removed:
                observation.point3D_id = 2**64 - 1
    validate_tracks(reconstruction)


def validate_tracks(reconstruction):
    associations = {}
    for pid, point in reconstruction.points3D.items():
        if not np.all(np.isfinite(point.xyz)) or len(set(point.track)) != len(point.track):
            raise ValueError(f"invalid point geometry or duplicate track: {pid}")
        associations[pid] = set(point.track)
        for image_id, index in point.track:
            if image_id not in reconstruction.images or not 0 <= index < len(
                reconstruction.images[image_id].points2D
            ):
                raise ValueError(f"invalid point-to-image track index: {pid}:{image_id}:{index}")
            if reconstruction.images[image_id].points2D[index].point3D_id != pid:
                raise ValueError(f"point-to-image track mismatch: {pid}")
    for image_id, image in reconstruction.images.items():
        if image.camera_id not in reconstruction.cameras or not np.all(
            np.isfinite([*image.qvec, *image.tvec])
        ):
            raise ValueError(f"invalid image camera/pose: {image_id}")
        for index, observation in enumerate(image.points2D):
            pid = observation.point3D_id
            if pid in {-1, 2**64 - 1}:
                continue
            if not np.all(np.isfinite([observation.x, observation.y])):
                raise ValueError(f"invalid observation pixel: {image_id}:{index}")
            if pid not in associations or (image_id, index) not in associations[pid]:
                raise ValueError(f"image-to-point track mismatch: {image_id}:{index}")
