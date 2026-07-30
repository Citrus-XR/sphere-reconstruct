"""既知の rig sensor baseline から COLMAP world の meter scale を推定する。"""

from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np

from .gravity_align import _quat_to_R
from .model import Reconstruction


def estimate_metric_scale(
    reconstruction: Reconstruction,
    image_catalog: dict,
    rig_config: list[dict],
    *,
    min_pairs: int = 8,
    min_stereo_points: int = 8,
    max_relative_mad: float = 0.02,
) -> dict:
    sensors = _sensor_centers(rig_config)
    if len(sensors) < 2:
        return {"applied": False, "metric": False, "reason": "rig_has_no_physical_baseline"}

    registered = {image.name.replace("\\", "/"): image for image in reconstruction.images.values()}
    captures: dict[tuple[str, int], list[tuple[str, np.ndarray]]] = defaultdict(list)
    observation_context: dict[int, tuple[tuple[str, int], str]] = {}
    for record in image_catalog["images"]:
        name = str(record["name"]).replace("\\", "/")
        image = registered.get(name)
        if image is None:
            continue
        prefix = next((item for item in sensors if name.startswith(item)), None)
        if prefix is None:
            continue
        capture_key = (str(record["source_id"]), int(record["capture_index"]))
        captures[capture_key].append(
            (prefix, np.asarray(image.camera_center, dtype=float))
        )
        observation_context[image.image_id] = (capture_key, prefix)

    ratios = []
    physical_distances = []
    reconstructed_distances = []
    for images in captures.values():
        for (prefix_a, center_a), (prefix_b, center_b) in itertools.combinations(images, 2):
            expected = float(np.linalg.norm(sensors[prefix_a] - sensors[prefix_b]))
            if expected < 0.005:
                continue
            reconstructed = float(np.linalg.norm(center_a - center_b))
            if reconstructed <= 1e-12:
                continue
            ratios.append(reconstructed / expected)
            physical_distances.append(expected)
            reconstructed_distances.append(reconstructed)

    if len(ratios) < min_pairs:
        return {
            "applied": False,
            "metric": False,
            "reason": "too_few_registered_rig_baselines",
            "baseline_pairs": len(ratios),
        }

    stereo_points, stereo_captures = _stereo_observability(
        reconstruction, observation_context
    )
    if stereo_points < min_stereo_points or stereo_captures < min_pairs:
        return {
            "applied": False,
            "metric": False,
            "reason": "rig_baseline_not_observable",
            "baseline_pairs": len(ratios),
            "stereo_points": stereo_points,
            "stereo_captures": stereo_captures,
            "imu_scale_role": "validation_only",
            "imu_scale_reason": "accelerometer_double_integration_is_drift_unbounded",
        }
    ratios_array = np.asarray(ratios)
    median = float(np.median(ratios_array))
    if not np.isfinite(median) or median <= 1e-12:
        raise ValueError(f"invalid reconstructed rig scale: {median}")
    absolute_deviation = np.abs(ratios_array - median)
    mad = float(np.median(absolute_deviation))
    relative_mad = mad / median
    threshold = max(5.0 * mad, median * 0.005)
    inliers = absolute_deviation <= threshold
    if np.count_nonzero(inliers) < min_pairs or relative_mad > max_relative_mad:
        return {
            "applied": False,
            "metric": False,
            "reason": "inconsistent_rig_baseline_scale",
            "baseline_pairs": len(ratios),
            "relative_mad": relative_mad,
        }
    model_units_per_meter = float(np.median(ratios_array[inliers]))
    scale_factor = 1.0 / model_units_per_meter
    return {
        "applied": abs(scale_factor - 1.0) > 1e-9,
        "metric": True,
        "method": "rig_baseline",
        "scale_factor": scale_factor,
        "model_units_per_meter": model_units_per_meter,
        "baseline_pairs": len(ratios),
        "inlier_pairs": int(np.count_nonzero(inliers)),
        "relative_mad": relative_mad,
        "physical_baseline_median_m": float(np.median(np.asarray(physical_distances)[inliers])),
        "reconstructed_baseline_median": float(
            np.median(np.asarray(reconstructed_distances)[inliers])
        ),
        "stereo_points": stereo_points,
        "stereo_captures": stereo_captures,
        "imu_scale_role": "validation_only",
        "imu_scale_reason": "accelerometer_double_integration_is_drift_unbounded",
    }


def _stereo_observability(
    reconstruction: Reconstruction,
    observation_context: dict[int, tuple[tuple[str, int], str]],
) -> tuple[int, int]:
    stereo_points = 0
    constrained_captures: set[tuple[str, int]] = set()
    for point in reconstruction.points3D.values():
        sensors_by_capture: dict[tuple[str, int], set[str]] = defaultdict(set)
        for image_id, _ in point.track:
            context = observation_context.get(image_id)
            if context is not None:
                capture_key, sensor = context
                sensors_by_capture[capture_key].add(sensor)
        point_constrains_scale = False
        for capture_key, sensors_in_capture in sensors_by_capture.items():
            if len(sensors_in_capture) >= 2:
                constrained_captures.add(capture_key)
                point_constrains_scale = True
        if point_constrains_scale:
            stereo_points += 1
    return stereo_points, len(constrained_captures)


def _sensor_centers(rig_config: list[dict]) -> dict[str, np.ndarray]:
    centers: dict[str, np.ndarray] = {}
    for rig in rig_config:
        for camera in rig.get("cameras", []):
            prefix = str(camera["image_prefix"]).replace("\\", "/")
            rotation = np.asarray(
                _quat_to_R(camera.get("cam_from_rig_rotation", [1.0, 0.0, 0.0, 0.0]))
            )
            translation = np.asarray(
                camera.get("cam_from_rig_translation", [0.0, 0.0, 0.0]), dtype=float
            )
            centers[prefix] = -rotation.T @ translation
    return centers
