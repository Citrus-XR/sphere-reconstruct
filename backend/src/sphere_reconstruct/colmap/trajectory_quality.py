"""連続 video の capture 順序から camera trajectory の不連続を検出する。"""

from __future__ import annotations

import math
from collections import defaultdict

from .model import Reconstruction


def evaluate_primary_trajectory(
    reconstruction: Reconstruction,
    image_records: list[dict],
    primary_source_id: str,
    *,
    max_step_ratio: float,
    minimum_steps: int = 20,
    step_baseline: str = "p95",
) -> dict:
    if max_step_ratio <= 1.0:
        raise ValueError("max_step_ratio must be greater than one")
    if step_baseline not in {"p95", "median"}:
        raise ValueError(f"unsupported trajectory step baseline: {step_baseline}")
    record_by_name = {
        str(record["name"]).replace("\\", "/"): record
        for record in image_records
        if record["source_id"] == primary_source_id
    }
    expected_captures = {int(record["capture_index"]) for record in record_by_name.values()}
    centers_by_capture: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    for image in reconstruction.images.values():
        record = record_by_name.get(image.name.replace("\\", "/"))
        if record is not None:
            centers_by_capture[int(record["capture_index"])].append(image.camera_center)

    capture_centers = {
        capture: tuple(sum(center[axis] for center in centers) / len(centers) for axis in range(3))
        for capture, centers in centers_by_capture.items()
    }
    ordered_captures = sorted(capture_centers)
    steps = [
        {
            "from_capture": first,
            "to_capture": second,
            "distance": math.dist(capture_centers[first], capture_centers[second]),
        }
        for first, second in zip(ordered_captures, ordered_captures[1:], strict=False)
        if second == first + 1
    ]
    if len(steps) < minimum_steps:
        short_input = max(0, len(expected_captures) - 1) < minimum_steps
        return {
            "available": False,
            "passed": short_input,
            "reason": (
                "input_sequence_too_short_for_statistical_gate"
                if short_input
                else "too_few_consecutive_registered_captures"
            ),
            "expected_captures": len(expected_captures),
            "registered_captures": len(capture_centers),
            "consecutive_steps": len(steps),
            "outlier_captures": [],
        }

    distances = sorted(float(step["distance"]) for step in steps)
    median = _percentile(distances, 0.5)
    p95 = _percentile(distances, 0.95)
    maximum = distances[-1]
    if step_baseline == "median":
        deviations = sorted(abs(distance - median) for distance in distances)
        mad = _percentile(deviations, 0.5)
        reference = median
        threshold = max(median * max_step_ratio, median + 6.0 * mad, 1e-9)
    else:
        mad = None
        reference = p95
        threshold = max(p95 * max_step_ratio, 1e-9)
    outliers = [step for step in steps if step["distance"] > threshold]
    outlier_captures = _outlier_captures(outliers)
    largest = sorted(steps, key=lambda step: step["distance"], reverse=True)[:10]
    return {
        "available": True,
        "passed": not outliers,
        "expected_captures": len(expected_captures),
        "registered_captures": len(capture_centers),
        "consecutive_steps": len(steps),
        "median_step": median,
        "p95_step": p95,
        "step_mad": mad,
        "step_baseline": step_baseline,
        "step_reference": reference,
        "maximum_step": maximum,
        "maximum_to_p95_ratio": maximum / max(p95, 1e-12),
        "max_step_ratio_limit": max_step_ratio,
        "outlier_threshold": threshold,
        "outlier_steps": len(outliers),
        "outlier_captures": outlier_captures,
        "outlier_capture_pairs": [
            f"{step['from_capture']}->{step['to_capture']}: {step['distance']:.6g}"
            for step in sorted(outliers, key=lambda item: item["distance"], reverse=True)
        ],
        "largest_steps": largest,
    }


def evaluate_source_path_consistency(
    reconstruction: Reconstruction,
    image_records: list[dict],
    primary_source_id: str,
    supplemental_source_id: str,
    *,
    minimum_captures: int = 20,
    robust_sigma_limit: float = 8.0,
    primary_radius_ratio: float = 2.0,
    primary_step_ratio: float = 5.0,
) -> dict:
    """Flag supplemental poses that sit well outside the primary camera path.

    The robust residual threshold adapts to normal source-to-source path separation.
    The primary-path envelope prevents a uniformly displaced supplemental track from
    inflating its own threshold and passing unchanged.
    """
    if minimum_captures < 2:
        raise ValueError("minimum_captures must be at least two")
    if robust_sigma_limit <= 0 or primary_radius_ratio <= 0 or primary_step_ratio <= 0:
        raise ValueError("source path consistency limits must be positive")

    primary = _capture_centers(reconstruction, image_records, primary_source_id)
    supplemental = _capture_centers(reconstruction, image_records, supplemental_source_id)
    if len(primary) < minimum_captures or len(supplemental) < minimum_captures:
        return {
            "available": False,
            "passed": True,
            "reason": "too_few_registered_captures",
            "primary_captures": len(primary),
            "supplemental_captures": len(supplemental),
            "outlier_captures": [],
        }

    primary_centers = [primary[capture] for capture in sorted(primary)]
    center = tuple(_percentile(sorted(point[axis] for point in primary_centers), 0.5) for axis in range(3))
    radii = sorted(math.dist(point, center) for point in primary_centers)
    primary_radius_p95 = _percentile(radii, 0.95)
    primary_steps = sorted(
        math.dist(primary[first], primary[second])
        for first, second in zip(sorted(primary), sorted(primary)[1:], strict=False)
        if second == first + 1
    )
    primary_step_p95 = _percentile(primary_steps, 0.95) if primary_steps else 0.0

    nearest = {
        capture: min(math.dist(point, reference) for reference in primary_centers)
        for capture, point in supplemental.items()
    }
    distances = sorted(nearest.values())
    median_distance = _percentile(distances, 0.5)
    deviations = sorted(abs(distance - median_distance) for distance in distances)
    distance_mad = _percentile(deviations, 0.5)
    robust_threshold = median_distance + robust_sigma_limit * 1.4826 * distance_mad
    sampling_floor = primary_step_ratio * primary_step_p95
    path_envelope = primary_radius_ratio * primary_radius_p95
    threshold = max(robust_threshold, sampling_floor, 1e-9)
    if path_envelope > 0:
        threshold = min(threshold, path_envelope)
    outliers = sorted(capture for capture, distance in nearest.items() if distance > threshold)
    return {
        "available": True,
        "passed": not outliers,
        "primary_captures": len(primary),
        "supplemental_captures": len(supplemental),
        "primary_radius_p95": primary_radius_p95,
        "primary_step_p95": primary_step_p95,
        "median_nearest_distance": median_distance,
        "nearest_distance_mad": distance_mad,
        "robust_sigma_limit": robust_sigma_limit,
        "robust_threshold": robust_threshold,
        "sampling_floor": sampling_floor,
        "primary_path_envelope": path_envelope,
        "maximum_allowed_distance": threshold,
        "p95_nearest_distance": _percentile(distances, 0.95),
        "maximum_nearest_distance": distances[-1],
        "outlier_captures": outliers,
        "outlier_count": len(outliers),
    }


def _capture_centers(
    reconstruction: Reconstruction,
    image_records: list[dict],
    source_id: str,
) -> dict[int, tuple[float, float, float]]:
    record_by_name = {
        str(record["name"]).replace("\\", "/"): record
        for record in image_records
        if record["source_id"] == source_id
    }
    centers_by_capture: defaultdict[int, list[tuple[float, float, float]]] = defaultdict(list)
    for image in reconstruction.images.values():
        record = record_by_name.get(image.name.replace("\\", "/"))
        if record is not None:
            centers_by_capture[int(record["capture_index"])].append(image.camera_center)
    return {
        capture: tuple(
            sum(center[axis] for center in centers) / len(centers)
            for axis in range(3)
        )
        for capture, centers in centers_by_capture.items()
    }


def _outlier_captures(outlier_steps: list[dict]) -> list[int]:
    """Choose captures that are themselves discontinuous, rather than both jump endpoints.

    A single bad pose normally creates two large edges (previous -> bad -> next).  The
    shared capture is therefore the safest deletion candidate.  For a one-sided jump,
    the destination is the first pose on the inconsistent branch.
    """
    if not outlier_steps:
        return []
    edges = {(int(step["from_capture"]), int(step["to_capture"])) for step in outlier_steps}
    incident: defaultdict[int, int] = defaultdict(int)
    for first, second in edges:
        incident[first] += 1
        incident[second] += 1
    selected: set[int] = set()
    for first, second in edges:
        if incident[first] >= 2 and incident[second] < incident[first]:
            selected.add(first)
        elif incident[second] >= 2 and incident[first] < incident[second]:
            selected.add(second)
        elif incident[first] >= 2 and incident[second] >= 2:
            selected.add(first if incident[first] > incident[second] else second)
        else:
            selected.add(second)
    return sorted(selected)


def _percentile(sorted_values: list[float], fraction: float) -> float:
    position = fraction * (len(sorted_values) - 1)
    low = int(position)
    high = min(len(sorted_values) - 1, low + 1)
    weight = position - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight
