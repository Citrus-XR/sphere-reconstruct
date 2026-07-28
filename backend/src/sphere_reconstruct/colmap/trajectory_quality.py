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
) -> dict:
    if max_step_ratio <= 1.0:
        raise ValueError("max_step_ratio must be greater than one")
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
        }

    distances = sorted(float(step["distance"]) for step in steps)
    median = _percentile(distances, 0.5)
    p95 = _percentile(distances, 0.95)
    maximum = distances[-1]
    threshold = max(p95 * max_step_ratio, 1e-9)
    outliers = [step for step in steps if step["distance"] > threshold]
    largest = sorted(steps, key=lambda step: step["distance"], reverse=True)[:10]
    return {
        "available": True,
        "passed": not outliers,
        "expected_captures": len(expected_captures),
        "registered_captures": len(capture_centers),
        "consecutive_steps": len(steps),
        "median_step": median,
        "p95_step": p95,
        "maximum_step": maximum,
        "maximum_to_p95_ratio": maximum / max(p95, 1e-12),
        "max_step_ratio_limit": max_step_ratio,
        "outlier_threshold": threshold,
        "outlier_steps": len(outliers),
        "outlier_capture_pairs": [
            f"{step['from_capture']}->{step['to_capture']}: {step['distance']:.6g}"
            for step in sorted(outliers, key=lambda item: item["distance"], reverse=True)
        ],
        "largest_steps": largest,
    }


def _percentile(sorted_values: list[float], fraction: float) -> float:
    position = fraction * (len(sorted_values) - 1)
    low = int(position)
    high = min(len(sorted_values) - 1, low + 1)
    weight = position - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight
