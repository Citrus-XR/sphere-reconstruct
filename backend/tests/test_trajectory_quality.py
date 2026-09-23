"""連続 capture の異常な camera-center jump を quality gate で検出する。"""

from __future__ import annotations

import math

from sphere_reconstruct.colmap.model import Camera, Image, Reconstruction
from sphere_reconstruct.colmap.trajectory_quality import (
    evaluate_primary_trajectory,
    evaluate_source_path_consistency,
)


def _sequence(*, jump_at: int | None = None) -> tuple[Reconstruction, list[dict]]:
    camera = Camera(1, "PINHOLE", 64, 64, [20, 20, 32, 32], 1)
    images = {}
    records = []
    offset = 0.0
    for capture in range(50):
        if capture == jump_at:
            offset += 40.0
        for sensor_index, sensor_offset in enumerate((0.0, 0.032)):
            image_id = capture * 2 + sensor_index + 1
            name = f"sources/primary/sensor_{sensor_index}/frame_{capture:06d}.jpg"
            center = (capture * 0.8 + offset, 1.7, sensor_offset)
            images[image_id] = Image(
                image_id,
                (1.0, 0.0, 0.0, 0.0),
                tuple(-value for value in center),
                1,
                name,
            )
            records.append({"name": name, "source_id": "primary", "capture_index": capture})
    return Reconstruction({1: camera}, images, {}), records


def test_smooth_rig_trajectory_passes():
    reconstruction, records = _sequence()

    result = evaluate_primary_trajectory(reconstruction, records, "primary", max_step_ratio=10.0)

    assert result["available"] is True
    assert result["passed"] is True
    assert result["outlier_steps"] == 0
    assert abs(result["median_step"] - 0.8) < 1e-9


def test_piecewise_translation_jump_fails_with_capture_ids():
    reconstruction, records = _sequence(jump_at=25)

    result = evaluate_primary_trajectory(reconstruction, records, "primary", max_step_ratio=10.0)

    assert result["passed"] is False
    assert result["outlier_steps"] == 1
    assert result["maximum_to_p95_ratio"] > 40
    assert result["largest_steps"][0]["from_capture"] == 24
    assert result["largest_steps"][0]["to_capture"] == 25
    assert result["outlier_captures"] == [25]


def test_short_sequence_does_not_create_a_false_failure():
    reconstruction, records = _sequence()
    reconstruction.images = dict(list(reconstruction.images.items())[:10])
    records = records[:10]

    result = evaluate_primary_trajectory(reconstruction, records, "primary", max_step_ratio=10.0)

    assert result["available"] is False
    assert result["passed"] is True


def test_long_input_without_enough_consecutive_registration_fails():
    reconstruction, records = _sequence()
    reconstruction.images = dict(list(reconstruction.images.items())[:10])

    result = evaluate_primary_trajectory(reconstruction, records, "primary", max_step_ratio=10.0)

    assert result["available"] is False
    assert result["passed"] is False
    assert result["reason"] == "too_few_consecutive_registered_captures"


def test_median_step_baseline_is_not_inflated_by_repeated_teleports():
    camera = Camera(1, "PINHOLE", 64, 64, [20, 20, 32, 32], 1)
    bad_captures = {10, 20, 30, 40, 50, 60, 70}
    images = {}
    records = []
    for capture in range(100):
        image_id = capture + 1
        y = 100.0 if capture in bad_captures else 0.0
        center = (capture * 0.1, y, 0.0)
        name = f"phone/frame_{capture:06d}.jpg"
        images[image_id] = Image(
            image_id,
            (1.0, 0.0, 0.0, 0.0),
            tuple(-value for value in center),
            1,
            name,
        )
        records.append({"name": name, "source_id": "phone", "capture_index": capture})
    reconstruction = Reconstruction({1: camera}, images, {})

    result = evaluate_primary_trajectory(
        reconstruction,
        records,
        "phone",
        max_step_ratio=10.0,
        step_baseline="median",
    )

    assert result["passed"] is False
    assert result["p95_step"] > result["outlier_threshold"]
    assert result["outlier_captures"] == sorted(bad_captures)


def test_supplemental_path_outliers_are_measured_against_primary_360_path():
    camera = Camera(1, "PINHOLE", 64, 64, [20, 20, 32, 32], 1)
    images = {}
    records = []
    image_id = 1
    primary_centers = []
    for capture in range(40):
        angle = capture * 2 * math.pi / 40
        center = (2 * math.cos(angle), 1.0, 2 * math.sin(angle))
        primary_centers.append(center)
        name = f"360/frame_{capture:06d}.jpg"
        images[image_id] = Image(
            image_id, (1.0, 0.0, 0.0, 0.0), tuple(-value for value in center), 1, name
        )
        records.append({"name": name, "source_id": "primary", "capture_index": capture})
        image_id += 1
    for capture in range(30):
        center = list(primary_centers[capture])
        if capture in {10, 20}:
            center[1] += 100
        name = f"phone/frame_{capture:06d}.jpg"
        images[image_id] = Image(
            image_id, (1.0, 0.0, 0.0, 0.0), tuple(-value for value in center), 1, name
        )
        records.append({"name": name, "source_id": "phone", "capture_index": capture})
        image_id += 1
    reconstruction = Reconstruction({1: camera}, images, {})

    result = evaluate_source_path_consistency(
        reconstruction,
        records,
        "primary",
        "phone",
        minimum_captures=20,
    )

    assert result["available"] is True
    assert result["outlier_captures"] == [10, 20]
    assert result["maximum_nearest_distance"] > 90
