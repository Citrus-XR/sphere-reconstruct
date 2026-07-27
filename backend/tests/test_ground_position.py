"""撮影経路近傍の広域 point mode だけを地面として採用することを検証する。"""

from __future__ import annotations

import numpy as np

from sphere_reconstruct.colmap.ground_position import estimate_ground_position
from sphere_reconstruct.colmap.model import Camera, Image, Point3D, Reconstruction


def _reconstruction(*, broad_ground: bool) -> Reconstruction:
    camera = Camera(1, "PINHOLE", 64, 64, [20, 20, 32, 32], 1)
    images = {
        index + 1: Image(
            index + 1,
            (1.0, 0.0, 0.0, 0.0),
            (-float(index), -1.7, 0.0),
            1,
            f"sources/primary/frame_{index:06d}.jpg",
        )
        for index in range(20)
    }
    rng = np.random.default_rng(7)
    count = 6000
    horizontal_scale = 20.0 if broad_ground else 0.1
    ground = np.column_stack(
        (
            rng.uniform(0.0, horizontal_scale, count),
            rng.normal(5.0, 0.03, count),
            rng.uniform(-2.0, 2.0 if broad_ground else -1.9, count),
        )
    )
    clutter = np.column_stack(
        (
            rng.uniform(0.0, 20.0, 2000),
            rng.uniform(-2.0, 10.0, 2000),
            rng.uniform(-2.0, 2.0, 2000),
        )
    )
    points = {
        index + 1: Point3D(index + 1, tuple(xyz), (128, 128, 128), 0.5)
        for index, xyz in enumerate(np.vstack((ground, clutter)))
    }
    return Reconstruction({1: camera}, images, points)


def test_broad_ground_mode_moves_ground_to_zero():
    result = estimate_ground_position(
        _reconstruction(broad_ground=True), reference_image_prefix="sources/primary/"
    )

    assert result["applied"] is True
    assert result["method"] == "trajectory_local_ground_mode"
    assert abs(result["ground_y"] - 5.0) < 0.03
    assert abs(result["translation"][1] + 5.0) < 0.03
    assert result["camera_height_median_m"] > 3.0
    assert result["horizontal_span_ratio"] > 0.8


def test_small_horizontal_patch_is_not_accepted_as_ground():
    result = estimate_ground_position(_reconstruction(broad_ground=False))

    assert result["applied"] is False
    assert result["reason"] == "no_trajectory_local_ground_mode"


def test_sloped_walking_surface_wins_over_distant_flat_plane():
    camera = Camera(1, "PINHOLE", 64, 64, [20, 20, 32, 32], 1)
    images = {}
    rng = np.random.default_rng(12)
    points = []
    for index in range(30):
        camera_y = index * 0.2
        images[index + 1] = Image(
            index + 1,
            (1.0, 0.0, 0.0, 0.0),
            (-float(index), -camera_y, 0.0),
            1,
            f"sources/primary/frame_{index:06d}.jpg",
        )
        points.extend(
            zip(
                rng.normal(index, 0.25, 240),
                rng.normal(camera_y + 1.8, 0.025, 240),
                rng.uniform(-1.5, 1.5, 240),
                strict=True,
            )
        )
    points.extend(
        zip(
            rng.uniform(0.0, 29.0, 10000),
            rng.normal(8.0, 0.02, 10000),
            rng.uniform(8.0, 14.0, 10000),
            strict=True,
        )
    )
    reconstruction = Reconstruction(
        {1: camera},
        images,
        {index + 1: Point3D(index + 1, tuple(xyz), (128, 128, 128), 0.5) for index, xyz in enumerate(points)},
    )

    result = estimate_ground_position(
        reconstruction, reference_image_prefix="sources/primary/", max_path_distance_m=4.0
    )

    assert result["applied"] is True
    assert abs(result["camera_height_median_m"] - 1.8) < 0.05
    assert abs(result["ground_y"] - 4.7) < 0.1
    assert result["trajectory_coverage_ratio"] > 0.9
    assert result["path_distance_median_m"] < 2.0


def test_missing_reference_source_is_reported():
    result = estimate_ground_position(
        _reconstruction(broad_ground=True), reference_image_prefix="sources/unknown/"
    )

    assert result == {"applied": False, "reason": "reference_trajectory_unavailable"}
