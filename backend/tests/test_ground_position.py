"""広域な水平 point mode だけを地面として採用することを検証する。"""

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
            f"frame_{index:06d}.jpg",
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
    result = estimate_ground_position(_reconstruction(broad_ground=True))

    assert result["applied"] is True
    assert result["method"] == "gravity_constrained_point_mode"
    assert abs(result["ground_y"] - 5.0) < 0.03
    assert abs(result["translation"][1] + 5.0) < 0.03
    assert result["camera_height_median_m"] > 3.0
    assert result["horizontal_span_ratio"] > 0.8


def test_small_horizontal_patch_is_not_accepted_as_ground():
    result = estimate_ground_position(_reconstruction(broad_ground=False))

    assert result["applied"] is False
    assert result["reason"] == "no_broad_horizontal_ground_mode"
