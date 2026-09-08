"""重力整列後の bounded Manhattan yaw 推定を検証する。"""

from __future__ import annotations

import numpy as np

from sphere_reconstruct.colmap.scene_orientation import estimate_scene_yaw, yaw_rotation_matrix


def _rotated_room(yaw_deg: float) -> np.ndarray:
    rng = np.random.default_rng(31)
    count = 3000
    vertical = rng.uniform(-1.5, 1.5, count)
    wall_x_low = np.column_stack((np.full(count, -3.0), vertical, rng.uniform(-2.0, 2.0, count)))
    wall_x_high = np.column_stack((np.full(count, 3.0), vertical, rng.uniform(-2.0, 2.0, count)))
    wall_z_low = np.column_stack((rng.uniform(-3.0, 3.0, count), vertical, np.full(count, -2.0)))
    wall_z_high = np.column_stack((rng.uniform(-3.0, 3.0, count), vertical, np.full(count, 2.0)))
    floor = np.column_stack(
        (rng.uniform(-3.0, 3.0, count), np.full(count, 1.5), rng.uniform(-2.0, 2.0, count))
    )
    points = np.vstack((wall_x_low, wall_x_high, wall_z_low, wall_z_high, floor))
    points += rng.normal(0.0, 0.003, points.shape)
    rotation = yaw_rotation_matrix(-np.deg2rad(yaw_deg))
    return (rotation @ points.T).T


def test_orthogonal_room_walls_recover_manhattan_yaw():
    points = _rotated_room(27.0)

    result = estimate_scene_yaw(points, ground_y=1.5, ground_band=0.05)

    assert result["applied"] is True
    assert result["method"] == "orthogonal_vertical_planes"
    assert abs(result["yaw_deg"] - 27.0) < 2.0
    assert result["orthogonality_residual_deg"] < 2.0
    assert result["confidence"] > 0.98
    assert result["analysis_points"] <= 20_000


def test_unstructured_outdoor_points_do_not_force_yaw():
    rng = np.random.default_rng(32)
    points = rng.uniform((-20.0, -3.0, -20.0), (20.0, 5.0, 20.0), size=(30_000, 3))

    result = estimate_scene_yaw(points, ground_y=None, ground_band=0.1)

    assert result["applied"] is False
    assert result["reason"] == "orthogonal_vertical_planes_unavailable"
    assert result["analysis_points"] == 20_000
