from __future__ import annotations

import math

import numpy as np
import pytest

from sphere_reconstruct.colmap.temporal_pose import (
    bidirectional_pose,
    local_pose_diagnostics,
    unique_correspondences,
    validated_anchor_support,
)


def _samples():
    return [{"image_id": i + 1, "capture": 400 + i, "rank": i,
             "timestamp_sec": i * 0.4, "center": [i * 0.2, 0.0, 0.0]} for i in range(60)]


def test_repeated_alternating_wrong_branch_does_not_blame_correct_intermediate_frames():
    samples = _samples()
    for index in [15, 17, 30, 32, 45]:
        samples[index]["center"][1] += 1.6
    result = local_pose_diagnostics(samples)
    assert {row["rank"] for row in result if row["suspect"]} == {15, 17, 30, 32, 45}


def test_motion_speed_uses_timestamps_and_does_not_interpolate_missing_captures():
    samples = _samples()
    for sample in samples:
        sample["timestamp_sec"] = sample["rank"] ** 2 * 0.1
        sample["center"][0] = sample["timestamp_sec"] * 0.2
    samples = [sample for sample in samples if sample["rank"] not in {12, 13, 34}]
    assert not any(row["suspect"] for row in local_pose_diagnostics(samples))


def test_smooth_camera_turn_is_not_a_spike():
    samples = _samples()
    for sample in samples:
        angle = sample["rank"] * 0.06
        sample["center"] = [3 * math.sin(angle), 0, 3 * math.cos(angle)]
    assert not any(row["suspect"] for row in local_pose_diagnostics(samples))


def test_feature_vote_ties_cannot_inflate_pnp_support():
    votes = {(0, 10): {2, 3}, (0, 11): {2}, (1, 20): {2}, (1, 21): {3}, (2, 30): {3}}
    assert unique_correspondences(votes) == [(0, 10), (2, 30)]


def test_duplicate_three_dimensional_points_cannot_inflate_pnp_support():
    assert unique_correspondences({(0, 10): {2}, (1, 10): {3}}) == []


def test_anchor_validation_counts_only_distinct_surviving_inliers():
    supporters = [{10, 11}] * 10 + [{12}] * 9 + [{13}] * 20
    errors = np.asarray([0.5] * 19 + [20] * 20)
    assert validated_anchor_support(errors, supporters) == {10: 10, 11: 10}


def test_two_sided_pnp_recovers_pose_from_pixels_without_a_position_prior():
    pycolmap = pytest.importorskip("pycolmap")
    camera = pycolmap.Camera(model="SIMPLE_RADIAL", width=1000, height=1000, params=[700, 500, 500, 0.01])
    rng = np.random.default_rng(4)
    xyz = rng.uniform([-2, -2, 4], [2, 2, 8], (160, 3))
    correct = pycolmap.Rigid3d()
    wrong = pycolmap.Rigid3d(pycolmap.Rotation3d(), [1.6, 0, 0])
    pixels = camera.img_from_cam(xyz) + rng.normal(0, 0.2, (160, 2))
    diagnostic = {"predicted_center": [0, 0, 0], "local_step": 0.2, "fit_scatter": 0.01,
                  "innovation": 1.6, "ambiguous_motion": False}
    result, report = bidirectional_pose(camera, wrong, (pixels[:80], xyz[:80]), (pixels[80:], xyz[80:]), diagnostic)
    assert report["accepted"] is True
    assert np.linalg.norm(result.inverse().translation - correct.inverse().translation) < 0.02
    np.testing.assert_array_equal(camera.params, [700, 500, 500, 0.01])

    result, report = bidirectional_pose(camera, correct, (pixels[:80], xyz[:80]), (pixels[80:], xyz[80:]), diagnostic)
    assert result is None
    assert report["reason"] == "no_clear_geometric_improvement"


def test_two_incompatible_neighbor_modes_are_not_blended():
    pycolmap = pytest.importorskip("pycolmap")
    camera = pycolmap.Camera(model="SIMPLE_RADIAL", width=1000, height=1000, params=[700, 500, 500, 0])
    rng = np.random.default_rng(5)
    xyz = rng.uniform([-2, -2, 4], [2, 2, 8], (100, 3))
    left = camera.img_from_cam(xyz)
    right = camera.img_from_cam(xyz + [1.6, 0, 0])
    diagnostic = {"predicted_center": [0, 0, 0], "local_step": 0.2, "fit_scatter": 0.01,
                  "innovation": 1.6, "ambiguous_motion": False}
    result, report = bidirectional_pose(camera, pycolmap.Rigid3d(), (left, xyz), (right, xyz), diagnostic)
    assert result is None
    assert report["reason"] == "two_sided_cross_validation_failed"
