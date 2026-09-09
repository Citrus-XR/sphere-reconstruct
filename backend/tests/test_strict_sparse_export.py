"""既存点の厳格除去と native fisheye 不確実性の検証。"""

import importlib.util
import sys
from pathlib import Path

import numpy as np


def load_script():
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("strict_sparse", scripts / "export_strict_sparse.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


def synthetic_track(script, positions, *, split_depth=None, focal_length=1000):
    model = script.model
    camera = model.Camera(1, "OPENCV_FISHEYE", 3840, 3840, [focal_length, focal_length, 1920, 1920, 0, 0, 0, 0])
    images, records = {}, {}
    for i, x in enumerate(positions):
        xyz = np.array([0., 0., 10. if split_depth is None or i < len(positions) // 2 else split_depth])
        uv = script.camera_rays_to_pixels(camera.model, camera.params, (xyz - [x, 0, 0])[None])[0]
        name = f"image{i}.png"
        images[i] = model.Image(i, (1, 0, 0, 0), (-x, 0, 0), 1, name,
                                [model.ImagePoint2D(*uv, 5)])
        records[name] = {"source_id": "source", "capture_index": i}
    point = model.Point3D(5, (0, 0, 10), (255, 120, 10), 0, [(i, 0) for i in images])
    reconstruction = model.Reconstruction({1: camera}, images, {5: point})
    return reconstruction, point, records


def test_strict_filter_distinguishes_stability_from_reprojection_fit():
    script = load_script()
    cases = [([-6, -4, -2, -1, 1, 2, 4, 6], None, "keep"),
             ([-1, 1], None, "insufficient_captures"),
             ([0.] * 8, None, "degenerate_split"),
             (np.arange(8) * .002, None, "conditional_uncertainty"),
             ([-.12, -.08, -.04, -.02, .02, .04, .08, .12], 12, "split_disagreement"),
             ([-6, -4, -2, -1, 1, 2, 4, 6], 12, "original_reprojection")]
    for positions, split_depth, expected in cases:
        reconstruction, point, records = synthetic_track(script, positions, split_depth=split_depth)
        reason, values = script.assess_point(point, script.geometry(reconstruction), records,
                                             relative_budget=.02, pixel_sigma=1, cross_limit=2)
        assert reason == expected, (expected, reason, values)


def test_original_position_must_pass_projection_even_when_split_fits_are_stable():
    script = load_script()
    reconstruction, point, records = synthetic_track(script, [-4, -2, 2, 4], focal_length=2000)
    reason, _ = script.assess_point(point, script.geometry(reconstruction), records,
                                    relative_budget=.02, pixel_sigma=1, cross_limit=2)
    assert reason == "keep"
    point.xyz = (.1, 0, 10)
    reason, values = script.assess_point(point, script.geometry(reconstruction), records,
                                        relative_budget=.02, pixel_sigma=1, cross_limit=2)
    assert reason == "original_reprojection"
    assert values[6] > 17


def test_conditional_radius_tracks_pixel_noise_and_camera_scale():
    script = load_script()
    reconstruction, point, _ = synthetic_track(script, [-6, -4, -2, -1, 1, 2, 4, 6])
    views = list(script.geometry(reconstruction).values())
    centers = np.array([view["center"] for view in views])
    rotations = np.array([view["rotation"] for view in views])
    cameras = [view["camera"] for view in views]
    _, jacobian = script.project_and_jacobian(point.xyz, centers, rotations, cameras)
    distance = np.median(np.linalg.norm(np.array(point.xyz) - centers, axis=1))
    radius = script.relative_radius95(jacobian, 1, distance)
    assert np.isfinite(radius) and radius > 0
    np.testing.assert_allclose(script.relative_radius95(jacobian, 2, distance), 2 * radius)
    pixels, scaled = script.project_and_jacobian(np.array(point.xyz) * 7, centers * 7, rotations, cameras)
    np.testing.assert_allclose(script.relative_radius95(scaled, 1, distance * 7), radius, rtol=1e-8)
    expected = [[view["image"].points2D[0].x, view["image"].points2D[0].y] for view in views]
    np.testing.assert_allclose(pixels, expected, atol=1e-8)
    angle = np.pi / 3
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                         [-np.sin(angle), 0, np.cos(angle)]])
    pixels_rotated, jacobian_rotated = script.project_and_jacobian(
        rotation @ point.xyz, centers @ rotation.T, rotations @ rotation.T, cameras)
    np.testing.assert_allclose(pixels_rotated, expected, atol=1e-8)
    np.testing.assert_allclose(script.relative_radius95(jacobian_rotated, 1, distance), radius, rtol=1e-8)


def test_capture_grouping_does_not_count_two_lenses_as_two_captures():
    script = load_script()
    reconstruction, point, records = synthetic_track(script, [-3, -2.97, 0, .03, 3, 3.03])
    for i, record in enumerate(records.values()):
        record["capture_index"] = i // 2
    reason, _ = script.assess_point(point, script.geometry(reconstruction), records,
                                    relative_budget=.02, pixel_sigma=1, cross_limit=2)
    assert reason == "insufficient_captures"


def test_deletion_preserves_survivor_and_clears_every_observation(tmp_path):
    script = load_script()
    reconstruction, point, _ = synthetic_track(script, [-6, -4, -2, -1, 1, 2, 4, 6])
    original_xyz, original_track = point.xyz, list(point.track)
    second = script.model.Point3D(9, (1, 0, 10), (10, 20, 30), 0, [(i, 1) for i in reconstruction.images])
    reconstruction.points3D[9] = second
    for image in reconstruction.images.values():
        image.points2D.append(script.model.ImagePoint2D(100, 100, 9))
    script.retain_points(reconstruction, {5})
    assert reconstruction.points3D[5].xyz == original_xyz
    assert reconstruction.points3D[5].track == original_track
    assert all(image.num_registered_points == 1 for image in reconstruction.images.values())
    script.model.write_cameras_bin(tmp_path / "cameras.bin", reconstruction.cameras)
    script.model.write_images_bin(tmp_path / "images.bin", reconstruction.images)
    script.model.write_points3D_bin(tmp_path / "points3D.bin", reconstruction.points3D)
    loaded = script.model.read_model(tmp_path)
    script.validate_tracks(loaded)
    assert set(loaded.points3D) == {5}


def test_full_track_retains_three_constrained_positions_but_rejects_weak_baseline():
    script = load_script()
    for positions, expected in [([-3, 0, 3], "keep"),
                                ([0, .001, .002], "conditional_uncertainty"),
                                ([0, 0, 0], "conditional_uncertainty"),
                                ([-3, 3], "insufficient_captures")]:
        reconstruction, point, records = synthetic_track(script, positions)
        reason, values = script.assess_point(point, script.geometry(reconstruction), records,
                                             relative_budget=.02, pixel_sigma=1, cross_limit=2,
                                             policy="full_track")
        assert reason == expected, (expected, reason, values)


def test_leave_one_out_predicts_the_withheld_capture_and_checks_original_xyz():
    script = load_script()
    reconstruction, point, records = synthetic_track(script, [-3, 0, 3])
    point.xyz = (.1, 0, 10)
    reason, _ = script.assess_point(point, script.geometry(reconstruction), records,
                                    relative_budget=.02, pixel_sigma=1, cross_limit=2, policy="full_track")
    assert reason == "original_reprojection"
    point.xyz = (0, 0, 10)
    reconstruction.images[1].points2D[0].x += 3.5
    # 元 XYZ の p95 は許容内でも、独立 capture への予測は一致しない。
    reason, values = script.assess_point(point, script.geometry(reconstruction), records,
                                         relative_budget=.02, pixel_sigma=1, cross_limit=3.4,
                                         policy="full_track")
    assert reason == "cross_reprojection", (reason, values)


def test_leave_one_out_groups_both_lenses_and_rejects_a_degenerate_remainder():
    script = load_script()
    reconstruction, point, records = synthetic_track(script, [-3, -2.97, 3, 3.03])
    for i, record in enumerate(records.values()):
        record["capture_index"] = i // 2
    reason, _ = script.assess_point(point, script.geometry(reconstruction), records,
                                    relative_budget=.02, pixel_sigma=1, cross_limit=2, policy="full_track")
    assert reason == "insufficient_captures"
    reconstruction, point, records = synthetic_track(script, [0, 0, 3])
    reason, _ = script.assess_point(point, script.geometry(reconstruction), records,
                                    relative_budget=.02, pixel_sigma=1, cross_limit=2, policy="full_track")
    assert reason == "degenerate_leave_one_out"
