"""固定 pose 実験の split と source database 保護。"""

import hashlib
import importlib.util
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_ray_triangulation_recovers_large_translated_scene_and_rejects_parallel():
    script = Path(__file__).resolve().parents[2] / "scripts" / "audit_fixed_pose.py"
    spec = importlib.util.spec_from_file_location("fixed_pose_audit", script)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    offset = np.array([1e6, -2e6, 3e6])
    centers = offset + np.array([[-1.0, 0, 0], [0.0, 0, 0], [2.0, 0, 0]])
    target = offset + np.array([0.1, 0.5, 8.0])
    directions = target - centers
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    np.testing.assert_allclose(audit.triangulate_rays(centers, directions), target, rtol=0, atol=1e-7)
    assert audit.triangulate_rays(centers, np.tile([0.0, 0.0, 1.0], (3, 1))) is None
    assert audit.max_angle(np.array([[0.0, 0, 1.0], [0.0, 0, -1.0]])) == 0.0


def test_heldout_validation_separates_depth_aliases_from_supported_predictions():
    script = Path(__file__).resolve().parents[2] / "scripts" / "audit_fixed_pose.py"
    spec = importlib.util.spec_from_file_location("fixed_pose_audit", script)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    model = audit.model
    camera = model.Camera(1, "OPENCV_FISHEYE", 100, 100, [40, 40, 50, 50, 0, 0, 0, 0])
    xyz = np.array([[0.0, 0, 5.0], [0.0, 0, 10.0]])
    images, pixels = {}, {}
    for image_id, center in [(1, [-1.0, 0, 0]), (2, [1.0, 0, 0]), (3, [0.0, 0, 0])]:
        uv = audit.camera_rays_to_pixels(camera.model, camera.params, xyz - center)
        points = [model.ImagePoint2D(*row, -1 if image_id == 3 else pid)
                  for row, pid in zip(uv, [10, 20], strict=True)]
        images[image_id] = model.Image(image_id, (1, 0, 0, 0), tuple(-np.array(center)), 1, str(image_id), points)
        pixels[image_id] = uv
    reconstruction = model.Reconstruction({1: camera}, images, {
        10: model.Point3D(10, tuple(xyz[0]), (255, 0, 0), 0.0, [(1, 0), (2, 0)]),
        20: model.Point3D(20, tuple(xyz[1]), (0, 255, 0), 0.0, [(1, 1), (2, 1)])})
    links = np.array([[3, 0, 1, 0, 0], [3, 0, 2, 0, 1],
                      [3, 1, 1, 0, 0], [3, 1, 2, 0, 1],
                      [3, 1, 1, 1, 0], [3, 1, 2, 1, 1]])
    report, predictions, errors = audit.heldout_audit(reconstruction, {i: str(i) for i in images},
                                                     pixels, links, np.array([[3, 0], [3, 1]]), {"3"})
    assert report["prediction_coverage"] == 0.5
    assert report["validation_observations_in_model"] == 0
    assert report["ambiguous_with_two_plausible_depths"] == 1
    assert report["ambiguous_depths_differ_over_25pct"] == 1
    assert report["plausible_depth_ratio"]["p50"] == 2.0
    np.testing.assert_array_equal(predictions, [[3, 0, 10]])
    np.testing.assert_allclose(errors, 0)


def test_graph_filter_keeps_source_and_excludes_validation(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts" / "benchmark_fixed_pose.py"
    spec = importlib.util.spec_from_file_location("fixed_pose_benchmark", script)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    source = tmp_path / "source.db"
    destination = tmp_path / "filtered.db"
    pairs = [(1 * benchmark.MAX_IMAGE_ID + 2, 40),
             (1 * benchmark.MAX_IMAGE_ID + 3, 30),
             (1 * benchmark.MAX_IMAGE_ID + 4, 20)]
    with sqlite3.connect(source) as db:
        db.execute("CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT)")
        db.execute("CREATE TABLE matches (pair_id INTEGER PRIMARY KEY, rows INTEGER)")
        db.execute("CREATE TABLE two_view_geometries (pair_id INTEGER PRIMARY KEY, rows INTEGER)")
        db.executemany("INSERT INTO images VALUES (?, ?)", [(i, str(i)) for i in range(1, 5)])
        db.executemany("INSERT INTO matches VALUES (?, ?)", pairs)
        db.executemany("INSERT INTO two_view_geometries VALUES (?, ?)", pairs)
    before = source.read_bytes()
    records = {str(i): {"source_id": "source"} for i in range(1, 5)}
    centers = {str(i): np.array([i - 1.0, 0.0, 0.0]) for i in range(1, 5)}
    result = benchmark.filter_graph(source, destination, records, centers, {"4"}, 2.0)
    assert source.read_bytes() == before
    assert result["withheld_pairs"] == 1
    assert result["short_baseline_pairs"] == 1
    assert result["retained_pairs"] == 1
    assert result["retained_matches"] == 30
    with sqlite3.connect(destination) as db:
        assert db.execute("SELECT pair_id FROM matches").fetchall() == [(pairs[1][0],)]
        assert db.execute("SELECT pair_id FROM two_view_geometries").fetchall() == [(pairs[1][0],)]


def test_supplement_preserves_base_and_remaps_bidirectional_tracks(tmp_path):
    from sphere_reconstruct.colmap import model

    base, candidate, experiment = [tmp_path / name for name in ("base", "candidate", "experiment")]
    camera = model.Camera(1, "OPENCV_FISHEYE", 100, 100, [40, 40, 50, 50, 0, 0, 0, 0])
    for directory, image_ids, point_ids in [(base, [1, 2], [10, -1, -1]),
                                            (candidate, [101, 102], [20, 30, 40])]:
        directory.mkdir()
        images = {identifier: model.Image(identifier, (1, 0, 0, 0), (i, 0, 0), 1, f"view{i}",
                  [model.ImagePoint2D(50 + j, 50, pid) for j, pid in enumerate(point_ids)])
                  for i, identifier in enumerate(image_ids)}
        points = {pid: model.Point3D(pid, (j, 0, 10), (40, 50, 60), 0.1,
                                    [(identifier, j) for identifier in image_ids])
                  for j, pid in enumerate(point_ids) if pid >= 0}
        model.write_cameras_bin(directory / "cameras.bin", {1: camera})
        model.write_images_bin(directory / "images.bin", images)
        model.write_points3D_bin(directory / "points3D.bin", points)
    experiment.mkdir()
    (experiment / "reference").mkdir()
    for name in ("input_spec.json", "validation_names.json"):
        (experiment / name).write_text("{}")
    stability = tmp_path / "stability.npz"
    fingerprints = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in candidate.iterdir()}
    np.savez(stability, values=[[20, 4, .01, 1, 5], [30, 4, .01, 1, 5], [40, 4, .04, 1, 5]],
             model_fingerprint=json.dumps(fingerprints),
             columns=["point_id", "captures", "relative_xyz_disagreement", "cross_median_px", "half_minimum_angle_deg"])
    before = {path: path.read_bytes() for path in [*base.iterdir(), *candidate.iterdir()]}
    script = Path(__file__).resolve().parents[2] / "scripts" / "supplement_stable_points.py"
    subprocess.run([sys.executable, str(script), str(base), str(candidate), str(stability),
                    str(tmp_path / "output"), "--experiment-reference", str(experiment),
                    "--relative-error-budgets", "0.02", "0.05", "--max-cross-error-px", "2"],
                   check=True, capture_output=True, text=True)
    for suffix, count in [("02", 2), ("05", 3)]:
        result = model.read_model(tmp_path / "output" / f"heldout_supplement_{suffix}pct" / "sparse" / "0")
        assert len(result.points3D) == count
        assert result.points3D[10].xyz == (0, 0, 10)
        assert result.points3D[10].track == [(1, 0), (2, 0)]
        for pid, point in result.points3D.items():
            for image_id, index in point.track:
                assert result.images[image_id].points2D[index].point3D_id == pid
        for image_id, image in result.images.items():
            for index, point in enumerate(image.points2D):
                if point.point3D_id in result.points3D:
                    assert (image_id, index) in result.points3D[point.point3D_id].track
    assert all(path.read_bytes() == content for path, content in before.items())
    candidate_points = model.read_model(candidate).points3D
    candidate_points[30].xyz = (100, 100, 100)
    model.write_points3D_bin(candidate / "points3D.bin", candidate_points)
    rejected = subprocess.run([sys.executable, str(script), str(base), str(candidate), str(stability),
                                str(tmp_path / "mismatched"), "--experiment-reference", str(experiment),
                                "--relative-error-budgets", "0.02", "--max-cross-error-px", "2"],
                               capture_output=True, text=True)
    assert rejected.returncode != 0
    assert "stability archive belongs to a different candidate model" in rejected.stderr


def test_cohort_comparison_accounts_for_lost_coverage():
    script = Path(__file__).resolve().parents[2] / "scripts" / "compare_fixed_pose_validation.py"
    spec = importlib.util.spec_from_file_location("fixed_pose_comparison", script)
    comparison = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparison)
    base_keys = np.rec.fromarrays([[1, 1, 2], [0, 1, 0]], names="image,keypoint")
    keys = np.rec.fromarrays([[1, 2, 2], [0, 0, 1]], names="image,keypoint")
    result = comparison.compare(base_keys, np.array([1., 1., 4.]), keys, np.array([1., 4., .5]))
    assert result["common_max_absolute_error_change"] == 0
    assert result["added"]["within_2px"] == 1
    assert result["lost"]["within_2px"] == 1
    assert result["net_within_2px"] == 0
