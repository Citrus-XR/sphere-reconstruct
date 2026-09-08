"""撮影経路近傍の広域 point mode だけを地面として採用することを検証する。"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from sphere_reconstruct.colmap import model as colmap_model
from sphere_reconstruct.colmap.ground_position import estimate_ground_position, sample_analysis_points
from sphere_reconstruct.colmap.model import Camera, Image, Point3D, Reconstruction
from sphere_reconstruct.pipeline.stage import ProgressReporter, StageContext
from sphere_reconstruct.stages import scene_alignment as scene_alignment_module
from sphere_reconstruct.stages.scene_alignment import SceneAlignment


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


def test_nearest_plane_wins_when_multiple_dominant_horizontal_planes_exist():
    reconstruction = _reconstruction(broad_ground=True)
    rng = np.random.default_rng(21)
    first_id = max(reconstruction.points3D) + 1
    near_plane = zip(
        rng.uniform(0.0, 20.0, 6000),
        rng.normal(3.0, 0.03, 6000),
        rng.uniform(-2.0, 2.0, 6000),
        strict=True,
    )
    for offset, xyz in enumerate(near_plane):
        point_id = first_id + offset
        reconstruction.points3D[point_id] = Point3D(point_id, tuple(xyz), (128, 128, 128), 0.5)

    result = estimate_ground_position(reconstruction, reference_image_prefix="sources/primary/")

    assert result["applied"] is True
    assert result["plane_selection"] == "nearest_dominant_horizontal_plane"
    assert abs(result["ground_y"] - 3.0) < 0.05
    assert result["camera_height_median_m"] < 2.0


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


def test_scene_analysis_sampling_has_a_deterministic_hard_cap():
    reconstruction = _reconstruction(broad_ground=True)

    first = sample_analysis_points(reconstruction, maximum_points=128)
    second = sample_analysis_points(reconstruction, maximum_points=128)

    assert first.shape == (128, 3)
    assert np.array_equal(first, second)
    assert np.array_equal(first[0], next(iter(reconstruction.points3D.values())).xyz)
    assert np.array_equal(first[-1], list(reconstruction.points3D.values())[-1].xyz)


def test_scene_alignment_runs_ground_detection_without_metric_scale(tmp_path, monkeypatch):
    reconstruction = _reconstruction(broad_ground=True)
    input_model = tmp_path / "restore_metric_scale" / "sparse" / "0"
    input_model.mkdir(parents=True)
    colmap_model.write_cameras_bin(input_model / "cameras.bin", reconstruction.cameras)
    colmap_model.write_images_bin(input_model / "images.bin", reconstruction.images)
    colmap_model.write_points3D_bin(input_model / "points3D.bin", reconstruction.points3D)
    (tmp_path / "restore_metric_scale" / "scale_restoration.json").write_text(
        json.dumps({"metric": False, "reason": "rig_baseline_not_observable"}),
        encoding="utf-8",
    )
    stage_out = tmp_path / ".scene_alignment.tmp"
    stage_out.mkdir()
    applied_translation = None

    def materialize(context, _input_model, *, translation, **_kwargs):
        nonlocal applied_translation
        applied_translation = translation
        output_model = context.stage_out_dir / "sparse" / "0"
        output_model.mkdir(parents=True)
        colmap_model.write_cameras_bin(output_model / "cameras.bin", reconstruction.cameras)
        colmap_model.write_images_bin(output_model / "images.bin", reconstruction.images)
        colmap_model.write_points3D_bin(output_model / "points3D.bin", reconstruction.points3D)
        return output_model, reconstruction

    def write_preview(context, _reconstruction, **_kwargs):
        preview = context.stage_out_dir / "preview"
        preview.mkdir()
        (preview / "reconstruction.json").write_text("{}", encoding="utf-8")
        (preview / "points.bin").write_bytes(b"points")
        return SimpleNamespace(num_points_written=len(reconstruction.points3D))

    monkeypatch.setattr(scene_alignment_module.similarity_transform, "materialize_similarity", materialize)
    monkeypatch.setattr(scene_alignment_module.similarity_transform, "write_preview", write_preview)
    stage = SceneAlignment()
    context = StageContext(
        project_id="project",
        project_dir=tmp_path,
        stage_out_dir=stage_out,
        params=stage.normalize_params({}),
        sources=(),
        progress=ProgressReporter(lambda *_args: None),
        resolved_inputs=[],
    )

    manifest = stage.execute(context)

    assert manifest.extra["applied"] is True
    assert manifest.extra["ground"]["metric_scale_available"] is False
    assert manifest.extra["ground"]["plane_selection"] == "nearest_dominant_horizontal_plane"
    assert "camera_height_median_m" not in manifest.extra["ground"]
    assert manifest.extra["ground"]["camera_height_median_model_units"] > 3.0
    assert manifest.extra["orientation"]["applied"] is False
    assert applied_translation is not None
    assert applied_translation[1] < -4.9
