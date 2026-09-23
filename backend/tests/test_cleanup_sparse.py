"""全 track cleanup の mixed camera、capture 留保、成果物を検証する。"""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from sphere_reconstruct.colmap import model
from sphere_reconstruct.colmap.input_workspace import InputSpec
from sphere_reconstruct.colmap.point_stability import assess_point, geometry
from sphere_reconstruct.imaging.camera_geometry import camera_rays_to_pixels
from sphere_reconstruct.pipeline.stage import ProgressReporter, StageContext
from sphere_reconstruct.stages.cleanup_sparse import (
    CleanupSparse,
    _filter_reconstruction,
    _find_trajectory_outliers,
    _remove_trajectory_outlier_images,
)


def mixed_model():
    cameras = {
        1: model.Camera(1, "OPENCV_FISHEYE", 2000, 2000, [1000, 1000, 1000, 1000, 0.01, 0, 0, 0]),
        2: model.Camera(2, "PINHOLE", 4000, 3000, [2500, 2450, 2000, 1500]),
        3: model.Camera(3, "SIMPLE_RADIAL", 3000, 2000, [1800, 1500, 1000, 0.03]),
    }
    points = {
        1: model.Point3D(1, (0, 0, 10), (120, 130, 140), 0, []),
        2: model.Point3D(2, (0, 0, 10000), (40, 50, 60), 0, []),
    }
    images, records = {}, []
    for index, (camera_id, x) in enumerate([(1, -3), (2, -1), (1, 1), (3, 3)]):
        name = f"sources/source{camera_id}/frame_{index:06d}.jpg"
        observations = []
        for point in points.values():
            pixel = camera_rays_to_pixels(cameras[camera_id], (np.asarray(point.xyz) - [x, 0, 0])[None])[0]
            point.track.append((index, len(observations)))
            observations.append(model.ImagePoint2D(*pixel, point.point3D_id))
        images[index] = model.Image(index, (1, 0, 0, 0), (-x, 0, 0), camera_id, name, observations)
        records.append({"name": name, "source_id": f"source{camera_id}", "capture_index": index // 2})
    return model.Reconstruction(cameras, images, points), records


def context(tmp_path, **params):
    return SimpleNamespace(
        params=CleanupSparse().normalize_params(params),
        stage_out_dir=tmp_path,
        progress=ProgressReporter(lambda *args: None),
    )


def test_mixed_cleanup_keeps_constrained_geometry_and_removes_uncertain_depth(tmp_path):
    reconstruction, records = mixed_model()
    before = deepcopy(reconstruction)
    result = _filter_reconstruction(context(tmp_path), reconstruction, records)
    assert result["reason_counts"] == {"keep": 1, "conditional_uncertainty": 1}
    assert result["output_points"] == 1
    assert result["cleared_observations"] == 4
    assert reconstruction.cameras == before.cameras
    assert reconstruction.points3D[1] == before.points3D[1]
    assert all(image.num_registered_points == 1 for image in reconstruction.images.values())
    for image_id, image in reconstruction.images.items():
        assert image.qvec == before.images[image_id].qvec
        assert image.tvec == before.images[image_id].tvec
    with np.load(tmp_path / "point_assessment.npz") as assessment:
        assert assessment["point_ids"].tolist() == [1, 2]
        assert assessment["reasons"].tolist() == ["keep", "conditional_uncertainty"]


def test_trajectory_removal_assesses_short_tracks_and_reports_final_counts(tmp_path, monkeypatch):
    camera = model.Camera(1, "PINHOLE", 2000, 1500, [1200, 1200, 1000, 750])
    centers = [(-2, 0, 0), (-1, 0.2, 0), (0, 0, 0), (1, -0.2, 0), (2, 0, 0)]
    images, records = {}, []
    points = {
        1: model.Point3D(1, (0, 0, 8), (120, 130, 140), 0, []),
        2: model.Point3D(2, (0.5, 0.4, 6), (40, 50, 60), 0, []),
    }
    for capture, center in enumerate(centers):
        image_id = capture + 1
        observations = []
        observed_points = points.values() if capture < 3 else (points[1],)
        for point in observed_points:
            pixel = camera_rays_to_pixels(
                camera, (np.asarray(point.xyz) - np.asarray(center))[None]
            )[0]
            point.track.append((image_id, len(observations)))
            observations.append(model.ImagePoint2D(*pixel, point.point3D_id))
        name = f"phone/frame_{capture:06d}.jpg"
        images[image_id] = model.Image(
            image_id, (1, 0, 0, 0), tuple(-value for value in center), 1, name, observations
        )
        records.append({"name": name, "source_id": "phone", "capture_index": capture})
    reconstruction = model.Reconstruction({1: camera}, images, points)
    monkeypatch.setattr(
        "sphere_reconstruct.stages.cleanup_sparse._find_trajectory_outliers",
        lambda *_: {
            "outlier_captures_by_source": {"phone": [0, 1]},
            "trajectories": {"phone": {"test": "outliers"}},
        },
    )

    result = _filter_reconstruction(context(tmp_path), reconstruction, records)

    assert result["reason_counts"] == {"keep": 1, "insufficient_captures": 1}
    assert result["input_points"] == 2
    assert result["removed_points"] == 1
    assert result["output_points"] == len(reconstruction.points3D) == 1
    assert result["trajectory_removed_observations"] == 4
    assert result["cleared_observations"] == 5
    assert result["removed_images"] == 2
    assert len(reconstruction.points3D[1].track) == 3
    with np.load(tmp_path / "point_assessment.npz") as assessment:
        assert assessment["point_ids"].tolist() == [1, 2]
        assert assessment["reasons"].tolist() == ["keep", "insufficient_captures"]


def test_mixed_capture_indices_are_source_scoped_and_same_capture_views_are_grouped():
    reconstruction, records = mixed_model()
    for record in records:
        record["capture_index"] = 0
    mapped = {record["name"]: record for record in records}
    result, metrics = assess_point(
        reconstruction.points3D[1],
        geometry(reconstruction),
        mapped,
        relative_budget=0.02,
        pixel_sigma=1,
        cross_limit=2,
        policy="full_track",
    )
    assert result == "keep"
    assert metrics[0] == 3
    for record in records:
        record["source_id"] = "one_source"
    result, _ = assess_point(
        reconstruction.points3D[1],
        geometry(reconstruction),
        mapped,
        relative_budget=0.02,
        pixel_sigma=1,
        cross_limit=2,
        policy="full_track",
    )
    assert result == "insufficient_captures"


def test_cleanup_empty_result_fails_without_mutating_input(tmp_path):
    reconstruction, records = mixed_model()
    before = deepcopy(reconstruction)
    with pytest.raises(ValueError, match="no sparse points pass"):
        _filter_reconstruction(context(tmp_path, relative_error=1e-12), reconstruction, records)
    assert reconstruction == before
    assert (tmp_path / "point_assessment.npz").is_file()


@pytest.mark.parametrize("field", ["relative_error", "pixel_sigma", "max_cross_error"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_cleanup_validates_parameters(field, value):
    with pytest.raises(ValueError, match=field):
        CleanupSparse().normalize_params({field: value})


def test_cleanup_requires_capture_metadata(tmp_path):
    reconstruction, records = mixed_model()
    with pytest.raises(ValueError, match="missing capture metadata"):
        _filter_reconstruction(context(tmp_path), reconstruction, records[:-1])


def test_trajectory_outlier_image_removal_rewrites_tracks():
    camera = model.Camera(1, "PINHOLE", 100, 100, [80, 80, 50, 50])
    images = {
        image_id: model.Image(
            image_id,
            (1, 0, 0, 0),
            (-float(index), 0, 0),
            1,
            f"phone/frame_{index:06d}.jpg",
            [model.ImagePoint2D(50, 50, 1)],
        )
        for index, image_id in enumerate((1, 2, 3))
    }
    reconstruction = model.Reconstruction(
        {1: camera},
        images,
        {1: model.Point3D(1, (1, 0, 5), (100, 100, 100), 0, [(1, 0), (2, 0), (3, 0)])},
    )

    removed = _remove_trajectory_outlier_images(
        reconstruction,
        [
            {"name": image.name, "source_id": "phone", "capture_index": index}
            for index, image in enumerate(images.values())
        ],
        {"phone": [1]},
    )

    assert removed == {2}
    assert set(reconstruction.images) == {1, 3}
    assert reconstruction.points3D[1].track == [(1, 0), (3, 0)]
    assert all(image.num_registered_points == 1 for image in reconstruction.images.values())


def test_cleanup_combines_robust_steps_with_primary_path_consistency():
    camera = model.Camera(1, "PINHOLE", 100, 100, [80, 80, 50, 50])
    images, records = {}, []
    primary_centers = []
    image_id = 1
    for capture in range(40):
        angle = capture * 2 * np.pi / 40
        center = (2 * np.cos(angle), 1.0, 2 * np.sin(angle))
        primary_centers.append(center)
        name = f"360/frame_{capture:06d}.jpg"
        images[image_id] = model.Image(
            image_id, (1, 0, 0, 0), tuple(-value for value in center), 1, name
        )
        records.append({"name": name, "source_id": "primary", "capture_index": capture})
        image_id += 1
    for capture in range(30):
        center = list(primary_centers[capture])
        if capture in {10, 20}:
            center[1] += 100
        name = f"phone/frame_{capture:06d}.jpg"
        images[image_id] = model.Image(
            image_id, (1, 0, 0, 0), tuple(-value for value in center), 1, name
        )
        records.append({"name": name, "source_id": "phone", "capture_index": capture})
        image_id += 1
    reconstruction = model.Reconstruction({1: camera}, images, {})
    primary = SimpleNamespace(
        id="primary", role=SimpleNamespace(value="primary"), media_kind=SimpleNamespace(value="video")
    )
    supplemental = SimpleNamespace(
        id="phone", role=SimpleNamespace(value="supplemental"), media_kind=SimpleNamespace(value="images")
    )
    ctx = SimpleNamespace(
        params=CleanupSparse().normalize_params({}),
        sources=(primary, supplemental),
        primary_source=primary,
    )

    result = _find_trajectory_outliers(ctx, reconstruction, records)

    assert result["outlier_captures_by_source"] == {"phone": [10, 20]}
    phone = result["trajectories"]["phone"]
    assert phone["primary_path_consistency"]["maximum_allowed_distance"] < 10


def test_cleanup_rejects_unsupported_projection_before_filtering(tmp_path):
    reconstruction, records = mixed_model()
    reconstruction.cameras[1].model = "EQUIRECTANGULAR"
    with pytest.raises(ValueError, match="camera models.*EQUIRECTANGULAR"):
        _filter_reconstruction(context(tmp_path), reconstruction, records)


@pytest.mark.parametrize("enabled", [True, False])
def test_stage_exports_complete_model_and_declares_metadata_dependency(tmp_path, enabled):
    reconstruction, records = mixed_model()
    source = tmp_path / "scene_alignment" / "sparse" / "0"
    source.mkdir(parents=True)
    model.write_cameras_bin(source / "cameras.bin", reconstruction.cameras)
    model.write_images_bin(source / "images.bin", reconstruction.images)
    model.write_points3D_bin(source / "points3D.bin", reconstruction.points3D)
    (source / "rigs.bin").write_bytes(b"fixture rig")
    (source / "frames.bin").write_bytes(b"fixture frames")
    original_model = model.read_model(source)
    spec = InputSpec(
        3,
        "native_fisheye",
        4,
        3,
        "source1",
        [],
        [],
        records,
        [],
        "images",
        None,
        False,
        None,
        False,
        False,
        False,
    )
    (tmp_path / "extract_features").mkdir()
    spec.write(tmp_path / "extract_features" / "input_spec.json")
    output = tmp_path / ".cleanup_sparse.tmp"
    output.mkdir()
    progress = []
    ctx = StageContext(
        "fixture",
        tmp_path,
        output,
        CleanupSparse().normalize_params({"enabled": enabled}),
        (),
        ProgressReporter(lambda *args: progress.append(args)),
    )
    manifest = CleanupSparse().execute(ctx)
    assert any(ref.path == "extract_features/input_spec.json" for ref in manifest.inputs)
    assert manifest.impl_version == "2.3"
    loaded = model.read_model(output / "sparse" / "0")
    assert len(loaded.points3D) == (1 if enabled else 2)
    assert loaded.cameras == original_model.cameras
    for filename in ("rigs.bin", "frames.bin"):
        assert (output / "sparse" / "0" / filename).read_bytes() == (source / filename).read_bytes()
    assert (output / "preview" / "points.bin").is_file()
    assert bool([ref for ref in manifest.outputs if ref.path.endswith("point_assessment.npz")]) is enabled
    result = json.loads((output / "cleanup_sparse.json").read_text())
    assert result["enabled"] is enabled
    assert all(not ref.path.startswith(str(tmp_path)) for ref in manifest.outputs)
    assert model.read_model(source) == original_model
