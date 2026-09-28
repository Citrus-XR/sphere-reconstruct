from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

pycolmap = pytest.importorskip("pycolmap")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
refit = importlib.import_module("prepare_fixed_primary_reference").refit_primary


def test_primary_point_refit_removes_phone_influence_without_moving_primary_poses():
    reconstruction = pycolmap.synthesize_dataset(pycolmap.SyntheticDatasetOptions(
        num_rigs=2, num_cameras_per_rig=1, num_frames_per_rig=20,
        num_points3D=50, num_points2D_without_point3D=0,
    ))
    records = {image.name: {"source_id": "primary" if image.camera_id == 1 else "phone", "capture_index": image_id}
               for image_id, image in reconstruction.images.items()}
    correct = {key: point.xyz.copy() for key, point in reconstruction.points3D.items()}
    for point in reconstruction.points3D.values():
        point.xyz = point.xyz + [0.4, -0.2, 0.3]
    for image in reconstruction.images.values():
        if image.camera_id == 2:
            pose = image.cam_from_world()
            image.frame.set_cam_from_world(image.camera_id, pycolmap.Rigid3d(pose.rotation, pose.translation + [2, 3, 4]))

    report = refit(reconstruction, records, "primary", max_error=2, progress=lambda *_args: None)

    assert report["primary_poses_unchanged"]
    assert report["primary_images"] == 20
    assert report["retained_points"] == 50
    assert len(reconstruction.reg_image_ids()) == 20
    for key, point in reconstruction.points3D.items():
        np.testing.assert_allclose(point.xyz, correct[key], atol=1e-7)
        assert all(records[reconstruction.images[element.image_id].name]["source_id"] == "primary" for element in point.track.elements)


def test_primary_refit_preserves_long_distant_tracks_and_rejects_unconstrained_depth():
    reconstruction = pycolmap.synthesize_dataset(pycolmap.SyntheticDatasetOptions(
        num_rigs=1, num_cameras_per_rig=1, num_frames_per_rig=80,
        num_points3D=2, num_points2D_without_point3D=0,
    ))
    camera = next(iter(reconstruction.cameras.values()))
    camera.params = [2000, 512, 512, 0]
    point_ids = sorted(reconstruction.points3D)
    for point_id, depth in zip(point_ids, (25, 2500), strict=True):
        reconstruction.points3D[point_id].xyz = [0, 0, depth]
    records = {}
    for capture, (_image_id, image) in enumerate(sorted(reconstruction.images.items())):
        image.frame.set_cam_from_world(image.camera_id, pycolmap.Rigid3d(
            pycolmap.Rotation3d(), np.array([0.5 - capture / 79, 0, 0])))
        for observation in image.points2D:
            xyz = reconstruction.points3D[observation.point3D_id].xyz
            observation.xy = camera.img_from_cam(image.cam_from_world() * xyz)
        records[image.name] = {"source_id": "primary", "capture_index": capture}

    report = refit(reconstruction, records, "primary", max_error=2, progress=lambda *_: None)

    assert np.degrees(2 * np.arctan(0.5 / 25)) < 3
    assert set(reconstruction.points3D) == {point_ids[0]}
    assert report["reason_counts"] == {"keep": 1, "conditional_uncertainty": 1}
    assert report["primary_poses_unchanged"]
    np.testing.assert_allclose(reconstruction.points3D[point_ids[0]].xyz, [0, 0, 25], atol=1e-8)
