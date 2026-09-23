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

    report = refit(reconstruction, records, "primary", max_error=2, min_angle=3, progress=lambda *_args: None)

    assert report["primary_poses_unchanged"]
    assert report["primary_images"] == 20
    assert report["retained_points"] == 50
    assert len(reconstruction.reg_image_ids()) == 20
    for key, point in reconstruction.points3D.items():
        np.testing.assert_allclose(point.xyz, correct[key], atol=1e-7)
        assert all(records[reconstruction.images[element.image_id].name]["source_id"] == "primary" for element in point.track.elements)
