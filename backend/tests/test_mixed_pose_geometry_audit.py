from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

pycolmap = pytest.importorskip("pycolmap")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
audit = importlib.import_module("audit_mixed_pose_geometry")


def test_primary_alignment_does_not_use_erroneous_phone_centers(tmp_path):
    primary = pycolmap.synthesize_dataset(pycolmap.SyntheticDatasetOptions(
        num_rigs=1, num_cameras_per_rig=1, num_frames_per_rig=100,
        num_points3D=100, num_points2D_without_point3D=0,
    ))
    primary.write(str(tmp_path))
    model = pycolmap.Reconstruction(str(tmp_path))
    records = {image.name: {"capture_index": image_id} for image_id, image in primary.images.items()}
    model.transform(pycolmap.Sim3d(2.4, pycolmap.Rotation3d(np.array([0.2, 0.4, -0.3])), [1, 2, 3]))
    phone_id = max(primary.images)
    original_phone = model.images[phone_id]
    pose = original_phone.cam_from_world()
    original_phone.frame.set_cam_from_world(original_phone.camera_id, pycolmap.Rigid3d(pose.rotation, pose.translation + [100, 0, 0]))
    primary.deregister_frame(primary.images[phone_id].frame_id)

    result = audit.align_to_primary(model, primary, records)

    assert result["heldout_center_errors"]["maximum"] < 1e-10
    assert result["heldout_rotation_degrees"]["maximum"] < 1e-5
    assert phone_id in model.images
    for image_id in primary.reg_image_ids():
        image = primary.images[image_id]
        np.testing.assert_allclose(model.images[image_id].cam_from_world().matrix(), image.cam_from_world().matrix(), atol=1e-10)


def test_heldout_focal_probe_recovers_calibration_without_mutating_input():
    rng = np.random.default_rng(4)
    xyz = rng.uniform([-3, -2, 4], [3, 2, 10], (600, 3))
    camera = pycolmap.Camera(model="SIMPLE_RADIAL", width=1920, height=1440, params=[1350, 960, 720, 0.008])
    pixels = camera.img_from_cam(xyz)
    camera.params = [1500, 960, 720, 0.008]
    initial = camera.params.copy()

    pose, result = audit.heldout_pnp(camera, pixels, xyz, 11, refine_focal=True)

    assert pose is not None
    assert result["heldout"]["supported"]
    assert result["camera_params"][0] == pytest.approx(1350, abs=0.1)
    assert result["heldout"]["residual_px"]["maximum"] < 0.01
    np.testing.assert_array_equal(camera.params, initial)
    assert len(result["heldout_indices"]) == 200


def test_common_pair_comparison_reports_missing_cameras(tmp_path):
    baseline = pycolmap.synthesize_dataset(pycolmap.SyntheticDatasetOptions(
        num_rigs=1, num_cameras_per_rig=1, num_frames_per_rig=5, num_points3D=10,
    ))
    baseline.write(str(tmp_path))
    candidate = pycolmap.Reconstruction(str(tmp_path))
    ids = sorted(baseline.images)
    records = {image.name: {"capture_index": image_id, "source_id": "phone"}
               for image_id, image in baseline.images.items()}
    candidate.deregister_frame(candidate.images[ids[2]].frame_id)

    result = audit.trajectory_comparison({"baseline": baseline, "candidate": candidate}, records, ["phone"])["phone"]

    assert result["expected_steps"] == 4
    assert result["common_steps"] == 2
    assert result["models"]["baseline"] == result["models"]["candidate"]
