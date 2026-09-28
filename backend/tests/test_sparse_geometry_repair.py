from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

pycolmap = pytest.importorskip("pycolmap")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
module = importlib.import_module("repair_sparse_geometry")


def fixture():
    model = pycolmap.synthesize_dataset(pycolmap.SyntheticDatasetOptions(
        num_rigs=2, num_cameras_per_rig=1, num_frames_per_rig=10,
        num_points3D=10, num_points2D_without_point3D=0,
    ))
    records = {im.name: {"source_id": "primary" if im.camera_id == 1 else "phone", "capture_index": i}
               for i, im in model.images.items()}
    image = next(im for im in model.images.values() if im.camera_id == 2)
    correct = image.cam_from_world().matrix().copy()
    image.frame.set_cam_from_world(image.camera_id, pycolmap.Rigid3d(
        image.cam_from_world().rotation, correct[:, 3] + [0.4, 0.2, -0.1]))
    old = image.cam_from_world().matrix()
    row = {"image_id": image.image_id, "old_R": old[:, :3].tolist(), "old_t": old[:, 3].tolist(),
           "R": correct[:, :3].tolist(), "t": correct[:, 3].tolist()}
    return model, records, row, correct


def test_repair_retriangulates_points_and_persists_frame_pose(tmp_path):
    model, records, row, correct = fixture()
    xyz = {i: pt.xyz.copy() for i, pt in model.points3D.items()}
    for point in model.points3D.values():
        point.xyz += [0.1, -0.1, 0.1]
    report = module.repair(model, records, "primary", [row], [], progress=lambda *_: None)
    assert report["affected_points"] == 10
    assert report["refit_reasons"] == {"keep": 10}
    assert report["other_poses_unchanged"]
    assert report["preserved_images"] == 20
    for i, pt in model.points3D.items():
        np.testing.assert_allclose(pt.xyz, xyz[i], atol=1e-7)
    model.write(str(tmp_path))
    loaded = pycolmap.Reconstruction(str(tmp_path))
    np.testing.assert_allclose(loaded.images[row["image_id"]].cam_from_world().matrix(), correct, atol=1e-10)


def test_repair_refuses_stale_pose_evidence():
    model, records, row, _ = fixture()
    row["old_t"][0] += 1
    with pytest.raises(ValueError, match="different model"):
        module.repair(model, records, "primary", [row], [], progress=lambda *_: None)


def test_repair_refuses_primary_pose_changes():
    model, records, row, _ = fixture()
    records[model.images[row["image_id"]].name]["source_id"] = "primary"
    with pytest.raises(ValueError, match="primary camera"):
        module.repair(model, records, "primary", [row], [], progress=lambda *_: None)


def test_reference_alignment_validates_all_primary_poses():
    model, _, _, _ = fixture()
    reference = pycolmap.Reconstruction(model)
    transform = pycolmap.Sim3d(1.5, pycolmap.Rotation3d([0.1, -0.2, 0.1]), [1., 2., 3.])
    model.transform(transform)
    primary = {i for i, im in model.images.items() if im.camera_id == 1}
    module.align_to_reference(model, reference, primary)
    for i in reference.images:
        np.testing.assert_allclose(model.images[i].cam_from_world().matrix(),
                                   reference.images[i].cam_from_world().matrix(), atol=1e-9)


def test_recovery_refits_primary_tracks_and_refuses_existing_observation_conflicts():
    model, records, _, _ = fixture()
    point_id = next(iter(model.points3D))
    point = model.points3D[point_id]
    expected = point.xyz.copy()
    row = {"old_id": point_id, "rgb": point.color.tolist(),
           "track": [[e.image_id, e.point2D_idx] for e in point.track.elements
                     if records[model.images[e.image_id].name]["source_id"] == "primary"]}
    with pytest.raises(ValueError, match="already belongs"):
        module.repair(model, records, "primary", [], [row], progress=lambda *_: None)
    model.delete_point3D(point_id)
    report = module.repair(model, records, "primary", [], [row], progress=lambda *_: None)
    assert report["affected_points"] == 0
    restored = model.points3D[report["restored_point_ids"][0]]
    np.testing.assert_allclose(restored.xyz, expected, atol=1e-7)
    assert restored.track.length() == 10
