from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pycolmap = pytest.importorskip("pycolmap")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
script = importlib.import_module("prepare_pose_training_comparison")


def test_paired_training_preserves_primary_geometry_and_reports_omissions(tmp_path):
    baseline = pycolmap.synthesize_dataset(pycolmap.SyntheticDatasetOptions(
        num_rigs=2, num_cameras_per_rig=1, num_frames_per_rig=50,
        num_points3D=30, num_points2D_without_point3D=0,
    ))
    baseline_path = tmp_path / "original"
    baseline_path.mkdir()
    baseline.write(str(baseline_path))
    primary = pycolmap.Reconstruction(str(baseline_path))
    candidate = pycolmap.Reconstruction(str(baseline_path))
    phone_ids = sorted(i for i, image in baseline.images.items() if image.camera_id == 2)
    for image_id in phone_ids:
        primary.deregister_frame(primary.images[image_id].frame_id)
    for image_id in phone_ids[1:]:
        image = candidate.images[image_id]
        pose = image.cam_from_world()
        image.frame.set_cam_from_world(image.camera_id, pycolmap.Rigid3d(pose.rotation, pose.translation + [0.1, 0, 0]))
    candidate.deregister_frame(candidate.images[phone_ids[0]].frame_id)
    for name, model in [("primary", primary), ("candidate", candidate)]:
        (tmp_path / name).mkdir()
        model.write(str(tmp_path / name))
    project = tmp_path / "project"
    (project / "extract_features").mkdir(parents=True)
    (project / "extract_features/input_spec.json").write_text(json.dumps({"image_path": "images"}))
    catalog = {"primary_source_id": "primary", "images": [
        {"name": image.name, "capture_index": image_id,
         "source_id": "primary" if image.camera_id == 1 else "phone"}
        for image_id, image in baseline.images.items()
    ]}
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog))
    mask_records = []
    for image in baseline.images.values():
        path = project / "extract_features/images" / image.name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image content")
        mask = project / "generate_training_masks" / (image.name + ".png")
        mask.parent.mkdir(parents=True, exist_ok=True)
        mask.write_bytes(b"mask content")
        mask_records.append({"name": image.name, "path": str(mask.relative_to(project))})
    (project / "generate_training_masks/manifest_masks.json").write_text(json.dumps({
        "version": 3, "purpose": "training", "images": mask_records,
    }))
    output = tmp_path / "output"
    output.mkdir()
    args = SimpleNamespace(project=project, primary=tmp_path / "primary", baseline=baseline_path,
                           candidate=tmp_path / "candidate", cohort=baseline_path, catalog=catalog_path,
                           output=output, backend_src=Path(__file__).resolve().parents[1] / "src")

    script.execute(args)

    report = json.loads((output / "report.json").read_text())
    assert report["primary_images"] == 50
    assert report["requested_phone_images"] == 50
    assert report["common_phone_images"] == 49
    assert report["omitted_phone_images"] == [baseline.images[phone_ids[0]].name]
    first = pycolmap.Reconstruction(str(output / "baseline/sparse/0"))
    second = pycolmap.Reconstruction(str(output / "candidate/sparse/0"))
    assert len(first.reg_image_ids()) == len(second.reg_image_ids()) == 99
    assert (output / "baseline/sparse/0/points3D.bin").read_bytes() == (output / "candidate/sparse/0/points3D.bin").read_bytes()
    for image_id in primary.reg_image_ids():
        np.testing.assert_allclose(first.images[image_id].cam_from_world().matrix(), second.images[image_id].cam_from_world().matrix(), atol=1e-10)
    for image_id in phone_ids[1:]:
        assert first.images[image_id].num_points2D() == 0
        assert second.images[image_id].num_points2D() == 0
        assert np.linalg.norm(first.images[image_id].projection_center() - second.images[image_id].projection_center()) == pytest.approx(0.1)
