"""Mixed-camera image catalog を InputSpec v2 へ変換する契約を検証する。"""

import json

import pytest
from PIL import Image as PilImage

from sphere_reconstruct.colmap import input_workspace


def _write_feature_masks(project, names):
    root = project / "generate_feature_masks"
    records = []
    for index, name in enumerate(names):
        path = root / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        PilImage.new("L", (100, 100), 255 - index).save(path, format="PNG")
        records.append(
            {
                "name": name,
                "source_id": "primary" if index == 0 else "phone",
                "capture_index": index,
                "path": str(path.relative_to(project)),
                "coverage": 0.1,
                "coverage_warning": False,
                "detections": {},
            }
        )
    (root / "manifest_masks.json").write_text(
        json.dumps({"version": 3, "purpose": "feature", "images": records})
    )


def test_mixed_workspace_materializes_batches_masks_and_rig(tmp_path):
    project = tmp_path / "project"
    prepared = project / "rectify_fisheye"
    fisheye = prepared / "fisheye.jpg"
    phone = prepared / "phone.jpg"
    prepared.mkdir(parents=True)
    PilImage.new("RGB", (100, 100), (10, 20, 30)).save(fisheye, format="JPEG")
    PilImage.new("RGB", (100, 100), (40, 50, 60)).save(phone, format="JPEG")
    names = [
        "sources/primary/front/frame_000000.jpg",
        "sources/phone/camera_00/frame_000001.jpg",
    ]
    catalog = {
        "version": 1,
        "reconstruction_mode": "native_fisheye",
        "primary_source_id": "primary",
        "sources": [
            {"id": "primary", "label": "360", "role": "primary", "projection": "dual_fisheye"},
            {"id": "phone", "label": "Phone", "role": "supplemental", "projection": "perspective"},
        ],
        "rig_config_path": "rig_config.json",
        "camera_groups": [
            {
                "id": "primary:fish",
                "source_id": "primary",
                "camera_model": "OPENCV_FISHEYE",
                "camera_params": [20, 20, 50, 50, 0, 0, 0, 0],
                "single_camera": False,
                "single_camera_per_folder": True,
                "refine_intrinsics": True,
                "image_names": [names[0]],
            },
            {
                "id": "phone:perspective",
                "source_id": "phone",
                "camera_model": "SIMPLE_RADIAL",
                "camera_params": [80, 50, 50, 0],
                "single_camera": True,
                "single_camera_per_folder": False,
                "refine_intrinsics": True,
                "image_names": [names[1]],
            },
        ],
        "images": [
            {
                "name": names[0],
                "path": str(fisheye.relative_to(project)),
                "source_id": "primary",
                "source_role": "primary",
                "capture_index": 0,
                "sensor_id": "front",
                "width": 100,
                "height": 100,
                "valid_region": {
                    "kind": "fisheye",
                    "camera_model": "OPENCV_FISHEYE",
                    "params": [25.0, 25.0, 55.0, 50.0, 0.0, 0.0, 0.0, 0.0],
                    "max_theta_rad": 1.4,
                    "physical_circle": {"cx": 0.5, "cy": 0.5, "r": 0.49},
                },
            },
            {
                "name": names[1],
                "path": str(phone.relative_to(project)),
                "source_id": "phone",
                "source_role": "supplemental",
                "capture_index": 1,
                "sensor_id": "main",
                "width": 100,
                "height": 100,
                "valid_region": {"kind": "full"},
            },
        ],
    }
    (prepared / "image_catalog.json").write_text(json.dumps(catalog))
    (prepared / "rig_config.json").write_text("[]")
    _write_feature_masks(project, names)
    output = tmp_path / "output"
    output.mkdir()

    spec = input_workspace.build(project, output, use_feature_masks=True)

    assert spec.version == 3
    assert spec.feature_masks_enabled is True
    assert spec.image_count == 2
    assert spec.source_count == 2
    assert [batch.camera_model for batch in spec.feature_batches] == [
        "OPENCV_FISHEYE",
        "SIMPLE_RADIAL",
    ]
    assert (output / "images" / names[0]).is_file()
    assert (output / "images" / names[1]).is_file()
    assert (output / "masks" / f"{names[0]}.png").is_file()
    assert (output / "masks" / f"{names[1]}.png").is_file()
    assert (output / "rig_config.json").is_file()
    assert input_workspace.InputSpec.read(output / "input_spec.json") == spec


def test_workspace_without_feature_masks_keeps_only_physical_valid_regions(tmp_path):
    project = tmp_path / "project"
    prepared = project / "rectify_fisheye"
    prepared.mkdir(parents=True)
    source = prepared / "image.jpg"
    PilImage.new("RGB", (100, 100), (10, 20, 30)).save(source, format="JPEG")
    catalog = {
        "version": 1,
        "reconstruction_mode": "native_fisheye",
        "primary_source_id": "primary",
        "sources": [
            {"id": "primary", "label": "360", "role": "primary", "projection": "dual_fisheye"}
        ],
        "rig_config_path": None,
        "camera_groups": [],
        "images": [
            {
                "name": "sources/primary/front/frame_000000.jpg",
                "path": str(source.relative_to(project)),
                "source_id": "primary",
                "source_role": "primary",
                "capture_index": 0,
                "sensor_id": "front",
                "width": 100,
                "height": 100,
                "valid_region": {
                    "kind": "fisheye",
                    "camera_model": "OPENCV_FISHEYE",
                    "params": [25.0, 25.0, 55.0, 50.0, 0.0, 0.0, 0.0, 0.0],
                    "max_theta_rad": 1.4,
                    "physical_circle": {"cx": 0.5, "cy": 0.5, "r": 0.49},
                },
            }
        ],
    }
    (prepared / "image_catalog.json").write_text(json.dumps(catalog))
    output = tmp_path / "output"
    output.mkdir()

    spec = input_workspace.build(project, output, use_feature_masks=False)

    assert spec.feature_masks_enabled is False
    assert spec.mask_path == "masks"
    mask_path = output / "masks" / "sources/primary/front/frame_000000.jpg.png"
    assert mask_path.is_file()
    with PilImage.open(mask_path) as mask:
        assert mask.getpixel((55, 50)) == 255
        assert mask.getpixel((1, 50)) == 0


def test_workspace_with_full_regions_and_feature_masks_disabled_has_no_mask_path(tmp_path):
    project = tmp_path / "project"
    prepared = project / "rectify_fisheye"
    prepared.mkdir(parents=True)
    source = prepared / "image.jpg"
    PilImage.new("RGB", (100, 100), (10, 20, 30)).save(source, format="JPEG")
    catalog = {
        "version": 1,
        "reconstruction_mode": "native_fisheye",
        "primary_source_id": "phone",
        "sources": [
            {"id": "phone", "label": "Phone", "role": "primary", "projection": "perspective"}
        ],
        "rig_config_path": None,
        "camera_groups": [],
        "images": [
            {
                "name": "sources/phone/main/image.jpg",
                "path": str(source.relative_to(project)),
                "source_id": "phone",
                "source_role": "primary",
                "capture_index": 0,
                "sensor_id": "main",
                "width": 100,
                "height": 100,
                "valid_region": {"kind": "full"},
            }
        ],
    }
    (prepared / "image_catalog.json").write_text(json.dumps(catalog))
    output = tmp_path / "output"
    output.mkdir()

    spec = input_workspace.build(project, output, use_feature_masks=False)

    assert spec.mask_path is None
    assert not (output / "masks").exists()


def test_workspace_rejects_image_dimensions_changed_after_prepare(tmp_path):
    project = tmp_path / "project"
    prepared = project / "rectify_fisheye"
    prepared.mkdir(parents=True)
    source = prepared / "image.jpg"
    PilImage.new("RGB", (50, 50), (10, 20, 30)).save(source, format="JPEG")
    catalog = {
        "version": 1,
        "reconstruction_mode": "native_fisheye",
        "primary_source_id": "primary",
        "sources": [
            {"id": "primary", "label": "Primary", "role": "primary", "projection": "dual_fisheye"}
        ],
        "rig_config_path": None,
        "camera_groups": [],
        "images": [
            {
                "name": "front/image.jpg",
                "path": str(source.relative_to(project)),
                "source_id": "primary",
                "source_role": "primary",
                "capture_index": 0,
                "sensor_id": "front",
                "width": 100,
                "height": 100,
                "valid_region": {"kind": "full"},
            }
        ],
    }
    (prepared / "image_catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()

    with pytest.raises(RuntimeError, match="dimensions changed"):
        input_workspace.build(project, output, use_feature_masks=False)
