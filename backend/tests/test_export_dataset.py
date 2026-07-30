"""LFStudio が選択ディレクトリをそのまま読み込める export layout を検証する."""

from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path

import pytest
from PIL import Image

from sphere_reconstruct.colmap import model as colmap_model
from sphere_reconstruct.colmap.input_workspace import InputSpec
from sphere_reconstruct.pipeline.stage import ProgressReporter, StageContext
from sphere_reconstruct.stages.export_dataset import ExportDataset


def _write_model(model_dir: Path, *, model_id: int = 5, width: int = 64, height: int = 64) -> None:
    parameter_counts = {5: 8, 12: 4}
    model_dir.mkdir(parents=True)
    with (model_dir / "cameras.bin").open("wb") as file:
        file.write(struct.pack("<Q", 1))
        file.write(struct.pack("<iiQQ", 1, model_id, width, height))
        params = [20, 20, width / 2, height / 2, 0, 0, 0, 0][: parameter_counts[model_id]]
        file.write(struct.pack(f"<{len(params)}d", *params))
    with (model_dir / "images.bin").open("wb") as file:
        file.write(struct.pack("<Q", 1))
        file.write(struct.pack("<idddddddi", 1, 1, 0, 0, 0, 1, 2, 3, 1))
        file.write(b"front/frame_000000.jpg\x00")
        file.write(struct.pack("<Q", 0))
    with (model_dir / "points3D.bin").open("wb") as file:
        file.write(struct.pack("<Q", 0))
    (model_dir / "rigs.bin").write_bytes(b"rig")
    (model_dir / "frames.bin").write_bytes(b"frame")


def _write_stationary_rig_model(model_dir: Path) -> None:
    model_dir.mkdir(parents=True)
    with (model_dir / "cameras.bin").open("wb") as file:
        file.write(struct.pack("<Q", 2))
        for camera_id in (1, 2):
            file.write(struct.pack("<iiQQ", camera_id, 5, 64, 64))
            file.write(struct.pack("<8d", 20, 20, 32, 32, 0, 0, 0, 0))
    records = [
        (1, 1, (0.0, 0.0, 0.0), "front/frame_000000.jpg"),
        (2, 1, (0.0, 0.0, 0.0), "front/frame_000001.jpg"),
        (3, 2, (-0.03, 0.0, 0.0), "back/frame_000000.jpg"),
        (4, 2, (-0.03, 0.0, 0.0), "back/frame_000001.jpg"),
    ]
    with (model_dir / "images.bin").open("wb") as file:
        file.write(struct.pack("<Q", len(records)))
        for image_id, camera_id, translation, name in records:
            file.write(
                struct.pack(
                    "<idddddddi",
                    image_id,
                    1,
                    0,
                    0,
                    0,
                    *translation,
                    camera_id,
                )
            )
            file.write(name.encode() + b"\x00")
            file.write(struct.pack("<Q", 0))
    with (model_dir / "points3D.bin").open("wb") as file:
        file.write(struct.pack("<Q", 0))
    (model_dir / "rigs.bin").write_bytes(b"rig")
    (model_dir / "frames.bin").write_bytes(b"frame")


def _write_preview(project: Path) -> None:
    preview = project / "position_ground" / "preview"
    preview.mkdir(parents=True)
    (preview / "reconstruction.json").write_text("{}")
    (preview / "points.bin").write_bytes(b"points")
    scale = project / "restore_metric_scale" / "scale_restoration.json"
    scale.parent.mkdir(parents=True)
    scale.write_text(json.dumps({"metric": True, "scale_factor": 1.0}))
    (project / "position_ground" / "ground_position.json").write_text(
        json.dumps({"applied": True, "ground_y": 0.0})
    )


def _write_rgb(path: Path, size: tuple[int, int] = (64, 64)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (40, 80, 120)).save(path, format="JPEG")


def _write_mask(path: Path, size: tuple[int, int] = (64, 64), value: int = 255) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, value).save(path, format="PNG")


def _write_mask_artifact(
    project: Path,
    purpose: str,
    *,
    image_name: str = "front/frame_000000.jpg",
    size: tuple[int, int] = (64, 64),
    value: int = 255,
    corrupt: bool = False,
) -> None:
    stage = f"generate_{purpose}_masks"
    mask = project / stage / f"{image_name}.png"
    mask.parent.mkdir(parents=True, exist_ok=True)
    if corrupt:
        mask.write_bytes(b"not a mask")
    else:
        _write_mask(mask, size, value)
    manifest = {
        "version": 3,
        "purpose": purpose,
        "prompt": ["person"],
        "max_inference_size": 1024,
        "dilate_px": 8,
        "images": [
            {
                "name": image_name,
                "source_id": "primary",
                "capture_index": 0,
                "path": str(mask.relative_to(project)),
                "coverage": 0.1,
                "coverage_warning": False,
                "detections": {"person": 1},
            }
        ],
    }
    (project / stage / "manifest_masks.json").write_text(json.dumps(manifest))


def _execute(project: Path, raw_params: dict | None = None) -> Path:
    output = project / ".export_dataset.tmp"
    output.mkdir()
    dense_model = project / "dense_initialization" / "sparse" / "0"
    if not dense_model.exists():
        shutil.copytree(project / "position_ground" / "sparse" / "0", dense_model)
    reconstruction = colmap_model.read_model(dense_model)
    names = [image.name for image in reconstruction.images.values()]
    spec = InputSpec(
        version=3,
        reconstruction_mode="native_fisheye",
        image_count=len(names),
        source_count=1,
        primary_source_id="primary",
        primary_image_names=names,
        sources=[{"id": "primary", "label": "Primary", "role": "primary", "projection": "dual_fisheye"}],
        images=[
            {
                "name": name,
                "source_id": "primary",
                "source_role": "primary",
                "capture_index": index,
                "sensor_id": "front",
            }
            for index, name in enumerate(names)
        ],
        feature_batches=[],
        image_path="images",
        mask_path="masks",
        feature_masks_enabled=True,
        rig_config_path="rig_config.json",
        refine_intrinsics=True,
        refine_rig=False,
        multiple_models=False,
    )
    input_spec = project / "extract_features" / "input_spec.json"
    input_spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write(input_spec)
    catalog_path = project / "prepare_images" / "image_catalog.json"
    if not catalog_path.is_file():
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        catalog_path.write_text(
            json.dumps(
                {
                    "images": [
                        {
                            "name": image.name,
                            "width": reconstruction.cameras[image.camera_id].width,
                            "height": reconstruction.cameras[image.camera_id].height,
                            "valid_region": {"kind": "full"},
                        }
                        for image in reconstruction.images.values()
                    ]
                }
            ),
            encoding="utf-8",
        )
    stage = ExportDataset()
    ctx = StageContext(
        project_id="test",
        project_dir=project,
        stage_out_dir=output,
        params=stage.normalize_params(raw_params or {}),
        sources=(),
        progress=ProgressReporter(lambda *_args: None),
    )
    stage.execute(ctx)
    return output


def test_export_root_is_directly_loadable_by_lf_studio(tmp_path: Path):
    project = tmp_path / "project"
    _write_model(project / "position_ground" / "sparse" / "0")
    _write_preview(project)
    image = project / "extract_features" / "images" / "front" / "frame_000000.jpg"
    _write_rgb(image)
    _write_mask_artifact(project, "feature", value=64)
    _write_mask_artifact(project, "training", value=192)

    output = _execute(project)

    assert not (output / "dataset").exists()
    assert (output / "images" / "front" / "frame_000000.jpg").is_file()
    assert (output / "masks" / "front" / "frame_000000.jpg.png").is_file()
    for name in ("rigs.bin", "cameras.bin", "frames.bin", "images.bin", "points3D.bin"):
        assert (output / "sparse" / "0" / name).is_file()

    export_manifest = json.loads((output / "export_manifest.json").read_text())
    assert export_manifest["load_in_lichtfeld_studio"] == "."
    assert export_manifest["validation"]["loadable"] is True
    assert export_manifest["validation"]["unique_camera_centers"] == 1
    assert export_manifest["validation"]["matched_mask_count"] == 1
    assert export_manifest["mask_source"] == "training"
    with Image.open(output / "masks" / "front" / "frame_000000.jpg.png") as exported_mask:
        assert exported_mask.getpixel((0, 0)) == 192

    recommendation = json.loads((output / "train_configs" / "recommendations.json").read_text())
    assert "--data-path <export_dataset>" in recommendation["usage"]
    assert "--output-path" not in recommendation["usage"]
    assert "command_template" not in recommendation
    assert "training_output_dir" not in recommendation
    assert recommendation["gui_integration"]["train_configs_auto_applied"] is False
    required_settings = recommendation["gui_integration"]["required_settings"]
    assert required_settings["ppisp"] is False
    assert required_settings["ppisp_controller"] is False
    config = json.loads((output / "train_configs" / "train_config.mrnf.json").read_text())
    assert config["mask_mode"] == "segment"
    assert config["use_ppisp"] is False
    assert config["ppisp_use_controller"] is False


def test_export_losslessly_crops_fisheye_training_dataset(tmp_path: Path):
    if shutil.which("jpegtran") is None:
        pytest.skip("jpegtran is not installed on this test host")
    project = tmp_path / "project"
    _write_model(project / "position_ground" / "sparse" / "0")
    _write_preview(project)
    image_name = "front/frame_000000.jpg"
    _write_rgb(project / "extract_features" / "images" / image_name)
    _write_mask_artifact(project, "training")
    catalog = {
        "images": [
            {
                "name": image_name,
                "width": 64,
                "height": 64,
                "valid_region": {"kind": "circle", "cx": 0.5, "cy": 0.5, "r": 0.25},
            }
        ]
    }
    catalog_path = project / "prepare_images" / "image_catalog.json"
    catalog_path.parent.mkdir(parents=True)
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

    output = _execute(project)
    exported = colmap_model.read_model(output / "sparse" / "0")

    assert (exported.cameras[1].width, exported.cameras[1].height) == (32, 32)
    assert exported.cameras[1].params[2:4] == [16.0, 16.0]
    with Image.open(output / "images" / image_name) as image:
        assert image.size == (32, 32)
    with Image.open(output / "masks" / f"{image_name}.png") as mask:
        assert mask.size == (32, 32)
    manifest = json.loads((output / "export_manifest.json").read_text())
    assert manifest["training_crop"]["enabled"] is True
    assert manifest["training_crop"]["lossless"] is True


@pytest.mark.parametrize("failure", ["corrupt", "wrong_size"])
def test_export_rejects_invalid_registered_image(tmp_path: Path, failure: str):
    project = tmp_path / "project"
    _write_model(project / "position_ground" / "sparse" / "0")
    _write_preview(project)
    image = project / "extract_features" / "images" / "front" / "frame_000000.jpg"
    image.parent.mkdir(parents=True)
    if failure == "corrupt":
        image.write_bytes(b"not an image")
    else:
        _write_rgb(image, (32, 64))

    with pytest.raises(RuntimeError, match="LFStudio export"):
        _execute(
            project,
            {
                "emit_train_configs": False,
                "feature_masks_enabled": False,
                "training_masks_enabled": False,
            },
        )


@pytest.mark.parametrize("failure", ["corrupt", "wrong_size"])
def test_export_rejects_invalid_registered_mask(tmp_path: Path, failure: str):
    project = tmp_path / "project"
    _write_model(project / "position_ground" / "sparse" / "0")
    _write_preview(project)
    _write_rgb(project / "extract_features" / "images" / "front" / "frame_000000.jpg")
    _write_mask_artifact(
        project,
        "training",
        corrupt=failure == "corrupt",
        size=(63, 64) if failure == "wrong_size" else (64, 64),
    )

    with pytest.raises(RuntimeError, match="LFStudio export"):
        _execute(
            project,
            {
                "emit_train_configs": False,
                "feature_masks_enabled": False,
                "training_masks_enabled": True,
            },
        )


def test_export_rejects_camera_model_unsupported_by_lf_studio(tmp_path: Path):
    project = tmp_path / "project"
    _write_model(project / "position_ground" / "sparse" / "0", model_id=12)
    _write_preview(project)
    _write_rgb(project / "extract_features" / "images" / "front" / "frame_000000.jpg")

    with pytest.raises(RuntimeError, match="unsupported_camera_models"):
        _execute(
            project,
            {
                "emit_train_configs": False,
                "feature_masks_enabled": False,
                "training_masks_enabled": False,
            },
        )


def test_export_rejects_selected_mask_channel_missing_registered_images(tmp_path: Path):
    project = tmp_path / "project"
    _write_model(project / "position_ground" / "sparse" / "0")
    _write_preview(project)
    _write_rgb(project / "extract_features" / "images" / "front" / "frame_000000.jpg")
    _write_mask_artifact(project, "training", image_name="unrelated.jpg")

    with pytest.raises(RuntimeError, match="missing registered images"):
        _execute(project)


@pytest.mark.parametrize(
    ("feature_enabled", "training_enabled", "expected_source", "expected_value"),
    [
        (True, True, "training", 192),
        (True, False, "feature", 64),
        (False, True, "training", 192),
        (False, False, "physical", 255),
    ],
)
def test_export_uses_one_resolved_mask_channel(
    tmp_path: Path,
    feature_enabled: bool,
    training_enabled: bool,
    expected_source: str | None,
    expected_value: int | None,
):
    project = tmp_path / "project"
    _write_model(project / "position_ground" / "sparse" / "0")
    _write_preview(project)
    _write_rgb(project / "extract_features" / "images" / "front" / "frame_000000.jpg")
    _write_mask_artifact(project, "feature", value=64)
    _write_mask_artifact(project, "training", value=192)

    output = _execute(
        project,
        {
            "feature_masks_enabled": feature_enabled,
            "training_masks_enabled": training_enabled,
        },
    )

    manifest = json.loads((output / "export_manifest.json").read_text())
    exported = output / "masks" / "front" / "frame_000000.jpg.png"
    assert manifest["mask_source"] == expected_source
    assert exported.exists() is (expected_source is not None)
    if expected_value is not None:
        with Image.open(exported) as mask:
            assert mask.getpixel((0, 0)) == expected_value


def test_export_keeps_physical_fisheye_mask_when_both_sam_steps_are_disabled(tmp_path: Path):
    project = tmp_path / "project"
    _write_model(project / "position_ground" / "sparse" / "0")
    _write_preview(project)
    _write_rgb(project / "extract_features" / "images" / "front" / "frame_000000.jpg")
    catalog = {
        "images": [
            {
                "name": "front/frame_000000.jpg",
                "width": 64,
                "height": 64,
                "valid_region": {"kind": "circle", "cx": 0.5, "cy": 0.5, "r": 0.4},
            }
        ]
    }
    catalog_path = project / "prepare_images" / "image_catalog.json"
    catalog_path.parent.mkdir(parents=True)
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

    output = _execute(
        project,
        {"feature_masks_enabled": False, "training_masks_enabled": False},
    )

    manifest = json.loads((output / "export_manifest.json").read_text())
    assert manifest["mask_source"] == "physical"
    with Image.open(output / "masks" / "front" / "frame_000000.jpg.png") as mask:
        assert mask.getpixel((32, 32)) == 255
        assert mask.getpixel((0, 0)) == 0


def test_stationary_rig_is_detected_from_reference_sensor_trajectory(tmp_path: Path):
    project = tmp_path / "project"
    _write_stationary_rig_model(project / "position_ground" / "sparse" / "0")
    _write_preview(project)
    for lens in ("front", "back"):
        for index in range(2):
            _write_rgb(project / "extract_features" / "images" / lens / f"frame_{index:06d}.jpg")

    output = _execute(
        project,
        {
            "emit_train_configs": False,
            "feature_masks_enabled": False,
            "training_masks_enabled": False,
        },
    )

    validation = json.loads((output / "export_manifest.json").read_text())["validation"]
    assert validation["loadable"] is True
    assert validation["training_ready"] is False
    assert validation["reference_camera_trajectory_diameter"] == 0.0
    assert validation["reference_unique_camera_centers"] == 1
    assert validation["camera_trajectory_diameter"] > 0.0
    assert "collapsed_reference_camera_trajectory" in validation["warnings"]
