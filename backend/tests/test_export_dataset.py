"""LFStudio が選択ディレクトリをそのまま読み込める export layout を検証する."""

from __future__ import annotations

import json
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
    preview = project / "align_reconstruction" / "preview"
    preview.mkdir(parents=True)
    (preview / "reconstruction.json").write_text("{}")
    (preview / "points.bin").write_bytes(b"points")


def _write_rgb(path: Path, size: tuple[int, int] = (64, 64)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (40, 80, 120)).save(path, format="JPEG")


def _write_mask(path: Path, size: tuple[int, int] = (64, 64)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, 255).save(path, format="PNG")


def _execute(project: Path, raw_params: dict | None = None) -> Path:
    output = project / ".export_dataset.tmp"
    output.mkdir()
    reconstruction = colmap_model.read_model(project / "align_reconstruction" / "sparse" / "0")
    names = [image.name for image in reconstruction.images.values()]
    spec = InputSpec(
        version=2,
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
        rig_config_path="rig_config.json",
        refine_intrinsics=True,
        refine_rig=False,
        multiple_models=False,
    )
    input_spec = project / "extract_features" / "input_spec.json"
    input_spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write(input_spec)
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
    _write_model(project / "align_reconstruction" / "sparse" / "0")
    _write_preview(project)
    image = project / "extract_features" / "images" / "front" / "frame_000000.jpg"
    _write_rgb(image)
    mask = project / "extract_features" / "masks" / "front" / "frame_000000.jpg.png"
    _write_mask(mask)

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

    recommendation = json.loads((output / "train_configs" / "recommendations.json").read_text())
    assert "--data-path <export_dataset>" in recommendation["usage"]
    assert "--output-path" not in recommendation["usage"]
    assert "command_template" not in recommendation
    assert "training_output_dir" not in recommendation
    assert recommendation["gui_integration"]["train_configs_auto_applied"] is False
    required_settings = recommendation["gui_integration"]["required_settings"]
    assert required_settings["ppisp"] is True
    assert required_settings["ppisp_controller"] is True
    config = json.loads((output / "train_configs" / "train_config.mrnf.json").read_text())
    assert config["mask_mode"] == "segment"
    assert config["use_ppisp"] is True
    assert config["ppisp_use_controller"] is True


@pytest.mark.parametrize("failure", ["corrupt", "wrong_size"])
def test_export_rejects_invalid_registered_image(tmp_path: Path, failure: str):
    project = tmp_path / "project"
    _write_model(project / "align_reconstruction" / "sparse" / "0")
    _write_preview(project)
    image = project / "extract_features" / "images" / "front" / "frame_000000.jpg"
    image.parent.mkdir(parents=True)
    if failure == "corrupt":
        image.write_bytes(b"not an image")
    else:
        _write_rgb(image, (32, 64))

    with pytest.raises(RuntimeError, match="LFStudio export"):
        _execute(project, {"emit_train_configs": False})


@pytest.mark.parametrize("failure", ["corrupt", "wrong_size"])
def test_export_rejects_invalid_registered_mask(tmp_path: Path, failure: str):
    project = tmp_path / "project"
    _write_model(project / "align_reconstruction" / "sparse" / "0")
    _write_preview(project)
    _write_rgb(project / "extract_features" / "images" / "front" / "frame_000000.jpg")
    mask = project / "extract_features" / "masks" / "front" / "frame_000000.jpg.png"
    mask.parent.mkdir(parents=True)
    if failure == "corrupt":
        mask.write_bytes(b"not a mask")
    else:
        _write_mask(mask, (63, 64))

    with pytest.raises(RuntimeError, match="LFStudio export"):
        _execute(project, {"emit_train_configs": False})


def test_export_rejects_camera_model_unsupported_by_lf_studio(tmp_path: Path):
    project = tmp_path / "project"
    _write_model(project / "align_reconstruction" / "sparse" / "0", model_id=12)
    _write_preview(project)
    _write_rgb(project / "extract_features" / "images" / "front" / "frame_000000.jpg")

    with pytest.raises(RuntimeError, match="unsupported_camera_models"):
        _execute(project, {"emit_train_configs": False})


def test_unmatched_mask_does_not_enable_segment_mode(tmp_path: Path):
    project = tmp_path / "project"
    _write_model(project / "align_reconstruction" / "sparse" / "0")
    _write_preview(project)
    _write_rgb(project / "extract_features" / "images" / "front" / "frame_000000.jpg")
    _write_mask(project / "extract_features" / "masks" / "unrelated.png")

    output = _execute(project)

    config = json.loads((output / "train_configs" / "train_config.mrnf.json").read_text())
    exported = json.loads((output / "export_manifest.json").read_text())
    assert config["mask_mode"] == "none"
    assert exported["masks"] == 0
    assert not (output / "masks" / "unrelated.png").exists()


def test_stationary_rig_is_detected_from_reference_sensor_trajectory(tmp_path: Path):
    project = tmp_path / "project"
    _write_stationary_rig_model(project / "align_reconstruction" / "sparse" / "0")
    _write_preview(project)
    for lens in ("front", "back"):
        for index in range(2):
            _write_rgb(project / "extract_features" / "images" / lens / f"frame_{index:06d}.jpg")

    output = _execute(project, {"emit_train_configs": False})

    validation = json.loads((output / "export_manifest.json").read_text())["validation"]
    assert validation["loadable"] is True
    assert validation["training_ready"] is False
    assert validation["reference_camera_trajectory_diameter"] == 0.0
    assert validation["reference_unique_camera_centers"] == 1
    assert validation["camera_trajectory_diameter"] > 0.0
    assert "collapsed_reference_camera_trajectory" in validation["warnings"]
