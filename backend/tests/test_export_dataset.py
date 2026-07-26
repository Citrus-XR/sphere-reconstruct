"""LFStudio が選択ディレクトリをそのまま読み込める export layout を検証する."""

from __future__ import annotations

import json
import struct
from pathlib import Path

from sphere_reconstruct.pipeline.stage import ProgressReporter, StageContext
from sphere_reconstruct.stages.export_dataset import ExportDataset


def _write_model(model_dir: Path) -> None:
    model_dir.mkdir(parents=True)
    with (model_dir / "cameras.bin").open("wb") as file:
        file.write(struct.pack("<Q", 1))
        file.write(struct.pack("<iiQQ", 1, 5, 64, 64))
        file.write(struct.pack("<8d", 20, 20, 32, 32, 0, 0, 0, 0))
    with (model_dir / "images.bin").open("wb") as file:
        file.write(struct.pack("<Q", 1))
        file.write(struct.pack("<idddddddi", 1, 1, 0, 0, 0, 1, 2, 3, 1))
        file.write(b"front/frame_000000.jpg\x00")
        file.write(struct.pack("<Q", 0))
    with (model_dir / "points3D.bin").open("wb") as file:
        file.write(struct.pack("<Q", 0))
    (model_dir / "rigs.bin").write_bytes(b"rig")
    (model_dir / "frames.bin").write_bytes(b"frame")


def test_export_root_is_directly_loadable_by_lf_studio(tmp_path: Path):
    project = tmp_path / "project"
    _write_model(project / "align_reconstruction" / "sparse" / "0")
    preview = project / "align_reconstruction" / "preview"
    preview.mkdir(parents=True)
    (preview / "reconstruction.json").write_text("{}")
    (preview / "points.bin").write_bytes(b"points")
    image = project / "extract_features" / "images" / "front" / "frame_000000.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    mask = project / "extract_features" / "masks" / "front" / "frame_000000.jpg.png"
    mask.parent.mkdir(parents=True)
    mask.write_bytes(b"mask")

    output = project / ".export_dataset.tmp"
    output.mkdir()
    reporter = ProgressReporter(lambda *_args: None)
    stage = ExportDataset()
    ctx = StageContext(
        project_id="test",
        project_dir=project,
        stage_out_dir=output,
        params=stage.normalize_params({}),
        source_path=None,
        source_kind="insv",
        progress=reporter,
    )
    stage.execute(ctx)

    assert not (output / "dataset").exists()
    assert (output / "images" / "front" / "frame_000000.jpg").is_file()
    assert (output / "masks" / "front" / "frame_000000.jpg.png").is_file()
    for name in ("rigs.bin", "cameras.bin", "frames.bin", "images.bin", "points3D.bin"):
        assert (output / "sparse" / "0" / name).is_file()

    export_manifest = json.loads((output / "export_manifest.json").read_text())
    assert export_manifest["load_in_lichtfeld_studio"] == "."
    assert export_manifest["validation"]["loadable"] is True
    assert export_manifest["validation"]["unique_camera_centers"] == 1

    recommendation = json.loads((output / "train_configs" / "recommendations.json").read_text())
    assert recommendation["usage"].find("--output-path") >= 0
    config = json.loads((output / "train_configs" / "train_config.mrnf.json").read_text())
    assert config["mask_mode"] == "segment"
