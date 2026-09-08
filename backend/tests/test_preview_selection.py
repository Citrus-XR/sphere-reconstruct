"""変換 Step ごとに最新の Scene View preview が選ばれることを検証する。"""

import pytest
from fastapi import HTTPException
from PIL import Image

from sphere_reconstruct.api.previews import _image_source_dimensions, _latest_transform_preview


def test_latest_transform_preview_follows_pipeline_order(tmp_path):
    reconstructed = tmp_path / "reconstruct" / "preview" / "reconstruction.json"
    reconstructed.parent.mkdir(parents=True)
    reconstructed.write_text("reconstructed")
    assert _latest_transform_preview(tmp_path, "reconstruction.json") == reconstructed

    aligned = tmp_path / "align_reconstruction" / "preview" / "reconstruction.json"
    aligned.parent.mkdir(parents=True)
    aligned.write_text("aligned")
    assert _latest_transform_preview(tmp_path, "reconstruction.json") == aligned

    scaled = tmp_path / "restore_metric_scale" / "preview" / "reconstruction.json"
    scaled.parent.mkdir(parents=True)
    scaled.write_text("scaled")
    assert _latest_transform_preview(tmp_path, "reconstruction.json") == scaled

    scene_aligned = tmp_path / "scene_alignment" / "preview" / "reconstruction.json"
    scene_aligned.parent.mkdir(parents=True)
    scene_aligned.write_text("scene-aligned")
    assert _latest_transform_preview(tmp_path, "reconstruction.json") == scene_aligned

    cleaned = tmp_path / "cleanup_sparse" / "preview" / "reconstruction.json"
    cleaned.parent.mkdir(parents=True)
    cleaned.write_text("cleaned")
    assert _latest_transform_preview(tmp_path, "reconstruction.json") == cleaned

    dense = tmp_path / "dense_initialization" / "preview" / "reconstruction.json"
    dense.parent.mkdir(parents=True)
    dense.write_text("dense")
    assert _latest_transform_preview(tmp_path, "reconstruction.json") == dense


def test_latest_transform_preview_requires_reconstruction_or_transform(tmp_path):
    with pytest.raises(HTTPException, match="preview not available"):
        _latest_transform_preview(tmp_path, "points.bin")


def test_image_source_dimensions_reports_first_image_and_count(tmp_path):
    Image.new("RGB", (4032, 3024)).save(tmp_path / "a.jpg")
    Image.new("RGB", (1920, 1080)).save(tmp_path / "b.png")
    (tmp_path / "ignored.txt").write_text("x")

    assert _image_source_dimensions(tmp_path) == (4032, 3024, 2)
