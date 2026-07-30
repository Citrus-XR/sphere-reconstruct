"""変換 Step ごとに最新の Scene View preview が選ばれることを検証する。"""

import pytest
from fastapi import HTTPException

from sphere_reconstruct.api.previews import _latest_transform_preview


def test_latest_transform_preview_follows_pipeline_order(tmp_path):
    aligned = tmp_path / "align_reconstruction" / "preview" / "reconstruction.json"
    aligned.parent.mkdir(parents=True)
    aligned.write_text("aligned")
    assert _latest_transform_preview(tmp_path, "reconstruction.json") == aligned

    scaled = tmp_path / "restore_metric_scale" / "preview" / "reconstruction.json"
    scaled.parent.mkdir(parents=True)
    scaled.write_text("scaled")
    assert _latest_transform_preview(tmp_path, "reconstruction.json") == scaled

    grounded = tmp_path / "position_ground" / "preview" / "reconstruction.json"
    grounded.parent.mkdir(parents=True)
    grounded.write_text("grounded")
    assert _latest_transform_preview(tmp_path, "reconstruction.json") == grounded

    dense = tmp_path / "dense_initialization" / "preview" / "reconstruction.json"
    dense.parent.mkdir(parents=True)
    dense.write_text("dense")
    assert _latest_transform_preview(tmp_path, "reconstruction.json") == dense


def test_latest_transform_preview_requires_one_completed_transform(tmp_path):
    with pytest.raises(HTTPException, match="preview not available"):
        _latest_transform_preview(tmp_path, "points.bin")
