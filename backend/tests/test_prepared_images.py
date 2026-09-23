"""Internal image-normalization artifact の error contract。"""

import json

import pytest

from sphere_reconstruct.pipeline import prepared_images
from sphere_reconstruct.pipeline.errors import LocalizedError


def test_missing_rectified_catalog_has_localization_key(tmp_path):
    with pytest.raises(LocalizedError) as captured:
        prepared_images.load_catalog(tmp_path)

    assert captured.value.key == "error.rectify_required"


def test_load_catalog_migrates_legacy_focal_provenance_without_rewriting_artifact(tmp_path):
    catalog = {
        "camera_groups": [
            {"id": "calibrated", "refine_intrinsics": False},
            {"id": "unknown", "refine_intrinsics": True},
            {"id": "exif", "refine_intrinsics": True, "has_prior_focal_length": True},
        ],
    }
    path = prepared_images.catalog_path(tmp_path)
    path.parent.mkdir()
    artifact = json.dumps(catalog)
    path.write_text(artifact, encoding="utf-8")

    migrated = prepared_images.load_catalog(tmp_path)

    assert [group["has_prior_focal_length"] for group in migrated["camera_groups"]] == [True, False, True]
    assert path.read_text(encoding="utf-8") == artifact


def test_load_catalog_preserves_partial_image_catalog(tmp_path):
    path = prepared_images.catalog_path(tmp_path)
    path.parent.mkdir()
    path.write_text(json.dumps({"images": []}), encoding="utf-8")

    assert prepared_images.load_catalog(tmp_path) == {"images": []}
