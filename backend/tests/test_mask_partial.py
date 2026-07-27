"""SAM3 の生成途中 preview manifest は complete record だけを公開する。"""

from __future__ import annotations

from sphere_reconstruct.domain.mask_artifact import (
    MASK_MANIFEST_VERSION,
    MaskPurpose,
    append_partial_mask_record,
    initialise_partial_mask_manifest,
    load_partial_mask_manifest,
    remove_partial_mask_manifest,
)


def test_partial_mask_manifest_is_incrementally_readable(tmp_path):
    stage_dir = tmp_path / ".generate_feature_masks.tmp"
    stage_dir.mkdir()
    records_path = initialise_partial_mask_manifest(
        stage_dir,
        {
            "version": MASK_MANIFEST_VERSION,
            "purpose": "feature",
            "revision": "run-1",
            "prompt": ["person"],
            "max_inference_size": 2048,
            "dilate_px": 8,
            "total_images": 2,
        },
    )
    record = {
        "name": "sources/source/front/frame_000000.jpg",
        "source_id": "source",
        "capture_index": 0,
        "path": "generate_feature_masks/sources/source/front/frame_000000.jpg.png",
        "coverage": 0.1,
        "coverage_warning": False,
        "detections": {"person": 1},
    }
    append_partial_mask_record(records_path, record)

    document = load_partial_mask_manifest(stage_dir, MaskPurpose.FEATURE)

    assert document is not None
    assert document["complete"] is False
    assert document["generated_images"] == 1
    assert document["total_images"] == 2
    assert document["revision"] == "run-1"
    assert document["images"] == [record]

    remove_partial_mask_manifest(stage_dir)
    assert load_partial_mask_manifest(stage_dir, MaskPurpose.FEATURE) is None
