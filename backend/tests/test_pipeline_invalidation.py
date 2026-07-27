"""分割 stage の transitive artifact invalidation を検証する."""

import pytest

from sphere_reconstruct.domain.artifacts import manifest_path
from sphere_reconstruct.domain.pipeline_state import STAGE_ORDER, PipelineState, StageName
from sphere_reconstruct.pipeline.invalidation import (
    assert_export_is_managed,
    derive_pipeline_state,
    invalidate_from,
    is_stale,
)


def _populate(project_dir):
    for stage in STAGE_ORDER:
        (project_dir / stage.value).mkdir(parents=True)
        manifest = manifest_path(project_dir, stage.value)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("{}")


def test_matching_change_preserves_features_and_invalidates_consumers(tmp_path):
    _populate(tmp_path)
    invalidated = invalidate_from(tmp_path, StageName.MATCH_FEATURES, include_self=True)
    assert invalidated[0] == StageName.MATCH_FEATURES
    assert (tmp_path / StageName.EXTRACT_FEATURES.value).is_dir()
    assert not is_stale(tmp_path, StageName.EXTRACT_FEATURES)
    assert manifest_path(tmp_path, StageName.EXTRACT_FEATURES.value).is_file()
    for stage in invalidated:
        assert not (tmp_path / stage.value).exists()
        assert not manifest_path(tmp_path, stage.value).exists()
        if stage != StageName.MATCH_FEATURES:
            assert is_stale(tmp_path, stage)


def test_mapper_change_preserves_matching(tmp_path):
    _populate(tmp_path)
    invalidate_from(tmp_path, StageName.RECONSTRUCT, include_self=True)
    assert (tmp_path / StageName.MATCH_FEATURES.value).is_dir()
    assert not (tmp_path / StageName.RECONSTRUCT.value).exists()
    assert not (tmp_path / StageName.ALIGN_RECONSTRUCTION.value).exists()
    assert not (tmp_path / StageName.EXPORT_DATASET.value).exists()


def test_feature_change_invalidates_every_consumer(tmp_path):
    _populate(tmp_path)
    invalidated = invalidate_from(tmp_path, StageName.EXTRACT_FEATURES, include_self=False)
    assert StageName.MATCH_FEATURES in invalidated
    assert StageName.EXPORT_DATASET in invalidated
    assert derive_pipeline_state(tmp_path) == PipelineState.FEATURES_EXTRACTED


def test_feature_mask_change_preserves_training_masks_and_invalidates_sfm(tmp_path):
    _populate(tmp_path)
    invalidated = invalidate_from(tmp_path, StageName.GENERATE_FEATURE_MASKS, include_self=True)

    assert StageName.EXTRACT_FEATURES in invalidated
    assert StageName.EXPORT_DATASET in invalidated
    assert StageName.GENERATE_TRAINING_MASKS not in invalidated
    assert (tmp_path / StageName.GENERATE_TRAINING_MASKS.value).is_dir()


def test_training_mask_change_only_invalidates_export(tmp_path):
    _populate(tmp_path)
    invalidated = invalidate_from(tmp_path, StageName.GENERATE_TRAINING_MASKS, include_self=True)

    assert invalidated == [StageName.GENERATE_TRAINING_MASKS, StageName.EXPORT_DATASET]
    assert (tmp_path / StageName.EXTRACT_FEATURES.value).is_dir()
    assert (tmp_path / StageName.ALIGN_RECONSTRUCTION.value).is_dir()


def test_summary_state_follows_last_main_branch_artifact(tmp_path):
    _populate(tmp_path)
    (tmp_path / StageName.EXPORT_DATASET.value).rmdir()
    manifest_path(tmp_path, StageName.EXPORT_DATASET.value).unlink()
    assert derive_pipeline_state(tmp_path) == PipelineState.ALIGNED
    (tmp_path / StageName.ALIGN_RECONSTRUCTION.value).rmdir()
    manifest_path(tmp_path, StageName.ALIGN_RECONSTRUCTION.value).unlink()
    assert derive_pipeline_state(tmp_path) == PipelineState.RECONSTRUCTED


def test_missing_consumers_are_not_marked_stale(tmp_path):
    (tmp_path / StageName.INSPECT_SOURCE.value).mkdir()
    invalidate_from(tmp_path, StageName.INSPECT_SOURCE, include_self=False)
    for stage in STAGE_ORDER[1:]:
        assert not is_stale(tmp_path, stage)


def test_unmanaged_lfstudio_outputs_block_destructive_export_replacement(tmp_path):
    export = tmp_path / "export_dataset"
    (export / "images").mkdir(parents=True)
    (export / "images" / "managed.jpg").write_bytes(b"managed")
    checkpoint = export / "output" / "checkpoints" / "checkpoint.resume"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    custom = export / "my_training" / "model.ply"
    custom.parent.mkdir()
    custom.write_bytes(b"model")

    with pytest.raises(RuntimeError, match="管理外"):
        assert_export_is_managed(tmp_path)

    assert checkpoint.read_bytes() == b"checkpoint"
    assert custom.read_bytes() == b"model"
    assert (export / "images" / "managed.jpg").is_file()


def test_unmanaged_export_blocks_invalidation_before_any_stage_is_removed(tmp_path):
    _populate(tmp_path)
    external = tmp_path / "export_dataset" / "external-training" / "model.ply"
    external.parent.mkdir()
    external.write_bytes(b"model")

    with pytest.raises(RuntimeError, match="管理外"):
        invalidate_from(tmp_path, StageName.MATCH_FEATURES, include_self=True)

    assert (tmp_path / StageName.MATCH_FEATURES.value).is_dir()
    assert (tmp_path / StageName.RECONSTRUCT.value).is_dir()
    assert external.read_bytes() == b"model"
