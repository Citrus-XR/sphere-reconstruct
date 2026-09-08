"""分割 stage の transitive artifact invalidation を検証する."""

from pathlib import Path

import pytest

from sphere_reconstruct.domain.artifacts import manifest_path
from sphere_reconstruct.domain.pipeline_state import (
    STAGE_ORDER,
    PipelineState,
    StageName,
    requested_stage_plan,
)
from sphere_reconstruct.pipeline.invalidation import (
    ArtifactBusyError,
    clear_stages,
    derive_pipeline_state,
    invalidate_from,
    is_stale,
    pending_cleanup_paths,
)


def _populate(project_dir):
    for stage in STAGE_ORDER:
        (project_dir / stage.value).mkdir(parents=True)
        manifest = manifest_path(project_dir, stage.value)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("{}")


@pytest.mark.parametrize(
    "target",
    (
        StageName.GENERATE_FEATURE_MASKS,
        StageName.GENERATE_TRAINING_MASKS,
        StageName.EXTRACT_FEATURES,
    ),
)
def test_direct_consumer_request_runs_hidden_rectification_first(target):
    assert requested_stage_plan(target) == (StageName.RECTIFY_FISHEYE, target)


def test_visible_stage_without_internal_dependency_runs_alone():
    assert requested_stage_plan(StageName.MATCH_FEATURES) == (StageName.MATCH_FEATURES,)


def test_clear_after_frames_preserves_source_and_frame_artifacts(tmp_path):
    _populate(tmp_path)

    cleared = clear_stages(tmp_path, [StageName.PREPARE_IMAGES])

    assert cleared == list(STAGE_ORDER[2:])
    assert (tmp_path / StageName.INSPECT_SOURCE.value).is_dir()
    assert (tmp_path / StageName.EXTRACT_FRAMES.value).is_dir()
    assert not (tmp_path / StageName.PREPARE_IMAGES.value).exists()
    assert not (tmp_path / StageName.EXPORT_DATASET.value).exists()


def test_custom_clear_expands_consumer_closure_without_clearing_independent_branch(tmp_path):
    _populate(tmp_path)

    cleared = clear_stages(tmp_path, [StageName.GENERATE_TRAINING_MASKS])

    assert cleared == [StageName.GENERATE_TRAINING_MASKS, StageName.EXPORT_DATASET]
    assert (tmp_path / StageName.GENERATE_FEATURE_MASKS.value).is_dir()
    assert (tmp_path / StageName.RECONSTRUCT.value).is_dir()


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
    assert not (tmp_path / StageName.RESTORE_METRIC_SCALE.value).exists()
    assert not (tmp_path / StageName.SCENE_ALIGNMENT.value).exists()
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
    assert derive_pipeline_state(tmp_path) == PipelineState.DENSIFIED
    (tmp_path / StageName.DENSE_INITIALIZATION.value).rmdir()
    manifest_path(tmp_path, StageName.DENSE_INITIALIZATION.value).unlink()
    assert derive_pipeline_state(tmp_path) == PipelineState.CLEANED
    (tmp_path / StageName.CLEANUP_SPARSE.value).rmdir()
    manifest_path(tmp_path, StageName.CLEANUP_SPARSE.value).unlink()
    assert derive_pipeline_state(tmp_path) == PipelineState.SCENE_ALIGNED
    (tmp_path / StageName.SCENE_ALIGNMENT.value).rmdir()
    manifest_path(tmp_path, StageName.SCENE_ALIGNMENT.value).unlink()
    assert derive_pipeline_state(tmp_path) == PipelineState.SCALE_RESTORED
    (tmp_path / StageName.RESTORE_METRIC_SCALE.value).rmdir()
    manifest_path(tmp_path, StageName.RESTORE_METRIC_SCALE.value).unlink()
    assert derive_pipeline_state(tmp_path) == PipelineState.ALIGNED
    (tmp_path / StageName.ALIGN_RECONSTRUCTION.value).rmdir()
    manifest_path(tmp_path, StageName.ALIGN_RECONSTRUCTION.value).unlink()
    assert derive_pipeline_state(tmp_path) == PipelineState.RECONSTRUCTED


def test_missing_consumers_are_not_marked_stale(tmp_path):
    (tmp_path / StageName.INSPECT_SOURCE.value).mkdir()
    invalidate_from(tmp_path, StageName.INSPECT_SOURCE, include_self=False)
    for stage in STAGE_ORDER[1:]:
        assert not is_stale(tmp_path, stage)


def test_clearing_export_removes_lfstudio_outputs(tmp_path):
    export = tmp_path / "export_dataset"
    (export / "images").mkdir(parents=True)
    (export / "images" / "managed.jpg").write_bytes(b"managed")
    checkpoint = export / "output" / "checkpoints" / "checkpoint.resume"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    project = export / "project.licht"
    project.write_bytes(b"project")

    cleared = clear_stages(tmp_path, [StageName.EXPORT_DATASET])

    assert cleared == [StageName.EXPORT_DATASET]
    assert not export.exists()
    assert not checkpoint.exists()
    assert not project.exists()


def test_invalidation_removes_lfstudio_outputs_with_export(tmp_path):
    _populate(tmp_path)
    external = tmp_path / "export_dataset" / "external-training" / "model.ply"
    external.parent.mkdir()
    external.write_bytes(b"model")

    invalidated = invalidate_from(tmp_path, StageName.MATCH_FEATURES, include_self=True)

    assert StageName.EXPORT_DATASET in invalidated
    assert not (tmp_path / StageName.MATCH_FEATURES.value).exists()
    assert not (tmp_path / StageName.RECONSTRUCT.value).exists()
    assert not external.exists()
    assert not (tmp_path / StageName.EXPORT_DATASET.value).exists()


def test_locked_downstream_artifact_rolls_back_every_quarantined_stage(tmp_path, monkeypatch):
    _populate(tmp_path)
    original_rename = Path.rename

    def fail_on_export(source: Path, target: Path):
        if source == tmp_path / StageName.EXPORT_DATASET.value:
            raise PermissionError(13, "locked export")
        return original_rename(source, target)

    monkeypatch.setattr(Path, "rename", fail_on_export)

    with pytest.raises(ArtifactBusyError, match="in use by another process"):
        invalidate_from(tmp_path, StageName.DENSE_INITIALIZATION, include_self=True)

    for stage in (StageName.DENSE_INITIALIZATION, StageName.EXPORT_DATASET):
        assert (tmp_path / stage.value).is_dir()
        assert manifest_path(tmp_path, stage.value).is_file()


def test_locked_file_after_quarantine_is_reported_as_pending_cleanup(tmp_path, monkeypatch):
    from sphere_reconstruct.pipeline import invalidation as module

    _populate(tmp_path)
    original_rmtree = module.shutil.rmtree

    def keep_quarantine(path, *args, **kwargs):
        if ".pipeline" in Path(path).parts and "trash" in Path(path).parts:
            raise PermissionError(13, "locked file")
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(module.shutil, "rmtree", keep_quarantine)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    invalidated = invalidate_from(tmp_path, StageName.DENSE_INITIALIZATION, include_self=True)

    assert invalidated == [StageName.DENSE_INITIALIZATION, StageName.EXPORT_DATASET]
    assert not (tmp_path / StageName.DENSE_INITIALIZATION.value).exists()
    assert not (tmp_path / StageName.EXPORT_DATASET.value).exists()
    pending = pending_cleanup_paths(tmp_path)
    assert len(pending) == 1
    assert (tmp_path / pending[0] / "pending_cleanup.json").is_file()


def test_incomplete_rollback_preserves_backups_across_later_cleanup(tmp_path, monkeypatch):
    _populate(tmp_path)
    dense = tmp_path / StageName.DENSE_INITIALIZATION.value
    (dense / "payload.bin").write_bytes(b"recoverable")
    original_rename = Path.rename

    def fail_quarantine_and_rollback(source, target):
        if source == tmp_path / StageName.EXPORT_DATASET.value or target == dense:
            raise PermissionError("locked during rollback")
        return original_rename(source, target)

    with monkeypatch.context() as failure:
        failure.setattr(Path, "rename", fail_quarantine_and_rollback)
        with pytest.raises(RuntimeError, match="rollback failed"):
            invalidate_from(tmp_path, StageName.DENSE_INITIALIZATION, include_self=True)
    backups = list((tmp_path / ".pipeline" / "trash").rglob("payload.bin"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == b"recoverable"
    assert list((tmp_path / ".pipeline" / "trash").rglob("recovery_required.json"))
    assert pending_cleanup_paths(tmp_path) == []

    clear_stages(tmp_path, [StageName.EXPORT_DATASET])
    assert backups[0].read_bytes() == b"recoverable"


def test_cleanup_does_not_remove_an_in_progress_quarantine(tmp_path):
    active = tmp_path / ".pipeline" / "trash" / "active-quarantine"
    active.mkdir(parents=True)
    (active / "payload.bin").write_bytes(b"active")
    clear_stages(tmp_path, [StageName.EXPORT_DATASET])
    assert (active / "payload.bin").read_bytes() == b"active"
    assert pending_cleanup_paths(tmp_path) == []
