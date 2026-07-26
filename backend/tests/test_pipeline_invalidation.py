"""分割 stage の transitive artifact invalidation を検証する."""

from sphere_reconstruct.domain.artifacts import manifest_path
from sphere_reconstruct.domain.pipeline_state import STAGE_ORDER, PipelineState, StageName
from sphere_reconstruct.pipeline.invalidation import (
    derive_pipeline_state,
    invalidate_from,
    is_stale,
    preserve_export_outputs,
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
    assert (tmp_path / StageName.DENOISE_FRAMES.value).is_dir()
    assert manifest_path(tmp_path, StageName.DENOISE_FRAMES.value).is_file()
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
    assert (tmp_path / StageName.DENOISE_FRAMES.value).is_dir()


def test_feature_change_preserves_independent_denoise_branch(tmp_path):
    _populate(tmp_path)
    invalidated = invalidate_from(tmp_path, StageName.EXTRACT_FEATURES, include_self=False)
    assert StageName.DENOISE_FRAMES not in invalidated
    assert StageName.EXPORT_DATASET in invalidated
    assert (tmp_path / StageName.DENOISE_FRAMES.value).is_dir()
    assert derive_pipeline_state(tmp_path) == PipelineState.FEATURES_EXTRACTED


def test_summary_state_requires_main_branch_even_when_denoise_exists(tmp_path):
    _populate(tmp_path)
    (tmp_path / StageName.EXPORT_DATASET.value).rmdir()
    manifest_path(tmp_path, StageName.EXPORT_DATASET.value).unlink()
    assert derive_pipeline_state(tmp_path) == PipelineState.DENOISED
    (tmp_path / StageName.ALIGN_RECONSTRUCTION.value).rmdir()
    manifest_path(tmp_path, StageName.ALIGN_RECONSTRUCTION.value).unlink()
    assert derive_pipeline_state(tmp_path) == PipelineState.RECONSTRUCTED


def test_missing_consumers_are_not_marked_stale(tmp_path):
    (tmp_path / StageName.INSPECT_SOURCE.value).mkdir()
    invalidate_from(tmp_path, StageName.INSPECT_SOURCE, include_self=False)
    for stage in STAGE_ORDER[1:]:
        assert not is_stale(tmp_path, stage)


def test_lfstudio_outputs_are_moved_outside_managed_export(tmp_path):
    export = tmp_path / "export_dataset"
    (export / "images").mkdir(parents=True)
    (export / "images" / "managed.jpg").write_bytes(b"managed")
    checkpoint = export / "output" / "checkpoints" / "checkpoint.resume"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    custom = export / "my_training" / "model.ply"
    custom.parent.mkdir()
    custom.write_bytes(b"model")

    preserved = preserve_export_outputs(tmp_path)

    assert {path.name for path in preserved} == {"output", "my_training"}
    rescued_checkpoint = tmp_path / "training_outputs" / "output" / "checkpoints" / "checkpoint.resume"
    assert rescued_checkpoint.read_bytes() == b"checkpoint"
    assert (tmp_path / "training_outputs" / "my_training" / "model.ply").read_bytes() == b"model"
    assert (export / "images" / "managed.jpg").is_file()
