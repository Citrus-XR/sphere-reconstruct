"""分割 stage の transitive artifact invalidation を検証する."""

from sphere_reconstruct.domain.artifacts import manifest_path
from sphere_reconstruct.domain.pipeline_state import STAGE_ORDER, StageName
from sphere_reconstruct.pipeline.invalidation import invalidate_from, is_stale


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


def test_missing_consumers_are_not_marked_stale(tmp_path):
    (tmp_path / StageName.INSPECT_SOURCE.value).mkdir()
    invalidate_from(tmp_path, StageName.INSPECT_SOURCE, include_self=False)
    for stage in STAGE_ORDER[1:]:
        assert not is_stale(tmp_path, stage)
