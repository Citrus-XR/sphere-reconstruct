"""Stage artifact の削除と transitive invalidation を一か所に集約する."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..domain.artifacts import manifest_path
from ..domain.pipeline_state import STAGE_ORDER, STAGE_TO_STATE, PipelineState, StageName, downstream_of

_STALE_DIR = ".pipeline/stale"
_MANAGED_EXPORT_ENTRIES = {
    "images",
    "masks",
    "sparse",
    "preview",
    "train_configs",
    "export_manifest.json",
}


def remove_stage(project_dir: Path, stage: StageName) -> None:
    if stage == StageName.EXPORT_DATASET:
        assert_export_is_managed(project_dir)
    for directory in (project_dir / stage.value, project_dir / f".{stage.value}.tmp"):
        if directory.exists():
            shutil.rmtree(directory)
    stage_manifest = manifest_path(project_dir, stage.value)
    if stage_manifest.exists():
        stage_manifest.unlink()
    clear_stale(project_dir, stage)


def invalidate_from(
    project_dir: Path,
    stage: StageName,
    *,
    include_self: bool,
) -> list[StageName]:
    invalidated = downstream_of(stage, include_self=include_self)
    if StageName.EXPORT_DATASET in invalidated:
        assert_export_is_managed(project_dir)
    for index, item in enumerate(invalidated):
        had_artifact = (project_dir / item.value).exists() or manifest_path(project_dir, item.value).exists()
        remove_stage(project_dir, item)
        if had_artifact and (index > 0 or not include_self):
            _mark_stale(project_dir, item, stage)
    return invalidated


def clear_pipeline(project_dir: Path) -> None:
    assert_export_is_managed(project_dir)
    for stage in STAGE_ORDER:
        remove_stage(project_dir, stage)
    stale_directory = project_dir / _STALE_DIR
    if stale_directory.exists():
        shutil.rmtree(stale_directory)


def derive_pipeline_state(project_dir: Path) -> PipelineState:
    """分岐 artifact を考慮して project の要約 state を導出する。"""
    present = {stage for stage in STAGE_ORDER if manifest_path(project_dir, stage.value).is_file()}
    if StageName.EXPORT_DATASET in present:
        return PipelineState.EXPORTED
    if StageName.ALIGN_RECONSTRUCTION in present:
        return PipelineState.ALIGNED
    for stage in reversed(STAGE_ORDER):
        if stage in present:
            return STAGE_TO_STATE[stage]
    return PipelineState.CREATED


def is_stale(project_dir: Path, stage: StageName) -> bool:
    return _stale_path(project_dir, stage).is_file()


def clear_stale(project_dir: Path, stage: StageName) -> None:
    path = _stale_path(project_dir, stage)
    if path.exists():
        path.unlink()


def _mark_stale(project_dir: Path, stage: StageName, cause: StageName) -> None:
    path = _stale_path(project_dir, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"stage": stage.value, "invalidated_by": cause.value}, indent=2),
        encoding="utf-8",
    )


def _stale_path(project_dir: Path, stage: StageName) -> Path:
    return project_dir / _STALE_DIR / f"{stage.value}.json"


def assert_export_is_managed(project_dir: Path) -> None:
    """再生成可能な export 以外が dataset root に混在していないことを保証する.

    LFStudio の出力先はこのアプリケーションでは管理しない。利用者が dataset root 内を
    学習出力先に選んだ場合も、stage の再生成や消去でその成果を暗黙に移動・削除しない。
    """
    export_dir = project_dir / StageName.EXPORT_DATASET.value
    if not export_dir.is_dir():
        return
    external = [child for child in export_dir.iterdir() if child.name not in _MANAGED_EXPORT_ENTRIES]
    if external:
        names = ", ".join(sorted(child.name for child in external))
        raise RuntimeError(
            "export_dataset 内にアプリケーション管理外のファイルがあります。"
            f"削除または再生成の前に別の場所へ移動してください: {names}"
        )
