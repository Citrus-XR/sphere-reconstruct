"""Stage artifact の削除と transitive invalidation を一か所に集約する."""

from __future__ import annotations

import json
import shutil
import uuid
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
        preserve_export_outputs(project_dir)
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
    for index, item in enumerate(invalidated):
        had_artifact = (project_dir / item.value).exists() or manifest_path(project_dir, item.value).exists()
        remove_stage(project_dir, item)
        if had_artifact and (index > 0 or not include_self):
            _mark_stale(project_dir, item, stage)
    return invalidated


def clear_pipeline(project_dir: Path) -> None:
    for stage in STAGE_ORDER:
        remove_stage(project_dir, stage)
    stale_directory = project_dir / _STALE_DIR
    if stale_directory.exists():
        shutil.rmtree(stale_directory)


def derive_pipeline_state(project_dir: Path) -> PipelineState:
    """分岐 artifact を考慮して project の要約 state を導出する。"""
    present = {
        stage for stage in STAGE_ORDER if manifest_path(project_dir, stage.value).is_file()
    }
    if StageName.EXPORT_DATASET in present:
        return PipelineState.EXPORTED
    if StageName.ALIGN_RECONSTRUCTION in present:
        return (
            PipelineState.DENOISED
            if StageName.DENOISE_FRAMES in present
            else PipelineState.ALIGNED
        )
    for stage in reversed(STAGE_ORDER):
        if stage != StageName.DENOISE_FRAMES and stage in present:
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


def preserve_export_outputs(project_dir: Path) -> list[Path]:
    """LFStudio が dataset root 内へ作った unmanaged output を stage 外へ退避する.

    export_dataset は再生成可能だが、LFStudio の既定 ``output/`` や利用者指定の training
    directory は再生成不能な利用者データである。既知の managed entry 以外は削除せず、
    project の永続 ``training_outputs/`` へ同一 volume 上で移動する。
    """
    export_dir = project_dir / StageName.EXPORT_DATASET.value
    if not export_dir.is_dir():
        return []
    external = [child for child in export_dir.iterdir() if child.name not in _MANAGED_EXPORT_ENTRIES]
    if not external:
        return []
    destination_root = project_dir / "training_outputs"
    destination_root.mkdir(parents=True, exist_ok=True)
    preserved: list[Path] = []
    for source in external:
        destination = destination_root / source.name
        if destination.exists():
            destination = destination_root / (
                f"{source.stem}-recovered-{uuid.uuid4().hex[:8]}{source.suffix}"
            )
        shutil.move(str(source), str(destination))
        preserved.append(destination)
    return preserved
