"""Stage artifact の削除と transitive invalidation を一か所に集約する."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..domain.artifacts import manifest_path
from ..domain.pipeline_state import STAGE_ORDER, StageName, downstream_of

_STALE_DIR = ".pipeline/stale"


def remove_stage(project_dir: Path, stage: StageName) -> None:
    for directory in (project_dir / stage.value, project_dir / f".{stage.value}.tmp"):
        shutil.rmtree(directory, ignore_errors=True)
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
    shutil.rmtree(project_dir / _STALE_DIR, ignore_errors=True)


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
