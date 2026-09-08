"""Stage artifact の削除と transitive invalidation を一か所に集約する."""

from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path

from ..domain.artifacts import manifest_path
from ..domain.pipeline_state import STAGE_ORDER, STAGE_TO_STATE, PipelineState, StageName, downstream_of

_STALE_DIR = ".pipeline/stale"
_TRASH_DIR = ".pipeline/trash"
class ArtifactBusyError(RuntimeError):
    def __init__(self, path: Path, error: OSError):
        self.path = path
        self.error = error
        super().__init__(f"artifact is in use by another process: {path} ({error})")


def remove_stage(project_dir: Path, stage: StageName) -> None:
    transaction = _quarantine_stage_artifacts(project_dir, [stage])
    _delete_quarantine(transaction)


def invalidate_from(
    project_dir: Path,
    stage: StageName,
    *,
    include_self: bool,
) -> list[StageName]:
    invalidated = downstream_of(stage, include_self=include_self)
    had_artifact = {
        item: (project_dir / item.value).exists() or manifest_path(project_dir, item.value).exists()
        for item in invalidated
    }
    transaction = _quarantine_stage_artifacts(project_dir, invalidated)
    for index, item in enumerate(invalidated):
        if had_artifact[item] and (index > 0 or not include_self):
            _mark_stale(project_dir, item, stage)
    _delete_quarantine(transaction)
    return invalidated


def clear_pipeline(project_dir: Path) -> None:
    transaction = _quarantine_stage_artifacts(project_dir, list(STAGE_ORDER))
    _delete_quarantine(transaction)


def clear_stages(project_dir: Path, requested: list[StageName]) -> list[StageName]:
    effective = {
        stage
        for requested_stage in requested
        for stage in downstream_of(requested_stage, include_self=True)
    }
    ordered = [stage for stage in STAGE_ORDER if stage in effective]
    transaction = _quarantine_stage_artifacts(project_dir, ordered)
    _delete_quarantine(transaction)
    return ordered


def pending_cleanup_paths(project_dir: Path) -> list[str]:
    trash = project_dir / _TRASH_DIR
    if not trash.is_dir():
        return []
    return [str(path.relative_to(project_dir)) for path in sorted(trash.iterdir()) if _can_retry_cleanup(path)]


def _quarantine_stage_artifacts(project_dir: Path, stages: list[StageName]) -> Path | None:
    _retry_pending_cleanup(project_dir)
    transaction = project_dir / _TRASH_DIR / uuid.uuid4().hex
    moved: list[tuple[Path, Path]] = []
    failed_source = project_dir
    try:
        for index, stage in enumerate(stages):
            destination = transaction / f"{index:02d}-{stage.value}"
            candidates = (
                (project_dir / stage.value, destination / "output"),
                (project_dir / f".{stage.value}.tmp", destination / "temporary"),
                (manifest_path(project_dir, stage.value), destination / "manifest.json"),
                (_stale_path(project_dir, stage), destination / "stale.json"),
            )
            for source, target in candidates:
                if not source.exists():
                    continue
                failed_source = source
                target.parent.mkdir(parents=True, exist_ok=True)
                source.rename(target)
                moved.append((source, target))
    except OSError as error:
        failed_source = source
        rollback_errors = []
        for source, target in reversed(moved):
            if not target.exists():
                continue
            try:
                source.parent.mkdir(parents=True, exist_ok=True)
                target.rename(source)
            except OSError as rollback_error:
                rollback_errors.append((source, target, rollback_error))
        if rollback_errors:
            details = ", ".join(f"{path}: {failure}" for path, _backup, failure in rollback_errors)
            diagnostic = f"artifact quarantine rollback failed: {details}; backups preserved at {transaction}"
            recovery = {
                "reason": "rollback_failed",
                "artifacts": [
                    {
                        "source": str(path.relative_to(project_dir)),
                        "backup": str(backup.relative_to(project_dir)),
                        "error": str(failure),
                    }
                    for path, backup, failure in rollback_errors
                ],
            }
            try:
                (transaction / "recovery_required.json").write_text(json.dumps(recovery, indent=2), encoding="utf-8")
            except OSError as marker_error:
                raise RuntimeError(f"{diagnostic}; recovery metadata failed: {marker_error}") from error
            raise RuntimeError(diagnostic) from error
        shutil.rmtree(transaction, ignore_errors=True)
        raise ArtifactBusyError(failed_source, error) from error
    if not moved:
        shutil.rmtree(transaction, ignore_errors=True)
        return None
    return transaction


def _delete_quarantine(transaction: Path | None) -> bool:
    if transaction is None or not transaction.exists():
        return True
    last_error: OSError | None = None
    for delay in (0.0, 0.05, 0.1, 0.2, 0.4, 0.8):
        if delay:
            time.sleep(delay)
        try:
            shutil.rmtree(transaction)
            return True
        except OSError as error:
            last_error = error
            continue
    marker = transaction / "pending_cleanup.json"
    marker.write_text(
        json.dumps(
            {"path": str(transaction), "reason": "artifact_locked", "error": str(last_error)},
            indent=2,
        ),
        encoding="utf-8",
    )
    return False


def _retry_pending_cleanup(project_dir: Path) -> None:
    trash = project_dir / _TRASH_DIR
    if not trash.is_dir():
        return
    for transaction in list(trash.iterdir()):
        if _can_retry_cleanup(transaction):
            _delete_quarantine(transaction)


def _can_retry_cleanup(transaction: Path) -> bool:
    # 未確定の quarantine と復旧待ち backup は自動削除しない。
    return (
        transaction.is_dir()
        and (transaction / "pending_cleanup.json").is_file()
        and not (transaction / "recovery_required.json").exists()
    )


def derive_pipeline_state(project_dir: Path) -> PipelineState:
    """分岐 artifact を考慮して project の要約 state を導出する。"""
    present = {stage for stage in STAGE_ORDER if manifest_path(project_dir, stage.value).is_file()}
    if StageName.EXPORT_DATASET in present:
        return PipelineState.EXPORTED
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
