"""Schema と filesystem artifact を新しい runtime contract へ一度だけ移行する。"""

from __future__ import annotations

import json
from pathlib import Path

import aiosqlite

_OLD_MASK_STAGE = "generate_masks"
_TRAINING_MASK_STAGE = "generate_training_masks"
_OLD_MASK_PARAM_KEYS = {"maskSize", "downsampleOn", "dilate", "dilateOn", "prompt"}


async def migrate_dual_mask_artifacts(
    connection: aiosqlite.Connection,
    workspace_root: Path,
) -> None:
    rows = await (await connection.execute("SELECT id, metadata_json FROM project")).fetchall()
    for row in rows:
        project_dir = workspace_root / "projects" / row["id"]
        source = await (
            await connection.execute(
                "SELECT id FROM project_source WHERE project_id=? AND role='primary' ORDER BY ordinal LIMIT 1",
                (row["id"],),
            )
        ).fetchone()
        _migrate_project_mask_files(project_dir, source["id"] if source is not None else None)
        metadata = _migrate_ui_metadata(json.loads(row["metadata_json"]))
        if metadata is not None:
            await connection.execute(
                "UPDATE project SET metadata_json=? WHERE id=?",
                (json.dumps(metadata, ensure_ascii=False), row["id"]),
            )

    await connection.execute(
        "UPDATE stage_run SET stage=?, manifest_path=replace(manifest_path, ?, ?) WHERE stage=?",
        (_TRAINING_MASK_STAGE, _OLD_MASK_STAGE, _TRAINING_MASK_STAGE, _OLD_MASK_STAGE),
    )
    await connection.execute(
        "UPDATE job SET stage=? WHERE stage=?",
        (_TRAINING_MASK_STAGE, _OLD_MASK_STAGE),
    )
    await connection.execute(
        "UPDATE event SET stage=? WHERE stage=?",
        (_TRAINING_MASK_STAGE, _OLD_MASK_STAGE),
    )


def _migrate_project_mask_files(project_dir: Path, source_id: str | None) -> None:
    old_directory = project_dir / _OLD_MASK_STAGE
    training_directory = project_dir / _TRAINING_MASK_STAGE
    migrated_directory = False
    if old_directory.exists():
        if training_directory.exists():
            raise RuntimeError(
                f"cannot migrate both {_OLD_MASK_STAGE} and {_TRAINING_MASK_STAGE}: {project_dir}"
            )
        old_directory.rename(training_directory)
        migrated_directory = True

    mask_manifest = training_directory / "manifest_masks.json"
    if mask_manifest.is_file():
        data = json.loads(mask_manifest.read_text(encoding="utf-8"))
        if migrated_directory or data.get("version") != 3:
            data = _upgrade_mask_manifest(data, source_id)
        else:
            data = None
    else:
        data = None
    if data is not None:
        mask_manifest.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    _migrate_json_file(
        project_dir / "manifests" / f"{_OLD_MASK_STAGE}.json",
        project_dir / "manifests" / f"{_TRAINING_MASK_STAGE}.json",
    )
    _migrate_json_file(
        project_dir / ".pipeline" / "stale" / f"{_OLD_MASK_STAGE}.json",
        project_dir / ".pipeline" / "stale" / f"{_TRAINING_MASK_STAGE}.json",
    )


def _migrate_json_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        return
    if destination.exists():
        raise RuntimeError(f"migration destination already exists: {destination}")
    data = _replace_stage(json.loads(source.read_text(encoding="utf-8")))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    source.unlink()


def _replace_stage(value):
    if isinstance(value, str):
        return value.replace(_OLD_MASK_STAGE, _TRAINING_MASK_STAGE)
    if isinstance(value, list):
        return [_replace_stage(item) for item in value]
    if isinstance(value, dict):
        return {key: _replace_stage(item) for key, item in value.items()}
    return value


def _upgrade_mask_manifest(data: dict, source_id: str | None) -> dict:
    migrated = _replace_stage(data)
    version = migrated.get("version")
    if version == 3:
        if migrated.get("purpose") != "training":
            raise ValueError(f"unexpected migrated mask purpose: {migrated.get('purpose')}")
        return migrated
    if version == 2:
        migrated["version"] = 3
        migrated["purpose"] = "training"
        return migrated
    if source_id is None:
        raise ValueError("legacy mask manifest requires a primary source")

    kind = migrated.get("kind")
    if kind == "sam3_fisheye_masks":
        records = _flatten_fisheye_masks(migrated, source_id)
    elif kind == "sam3_pinhole_masks":
        records = _flatten_pinhole_masks(migrated, source_id)
    elif kind == "sam3_erp_masks":
        records = _flatten_erp_masks(migrated, source_id)
    else:
        raise ValueError(f"unsupported legacy mask manifest kind: {kind}")
    return {
        "version": 3,
        "purpose": "training",
        "prompt": migrated.get("prompt", []),
        "max_inference_size": migrated.get("max_inference_size", 0),
        "dilate_px": migrated.get("dilate_px", 0),
        "images": records,
    }


def _flatten_fisheye_masks(data: dict, source_id: str) -> list[dict]:
    records = []
    for frame in data["frames"]:
        index = int(frame["index"])
        for lens in frame["lenses"]:
            sensor = "front" if int(lens["lens"]) == 0 else "back"
            records.append(_legacy_record(lens, f"{sensor}/frame_{index:06d}.jpg", source_id, index))
    return records


def _flatten_pinhole_masks(data: dict, source_id: str) -> list[dict]:
    records = []
    for frame in data["frames"]:
        index = int(frame["index"])
        for view in frame["views"]:
            name = f"{view['view']}_lens{int(view['lens'])}/frame_{index:06d}.jpg"
            records.append(_legacy_record(view, name, source_id, index))
    return records


def _flatten_erp_masks(data: dict, source_id: str) -> list[dict]:
    return [
        _legacy_record(frame, f"frame_{int(frame['index']):06d}.jpg", source_id, int(frame["index"]))
        for frame in data["frames"]
    ]


def _legacy_record(record: dict, name: str, source_id: str, capture_index: int) -> dict:
    coverage = float(record.get("coverage", 0.0))
    return {
        "name": name,
        "source_id": source_id,
        "capture_index": capture_index,
        "path": record["path"],
        "coverage": coverage,
        "coverage_warning": bool(record.get("coverage_warning", coverage > 0.5)),
        "detections": record.get("detections", {}),
    }


def _migrate_ui_metadata(metadata: dict) -> dict | None:
    ui = metadata.get("ui")
    if not isinstance(ui, dict):
        return None
    params = ui.get("params")
    disabled = ui.get("disabled")
    has_old_params = isinstance(params, dict) and bool(_OLD_MASK_PARAM_KEYS & params.keys())
    if not has_old_params and not isinstance(disabled, list):
        return None

    updated = dict(metadata)
    updated_ui = dict(ui)
    updated_params = dict(params) if isinstance(params, dict) else {}
    for key in _OLD_MASK_PARAM_KEYS:
        updated_params.pop(key, None)
    masks_disabled = isinstance(disabled, list) and _OLD_MASK_STAGE in disabled
    updated_params["featureMaskEnabled"] = not masks_disabled
    updated_params["trainingMaskEnabled"] = not masks_disabled
    updated_ui["params"] = updated_params
    updated_ui.pop("disabled", None)
    updated["ui"] = updated_ui
    return updated
