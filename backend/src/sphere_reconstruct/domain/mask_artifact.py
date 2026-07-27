"""Feature / training mask artifact の共通 contract。"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path

from .pipeline_state import StageName

MASK_MANIFEST_VERSION = 3
PARTIAL_MASK_HEADER = ".preview-mask-header.json"
PARTIAL_MASK_RECORDS = ".preview-mask-records.jsonl"


class MaskPurpose(StrEnum):
    FEATURE = "feature"
    TRAINING = "training"


_STAGE_BY_PURPOSE = {
    MaskPurpose.FEATURE: StageName.GENERATE_FEATURE_MASKS,
    MaskPurpose.TRAINING: StageName.GENERATE_TRAINING_MASKS,
}


def stage_for(purpose: MaskPurpose) -> StageName:
    return _STAGE_BY_PURPOSE[purpose]


def mask_manifest_path(project_dir: Path, purpose: MaskPurpose) -> Path:
    return project_dir / stage_for(purpose).value / "manifest_masks.json"


def load_mask_manifest(project_dir: Path, purpose: MaskPurpose) -> dict:
    path = mask_manifest_path(project_dir, purpose)
    data = json.loads(path.read_text(encoding="utf-8"))
    _validate_manifest(data, purpose)
    return data


def initialise_partial_mask_manifest(stage_dir: Path, document: dict) -> Path:
    """生成済み record だけを API process へ公開する append-only preview を初期化する。"""
    _validate_manifest(document, MaskPurpose(document["purpose"]))
    header = {**document, "complete": False, "images": []}
    header_path = stage_dir / PARTIAL_MASK_HEADER
    temporary = header_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(header, ensure_ascii=False), encoding="utf-8")
    temporary.replace(header_path)
    records_path = stage_dir / PARTIAL_MASK_RECORDS
    records_path.write_text("", encoding="utf-8")
    return records_path


def load_partial_mask_manifest(stage_dir: Path, purpose: MaskPurpose) -> dict | None:
    header_path = stage_dir / PARTIAL_MASK_HEADER
    records_path = stage_dir / PARTIAL_MASK_RECORDS
    if not header_path.is_file() or not records_path.is_file():
        return None
    data = json.loads(header_path.read_text(encoding="utf-8"))
    _validate_manifest(data, purpose)
    records = []
    with records_path.open(encoding="utf-8") as stream:
        for line in stream:
            # Writer crash / concurrent append 中の末尾だけは record として公開しない。
            if not line.endswith("\n"):
                break
            records.append(json.loads(line))
    data["images"] = records
    data["generated_images"] = len(records)
    data["complete"] = False
    records_by_name(data)
    return data


def append_partial_mask_record(records_path: Path, record: dict) -> None:
    with records_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()


def remove_partial_mask_manifest(stage_dir: Path) -> None:
    (stage_dir / PARTIAL_MASK_HEADER).unlink(missing_ok=True)
    (stage_dir / PARTIAL_MASK_RECORDS).unlink(missing_ok=True)


def _validate_manifest(data: dict, purpose: MaskPurpose) -> None:
    if data.get("version") != MASK_MANIFEST_VERSION:
        raise ValueError(f"unsupported mask manifest version: {data.get('version')}")
    if data.get("purpose") != purpose.value:
        raise ValueError(
            f"mask manifest purpose mismatch: expected {purpose.value}, got {data.get('purpose')}"
        )


def records_by_name(manifest: dict) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for record in manifest["images"]:
        name = str(record["name"])
        if name in records:
            raise ValueError(f"duplicate mask record: {name}")
        records[name] = record
    return records


def export_purpose(*, feature_enabled: bool, training_enabled: bool) -> MaskPurpose | None:
    if training_enabled:
        return MaskPurpose.TRAINING
    if feature_enabled:
        return MaskPurpose.FEATURE
    return None
