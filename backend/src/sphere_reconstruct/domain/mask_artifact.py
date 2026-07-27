"""Feature / training mask artifact の共通 contract。"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path

from .pipeline_state import StageName

MASK_MANIFEST_VERSION = 3


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
    if data.get("version") != MASK_MANIFEST_VERSION:
        raise ValueError(f"unsupported mask manifest version: {data.get('version')}")
    if data.get("purpose") != purpose.value:
        raise ValueError(
            f"mask manifest purpose mismatch: expected {purpose.value}, got {data.get('purpose')}"
        )
    return data


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
