"""ソース・sensor ごとの有効領域。座標は表示向きの画像に対して正規化する。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..domain.source import Projection

FILENAME = "source_regions.json"
REGION_VERSION = 1
MAX_CIRCLE_RADIUS = 0.5
DEFAULT_R_NORM = 0.5
MAX_CUSTOM_OPERATIONS = 2048


class RegionOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["add", "subtract"]
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    r: float = Field(ge=0.002, le=0.5)
    stroke_id: int = Field(ge=1, strict=True)


class FullRegion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["full"] = "full"
    operations: list[RegionOperation] = Field(default_factory=list, max_length=MAX_CUSTOM_OPERATIONS)


class CircleRegion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["circle"] = "circle"
    cx: Literal[0.5] = 0.5
    cy: Literal[0.5] = 0.5
    r: float = Field(default=DEFAULT_R_NORM, ge=0.01, le=MAX_CIRCLE_RADIUS)
    operations: list[RegionOperation] = Field(default_factory=list, max_length=MAX_CUSTOM_OPERATIONS)


class SourceRegion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    views: dict[str, Annotated[CircleRegion | FullRegion, Field(discriminator="kind")]]


def default_region(projection: Projection | str) -> dict:
    if Projection(projection) == Projection.DUAL_FISHEYE:
        return {"views": {sensor: CircleRegion().model_dump() for sensor in ("lens0", "lens1")}}
    return {"views": {"main": FullRegion().model_dump()}}


def validate_region(data: dict, projection: Projection | str) -> dict:
    region = SourceRegion.model_validate(data).model_dump()
    expected = default_region(projection)["views"]
    if region["views"].keys() != expected.keys():
        raise ValueError(f"source region requires views {list(expected)}")
    for sensor, view in region["views"].items():
        if view["kind"] != expected[sensor]["kind"]:
            raise ValueError(f"source view {sensor} requires region kind {expected[sensor]['kind']}")
    return region


def region_path(project_dir: Path) -> Path:
    return project_dir / FILENAME


def _read_document(project_dir: Path) -> dict:
    path = region_path(project_dir)
    if not path.exists():
        return {"version": REGION_VERSION, "sources": {}}
    document = json.loads(path.read_text(encoding="utf-8"))
    if document["version"] != REGION_VERSION:
        raise ValueError(f"unsupported source region version: {document['version']}")
    return document


def load_region(project_dir: Path, source_id: str, projection: Projection | str) -> dict:
    source = _read_document(project_dir)["sources"].get(source_id)
    return default_region(projection) if source is None else validate_region({"views": source["views"]}, projection)


def region_status(project_dir: Path, source_id: str) -> dict:
    source = _read_document(project_dir)["sources"].get(source_id)
    return {"saved": source is not None, "needs_review": bool(source and source.get("needs_review"))}


def save_region(project_dir: Path, source_id: str, projection: Projection | str, data: dict) -> dict:
    region = validate_region(data, projection)
    document = _read_document(project_dir)
    document["sources"][source_id] = {**region, "needs_review": False}
    _write_document(project_dir, document)
    return region


def _write_document(project_dir: Path, document: dict) -> None:
    path = region_path(project_dir)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, indent=2), encoding="utf-8")
    temporary.replace(path)


def migrate_source_regions(project_dir: Path) -> int:
    """旧領域を移行し、半径上限を超えていた円の数を返す。"""
    legacy = project_dir / "fisheye_regions.json"
    document = _read_document(project_dir)
    capped = 0
    for source in document["sources"].values():
        for view in source["views"].values():
            if view["kind"] == "circle" and view["r"] > MAX_CIRCLE_RADIUS:
                view["r"] = MAX_CIRCLE_RADIUS
                capped += 1
    if not legacy.is_file():
        if capped:
            _write_document(project_dir, document)
        return capped
    old = json.loads(legacy.read_text(encoding="utf-8"))
    for source_id, source in old.get("sources", {}).items():
        if source_id in document["sources"]:
            continue
        region = default_region(Projection.DUAL_FISHEYE)
        for sensor, view in region["views"].items():
            saved = source.get(sensor, {})
            radius = float(saved.get("r", DEFAULT_R_NORM))
            capped += int(radius > MAX_CIRCLE_RADIUS)
            view["r"] = max(0.01, min(MAX_CIRCLE_RADIUS, radius))
            operations = saved.get("operations", [])
            next_id = max((operation.get("stroke_id") or 0 for operation in operations), default=0) + 1
            for operation in operations:
                migrated = {**operation}
                if migrated.get("stroke_id") is None:
                    migrated["stroke_id"] = next_id
                    next_id += 1
                view["operations"].append(migrated)
        document["sources"][source_id] = {
            **validate_region(region, Projection.DUAL_FISHEYE),
            "needs_review": source.get("_coordinate_version") != 3,
        }
    _write_document(project_dir, document)
    legacy.unlink()
    return capped
