"""Project に属する入力 source の型と永続化。"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, model_validator

from ..infrastructure.database import Database


class SourceRole(StrEnum):
    PRIMARY = "primary"
    SUPPLEMENTAL = "supplemental"


class SourceAdapter:
    """永続化 ID の定数。型を閉じた Enum にせず adapter registry を拡張可能にする。"""

    INSTA360 = "insta360"
    GENERIC_VIDEO = "generic_video"
    GENERIC_IMAGES = "generic_images"


class MediaKind(StrEnum):
    VIDEO = "video"
    IMAGES = "images"


class Projection(StrEnum):
    DUAL_FISHEYE = "dual_fisheye"
    EQUIRECTANGULAR = "equirectangular"
    PERSPECTIVE = "perspective"


@dataclass(frozen=True)
class SourceAdapterDefinition:
    id: str
    media_kind: MediaKind
    projections: frozenset[Projection]


SOURCE_ADAPTERS = {
    definition.id: definition
    for definition in (
        SourceAdapterDefinition(
            id=SourceAdapter.INSTA360,
            media_kind=MediaKind.VIDEO,
            projections=frozenset({Projection.DUAL_FISHEYE}),
        ),
        SourceAdapterDefinition(
            id=SourceAdapter.GENERIC_VIDEO,
            media_kind=MediaKind.VIDEO,
            projections=frozenset({Projection.EQUIRECTANGULAR, Projection.PERSPECTIVE}),
        ),
        SourceAdapterDefinition(
            id=SourceAdapter.GENERIC_IMAGES,
            media_kind=MediaKind.IMAGES,
            projections=frozenset({Projection.EQUIRECTANGULAR, Projection.PERSPECTIVE}),
        ),
    )
}


def require_adapter(adapter_id: str) -> SourceAdapterDefinition:
    try:
        return SOURCE_ADAPTERS[adapter_id]
    except KeyError as error:
        raise ValueError(f"未登録の source adapter です: {adapter_id}") from error


class ProjectSource(BaseModel):
    id: str
    project_id: str
    label: str
    role: SourceRole
    adapter: str
    media_kind: MediaKind
    projection: Projection
    path: str
    ordinal: int
    enabled: bool

    @property
    def filesystem_path(self) -> Path:
        return Path(self.path)

    @model_validator(mode="after")
    def validate_combination(self) -> ProjectSource:
        definition = require_adapter(self.adapter)
        if self.media_kind != definition.media_kind:
            raise ValueError(
                f"{self.adapter} の media_kind は {definition.media_kind.value} でなければなりません"
            )
        if self.projection not in definition.projections:
            supported = ", ".join(sorted(projection.value for projection in definition.projections))
            raise ValueError(
                f"{self.adapter} は projection {self.projection.value} に対応していません: {supported}"
            )
        return self


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def source_from_row(row) -> ProjectSource:
    return ProjectSource(
        id=row["id"],
        project_id=row["project_id"],
        label=row["label"],
        role=SourceRole(row["role"]),
        adapter=str(row["adapter"]),
        media_kind=MediaKind(row["media_kind"]),
        projection=Projection(row["projection"]),
        path=row["path"],
        ordinal=int(row["ordinal"]),
        enabled=bool(row["enabled"]),
    )


async def list_sources(db: Database, project_id: str) -> list[ProjectSource]:
    cursor = await db.conn.execute(
        "SELECT * FROM project_source WHERE project_id=? ORDER BY ordinal, created_at, id",
        (project_id,),
    )
    return [source_from_row(row) for row in await cursor.fetchall()]


async def add_source(
    db: Database,
    project_id: str,
    *,
    label: str,
    role: SourceRole,
    adapter: str,
    media_kind: MediaKind,
    projection: Projection,
    path: str,
) -> ProjectSource:
    sources = await list_sources(db, project_id)
    effective_role = SourceRole.PRIMARY if not sources else role
    if effective_role == SourceRole.PRIMARY and any(source.role == SourceRole.PRIMARY for source in sources):
        raise ValueError("primary source は project ごとに 1 個だけ指定できます")
    source = ProjectSource(
        id=str(uuid.uuid4()),
        project_id=project_id,
        label=label.strip() or Path(path).name,
        role=effective_role,
        adapter=adapter,
        media_kind=media_kind,
        projection=projection,
        path=path,
        ordinal=len(sources),
        enabled=True,
    )
    now = _now_iso()
    async with db.transaction() as connection:
        await connection.execute(
            """
            INSERT INTO project_source
                (id, project_id, label, role, adapter, media_kind, projection, path,
                 ordinal, enabled, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
            """,
            (
                source.id,
                source.project_id,
                source.label,
                source.role.value,
                source.adapter,
                source.media_kind.value,
                source.projection.value,
                source.path,
                source.ordinal,
                now,
                now,
            ),
        )
        await connection.execute(
            "UPDATE project SET state='created', updated_at=? WHERE id=?",
            (now, project_id),
        )
    return source


async def remove_source(db: Database, project_id: str, source_id: str) -> None:
    sources = await list_sources(db, project_id)
    target = next((source for source in sources if source.id == source_id), None)
    if target is None:
        raise LookupError(source_id)
    if target.role == SourceRole.PRIMARY and len(sources) > 1:
        raise ValueError("primary source を削除する前に別の source を primary にしてください")
    now = _now_iso()
    async with db.transaction() as connection:
        await connection.execute(
            "DELETE FROM project_source WHERE project_id=? AND id=?",
            (project_id, source_id),
        )
        await connection.execute(
            "UPDATE project SET state='created', updated_at=? WHERE id=?",
            (now, project_id),
        )


async def make_primary(db: Database, project_id: str, source_id: str) -> None:
    sources = await list_sources(db, project_id)
    if not any(source.id == source_id for source in sources):
        raise LookupError(source_id)
    now = _now_iso()
    async with db.transaction() as connection:
        await connection.execute(
            "UPDATE project_source SET role='supplemental', updated_at=? "
            "WHERE project_id=? AND role='primary'",
            (now, project_id),
        )
        await connection.execute(
            "UPDATE project_source SET role='primary', updated_at=? WHERE project_id=? AND id=?",
            (now, project_id, source_id),
        )
        await connection.execute(
            "UPDATE project SET state='created', updated_at=? WHERE id=?",
            (now, project_id),
        )
