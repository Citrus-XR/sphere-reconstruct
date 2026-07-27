"""Project に属する入力 source の型と永続化。"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, model_validator

from ..infrastructure.database import Database


class SourceRole(StrEnum):
    PRIMARY = "primary"
    SUPPLEMENTAL = "supplemental"


class SourceAdapter(StrEnum):
    INSTA360_INSV = "insta360_insv"
    GENERIC_VIDEO = "generic_video"
    GENERIC_IMAGES = "generic_images"


class MediaKind(StrEnum):
    VIDEO = "video"
    IMAGES = "images"


class Projection(StrEnum):
    DUAL_FISHEYE = "dual_fisheye"
    EQUIRECTANGULAR = "equirectangular"
    PERSPECTIVE = "perspective"


class ProjectSource(BaseModel):
    id: str
    project_id: str
    label: str
    role: SourceRole
    adapter: SourceAdapter
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
        if self.adapter == SourceAdapter.INSTA360_INSV:
            if self.media_kind != MediaKind.VIDEO or self.projection != Projection.DUAL_FISHEYE:
                raise ValueError("Insta360 INSV は dual-fisheye video として指定してください")
        elif self.adapter == SourceAdapter.GENERIC_VIDEO and self.media_kind != MediaKind.VIDEO:
            raise ValueError("generic_video の media_kind は video でなければなりません")
        elif self.adapter == SourceAdapter.GENERIC_IMAGES and self.media_kind != MediaKind.IMAGES:
            raise ValueError("generic_images の media_kind は images でなければなりません")
        if self.adapter != SourceAdapter.INSTA360_INSV and self.projection == Projection.DUAL_FISHEYE:
            raise ValueError("raw dual-fisheye は Insta360 INSV adapter だけに対応しています")
        return self


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def source_from_row(row) -> ProjectSource:
    return ProjectSource(
        id=row["id"],
        project_id=row["project_id"],
        label=row["label"],
        role=SourceRole(row["role"]),
        adapter=SourceAdapter(row["adapter"]),
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
    adapter: SourceAdapter,
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
                source.adapter.value,
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
