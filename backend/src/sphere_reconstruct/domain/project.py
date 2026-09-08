"""Project ドメインモデルと DB アクセス.

ドメインモデルは pydantic. DB row との変換はこのファイルに集約.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from ..infrastructure.database import Database
from ..settings import workspace_root
from .pipeline_state import PipelineState
from .source import ProjectSource, SourceRole, list_sources


class Project(BaseModel):
    id: str
    name: str
    created_at: datetime
    updated_at: datetime
    sources: list[ProjectSource] = Field(default_factory=list)
    state: PipelineState = PipelineState.CREATED
    metadata: dict = Field(default_factory=dict)

    @property
    def workspace_dir(self) -> Path:
        return workspace_root() / "projects" / self.id

    @property
    def primary_source(self) -> ProjectSource | None:
        return next((source for source in self.sources if source.role == SourceRole.PRIMARY), None)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _project_from_row(row, sources: list[ProjectSource]) -> Project:
    import json

    return Project(
        id=row["id"],
        name=row["name"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        sources=sources,
        state=PipelineState(row["state"]),
        metadata=json.loads(row["metadata_json"] or "{}"),
    )


async def create_project(db: Database, name: str) -> Project:
    import json

    pid = str(uuid.uuid4())
    now = _now_iso()
    async with db.transaction() as conn:
        await conn.execute(
            """
            INSERT INTO project (id, name, created_at, updated_at, state, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (pid, name, now, now, PipelineState.CREATED.value, json.dumps({})),
        )
    project = await get_project(db, pid)
    assert project is not None
    project.workspace_dir.mkdir(parents=True, exist_ok=True)
    return project


async def list_projects(db: Database) -> list[Project]:
    cur = await db.conn.execute("SELECT * FROM project ORDER BY created_at DESC")
    rows = await cur.fetchall()
    return [_project_from_row(row, await list_sources(db, row["id"])) for row in rows]


async def get_project(db: Database, project_id: str) -> Project | None:
    cur = await db.conn.execute("SELECT * FROM project WHERE id = ?", (project_id,))
    row = await cur.fetchone()
    return _project_from_row(row, await list_sources(db, project_id)) if row else None


async def update_state(db: Database, project_id: str, state: PipelineState) -> None:
    async with db.transaction() as conn:
        await conn.execute(
            "UPDATE project SET state=?, updated_at=? WHERE id=?",
            (state.value, _now_iso(), project_id),
        )


async def patch_ui_state(db: Database, project_id: str, ui: dict) -> None:
    """設定の読み取りと部分更新を同じ transaction で行う。"""
    from .ui_state import merge_ui_patch

    async with db.transaction() as conn:
        row = await (await conn.execute(
            "SELECT metadata_json FROM project WHERE id=?", (project_id,)
        )).fetchone()
        if row is None:
            raise LookupError(f"project {project_id} not found")
        meta = json.loads(row[0])
        meta["ui"] = merge_ui_patch(meta.get("ui", {}), ui)
        await conn.execute(
            "UPDATE project SET metadata_json=?, updated_at=? WHERE id=?",
            (json.dumps(meta, ensure_ascii=False), _now_iso(), project_id),
        )
