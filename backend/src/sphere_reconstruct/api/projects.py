"""projects API.

CRUD 系. ここに書く HTTP モデルはあくまで API 用 DTO. domain.project.Project と
1:1 マップだが, 内部モデルを直接 dump しない (外側に無用に露出させない).
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from ..domain import project as project_domain
from ..domain.pipeline_state import PipelineState
from ..infrastructure.database import get_db
from ..infrastructure.filesystem import PathNotAllowedError, ensure_within_any
from ..pipeline.invalidation import clear_pipeline
from ..settings import get_settings

router = APIRouter(prefix="/api/projects", tags=["projects"])


class ProjectCreate(BaseModel):
    name: str


class ProjectRead(BaseModel):
    id: str
    name: str
    created_at: datetime
    updated_at: datetime
    source_kind: str | None
    source_path: str | None
    state: PipelineState
    ui_state: dict | None


class SetSourceBody(BaseModel):
    kind: project_domain.SourceKind
    path: str  # 絶対パス. サーバサイドで allowed_roots チェック.


def _to_read(p: project_domain.Project) -> ProjectRead:
    return ProjectRead(
        id=p.id,
        name=p.name,
        created_at=p.created_at,
        updated_at=p.updated_at,
        source_kind=p.source_kind.value if p.source_kind else None,
        source_path=p.source_path,
        state=p.state,
        ui_state=p.metadata.get("ui"),
    )


@router.post("", response_model=ProjectRead)
async def create_project(body: ProjectCreate) -> ProjectRead:
    db = get_db()
    project = await project_domain.create_project(db, body.name)
    return _to_read(project)


@router.get("", response_model=list[ProjectRead])
async def list_projects() -> list[ProjectRead]:
    db = get_db()
    return [_to_read(p) for p in await project_domain.list_projects(db)]


@router.get("/{project_id}", response_model=ProjectRead)
async def get_project(project_id: str) -> ProjectRead:
    db = get_db()
    p = await project_domain.get_project(db, project_id)
    if p is None:
        raise HTTPException(status_code=404, detail="project not found")
    return _to_read(p)


@router.post("/{project_id}/source", response_model=ProjectRead)
async def set_source(project_id: str, body: SetSourceBody) -> ProjectRead:
    db = get_db()
    existing = await project_domain.get_project(db, project_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="project not found")

    settings = get_settings()
    try:
        resolved = ensure_within_any(settings.filesystem.allowed_roots, Path(body.path))
    except PathNotAllowedError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
    if not resolved.exists():
        raise HTTPException(status_code=404, detail=f"source not found: {resolved}")

    p = await project_domain.set_source(db, project_id, kind=body.kind, path=str(resolved))
    await run_in_threadpool(clear_pipeline, p.workspace_dir)
    region = p.workspace_dir / "fisheye_region.json"
    if region.exists():
        region.unlink()
    return _to_read(p)


class UiStateBody(BaseModel):
    ui: dict


@router.put("/{project_id}/ui-state", response_model=ProjectRead)
async def put_ui_state(project_id: str, body: UiStateBody) -> ProjectRead:
    """工程ごとの UI 設定 (step パラメータ / モード / 無効化) を保存する."""
    db = get_db()
    existing = await project_domain.get_project(db, project_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="project not found")
    await project_domain.set_ui_state(db, project_id, body.ui)
    p = await project_domain.get_project(db, project_id)
    assert p is not None
    return _to_read(p)


@router.delete("/{project_id}")
async def delete_project(project_id: str) -> dict:
    """プロジェクトをディスクから完全に削除する (中間成果物含む).

    ソース動画はプロジェクト外のパス参照なので削除されない. DB 行削除で stage_run /
    job / event は ON DELETE CASCADE で消える.
    """
    db = get_db()
    p = await project_domain.get_project(db, project_id)
    if p is None:
        raise HTTPException(status_code=404, detail="project not found")
    if p.workspace_dir.exists():
        await run_in_threadpool(shutil.rmtree, p.workspace_dir)
    await db.conn.execute("DELETE FROM project WHERE id=?", (project_id,))
    await db.conn.commit()
    return {"deleted": project_id}
