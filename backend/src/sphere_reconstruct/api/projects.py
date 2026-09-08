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
from ..domain import source as source_domain
from ..domain.pipeline_state import PipelineState
from ..domain.ui_state import ProjectUiState
from ..infrastructure.database import get_db
from ..infrastructure.filesystem import PathNotAllowedError, ensure_within_any
from ..pipeline.invalidation import ArtifactBusyError, clear_pipeline
from ..settings import get_settings
from .project_access import MutableProject

router = APIRouter(prefix="/api/projects", tags=["projects"])


class ProjectCreate(BaseModel):
    name: str


class SourceRead(BaseModel):
    id: str
    label: str
    role: source_domain.SourceRole
    adapter: str
    media_kind: source_domain.MediaKind
    projection: source_domain.Projection
    path: str
    ordinal: int
    enabled: bool


class ProjectRead(BaseModel):
    id: str
    name: str
    created_at: datetime
    updated_at: datetime
    sources: list[SourceRead]
    state: PipelineState
    ui_state: dict | None


class SourceCreateBody(BaseModel):
    label: str = ""
    role: source_domain.SourceRole = source_domain.SourceRole.SUPPLEMENTAL
    adapter: str
    media_kind: source_domain.MediaKind
    projection: source_domain.Projection
    path: str


def _to_source_read(source: source_domain.ProjectSource) -> SourceRead:
    return SourceRead(**source.model_dump(exclude={"project_id"}))


def _to_read(p: project_domain.Project) -> ProjectRead:
    return ProjectRead(
        id=p.id,
        name=p.name,
        created_at=p.created_at,
        updated_at=p.updated_at,
        sources=[_to_source_read(source) for source in p.sources],
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


async def _invalidate_for_source_mutation(project: project_domain.Project) -> None:
    try:
        await run_in_threadpool(clear_pipeline, project.workspace_dir)
    except ArtifactBusyError as error:
        raise HTTPException(status_code=423, detail=str(error)) from error


def _resolve_source_path(path: str) -> Path:
    settings = get_settings()
    try:
        resolved = ensure_within_any(settings.filesystem.allowed_roots, Path(path))
    except PathNotAllowedError as error:
        raise HTTPException(status_code=403, detail=str(error)) from error
    if not resolved.exists():
        raise HTTPException(status_code=404, detail=f"source not found: {resolved}")
    return resolved


@router.post("/{project_id}/sources", response_model=ProjectRead)
async def add_source(project_id: str, body: SourceCreateBody, existing: MutableProject) -> ProjectRead:
    resolved = _resolve_source_path(body.path)
    if body.media_kind == source_domain.MediaKind.IMAGES and not resolved.is_dir():
        raise HTTPException(status_code=400, detail="image source must be a directory")
    if body.media_kind == source_domain.MediaKind.VIDEO and not resolved.is_file():
        raise HTTPException(status_code=400, detail="video source must be a file")
    if any(Path(source.path) == resolved for source in existing.sources):
        raise HTTPException(status_code=409, detail="source path is already registered")
    if body.role == source_domain.SourceRole.PRIMARY and any(
        source.role == source_domain.SourceRole.PRIMARY for source in existing.sources
    ):
        raise HTTPException(status_code=409, detail="project already has a primary source")
    try:
        source_domain.ProjectSource(
            id="validation",
            project_id=project_id,
            label=body.label or resolved.name,
            role=body.role,
            adapter=body.adapter,
            media_kind=body.media_kind,
            projection=body.projection,
            path=str(resolved),
            ordinal=len(existing.sources),
            enabled=True,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    await _invalidate_for_source_mutation(existing)

    try:
        await source_domain.add_source(
            get_db(),
            project_id,
            label=body.label,
            role=body.role,
            adapter=body.adapter,
            media_kind=body.media_kind,
            projection=body.projection,
            path=str(resolved),
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    project = await project_domain.get_project(get_db(), project_id)
    assert project is not None
    return _to_read(project)


@router.delete("/{project_id}/sources/{source_id}", response_model=ProjectRead)
async def delete_source(project_id: str, source_id: str, existing: MutableProject) -> ProjectRead:
    target = next((source for source in existing.sources if source.id == source_id), None)
    if target is None:
        raise HTTPException(status_code=404, detail="source not found")
    if target.role == source_domain.SourceRole.PRIMARY and len(existing.sources) > 1:
        raise HTTPException(
            status_code=409,
            detail="primary source を削除する前に別の source を primary にしてください",
        )
    await _invalidate_for_source_mutation(existing)
    try:
        await source_domain.remove_source(get_db(), project_id, source_id)
    except LookupError as error:
        raise HTTPException(status_code=404, detail="source not found") from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    project = await project_domain.get_project(get_db(), project_id)
    assert project is not None
    return _to_read(project)


@router.post("/{project_id}/sources/{source_id}/make-primary", response_model=ProjectRead)
async def make_primary_source(project_id: str, source_id: str, existing: MutableProject) -> ProjectRead:
    if not any(source.id == source_id for source in existing.sources):
        raise HTTPException(status_code=404, detail="source not found")
    await _invalidate_for_source_mutation(existing)
    try:
        await source_domain.make_primary(get_db(), project_id, source_id)
    except LookupError as error:
        raise HTTPException(status_code=404, detail="source not found") from error
    project = await project_domain.get_project(get_db(), project_id)
    assert project is not None
    return _to_read(project)


class UiStateBody(BaseModel):
    ui: ProjectUiState


@router.patch("/{project_id}/ui-state", response_model=ProjectRead)
async def patch_ui_state(project_id: str, body: UiStateBody) -> ProjectRead:
    """工程ごとの UI 設定を部分更新する。省略されたフィールドは変更しない。"""
    db = get_db()
    existing = await project_domain.get_project(db, project_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="project not found")
    await project_domain.patch_ui_state(db, project_id, body.ui.model_dump(exclude_unset=True))
    p = await project_domain.get_project(db, project_id)
    assert p is not None
    return _to_read(p)


@router.delete("/{project_id}")
async def delete_project(project_id: str, project: MutableProject) -> dict:
    """プロジェクトをディスクから完全に削除する (中間成果物含む).

    ソース動画はプロジェクト外のパス参照なので削除されない. DB 行削除で stage_run /
    job / event は ON DELETE CASCADE で消える.
    """
    db = get_db()
    if project.workspace_dir.exists():
        await run_in_threadpool(shutil.rmtree, project.workspace_dir)
    await db.conn.execute("DELETE FROM project WHERE id=?", (project_id,))
    await db.conn.commit()
    return {"deleted": project_id}
