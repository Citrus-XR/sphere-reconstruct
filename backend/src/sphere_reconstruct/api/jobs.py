"""jobs API.

- POST /api/projects/{id}/run                 : パイプライン全体を非同期実行 (job を返す)
- POST /api/projects/{id}/rerun/{stage}       : 指定ステージからやり直す
- POST /api/jobs/{id}/cancel                  : 実行中 Worker を殺す
- GET  /api/jobs/{id}                         : Job 状態取得
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..domain import project as project_domain
from ..domain.pipeline_state import StageName
from ..infrastructure.database import get_db
from ..job_supervisor import get_supervisor

router = APIRouter(tags=["jobs"])


class JobRead(BaseModel):
    id: str
    project_id: str
    kind: str
    stage: str | None
    status: str
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    error_text: str | None
    pid: int | None


class JobStarted(BaseModel):
    job_id: str


class RunBody(BaseModel):
    # ステージ名 -> パラメータ dict. 省略可.
    # 例: {"reproject_views": {"size": 1024, "max_frames": 10},
    #      "generate_masks": {"prompt": "person,tripod"}}
    params_by_stage: dict[str, dict] | None = None
    # Batch pipeline API で実行対象から外す optional stage.
    skip: list[str] | None = None


@router.post("/api/projects/{project_id}/run", response_model=JobStarted)
async def run_pipeline(project_id: str, body: RunBody | None = None) -> JobStarted:
    db = get_db()
    p = await project_domain.get_project(db, project_id)
    if p is None:
        raise HTTPException(status_code=404, detail="project not found")
    sup = get_supervisor()
    job_id = await sup.enqueue_run_pipeline(
        project_id=project_id,
        stage=None,
        params_by_stage=body.params_by_stage if body else None,
        skip=body.skip if body else None,
    )
    return JobStarted(job_id=job_id)


@router.post("/api/projects/{project_id}/rerun/{stage}", response_model=JobStarted)
async def rerun_stage(project_id: str, stage: str, body: RunBody | None = None) -> JobStarted:
    if stage not in [s.value for s in StageName]:
        raise HTTPException(status_code=400, detail=f"unknown stage: {stage}")
    db = get_db()
    p = await project_domain.get_project(db, project_id)
    if p is None:
        raise HTTPException(status_code=404, detail="project not found")
    sup = get_supervisor()
    job_id = await sup.enqueue_run_pipeline(
        project_id=project_id,
        stage=stage,
        params_by_stage=body.params_by_stage if body else None,
    )
    return JobStarted(job_id=job_id)


@router.get("/api/jobs/{job_id}", response_model=JobRead)
async def get_job(job_id: str) -> JobRead:
    db = get_db()
    cur = await db.conn.execute("SELECT * FROM job WHERE id=?", (job_id,))
    row = await cur.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="job not found")
    return _row_to_job(row)


@router.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict[str, Any]:
    sup = get_supervisor()
    ok = await sup.cancel_job(job_id)
    return {"cancelled": ok}


def _row_to_job(row) -> JobRead:
    return JobRead(
        id=row["id"],
        project_id=row["project_id"],
        kind=row["kind"],
        stage=row["stage"],
        status=row["status"],
        created_at=datetime.fromisoformat(row["created_at"]),
        started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
        finished_at=datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None,
        error_text=row["error_text"],
        pid=row["pid"],
    )
