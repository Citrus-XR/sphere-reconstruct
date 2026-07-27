"""stages API.

パイプラインを「独立した工程」として扱うための状態取得と, 個別クリア.

- GET  /api/projects/{id}/stages                 : 各ステージの状態一覧 (hierarchy 用).
- POST /api/projects/{id}/stages/{stage}/clear   : 指定ステージの出力を消してやり直せる状態にする.

実行 (再生成) は既存の POST /api/projects/{id}/rerun/{stage} を使う.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from ..domain import project as project_domain
from ..domain.artifacts import manifest_path
from ..domain.pipeline_state import (
    STAGE_ORDER,
    PipelineState,
    StageName,
)
from ..infrastructure.database import get_db
from ..pipeline.invalidation import clear_pipeline, derive_pipeline_state, invalidate_from, is_stale
from ..settings import workspace_root

router = APIRouter(tags=["stages"])


def _project_dir(project_id: str):
    return workspace_root() / "projects" / project_id


@router.get("/api/projects/{project_id}/stages")
async def list_stages(project_id: str) -> dict:
    db = get_db()
    p = await project_domain.get_project(db, project_id)
    if p is None:
        raise HTTPException(status_code=404, detail="project not found")

    # 各ステージの最新 stage_run を引く.
    cur = await db.conn.execute(
        "SELECT stage, status, error_text, started_at, finished_at, job_id FROM stage_run "
        "WHERE project_id=? ORDER BY started_at",
        (project_id,),
    )
    latest: dict[str, dict] = {}
    for row in await cur.fetchall():
        latest[row["stage"]] = {
            "status": row["status"],
            "error_text": row["error_text"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "job_id": row["job_id"],
        }

    progress_by_job: dict[tuple[str, str], dict] = {}
    progress_rows = await (
        await db.conn.execute(
            """
            WITH activity AS (
                SELECT stage, job_id, MAX(id) AS event_id
                FROM event
                WHERE project_id=? AND stage IS NOT NULL AND job_id IS NOT NULL
                GROUP BY stage, job_id
            ), numeric_progress AS (
                SELECT stage, job_id, MAX(id) AS event_id
                FROM event
                WHERE project_id=? AND stage IS NOT NULL AND job_id IS NOT NULL
                  AND progress IS NOT NULL
                GROUP BY stage, job_id
            )
            SELECT
                activity.stage,
                activity.job_id,
                activity_event.id,
                activity_event.level,
                activity_event.message,
                activity_event.msg_key,
                activity_event.msg_args,
                activity_event.kind,
                activity_event.ts,
                numeric_event.progress
            FROM activity
            JOIN event AS activity_event ON activity_event.id=activity.event_id
            LEFT JOIN numeric_progress
              ON numeric_progress.stage=activity.stage AND numeric_progress.job_id=activity.job_id
            LEFT JOIN event AS numeric_event ON numeric_event.id=numeric_progress.event_id
            """,
            (project_id, project_id),
        )
    ).fetchall()
    for row in progress_rows:
        progress_by_job[(row["stage"], row["job_id"])] = {
            "id": row["id"],
            "job_id": row["job_id"],
            "project_id": project_id,
            "stage": row["stage"],
            "level": row["level"],
            "message": row["message"],
            "msg_key": row["msg_key"],
            "msg_args": json.loads(row["msg_args"]) if row["msg_args"] else None,
            "progress": row["progress"],
            "kind": row["kind"],
            "ts": row["ts"],
        }

    proj_dir = _project_dir(project_id)
    stages = []
    for st in STAGE_ORDER:
        mf = manifest_path(proj_dir, st.value)
        has_output = mf.exists()
        params = None
        extra = None
        if has_output:
            try:
                manifest_data = json.loads(mf.read_text())
                params = manifest_data.get("params")
                extra = manifest_data.get("extra")
            except (OSError, json.JSONDecodeError):
                params = None
        run = latest.get(st.value, {})
        job_id = run.get("job_id")
        status = _resolve_stage_status(
            has_output=has_output,
            stale=is_stale(proj_dir, st),
            run_status=run.get("status"),
        )
        stages.append(
            {
                "stage": st.value,
                "has_output": has_output,
                "status": status,
                "error_text": run.get("error_text"),
                "job_id": run.get("job_id"),
                "started_at": run.get("started_at"),
                "finished_at": run.get("finished_at"),
                "params": params,
                "extra": extra,
                "progress_event": progress_by_job.get((st.value, job_id)) if job_id else None,
            }
        )
    return {"project_id": project_id, "state": p.state.value, "stages": stages}


def _resolve_stage_status(*, has_output: bool, stale: bool, run_status: str | None) -> str | None:
    if run_status in {"running", "queued", "failed"}:
        return run_status
    if not has_output and stale:
        return "stale"
    if not has_output and run_status == "succeeded":
        return None
    return run_status


@router.post("/api/projects/{project_id}/stages/{stage}/clear")
async def clear_stage(project_id: str, stage: str) -> dict:
    if stage not in [s.value for s in StageName]:
        raise HTTPException(status_code=400, detail=f"unknown stage: {stage}")
    db = get_db()
    p = await project_domain.get_project(db, project_id)
    if p is None:
        raise HTTPException(status_code=404, detail="project not found")

    proj_dir = _project_dir(project_id)
    invalidated = await run_in_threadpool(invalidate_from, proj_dir, StageName(stage), include_self=True)

    new_state = derive_pipeline_state(proj_dir)
    await db.conn.execute(
        "UPDATE project SET state=?, updated_at=datetime('now') WHERE id=?",
        (new_state.value, project_id),
    )
    await db.conn.commit()
    return {
        "cleared": stage,
        "invalidated": [item.value for item in invalidated],
        "state": new_state.value,
    }


@router.post("/api/projects/{project_id}/clear-outputs")
async def clear_all_outputs(project_id: str) -> dict:
    """処理済みの中間成果物を全ステージ分クリアする (パラメータは保持. UI 側の状態なので不変)."""
    db = get_db()
    p = await project_domain.get_project(db, project_id)
    if p is None:
        raise HTTPException(status_code=404, detail="project not found")
    proj_dir = _project_dir(project_id)
    await run_in_threadpool(clear_pipeline, proj_dir)
    await db.conn.execute(
        "UPDATE project SET state=?, updated_at=datetime('now') WHERE id=?",
        (PipelineState.CREATED.value, project_id),
    )
    await db.conn.commit()
    return {"cleared": "all", "state": PipelineState.CREATED.value}
