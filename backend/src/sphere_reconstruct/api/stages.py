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
from pydantic import BaseModel

from ..domain import project as project_domain
from ..domain.artifacts import manifest_path
from ..domain.pipeline_state import (
    STAGE_ORDER,
    StageName,
)
from ..infrastructure.database import get_db
from ..pipeline.invalidation import (
    ArtifactBusyError,
    clear_stages,
    derive_pipeline_state,
    invalidate_from,
    is_stale,
    pending_cleanup_paths,
)
from ..settings import workspace_root
from .project_access import MutableProject

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
        """
        SELECT stage_run.stage, stage_run.status, stage_run.error_text,
               stage_run.started_at, stage_run.finished_at, stage_run.job_id,
               job_request.params_by_stage_json
        FROM stage_run
        LEFT JOIN job_request ON job_request.job_id=stage_run.job_id
        WHERE stage_run.project_id=?
        ORDER BY stage_run.started_at
        """,
        (project_id,),
    )
    latest: dict[str, dict] = {}
    for row in await cur.fetchall():
        params_by_stage = (
            json.loads(row["params_by_stage_json"])
            if row["params_by_stage_json"] is not None
            else {}
        )
        latest[row["stage"]] = {
            "status": row["status"],
            "error_text": row["error_text"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "job_id": row["job_id"],
            "requested_params": params_by_stage.get(row["stage"]),
        }

    snapshots_by_job: dict[tuple[str, str], dict] = {}
    snapshot_rows = await (
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
                activity_event.id AS activity_id,
                activity_event.level AS activity_level,
                activity_event.message AS activity_message,
                activity_event.msg_key AS activity_msg_key,
                activity_event.msg_args AS activity_msg_args,
                activity_event.progress AS activity_progress,
                activity_event.kind AS activity_kind,
                activity_event.ts AS activity_ts,
                numeric_event.id AS numeric_id,
                numeric_event.level AS numeric_level,
                numeric_event.message AS numeric_message,
                numeric_event.msg_key AS numeric_msg_key,
                numeric_event.msg_args AS numeric_msg_args,
                numeric_event.progress AS numeric_progress,
                numeric_event.kind AS numeric_kind,
                numeric_event.ts AS numeric_ts
            FROM activity
            JOIN event AS activity_event ON activity_event.id=activity.event_id
            LEFT JOIN numeric_progress
              ON numeric_progress.stage=activity.stage AND numeric_progress.job_id=activity.job_id
            LEFT JOIN event AS numeric_event ON numeric_event.id=numeric_progress.event_id
            """,
            (project_id, project_id),
        )
    ).fetchall()
    for row in snapshot_rows:
        snapshots_by_job[(row["stage"], row["job_id"])] = {
            "activity_event": _event_payload(row, "activity", project_id),
            "progress_event": _event_payload(row, "numeric", project_id),
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
        snapshots = snapshots_by_job.get((st.value, job_id), {}) if job_id else {}
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
                "active_params": (
                    run.get("requested_params")
                    if status in {"running", "queued"}
                    else None
                ),
                "extra": extra,
                "activity_event": snapshots.get("activity_event"),
                "progress_event": snapshots.get("progress_event"),
            }
        )
    return {"project_id": project_id, "state": p.state.value, "stages": stages}


def _event_payload(row, prefix: str, project_id: str) -> dict | None:
    event_id = row[f"{prefix}_id"]
    if event_id is None:
        return None
    encoded_args = row[f"{prefix}_msg_args"]
    return {
        "id": event_id,
        "job_id": row["job_id"],
        "project_id": project_id,
        "stage": row["stage"],
        "level": row[f"{prefix}_level"],
        "message": row[f"{prefix}_message"],
        "msg_key": row[f"{prefix}_msg_key"],
        "msg_args": json.loads(encoded_args) if encoded_args else None,
        "progress": row[f"{prefix}_progress"],
        "kind": row[f"{prefix}_kind"],
        "ts": row[f"{prefix}_ts"],
    }


def _resolve_stage_status(*, has_output: bool, stale: bool, run_status: str | None) -> str | None:
    if run_status in {"running", "queued", "failed"}:
        return run_status
    if not has_output and stale:
        return "stale"
    if not has_output and run_status == "succeeded":
        return None
    return run_status


@router.post("/api/projects/{project_id}/stages/{stage}/clear")
async def clear_stage(project_id: str, stage: str, project: MutableProject) -> dict:
    if stage not in [s.value for s in StageName]:
        raise HTTPException(status_code=400, detail=f"unknown stage: {stage}")
    db = get_db()
    proj_dir = project.workspace_dir
    try:
        invalidated = await run_in_threadpool(invalidate_from, proj_dir, StageName(stage), include_self=True)
    except ArtifactBusyError as error:
        raise HTTPException(status_code=423, detail=str(error)) from error

    new_state = derive_pipeline_state(proj_dir)
    await db.conn.execute(
        "UPDATE project SET state=?, updated_at=datetime('now') WHERE id=?",
        (new_state.value, project_id),
    )
    await db.conn.commit()
    return {
        "cleared": stage,
        "invalidated": [item.value for item in invalidated],
        "pending_cleanup": pending_cleanup_paths(proj_dir),
        "state": new_state.value,
    }


class ClearOutputsBody(BaseModel):
    stages: list[StageName]


@router.post("/api/projects/{project_id}/clear-outputs")
async def clear_outputs(project_id: str, body: ClearOutputsBody, project: MutableProject) -> dict:
    """選択 Stage と consumer closure を同一 transaction で削除する。"""
    if not body.stages:
        raise HTTPException(status_code=400, detail="at least one stage must be selected")
    db = get_db()
    proj_dir = project.workspace_dir
    try:
        cleared = await run_in_threadpool(clear_stages, proj_dir, body.stages)
    except ArtifactBusyError as error:
        raise HTTPException(status_code=423, detail=str(error)) from error
    new_state = derive_pipeline_state(proj_dir)
    await db.conn.execute(
        "UPDATE project SET state=?, updated_at=datetime('now') WHERE id=?",
        (new_state.value, project_id),
    )
    await db.conn.commit()
    return {
        "requested": [stage.value for stage in body.stages],
        "cleared": [stage.value for stage in cleared],
        "pending_cleanup": pending_cleanup_paths(proj_dir),
        "state": new_state.value,
    }
