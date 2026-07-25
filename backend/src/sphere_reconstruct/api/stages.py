"""stages API.

パイプラインを「独立した工程」として扱うための状態取得と, 個別クリア.

- GET  /api/projects/{id}/stages                 : 各ステージの状態一覧 (hierarchy 用).
- POST /api/projects/{id}/stages/{stage}/clear   : 指定ステージの出力を消してやり直せる状態にする.

実行 (再生成) は既存の POST /api/projects/{id}/rerun/{stage} を使う.
"""

from __future__ import annotations

import json
import shutil

from fastapi import APIRouter, HTTPException

from ..domain import project as project_domain
from ..domain.artifacts import manifest_path
from ..domain.pipeline_state import STAGE_ORDER, STAGE_TO_STATE, PipelineState, StageName
from ..infrastructure.database import get_db
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

    proj_dir = _project_dir(project_id)
    stages = []
    for st in STAGE_ORDER:
        mf = manifest_path(proj_dir, st.value)
        has_output = mf.exists()
        params = None
        if has_output:
            try:
                params = json.loads(mf.read_text()).get("params")
            except (OSError, json.JSONDecodeError):
                params = None
        run = latest.get(st.value, {})
        stages.append(
            {
                "stage": st.value,
                "has_output": has_output,
                "status": run.get("status"),  # None = 未実行
                "error_text": run.get("error_text"),
                "job_id": run.get("job_id"),
                "started_at": run.get("started_at"),
                "finished_at": run.get("finished_at"),
                "params": params,
            }
        )
    return {"project_id": project_id, "state": p.state.value, "stages": stages}


@router.post("/api/projects/{project_id}/stages/{stage}/clear")
async def clear_stage(project_id: str, stage: str) -> dict:
    if stage not in [s.value for s in StageName]:
        raise HTTPException(status_code=400, detail=f"unknown stage: {stage}")
    db = get_db()
    p = await project_domain.get_project(db, project_id)
    if p is None:
        raise HTTPException(status_code=404, detail="project not found")

    proj_dir = _project_dir(project_id)
    # 出力ディレクトリ + tmp + manifest を削除.
    for d in (proj_dir / stage, proj_dir / f".{stage}.tmp"):
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
    mf = manifest_path(proj_dir, stage)
    if mf.exists():
        mf.unlink()
    await db.conn.execute(
        "DELETE FROM stage_run WHERE project_id=? AND stage=?", (project_id, stage)
    )
    await db.conn.commit()

    # state を「先頭から連続して manifest が残っている最後のステージ」に戻す.
    new_state = PipelineState.CREATED
    for st in STAGE_ORDER:
        if manifest_path(proj_dir, st.value).exists():
            new_state = STAGE_TO_STATE[st]
        else:
            break
    await db.conn.execute(
        "UPDATE project SET state=?, updated_at=datetime('now') WHERE id=?",
        (new_state.value, project_id),
    )
    await db.conn.commit()
    return {"cleared": stage, "state": new_state.value}


@router.post("/api/projects/{project_id}/clear-outputs")
async def clear_all_outputs(project_id: str) -> dict:
    """処理済みの中間成果物を全ステージ分クリアする (パラメータは保持. UI 側の状態なので不変)."""
    db = get_db()
    p = await project_domain.get_project(db, project_id)
    if p is None:
        raise HTTPException(status_code=404, detail="project not found")
    proj_dir = _project_dir(project_id)
    for st in STAGE_ORDER:
        for d in (proj_dir / st.value, proj_dir / f".{st.value}.tmp"):
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
        mf = manifest_path(proj_dir, st.value)
        if mf.exists():
            mf.unlink()
    await db.conn.execute("DELETE FROM stage_run WHERE project_id=?", (project_id,))
    await db.conn.execute(
        "UPDATE project SET state=?, updated_at=datetime('now') WHERE id=?",
        (PipelineState.CREATED.value, project_id),
    )
    await db.conn.commit()
    return {"cleared": "all", "state": PipelineState.CREATED.value}
