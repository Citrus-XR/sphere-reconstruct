"""Worker サブプロセスのエントリポイント.

spawn_worker() から呼ばれる. FastAPI プロセスとは別プロセスなので, ここで
初めて重い import (Torch など) が走る想定. 現状は insta360 モジュールのみ.

規約:
- 例外は必ず event テーブルに書き出してから re-raise (プロセスは非ゼロ終了).
- 終了時に job テーブルの status を更新するのは Engine の外, job_supervisor 側.
"""

from __future__ import annotations

import sqlite3
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# stages パッケージ import で `@register` が走り, レジストリが埋まる.
from . import stages as _stages  # noqa: F401
from .domain.pipeline_state import StageName, requested_stage_plan
from .pipeline.engine import Engine


def run_pipeline_entry(
    *,
    db_path: str,
    workspace_root: str,
    project_id: str,
    job_id: str,
    stage: str | None = None,
    params_by_stage: dict[str, dict[str, Any]] | None = None,
    skip: list[str] | None = None,
) -> None:
    """Worker のエントリ. spawn 経由で呼ばれるため引数は全て pickle 可能な素の型."""
    db = Path(db_path)
    ws = Path(workspace_root)

    _update_job_status(db, job_id, "running", started=True)

    engine: Engine | None = None
    try:
        engine = Engine(db, ws, project_id, job_id)
        if stage:
            target = StageName(stage)
            for planned in requested_stage_plan(target):
                engine.run_stage(planned, (params_by_stage or {}).get(planned.value, {}))
        else:
            typed_params = {StageName(k): v for k, v in (params_by_stage or {}).items()}
            skip_set = {StageName(s) for s in (skip or [])}
            engine.run_all(typed_params, skip=skip_set)
    except BaseException as e:
        tb = traceback.format_exc()
        _emit_error(db, job_id, project_id, f"{type(e).__name__}: {e}\n{tb}")
        _update_job_status(db, job_id, "failed", error=f"{type(e).__name__}: {e}", finished=True)
        raise
    else:
        _update_job_status(db, job_id, "succeeded", finished=True)
    finally:
        if engine is not None:
            engine.close()


def _emit_error(db_path: Path, job_id: str, project_id: str, message: str) -> None:
    try:
        conn = sqlite3.connect(str(db_path), isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(
            """
            INSERT INTO event (job_id, project_id, stage, level, message, progress, ts)
            VALUES (?, ?, NULL, 'error', ?, NULL, ?)
            """,
            (job_id, project_id, message, _iso_now()),
        )
        conn.close()
    except Exception:
        # 最終手段のログ出力先すら死んでたら諦める.
        pass


def _update_job_status(
    db_path: Path,
    job_id: str,
    status: str,
    *,
    started: bool = False,
    finished: bool = False,
    error: str | None = None,
) -> None:
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL;")
    fields = ["status=?"]
    vals: list[Any] = [status]
    if started:
        fields.append("started_at=?")
        vals.append(_iso_now())
    if finished:
        fields.append("finished_at=?")
        vals.append(_iso_now())
    if error is not None:
        fields.append("error_text=?")
        vals.append(error)
    vals.append(job_id)
    conn.execute(f"UPDATE job SET {', '.join(fields)} WHERE id=?", vals)
    conn.close()


def _iso_now() -> str:
    return datetime.now(UTC).isoformat()
