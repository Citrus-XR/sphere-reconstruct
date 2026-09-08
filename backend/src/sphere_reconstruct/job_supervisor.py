"""Job supervisor.

FastAPI プロセス側で Job のライフサイクルを管理する.

- POST /api/projects/{id}/run が来ると Job を作り, Worker サブプロセスを spawn.
- Job ID とプロセスハンドルを保持し, cancel API で殺せるようにする.
- Worker の通常終了状態を引き継ぎ、起動失敗・異常終了・cancel と一時出力の回収を管理する。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path

from .infrastructure.database import Database
from .infrastructure.processes import WorkerHandle, cancel, spawn_worker, wait_for
from .infrastructure.project_lock import project_lock, require_idle_project
from .settings import workspace_root

_logger = logging.getLogger(__name__)


class JobSupervisor:
    """プロセス内で唯一のインスタンス. アプリライフサイクルに紐付ける."""

    def __init__(self, db: Database, db_path: Path) -> None:
        self._db = db
        self._db_path = db_path
        self._handles: dict[str, WorkerHandle] = {}
        self._tasks: dict[str, asyncio.Task[int]] = {}
        self._cancelling: set[str] = set()

    async def enqueue_run_pipeline(
        self,
        *,
        project_id: str,
        stage: str | None = None,
        params_by_stage: dict[str, dict] | None = None,
        skip: list[str] | None = None,
    ) -> str:
        async with project_lock(project_id):
            await require_idle_project(self._db, project_id)
            return await self._start_pipeline(
                project_id=project_id, stage=stage, params_by_stage=params_by_stage, skip=skip
            )

    async def _start_pipeline(
        self,
        *,
        project_id: str,
        stage: str | None,
        params_by_stage: dict[str, dict] | None,
        skip: list[str] | None,
    ) -> str:
        job_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()
        async with self._db.transaction() as conn:
            await conn.execute(
                """
                INSERT INTO job (id, project_id, kind, stage, status, created_at)
                VALUES (?, ?, ?, ?, 'queued', ?)
                """,
                (
                    job_id,
                    project_id,
                    "rerun_stage" if stage else "run_pipeline",
                    stage,
                    now,
                ),
            )
            await conn.execute(
                """
                INSERT INTO job_request (job_id, params_by_stage_json, skip_json)
                VALUES (?, ?, ?)
                """,
                (
                    job_id,
                    json.dumps(params_by_stage or {}, ensure_ascii=False, sort_keys=True),
                    json.dumps(skip or [], ensure_ascii=False),
                ),
            )

        try:
            from .worker_entry import run_pipeline_entry

            handle = spawn_worker(
                run_pipeline_entry,
                job_id=job_id,
                kwargs={
                    "db_path": str(self._db_path),
                    "workspace_root": str(workspace_root()),
                    "project_id": project_id,
                    "job_id": job_id,
                    "stage": stage,
                    "params_by_stage": params_by_stage,
                    "skip": skip,
                },
            )
        except Exception as error:
            finished_at = datetime.now(UTC).isoformat()
            diagnostic = f"worker could not start: {type(error).__name__}: {error}"
            async with self._db.transaction() as conn:
                await conn.execute(
                    "UPDATE job SET status='failed', finished_at=?, error_text=? WHERE id=?",
                    (finished_at, diagnostic, job_id),
                )
                await conn.execute(
                    "INSERT INTO event (job_id, project_id, level, message, ts) VALUES (?, ?, 'error', ?, ?)",
                    (job_id, project_id, diagnostic, finished_at),
                )
            raise
        self._handles[job_id] = handle
        task = asyncio.create_task(self._await_worker(job_id, handle))
        self._tasks[job_id] = task
        async with self._db.transaction() as conn:
            await conn.execute("UPDATE job SET pid=? WHERE id=?", (handle.pid, job_id))
        return job_id

    async def _await_worker(self, job_id: str, handle: WorkerHandle) -> int:
        code = -1
        try:
            code = await wait_for(handle)
            return code
        finally:
            self._handles.pop(job_id, None)
            self._tasks.pop(job_id, None)
            # worker がネイティブクラッシュ (CUDA/ONNX の VRAM 不足など) や kill で異常終了すると
            # worker 内の except を通らず job/stage_run が 'running' のまま残る. ここで回収する.
            try:
                await self._finalize_if_orphaned(job_id, code)
            except Exception:
                _logger.exception("Could not finalize job %s after worker exit %s", job_id, code)
            self._cancelling.discard(job_id)
            await self.cleanup_scratch(job_id)

    async def cleanup_scratch(self, job_id: str) -> None:
        cur = await self._db.conn.execute("SELECT project_id FROM job WHERE id=?", (job_id,))
        row = await cur.fetchone()
        if row is None:
            return
        async with project_lock(row["project_id"]):
            await self._cleanup_scratch(job_id, row["project_id"])

    async def _cleanup_scratch(self, job_id: str, project_id: str) -> None:
        active = await (
            await self._db.conn.execute(
                "SELECT 1 FROM job WHERE project_id=? AND id<>? "
                "AND status IN ('queued', 'running') LIMIT 1",
                (project_id, job_id),
            )
        ).fetchone()
        if active is not None:
            return
        project_dir = workspace_root() / "projects" / project_id
        stage_rows = await (
            await self._db.conn.execute(
                "SELECT DISTINCT stage FROM stage_run WHERE job_id=?",
                (job_id,),
            )
        ).fetchall()
        scratch = [
            *(project_dir / f".{stage_row['stage']}.tmp" for stage_row in stage_rows),
            *project_dir.glob(".extract-frames-sharpness-*"),
            *project_dir.glob(".extract-frames-spatial-*"),
        ]

        def cleanup() -> None:
            for path in scratch:
                shutil.rmtree(path, ignore_errors=True)

        await asyncio.to_thread(cleanup)

    async def _finalize_if_orphaned(self, job_id: str, exit_code: int) -> None:
        """worker 終了後, job がまだ running/queued のままなら failed にして原因を通知する."""
        now = datetime.now(UTC).isoformat()
        async with self._db.transaction() as conn:
            cur = await conn.execute("SELECT status, project_id FROM job WHERE id=?", (job_id,))
            row = await cur.fetchone()
            if row is None:
                return
            if job_id in self._cancelling:
                await conn.execute(
                    "UPDATE job SET status='cancelled', finished_at=?, error_text=NULL WHERE id=?",
                    (now, job_id),
                )
                await conn.execute(
                    "UPDATE stage_run SET status='cancelled', finished_at=?, error_text=NULL "
                    "WHERE job_id=? AND status IN ('running', 'failed')",
                    (now, job_id),
                )
                return
            if row["status"] not in ("running", "queued"):
                return  # 正常に終了済み (worker が status を書けた).
            msg = (
                f"worker が異常終了しました (exit code {exit_code}). "
                "ネイティブ subprocess の失敗か手動停止です. stage log と直前の Console を確認し, "
                "GPU memory 不足なら画像上限・特徴数・SAM 解像度を下げてください."
            )
            await conn.execute(
                "UPDATE job SET status='failed', finished_at=?, error_text=? WHERE id=?",
                (now, msg, job_id),
            )
            await conn.execute(
                "UPDATE stage_run SET status='failed', finished_at=?, error_text=? "
                "WHERE job_id=? AND status='running'",
                (now, msg, job_id),
            )
            await conn.execute(
                "INSERT INTO event "
                "(job_id, project_id, stage, level, message, msg_key, msg_args, progress, ts) "
                "VALUES (?, ?, NULL, 'error', ?, ?, ?, NULL, ?)",
                (
                    job_id,
                    row["project_id"],
                    "⛔ " + msg,
                    "log.worker_orphaned",
                    json.dumps({"exit_code": exit_code}),
                    now,
                ),
            )

    async def cancel_job(self, job_id: str) -> bool:
        row = await (
            await self._db.conn.execute("SELECT project_id FROM job WHERE id=?", (job_id,))
        ).fetchone()
        if row is None:
            return False
        async with project_lock(row["project_id"]):
            return await self._cancel_job(job_id)

    async def _cancel_job(self, job_id: str) -> bool:
        handle = self._handles.get(job_id)
        if handle is None:
            return False
        async with self._db.transaction() as conn:
            cur = await conn.execute("SELECT status FROM job WHERE id=?", (job_id,))
            row = await cur.fetchone()
            if row is None or row["status"] not in ("queued", "running"):
                return False
            self._cancelling.add(job_id)
        await asyncio.to_thread(cancel, handle, grace_seconds=3.0)
        await self._finalize_if_orphaned(job_id, -1)
        return True

    async def shutdown(self) -> None:
        # プロセス終了時に走る. 未完了 Worker は kill.
        await asyncio.gather(
            *(asyncio.to_thread(cancel, handle, grace_seconds=1.0) for handle in self._handles.values())
        )
        for t in list(self._tasks.values()):
            t.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t


_supervisor: JobSupervisor | None = None


def init_supervisor(db: Database, db_path: Path) -> JobSupervisor:
    global _supervisor
    if _supervisor is not None:
        raise RuntimeError("supervisor already initialised")
    _supervisor = JobSupervisor(db, db_path)
    return _supervisor


def get_supervisor() -> JobSupervisor:
    if _supervisor is None:
        raise RuntimeError("supervisor not initialised")
    return _supervisor


async def shutdown_supervisor() -> None:
    global _supervisor
    if _supervisor is not None:
        await _supervisor.shutdown()
        _supervisor = None
