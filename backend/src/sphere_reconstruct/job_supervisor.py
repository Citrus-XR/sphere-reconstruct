"""Job supervisor.

FastAPI プロセス側で Job のライフサイクルを管理する.

- POST /api/projects/{id}/run が来ると Job を作り, Worker サブプロセスを spawn.
- Job ID とプロセスハンドルを保持し, cancel API で殺せるようにする.
- Worker 終了は非同期タスクで監視し, 事後処理 (無し. Worker 側で job.status を書く).
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from pathlib import Path

from .infrastructure.database import Database
from .infrastructure.processes import WorkerHandle, cancel, spawn_worker, wait_for
from .settings import get_settings, workspace_root


class JobSupervisor:
    """プロセス内で唯一のインスタンス. アプリライフサイクルに紐付ける."""

    def __init__(self, db: Database, db_path: Path) -> None:
        self._db = db
        self._db_path = db_path
        self._handles: dict[str, WorkerHandle] = {}
        self._tasks: dict[str, asyncio.Task[int]] = {}

    async def enqueue_run_pipeline(
        self,
        *,
        project_id: str,
        stage: str | None = None,
        params_by_stage: dict[str, dict] | None = None,
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

        from .worker_entry import run_pipeline_entry  # 遅延 import で subprocess pickle 用.

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
            },
        )
        # pid を job テーブルへ書き込む.
        async with self._db.transaction() as conn:
            await conn.execute(
                "UPDATE job SET pid=? WHERE id=?", (handle.pid, job_id)
            )
        self._handles[job_id] = handle
        task = asyncio.create_task(self._await_worker(job_id, handle))
        self._tasks[job_id] = task
        return job_id

    async def _await_worker(self, job_id: str, handle: WorkerHandle) -> int:
        try:
            return await wait_for(handle)
        finally:
            self._handles.pop(job_id, None)
            self._tasks.pop(job_id, None)

    def cancel_job(self, job_id: str) -> bool:
        handle = self._handles.get(job_id)
        if handle is None:
            return False
        cancel(handle, grace_seconds=3.0)
        return True

    async def shutdown(self) -> None:
        # プロセス終了時に走る. 未完了 Worker は kill.
        for h in list(self._handles.values()):
            cancel(h, grace_seconds=1.0)
        for t in list(self._tasks.values()):
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass


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


def _unused_settings_touch() -> None:
    # settings を import しておかないと循環回避のための遅延 import が忘れられがち.
    _ = get_settings()
