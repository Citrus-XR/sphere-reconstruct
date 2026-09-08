"""API の成果物変更と Worker の取得・解放を project 単位で直列化する。"""

import asyncio
from weakref import WeakValueDictionary

from .database import Database

_locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()


class ProjectBusyError(RuntimeError):
    pass


class ProjectNotFoundError(LookupError):
    pass


def project_lock(project_id: str) -> asyncio.Lock:
    lock = _locks.get(project_id)
    if lock is None:
        lock = asyncio.Lock()
        _locks[project_id] = lock
    return lock


async def require_idle_project(db: Database, project_id: str) -> None:
    row = await (
        await db.conn.execute(
            "SELECT project.id, job.id AS active_job FROM project "
            "LEFT JOIN job ON job.project_id=project.id AND job.status IN ('queued','running') "
            "WHERE project.id=? LIMIT 1",
            (project_id,),
        )
    ).fetchone()
    if row is None:
        raise ProjectNotFoundError("project not found")
    if row["active_job"] is not None:
        raise ProjectBusyError("project has a running or queued job")
