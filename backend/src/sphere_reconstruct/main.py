"""FastAPI エントリポイント.

    uv run uvicorn sphere_reconstruct.main:app --reload --host 127.0.0.1 --port 8787

- SQLite を workspace/state.db に置く.
- JobSupervisor を lifespan で初期化.
- frontend/dist があれば React SPA を同一 origin で配信する.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import events, jobs, previews, projects, stages, system
from .api import settings as settings_api
from .infrastructure.database import close_db, init_db
from .job_supervisor import init_supervisor, shutdown_supervisor
from .settings import get_settings, workspace_root


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    settings = get_settings()
    logging.basicConfig(level=settings.log.level.upper())
    logger = logging.getLogger("sphere_reconstruct")

    db_path = workspace_root() / "state.db"
    db = await init_db(db_path)
    init_supervisor(db, db_path)

    # クラッシュ復旧: 前回のプロセスが死んだ時点で running/queued だった job は,
    # その Worker プロセスがもう存在しないので 'failed' (interrupted) にする.
    # 成果物はステージ単位で原子的に確定しているので, 再実行すれば完了済みステージは
    # キャッシュヒットで飛ばし, 中断ステージから再開される.
    async with db.transaction() as conn:
        await conn.execute(
            "UPDATE job SET status='failed', error_text='interrupted by restart' "
            "WHERE status IN ('running', 'queued')"
        )
        await conn.execute(
            "UPDATE stage_run SET status='failed', error_text='interrupted by restart' WHERE status='running'"
        )

    logger.info("sphere-reconstruct backend started (db=%s)", db_path)

    try:
        yield
    finally:
        await shutdown_supervisor()
        await close_db()
        logger.info("sphere-reconstruct backend stopped")


app = FastAPI(
    title="sphere-reconstruct",
    version="0.0.1",
    lifespan=lifespan,
)

# 開発中は Vite dev server (localhost:5173) から fetch する.
# 本番配信 (FastAPI が静的 build を配る) では同一 origin なので CORS 不要になる.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

app.include_router(projects.router)
app.include_router(jobs.router)
app.include_router(events.router)
app.include_router(settings_api.router)
app.include_router(previews.router)
app.include_router(stages.router)
app.include_router(system.router)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


def _mount_frontend() -> None:
    """frontend/dist が存在すればそれを静的配信する (Electron 不要の単体アプリ).

    dev では Vite dev server を使うためこのマウントは不要 (dist が無ければ何もしない).
    SPA なので, API 以外の未知パスは index.html にフォールバックする.
    """
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    # backend/src/sphere_reconstruct/main.py -> repo ルート/frontend/dist
    repo_root = Path(__file__).resolve().parents[3]
    dist = repo_root / "frontend" / "dist"
    if not dist.is_dir():
        return

    app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def spa_fallback(full_path: str) -> FileResponse:
        # API パスはここに来ない (先に登録済みルータが処理する).
        candidate = dist / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(dist / "index.html")


_mount_frontend()
