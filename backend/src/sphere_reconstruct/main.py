"""FastAPI エントリポイント.

    uv run uvicorn sphere_reconstruct.main:app --reload --host 127.0.0.1 --port 8787

- SQLite を workspace/state.db に置く.
- JobSupervisor を lifespan で初期化.
- Frontend (React 静的 build) の配信は Phase 7 で追加予定.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import events, jobs, previews, projects, settings as settings_api
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


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
