"""SQLite (aiosqlite) データストア.

project / job / stage_run / event の 4 テーブル. WAL モードで運用.
大きな成果物 (画像, mask, 点群) はファイルシステム側に置き, DB には path とハッシュだけを持つ.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite

_SCHEMA = """
CREATE TABLE IF NOT EXISTS project (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    source_kind  TEXT,          -- 'insv' | 'erp_video' | 'erp_images' | NULL
    source_path  TEXT,          -- ソースへの絶対パス. 大きなファイルは複製しない.
    state        TEXT NOT NULL, -- pipeline_state.PipelineState の value
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS job (
    id           TEXT PRIMARY KEY,
    project_id   TEXT NOT NULL REFERENCES project(id) ON DELETE CASCADE,
    kind         TEXT NOT NULL, -- 'run_pipeline' | 'rerun_stage'
    stage        TEXT,          -- rerun 対象ステージ名
    status       TEXT NOT NULL, -- 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled'
    created_at   TEXT NOT NULL,
    started_at   TEXT,
    finished_at  TEXT,
    error_text   TEXT,
    pid          INTEGER        -- Worker プロセス PID
);

CREATE INDEX IF NOT EXISTS idx_job_project ON job(project_id);
CREATE INDEX IF NOT EXISTS idx_job_status ON job(status);

CREATE TABLE IF NOT EXISTS stage_run (
    id            TEXT PRIMARY KEY,
    project_id    TEXT NOT NULL REFERENCES project(id) ON DELETE CASCADE,
    job_id        TEXT REFERENCES job(id) ON DELETE SET NULL,
    stage         TEXT NOT NULL, -- ステージ名 (inspect_source 等)
    impl_version  TEXT NOT NULL, -- 実装バージョン. 変わったら invalidate.
    params_hash   TEXT NOT NULL,
    inputs_hash   TEXT NOT NULL,
    status        TEXT NOT NULL, -- 'running' | 'succeeded' | 'failed' | 'cancelled'
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    manifest_path TEXT,          -- artifacts manifest JSON へのパス
    error_text    TEXT
);

CREATE INDEX IF NOT EXISTS idx_stage_run_project ON stage_run(project_id, stage);

CREATE TABLE IF NOT EXISTS event (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      TEXT REFERENCES job(id) ON DELETE CASCADE,
    project_id  TEXT REFERENCES project(id) ON DELETE CASCADE,
    stage       TEXT,
    level       TEXT NOT NULL, -- 'debug' | 'info' | 'warn' | 'error'
    message     TEXT NOT NULL,
    progress    REAL,          -- 0.0 - 1.0 (nullable)
    ts          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_event_job ON event(job_id, id);
"""


class Database:
    """aiosqlite の薄いラッパ. 常に単一の接続を握って WAL で使う."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self._path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL;")
        await self._conn.execute("PRAGMA foreign_keys=ON;")
        await self._conn.executescript(_SCHEMA)
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database is not connected. call connect() first.")
        return self._conn

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[aiosqlite.Connection]:
        """明示的なトランザクション. with 抜けで commit, 例外なら rollback."""
        conn = self.conn
        try:
            yield conn
            await conn.commit()
        except BaseException:
            await conn.rollback()
            raise


# アプリ全体で 1 インスタンス.
_db: Database | None = None


async def init_db(path: Path) -> Database:
    global _db
    if _db is not None:
        raise RuntimeError("Database is already initialised.")
    _db = Database(path)
    await _db.connect()
    return _db


async def close_db() -> None:
    global _db
    if _db is not None:
        await _db.close()
        _db = None


def get_db() -> Database:
    if _db is None:
        raise RuntimeError("Database is not initialised.")
    return _db
