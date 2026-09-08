"""SQLite (aiosqlite) データストア.

Project / job / stage_run / event と関連設定を WAL モードで管理する。
大きな成果物 (画像, mask, 点群) はファイルシステム側に置き, DB には path とハッシュだけを持つ.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite

_SCHEMA = """
CREATE TABLE IF NOT EXISTS workspace_preferences (
    id INTEGER PRIMARY KEY CHECK(id=1),
    value_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS project (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    state        TEXT NOT NULL, -- pipeline_state.PipelineState の value
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS project_source (
    id           TEXT PRIMARY KEY,
    project_id   TEXT NOT NULL REFERENCES project(id) ON DELETE CASCADE,
    label        TEXT NOT NULL,
    role         TEXT NOT NULL,
    adapter      TEXT NOT NULL,
    media_kind   TEXT NOT NULL,
    projection   TEXT NOT NULL,
    path         TEXT NOT NULL,
    ordinal      INTEGER NOT NULL,
    enabled      INTEGER NOT NULL DEFAULT 1,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    UNIQUE(project_id, path)
);

CREATE INDEX IF NOT EXISTS idx_project_source_project
ON project_source(project_id, ordinal);

CREATE UNIQUE INDEX IF NOT EXISTS idx_project_source_primary
ON project_source(project_id) WHERE role='primary';

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

CREATE TABLE IF NOT EXISTS job_request (
    job_id                TEXT PRIMARY KEY REFERENCES job(id) ON DELETE CASCADE,
    params_by_stage_json  TEXT NOT NULL,
    skip_json             TEXT NOT NULL
);

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
    message     TEXT NOT NULL, -- レンダリング済みフォールバック文字列 (未 key 化の呼び出しでも壊れない)
    msg_key     TEXT,          -- i18n キー (log.*). フロントが view 時に翻訳する.
    msg_args    TEXT,          -- msg_key の補間引数 (JSON). null 可.
    progress    REAL,          -- 0.0 - 1.0 (nullable)
    kind        TEXT NOT NULL DEFAULT 'log', -- 'log' (Console 表示) | 'progress' (環形のみ, 非表示)
    ts          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_event_job ON event(job_id, id);
"""


class Database:
    """独立 SQL 用と明示 transaction 用の 2 接続を保持する。"""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._conn: aiosqlite.Connection | None = None
        self._transaction_conn: aiosqlite.Connection | None = None
        self._transaction_lock = asyncio.Lock()

    async def connect(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self._path, isolation_level=None)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL;")
        await self._conn.execute("PRAGMA foreign_keys=ON;")
        await self._conn.executescript(_SCHEMA)
        self._transaction_conn = await aiosqlite.connect(self._path, isolation_level=None)
        self._transaction_conn.row_factory = aiosqlite.Row
        await self._transaction_conn.execute("PRAGMA foreign_keys=ON;")

    async def close(self) -> None:
        if self._transaction_conn is not None:
            await self._transaction_conn.close()
            self._transaction_conn = None
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        """単一文の autocommit 接続。複数文の変更は transaction() を使う。"""
        if self._conn is None:
            raise RuntimeError("Database is not connected. call connect() first.")
        return self._conn

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[aiosqlite.Connection]:
        """専用接続を直列化し、他の request の commit / rollback から隔離する。"""
        if self._transaction_conn is None:
            raise RuntimeError("Database is not connected. call connect() first.")
        async with self._transaction_lock:
            conn = self._transaction_conn
            try:
                await conn.execute("BEGIN IMMEDIATE")
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
