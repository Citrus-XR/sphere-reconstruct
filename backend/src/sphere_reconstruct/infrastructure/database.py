"""SQLite (aiosqlite) データストア.

project / job / stage_run / event の 4 テーブル. WAL モードで運用.
大きな成果物 (画像, mask, 点群) はファイルシステム側に置き, DB には path とハッシュだけを持つ.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path

import aiosqlite

_SCHEMA = """
CREATE TABLE IF NOT EXISTS project (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    source_kind  TEXT,          -- migration 専用旧列. runtime は project_source を使う.
    source_path  TEXT,          -- migration 専用旧列.
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
        # 既存 DB (ALTER 前に作られたファイル) にも追加カラムを足す. マイグレーション層が
        # 無いので冪等な ADD COLUMN で吸収する (存在すれば OperationalError を握りつぶす).
        for col, decl in (("msg_key", "TEXT"), ("msg_args", "TEXT"), ("kind", "TEXT NOT NULL DEFAULT 'log'")):
            with suppress(aiosqlite.OperationalError):
                await self._conn.execute(f"ALTER TABLE event ADD COLUMN {col} {decl}")
        # 廃止済みの独立画像処理 branch は camera solve を変更しなかったため、旧要約状態は
        # main branch の最終成果である aligned へ一度だけ正規化する。
        await self._conn.execute("UPDATE project SET state='aligned' WHERE state='denoised'")
        await self._conn.execute("UPDATE project SET state='prepared' WHERE state='reprojected'")
        # 単一 source 列を正規化 table へ移し、以後は project_source だけを正とする。
        await self._conn.execute(
            """
            INSERT OR IGNORE INTO project_source
                (id, project_id, label, role, adapter, media_kind, projection, path,
                 ordinal, enabled, created_at, updated_at)
            SELECT
                'legacy-' || id,
                id,
                CASE
                    WHEN instr(replace(source_path, '\\', '/'), '/') > 0
                    THEN replace(source_path, '\\', '/')
                    ELSE source_path
                END,
                'primary',
                CASE source_kind
                    WHEN 'insv' THEN 'insta360_insv'
                    WHEN 'erp_video' THEN 'generic_video'
                    ELSE 'generic_images'
                END,
                CASE WHEN source_kind='erp_images' THEN 'images' ELSE 'video' END,
                CASE WHEN source_kind='insv' THEN 'dual_fisheye' ELSE 'equirectangular' END,
                source_path,
                0,
                1,
                created_at,
                updated_at
            FROM project
            WHERE source_path IS NOT NULL AND source_kind IS NOT NULL
            """
        )
        await self._conn.execute("UPDATE project SET source_kind=NULL, source_path=NULL")
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
