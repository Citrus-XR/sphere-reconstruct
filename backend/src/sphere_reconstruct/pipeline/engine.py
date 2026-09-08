"""パイプライン Engine (Worker プロセス側で動く).

責務:
- 指定 project について, 次に実行すべきステージを決める.
- 各ステージについて: 入力ハッシュ / パラメータハッシュ / 実装バージョンを比較,
  変わっていなければスキップ. 変わっていれば以降のステージを invalidate.
- 実行時は tmp ディレクトリに出力, 成功後 atomic replace.
- 進捗 / ログを SQLite の event テーブルに書き込む.
- キャンセルは Worker プロセス自体を殺すことで対応するため, 特別な cooperative
  cancel は今のところ実装しない (ステージ側で長時間ブロックしない工夫は必要).

このモジュールは Worker 側で使う想定. FastAPI からは触らない.

Note: ステージ実装内から `progress.info(...)` が別スレッドから呼ばれることがあるため
(例: ffmpeg 並列抽出), SQLite connection は check_same_thread=False + Lock で使う.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..domain.artifacts import StageManifest, manifest_path
from ..domain.pipeline_state import STAGE_ORDER, StageName
from ..domain.source import MediaKind, Projection, SourceRole
from ..infrastructure.filesystem import atomic_replace_dir
from .errors import LocalizedError
from .invalidation import clear_stale, derive_pipeline_state, invalidate_from
from .manifest import get as get_stage_cls
from .stage import ProgressReporter, SourceContext, StageContext


class PipelineError(RuntimeError):
    pass


class Engine:
    """Worker プロセス内で使う同期 API. DB は同期 sqlite3 で開く.

    FastAPI 側は独立した aiosqlite 接続を使い、SQLite WAL で変更を共有する。
    Worker 内の別 thread からの進捗書き込みは Lock で直列化する。
    """

    def __init__(self, db_path: Path, workspace_root: Path, project_id: str, job_id: str) -> None:
        self._db_path = db_path
        self._workspace_root = workspace_root
        self._project_id = project_id
        self._job_id = job_id
        # ステージ実装が別スレッドから progress を出すことがあるので check_same_thread=False.
        # 全 write を Lock で直列化する (SQLite の書き込み衝突を回避).
        self._conn = sqlite3.connect(str(db_path), isolation_level=None, check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")

    def _execute(self, sql: str, params: tuple = ()) -> None:
        with self._lock:
            self._conn.execute(sql, params)

    def _query_one(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        with self._lock:
            self._conn.row_factory = sqlite3.Row
            cur = self._conn.execute(sql, params)
            return cur.fetchone()

    # -- project 情報の読み取り ---------------------------------------------------
    def project_dir(self) -> Path:
        return self._workspace_root / "projects" / self._project_id

    def _project_row(self) -> sqlite3.Row:
        row = self._query_one("SELECT * FROM project WHERE id=?", (self._project_id,))
        if row is None:
            raise PipelineError(f"project {self._project_id} not found")
        return row

    def _project_sources(self) -> tuple[SourceContext, ...]:
        with self._lock:
            self._conn.row_factory = sqlite3.Row
            rows = self._conn.execute(
                "SELECT * FROM project_source WHERE project_id=? AND enabled=1 "
                "ORDER BY ordinal, created_at, id",
                (self._project_id,),
            ).fetchall()
        return tuple(
            SourceContext(
                id=row["id"],
                label=row["label"],
                role=SourceRole(row["role"]),
                adapter=str(row["adapter"]),
                media_kind=MediaKind(row["media_kind"]),
                projection=Projection(row["projection"]),
                path=Path(row["path"]),
                ordinal=int(row["ordinal"]),
                enabled=bool(row["enabled"]),
            )
            for row in rows
        )

    # -- ステージ実行 -------------------------------------------------------------
    def run_stage(self, stage_name: StageName, params: dict[str, Any] | None = None) -> None:
        params = params or {}
        stage_cls = get_stage_cls(stage_name)
        stage = stage_cls()

        project_dir = self.project_dir()
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "manifests").mkdir(parents=True, exist_ok=True)

        self._project_row()
        sources = self._project_sources()

        # 出力先は tmp ディレクトリ, 成功時に final に atomic replace.
        final_dir = project_dir / stage_name.value
        tmp_dir = project_dir / f".{stage_name.value}.tmp"

        stage_run_id = str(uuid.uuid4())
        started_at = _iso_now()

        def emit(
            level: str,
            progress: float | None,
            message: str,
            key: str | None = None,
            args: dict | None = None,
            kind: str = "log",
        ) -> None:
            self._execute(
                """
                INSERT INTO event
                    (job_id, project_id, stage, level, message, msg_key, msg_args, progress, kind, ts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self._job_id,
                    self._project_id,
                    stage_name.value,
                    level,
                    message,
                    key,
                    json.dumps(args, ensure_ascii=False) if args is not None else None,
                    progress,
                    kind,
                    _iso_now(),
                ),
            )

        reporter = ProgressReporter(_emit=emit)
        ctx = StageContext(
            project_id=self._project_id,
            project_dir=project_dir,
            stage_out_dir=tmp_dir,
            params=stage.normalize_params(params),
            sources=sources,
            progress=reporter,
        )

        # Fingerprint 中も UI が running / activity を復元できるよう先に lifecycle row を作る。
        self._execute(
            """
            INSERT INTO stage_run
                (id, project_id, job_id, stage, impl_version, params_hash, inputs_hash,
                 status, started_at, manifest_path, error_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'running', ?, NULL, NULL)
            """,
            (
                stage_run_id,
                self._project_id,
                self._job_id,
                stage_name.value,
                stage.impl_version,
                "",
                "",
                started_at,
            ),
        )

        try:
            if tmp_dir.exists():
                reporter.info(
                    f"removing stale temporary output: {stage_name.value}",
                    key="log.stage_cleanup",
                    args={"stage": stage_name.value},
                )
                shutil.rmtree(tmp_dir)
            tmp_dir.mkdir(parents=True)
            # 冪等チェックと実行で同じ fingerprint を共有し、大きな source/database を二重 hash しない。
            reporter.info(
                f"preparing stage inputs: {stage_name.value}",
                key="log.stage_preflight",
                args={"stage": stage_name.value},
            )
            current_inputs = ctx.inputs_for(stage)
            cached = self._cached_matches(stage, ctx, current_inputs)
            if cached is not None:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                self._refresh_project_state()
                self._execute(
                    """
                    UPDATE stage_run SET status='succeeded', finished_at=?, inputs_hash=?,
                        params_hash=?, manifest_path=? WHERE id=?
                    """,
                    (
                        _iso_now(),
                        cached.inputs_hash,
                        cached.params_hash,
                        str(manifest_path(project_dir, stage_name.value)),
                        stage_run_id,
                    ),
                )
                reporter.info(
                    f"stage {stage_name.value} skipped (cache hit)",
                    progress=1.0,
                    key="log.cache_hit",
                    args={"stage": stage_name.value},
                )
                return

            manifest = stage.execute(ctx)
            manifest.compute_hashes()
            manifest.finished_at = datetime.now(UTC)

            reporter.tick(
                progress=0.99,
                message=f"publishing stage artifacts: {stage_name.value}",
                key="log.stage_publishing",
                args={"stage": stage_name.value},
            )
            atomic_replace_dir(tmp_dir, final_dir)
            self._invalidate_downstream(stage_name)
            manifest_file = manifest_path(project_dir, stage_name.value)
            temporary_manifest = manifest_file.with_name(f".{stage_name.value}.{stage_run_id}.tmp")
            try:
                manifest.dump(temporary_manifest)
                temporary_manifest.replace(manifest_file)
            finally:
                temporary_manifest.unlink(missing_ok=True)
            clear_stale(project_dir, stage_name)
            self._refresh_project_state()
            self._execute(
                """
                UPDATE stage_run SET
                    status='succeeded',
                    finished_at=?,
                    inputs_hash=?,
                    params_hash=?,
                    manifest_path=?
                WHERE id=?
                """,
                (
                    _iso_now(),
                    manifest.inputs_hash,
                    manifest.params_hash,
                    str(manifest_file),
                    stage_run_id,
                ),
            )
            reporter.info(
                f"stage {stage_name.value} published",
                progress=1.0,
                key="log.stage_published",
                args={"stage": stage_name.value},
            )
        except BaseException as e:
            self._execute(
                """
                UPDATE stage_run SET status='failed', finished_at=?, error_text=? WHERE id=?
                """,
                (_iso_now(), _short_error(e), stage_run_id),
            )
            if isinstance(e, LocalizedError):
                reporter.error(str(e), key=e.key, args=e.formatting_args)
            else:
                reporter.error(
                    f"stage {stage_name.value} failed: {_short_error(e)}",
                    key="log.stage_failed",
                    args={"stage": stage_name.value, "error": _short_error(e)},
                )
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

    def _cached_matches(
        self,
        stage,
        ctx: StageContext,
        current_inputs: list,
    ) -> StageManifest | None:
        """既存 manifest が現在の入力/パラメータ/実装バージョンと一致するならそれを返す."""
        mf_path = manifest_path(self.project_dir(), stage.name.value)
        if not mf_path.exists():
            return None
        existing = StageManifest.load(mf_path)
        if existing.impl_version != stage.impl_version:
            return None
        # 現在の入力ハッシュを計算して比較する. 入力ファイル走査コスト vs. 実行コストを考えると許容.
        temp = StageManifest(
            stage=stage.name.value,
            impl_version=stage.impl_version,
            started_at=datetime.now(UTC),
            inputs=current_inputs,
            params=ctx.params,
        )
        temp.compute_hashes()
        if temp.inputs_hash != existing.inputs_hash:
            return None
        if temp.params_hash != existing.params_hash:
            return None
        for output in existing.outputs:
            path = self.project_dir() / output.path
            if not path.is_file() or path.stat().st_size != output.size:
                return None
        return existing

    def _invalidate_downstream(self, stage: StageName) -> None:
        invalidate_from(self.project_dir(), stage, include_self=False)

    def _refresh_project_state(self) -> None:
        state = derive_pipeline_state(self.project_dir())
        self._execute(
            "UPDATE project SET state=?, updated_at=? WHERE id=?",
            (state.value, _iso_now(), self._project_id),
        )

    # -- 全ステージ実行 -----------------------------------------------------------
    def run_all(
        self,
        params_by_stage: dict[StageName, dict[str, Any]] | None = None,
        skip: set[StageName] | None = None,
    ) -> None:
        params_by_stage = params_by_stage or {}
        skip = set(skip or set())
        for st in STAGE_ORDER:
            if st in skip:
                continue
            self.run_stage(st, params_by_stage.get(st, {}))

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _iso_now() -> str:
    return datetime.now(UTC).isoformat()


def _short_error(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"
