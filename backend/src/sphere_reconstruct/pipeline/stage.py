"""Stage 基底クラス.

各 Stage 実装は:
- name: StageName
- impl_version: 実装が変わるたびインクリメント
- collect_inputs(project) -> list[FileRef]
- normalize_params(raw) -> dict
- execute(ctx: StageContext) -> StageManifest

FastAPI プロセスからは import されない (Worker 側専用). 重い依存は
各実装ファイル内で lazy import する.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..domain.artifacts import FileRef, StageManifest
from ..domain.pipeline_state import StageName


@dataclass
class ProgressReporter:
    """Worker 内から Job / stage の進捗を出すためのコールバック.

    実体は Worker entrypoint で差し込む (SQLite の event テーブルへ書き込む実装).

    message は常にレンダリング済みのフォールバック文字列. key/args を渡すと i18n キー +
    補間引数として一緒に保存し, フロントが表示言語で翻訳する (message は未 key の呼び出しや
    翻訳欠落時のフォールバックとして残す).

    info/warn/error は「ログ行」(kind=log) として Console に出す. 進捗の細かい更新は tick()
    (kind=progress) を使う — これは環形インジケータの駆動のみで, Console には出さない (spam 防止).
    開始 / 完了 / 失敗など残すべき区切りは info/warn/error を使うこと.
    """

    _emit: Any  # Callable[[str, float|None, str, str|None, dict|None, str], None]

    def info(
        self, message: str, progress: float | None = None, *, key: str | None = None, args: dict | None = None
    ) -> None:
        self._emit("info", progress, message, key, args, "log")

    def warn(
        self, message: str, progress: float | None = None, *, key: str | None = None, args: dict | None = None
    ) -> None:
        self._emit("warn", progress, message, key, args, "log")

    def error(
        self, message: str, progress: float | None = None, *, key: str | None = None, args: dict | None = None
    ) -> None:
        self._emit("error", progress, message, key, args, "log")

    def tick(
        self,
        progress: float | None = None,
        *,
        message: str = "",
        key: str | None = None,
        args: dict | None = None,
    ) -> None:
        """進捗のみの一時イベント (kind=progress). Console には出さず, 環形インジケータだけ更新する."""
        self._emit("info", progress, message, key, args, "progress")


@dataclass
class StageContext:
    project_id: str
    project_dir: Path  # workspace/projects/<id>
    stage_out_dir: Path  # 出力書き込み先 (tmp, 成功後に atomic replace)
    params: dict[str, Any]
    source_path: Path | None
    source_kind: str | None
    progress: ProgressReporter


class Stage(ABC):
    name: StageName
    impl_version: str = "0"

    @abstractmethod
    def collect_inputs(self, ctx: StageContext) -> list[FileRef]:
        """このステージが入力として依存するファイル一覧. hash 込みで返す."""

    def normalize_params(self, raw: dict[str, Any]) -> dict[str, Any]:
        """パラメータ正規化. default 値差し込みなど. 冪等判定に使うため決定的に."""
        return dict(raw)

    @abstractmethod
    def execute(self, ctx: StageContext) -> StageManifest:
        """実処理. ctx.stage_out_dir に成果物を書き, manifest を返す.

        呼び出し側 (engine) が atomic replace, DB 更新を担当する. Stage は
        自分で final directory を触らない.
        """


def new_manifest(stage: StageName, impl_version: str) -> StageManifest:
    return StageManifest(
        stage=stage.value,
        impl_version=impl_version,
        started_at=datetime.now(UTC),
    )
