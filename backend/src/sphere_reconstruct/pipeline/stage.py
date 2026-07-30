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
from threading import Lock
from time import monotonic
from typing import Any

from ..domain.artifacts import FileRef, StageManifest
from ..domain.pipeline_state import StageName
from ..domain.source import MediaKind, Projection, SourceRole


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
    tick_min_interval: float = 0.1
    tick_min_progress: float = 0.0025

    def __post_init__(self) -> None:
        self._last_seen_progress: float | None = None
        self._last_emitted_progress: float | None = None
        self._last_activity_tick_at = float("-inf")
        self._last_numeric_tick_at = float("-inf")
        self._progress_lock = Lock()

    def _send(
        self,
        level: str,
        progress: float | None,
        message: str,
        key: str | None,
        args: dict | None,
        kind: str,
    ) -> None:
        now = monotonic()
        with self._progress_lock:
            if progress is not None:
                if not 0.0 <= progress <= 1.0:
                    raise ValueError(f"stage progress must be within 0..1: {progress}")
                if self._last_seen_progress is not None and progress < self._last_seen_progress - 1e-9:
                    raise ValueError(
                        f"stage progress moved backwards: {self._last_seen_progress} -> {progress}"
                    )
                self._last_seen_progress = progress

            if kind == "progress":
                if progress is None:
                    interval_ready = now - self._last_activity_tick_at >= self.tick_min_interval
                    if not interval_ready:
                        return
                elif progress < 1.0:
                    interval_ready = now - self._last_numeric_tick_at >= self.tick_min_interval
                    progress_ready = (
                        self._last_emitted_progress is None
                        or progress - self._last_emitted_progress >= self.tick_min_progress
                    )
                    if not interval_ready or not progress_ready:
                        return

            self._emit(level, progress, message, key, args, kind)
            if progress is not None:
                self._last_emitted_progress = progress
            if kind == "progress":
                if progress is None:
                    self._last_activity_tick_at = now
                else:
                    self._last_numeric_tick_at = now

    def info(
        self, message: str, progress: float | None = None, *, key: str | None = None, args: dict | None = None
    ) -> None:
        self._send("info", progress, message, key, args, "log")

    def warn(
        self, message: str, progress: float | None = None, *, key: str | None = None, args: dict | None = None
    ) -> None:
        self._send("warn", progress, message, key, args, "log")

    def error(
        self, message: str, progress: float | None = None, *, key: str | None = None, args: dict | None = None
    ) -> None:
        self._send("error", progress, message, key, args, "log")

    def tick(
        self,
        progress: float | None = None,
        *,
        message: str = "",
        key: str | None = None,
        args: dict | None = None,
    ) -> None:
        """進捗のみの一時イベント (kind=progress). Console には出さず, 環形インジケータだけ更新する."""
        self._send("info", progress, message, key, args, "progress")


@dataclass(frozen=True)
class ProgressSpan:
    """Stage 全体の一部へ local 0..1 progress を単調な絶対値として写像する。"""

    reporter: ProgressReporter
    low: float
    high: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.low <= self.high <= 1.0:
            raise ValueError(f"invalid progress span: {self.low}..{self.high}")

    def value(self, fraction: float) -> float:
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"progress fraction must be within 0..1: {fraction}")
        return self.low + (self.high - self.low) * fraction

    def child(self, low: float, high: float) -> ProgressSpan:
        return ProgressSpan(self.reporter, self.value(low), self.value(high))

    def tick(
        self,
        fraction: float,
        *,
        message: str,
        key: str | None = None,
        args: dict | None = None,
    ) -> None:
        self.reporter.tick(self.value(fraction), message=message, key=key, args=args)


@dataclass
class SourceContext:
    id: str
    label: str
    role: SourceRole
    adapter: str
    media_kind: MediaKind
    projection: Projection
    path: Path
    ordinal: int
    enabled: bool


@dataclass
class StageContext:
    project_id: str
    project_dir: Path  # workspace/projects/<id>
    stage_out_dir: Path  # 出力書き込み先 (tmp, 成功後に atomic replace)
    params: dict[str, Any]
    sources: tuple[SourceContext, ...]
    progress: ProgressReporter
    resolved_inputs: list[FileRef] | None = None

    @property
    def primary_source(self) -> SourceContext | None:
        return next((source for source in self.sources if source.role == SourceRole.PRIMARY), None)

    def inputs_for(self, stage: Stage) -> list[FileRef]:
        if self.resolved_inputs is None:
            self.resolved_inputs = stage.collect_inputs(self)
        return self.resolved_inputs


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
